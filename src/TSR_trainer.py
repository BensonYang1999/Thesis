import math
import os
import time

import cv2
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from torchinfo import summary

from torch.utils.tensorboard import SummaryWriter

# try:
#     from apex import amp
# except ImportError:
#     raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")


class EdgeAccuracy(torch.nn.Module):
    """
    Measures the accuracy of the edge map
    """

    def __init__(self, threshold=0.5):
        super(EdgeAccuracy, self).__init__()
        self.threshold = threshold

    def __call__(self, inputs, outputs):
        labels = (inputs > self.threshold)
        outputs = (outputs > self.threshold)

        relevant = torch.sum(labels.float())
        selected = torch.sum(outputs.float())

        if relevant == 0 and selected == 0:
            return torch.tensor(1), torch.tensor(1), torch.tensor(1)

        true_positive = ((outputs == labels) * labels).float()
        recall = torch.sum(true_positive) / (relevant + 1e-8)
        precision = torch.sum(true_positive) / (selected + 1e-8)
        f1_score = (2 * precision * recall) / (precision + recall + 1e-8)

        return precision * 100, recall * 100, f1_score * 100


class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1
    lr_decay = False
    warmup_iterations = 375e6
    final_iterations = 260e9
    iterations_per_epoch = 1e5
    # checkpoint settings
    ckpt_path = None
    num_workers = 0  # for DataLoader

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class TrainerForContinuousEdgeLine:
    def __init__(self, model, train_dataset, test_dataset, config, gpu, global_rank, iterations_per_epoch, logger=None):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.iterations_per_epoch = iterations_per_epoch
        self.config = config
        self.device = gpu

        self.model = model
        self.metric = EdgeAccuracy(threshold=0.5)
        self.global_rank = global_rank
        self.train_sampler = DistributedSampler(train_dataset, num_replicas=config.world_size,
                                                rank=global_rank, shuffle=True)
        self.logger = logger

    def save_checkpoint(self, epoch, optim, iterations, validation, edgeF1, lineF1, save_name):
        if self.global_rank == 0:  # Only save in global rank 0
            raw_model = self.model.module if hasattr(self.model, "module") else self.model
            save_url = os.path.join(self.config.ckpt_path, save_name + '.pth')
            self.logger.info("saving %s", save_url)
            torch.save({'model': raw_model.state_dict(),
                        'epoch': epoch,
                        'optimizer': optim.state_dict(),
                        'iterations': iterations,
                        'best_validation': validation,
                        'edgeF1': edgeF1,
                        'lineF1': lineF1}, save_url)

    def load_checkpoint(self, resume_path):
        if os.path.exists(resume_path):
            data = torch.load(resume_path)
            self.model.load_state_dict(data['model'])
            self.model = self.model.to(self.device)
            if self.global_rank == 0:
                self.logger.info('Finished reloading the Epoch %d model' % (data['epoch']))
            return data
        else:
            self.model = self.model.to(self.device)
            if self.global_rank == 0:
                self.logger.info('Warnning: There is no trained model found. An initialized model will be used.')
        return None

    def train(self, loaded_ckpt):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)

        # if self.config.AMP:  ## use AMP
        #     model, optimizer = amp.initialize(model, optimizer, num_losses=1, opt_level='O1')

        previous_epoch = -1
        bestAverageF1 = 0

        if loaded_ckpt is not None:
            optimizer.load_state_dict(loaded_ckpt['optimizer'])
            self.iterations = loaded_ckpt['iterations']
            bestAverageF1 = loaded_ckpt['best_validation']
            previous_epoch = loaded_ckpt['epoch']
            if self.global_rank == 0:
                self.logger.info('Finished reloading the Epoch %d optimizer' % (loaded_ckpt['epoch']))
        else:
            if self.global_rank == 0:
                self.logger.info(
                    'Warnning: There is no previous optimizer found. An initialized optimizer will be used.')

        train_loader = DataLoader(self.train_dataset, pin_memory=True,
                                  batch_size=config.batch_size // config.world_size,  # BS of each GPU
                                  num_workers=config.num_workers, sampler=self.train_sampler)
        test_loader = DataLoader(self.test_dataset, shuffle=False, pin_memory=True,
                                 batch_size=config.batch_size // config.world_size,
                                 num_workers=config.num_workers)

        if loaded_ckpt is None:
            self.iterations = 0  # counter used for learning rate decay

        for epoch in range(config.max_epochs):
            if previous_epoch != -1 and epoch <= previous_epoch:
                continue
            if epoch == previous_epoch + 1 and self.global_rank == 0:
                self.logger.info("Resume from Epoch %d" % (epoch))

            self.train_sampler.set_epoch(epoch)  ## Shuffle each epoch

            epoch_start = time.time()

            model.train()
            loader = train_loader
            losses = []
            not_show_tqdm = True
            for it, items in enumerate(tqdm(loader, disable=not_show_tqdm)):
                # place data on the correct device
                for k in items:
                    if type(items[k]) is torch.Tensor:
                        items[k] = items[k].to(self.device)

                edge, line, loss = model(items['img'], items['edge'], items['line'], items['edge'], items['line'],
                                         items['mask'])
                loss = loss.mean()  # collapse all losses if they are scattered on multiple gpus
                losses.append(loss.item())
                # backprop and update the parameters
                self.iterations += 1  # number of iterations processed this step (i.e. label is not -100)
                model.zero_grad()
                # if self.config.AMP:
                #     with amp.scale_loss(loss, optimizer, loss_id=0) as loss_scaled:
                #         loss_scaled.backward()
                #     torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), config.grad_norm_clip)
                # else:
                #     loss.backward()
                #     torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                optimizer.step()
                # decay the learning rate based on our progress
                if config.lr_decay:
                    if self.iterations < config.warmup_iterations:
                        # linear warmup
                        lr_mult = float(self.iterations) / float(max(1, config.warmup_iterations))
                    else:
                        # cosine learning rate decay
                        progress = float(self.iterations - config.warmup_iterations) / float(
                            max(1, config.final_iterations - config.warmup_iterations))
                        lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                    lr = config.learning_rate * lr_mult
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
                else:
                    lr = config.learning_rate

                if it % self.config.print_freq == 0 and self.global_rank == 0:
                    self.logger.info(
                        f"epoch {epoch + 1} iter {it}/{self.iterations_per_epoch}: train loss {loss.item():.5f}. lr {lr:e}")

                if self.iterations % 400 == 1 and self.global_rank == 0:
                    edge_output = edge[:4, :, :, :].squeeze(1).cpu()
                    edge_output = torch.cat(tuple(edge_output), dim=0)

                    line_output = line[:4, :, :, :].squeeze(1).cpu()
                    line_output = torch.cat(tuple(line_output), dim=0)

                    masked_edges = (items['edge'][:4, ...] * (1 - items['mask'][:4, ...])).squeeze(1).cpu()
                    original_edge = items['edge'][:4, ...].squeeze(1).cpu()
                    masked_lines = (items['line'][:4, ...] * (1 - items['mask'][:4, ...])).squeeze(1).cpu()
                    original_line = items['line'][:4, ...].squeeze(1).cpu()
                    masked_edges = torch.cat(tuple(masked_edges), dim=0)
                    original_edge = torch.cat(tuple(original_edge), dim=0)
                    masked_lines = torch.cat(tuple(masked_lines), dim=0)
                    original_line = torch.cat(tuple(original_line), dim=0)

                    output = torch.cat([original_edge.float(), original_line.float(), masked_edges.float(),
                                        masked_lines.float(), edge_output.float(), line_output.float()],
                                       dim=-1)[:, :, None].repeat(1, 1, 3)
                    output *= 255
                    output = output.detach().numpy().astype(np.uint8)
                    current_img = items['img'][:4, ...] * 0.5 + 0.5
                    current_img = current_img.permute(0, 2, 3, 1) * 255
                    original_img = np.concatenate(current_img.cpu().numpy().astype(np.uint8), axis=0)
                    mask = items['mask'][:4, ...].permute(0, 2, 3, 1)
                    current_img = (current_img * (1 - mask)).cpu().numpy().astype(np.uint8)
                    current_img = np.concatenate(current_img, axis=0)

                    output = np.concatenate([original_img, current_img, output], axis=1)
                    save_path = self.config.ckpt_path + '/samples'
                    os.makedirs(save_path, exist_ok=True)
                    cv2.imwrite(save_path + '/' + str(self.iterations) + '.jpg', output[:, :, ::-1])

                    # eval
                    model.eval()
                    edge_P, edge_R, edge_F1, line_P, line_R, line_F1 = self.val(model, test_loader)
                    model.train()

                    average_F1 = (edge_F1 + line_F1) / 2

                    self.logger.info("Epoch: %d, edge_P: %f, edge_R: %f, edge_F1: %f, line_P: %f, line_R: %f, "
                                     "line_F1: %f, ave_F1: %f time for 2k iter: %d seconds" %
                                     (epoch, edge_P, edge_R, edge_F1, line_P, line_R, line_F1, average_F1,
                                      time.time() - epoch_start))
                    # supports early stopping based on the test loss, or just save always if no test set is provided
                    good_model = self.test_dataset is None or average_F1 >= bestAverageF1
                    if self.config.ckpt_path is not None and good_model and self.global_rank == 0:  ## Validation on the global_rank==0 process
                        bestAverageF1 = average_F1
                        EdgeF1 = edge_F1
                        LineF1 = line_F1
                        self.logger.info("current best epoch is %d" % (epoch))
                        self.save_checkpoint(epoch, optimizer, self.iterations, bestAverageF1, EdgeF1, LineF1,
                                             save_name='best')

                    self.save_checkpoint(epoch, optimizer, self.iterations, average_F1, edge_F1, line_F1,
                                         save_name='latest')


    def val(self, model, dataloader):
        edge_precisions, edge_recalls, edge_f1s = [], [], []
        line_precisions, line_recalls, line_f1s = [], [], []
        for it, items in enumerate(tqdm(dataloader, disable=False)):
            # place data on the correct device
            for k in items:
                if type(items[k]) is torch.Tensor:
                    items[k] = items[k].to(self.device)
            with torch.no_grad():
                edge, line, _ = model(items['img'], items['edge'], items['line'], masks=items['mask'])

            edge_preds = edge
            line_preds = line
            precision, recall, f1 = self.metric(items['edge'] * items['mask'], edge_preds * items['mask'])
            edge_precisions.append(precision.item())
            edge_recalls.append(recall.item())
            edge_f1s.append(f1.item())
            precision, recall, f1 = self.metric(items['line'] * items['mask'],
                                                line_preds * items['mask'])
            line_precisions.append(precision.item())
            line_recalls.append(recall.item())
            line_f1s.append(f1.item())
        return float(np.mean(edge_precisions)), float(np.mean(edge_recalls)), float(np.mean(edge_f1s)), \
               float(np.mean(line_precisions)), float(np.mean(line_recalls)), float(np.mean(line_f1s))


class TrainerForEdgeLineFinetune(TrainerForContinuousEdgeLine):
    def __init__(self, model, train_dataset, test_dataset, config, gpu, global_rank, iterations_per_epoch, logger=None):
        super().__init__(model, train_dataset, test_dataset, config, gpu, global_rank, iterations_per_epoch, logger)

    def train(self, loaded_ckpt):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)

        # if self.config.AMP:  # use AMP
        #     model, optimizer = amp.initialize(model, optimizer, num_losses=1, opt_level='O1')

        previous_epoch = -1
        bestAverageF1 = 0

        if loaded_ckpt is not None:
            optimizer.load_state_dict(loaded_ckpt['optimizer'])
            self.iterations = loaded_ckpt['iterations']
            bestAverageF1 = loaded_ckpt['best_validation']
            previous_epoch = loaded_ckpt['epoch']
            if self.global_rank == 0:
                self.logger.info('Finished reloading the Epoch %d optimizer' % (loaded_ckpt['epoch']))
        else:
            if self.global_rank == 0:
                self.logger.info(
                    'Warnning: There is no previous optimizer found. An initialized optimizer will be used.')

        # TODO: Use different seeds to initialize each worker. (This issue is caused by the bug of pytorch itself)
        train_loader = DataLoader(self.train_dataset, pin_memory=True,
                                  batch_size=config.batch_size // config.world_size,  # BS of each GPU
                                  num_workers=config.num_workers, sampler=self.train_sampler)
        test_loader = DataLoader(self.test_dataset, shuffle=False, pin_memory=True,
                                 batch_size=config.batch_size // config.world_size,
                                 num_workers=config.num_workers)

        if loaded_ckpt is None:
            self.iterations = 0  # counter used for learning rate decay

        for epoch in range(config.max_epochs):

            if previous_epoch != -1 and epoch <= previous_epoch:
                continue

            if epoch == previous_epoch + 1 and self.global_rank == 0:
                self.logger.info("Resume from Epoch %d" % (epoch))

            self.train_sampler.set_epoch(epoch)  ## Shuffle each epoch

            epoch_start = time.time()

            model.train()
            loader = train_loader
            losses = []
            not_show_tqdm = True
            for it, items in enumerate(tqdm(loader, disable=not_show_tqdm)):
                # place data on the correct device
                for k in items:
                    if type(items[k]) is torch.Tensor:
                        items[k] = items[k].to(self.device)

                with torch.cuda.amp.autocast(enabled=self.config.AMP):
                    edge, line, loss = model(items['img'], items['edge'], items['line'], items['edge'],
                                             items['line'], items['mask'])
                    loss = loss.mean()  # collapse all losses if they are scattered on multiple gpus
                # if self.config.AMP:
                #     with torch.cuda.amp.autocast():
                #         edge, line, loss = model(items['img'], items['edge'], items['line'], items['edge'],
                #                                  items['line'], items['mask'])
                #         loss = loss.mean() # collapse all losses if they are scattered on multiple gpus
                # else:
                #     edge, line, loss = model(items['img'], items['edge'], items['line'], items['edge'],
                #                              items['line'], items['mask'])
                #     loss = loss.mean() # collapse all losses if they are scattered on multiple gpus
                # edge, line, loss = model(items['mask_img'], items['edge'], items['line'], items['edge'], items['line'],
                #                          items['erode_mask'])
                # loss = loss.mean()  # collapse all losses if they are scattered on multiple gpus
                losses.append(loss.item())

                # backprop and update the parameters
                self.iterations += 1  # number of iterations processed this step (i.e. label is not -100)
                model.zero_grad()
                # if self.config.AMP:
                #     with amp.scale_loss(loss, optimizer, loss_id=0) as loss_scaled:
                #         loss_scaled.backward()
                #     torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), config.grad_norm_clip)
                # else:
                #     loss.backward()
                #     torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                optimizer.step()
                # decay the learning rate based on our progress
                if config.lr_decay:
                    if self.iterations < config.warmup_iterations:
                        # linear warmup
                        lr_mult = float(self.iterations) / float(max(1, config.warmup_iterations))
                    else:
                        # cosine learning rate decay
                        progress = float(self.iterations - config.warmup_iterations) / float(
                            max(1, config.final_iterations - config.warmup_iterations))
                        lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                    lr = config.learning_rate * lr_mult
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
                else:
                    lr = config.learning_rate

                if it % self.config.print_freq == 0 and self.global_rank == 0:
                    self.logger.info(
                        f"epoch {epoch + 1} iter {it}/{self.iterations_per_epoch}: train loss {loss.item():.5f}. lr {lr:e}")

                if self.iterations % 2000 == 1 and self.global_rank == 0:
                    edge_output = edge[:4, :, :, :].squeeze(1).cpu()
                    edge_output = torch.cat(tuple(edge_output), dim=0)

                    line_output = line[:4, :, :, :].squeeze(1).cpu()
                    line_output = torch.cat(tuple(line_output), dim=0)

                    masked_edges = (items['edge'][:4, ...] * (1 - items['erode_mask'][:4, ...])).squeeze(1).cpu()
                    original_edge = items['edge'][:4, ...].squeeze(1).cpu()
                    masked_lines = (items['line'][:4, ...] * (1 - items['erode_mask'][:4, ...])).squeeze(1).cpu()
                    original_line = items['line'][:4, ...].squeeze(1).cpu()
                    masked_edges = torch.cat(tuple(masked_edges), dim=0)
                    original_edge = torch.cat(tuple(original_edge), dim=0)
                    masked_lines = torch.cat(tuple(masked_lines), dim=0)
                    original_line = torch.cat(tuple(original_line), dim=0)

                    output = torch.cat([original_edge.float(), original_line.float(), masked_edges.float(),
                                        masked_lines.float(), edge_output.float(), line_output.float()],
                                       dim=-1)[:, :, None].repeat(1, 1, 3)
                    output *= 255
                    output = output.detach().numpy().astype(np.uint8)
                    current_img = items['img'][:4, ...] * 0.5 + 0.5
                    current_img = current_img.permute(0, 2, 3, 1) * 255
                    original_img = np.concatenate(current_img.cpu().numpy().astype(np.uint8), axis=0)
                    mask = items['mask'][:4, ...].permute(0, 2, 3, 1)
                    current_img = (current_img * (1 - mask)).cpu().numpy().astype(np.uint8)
                    current_img = np.concatenate(current_img, axis=0)

                    output = np.concatenate([original_img, current_img, output], axis=1)
                    save_path = self.config.ckpt_path + '/samples'
                    os.makedirs(save_path, exist_ok=True)
                    cv2.imwrite(save_path + '/' + str(self.iterations) + '.jpg', output[:, :, ::-1])

                    # eval
                    model.eval()
                    edge_P, edge_R, edge_F1, line_P, line_R, line_F1 = self.val(model, test_loader)
                    model.train()

                    average_F1 = (edge_F1 + line_F1) / 2

                    self.logger.info("Epoch: %d, edge_P: %f, edge_R: %f, edge_F1: %f, line_P: %f, line_R: %f, "
                                     "line_F1: %f, ave_F1: %f time for 2k iter: %d seconds" %
                                     (epoch, edge_P, edge_R, edge_F1, line_P, line_R, line_F1, average_F1,
                                      time.time() - epoch_start))
                    # supports early stopping based on the test loss, or just save always if no test set is provided
                    good_model = self.test_dataset is None or average_F1 >= bestAverageF1
                    if self.config.ckpt_path is not None and good_model and self.global_rank == 0:  ## Validation on the global_rank==0 process
                        bestAverageF1 = average_F1
                        EdgeF1 = edge_F1
                        LineF1 = line_F1
                        self.logger.info("current best epoch is %d" % (epoch))
                        self.save_checkpoint(epoch, optimizer, self.iterations, bestAverageF1, EdgeF1, LineF1,
                                             save_name='best')

                    self.save_checkpoint(epoch, optimizer, self.iterations, average_F1, edge_F1, line_F1,
                                         save_name='latest')

    def val(self, model, dataloader):
        edge_precisions, edge_recalls, edge_f1s = [], [], []
        line_precisions, line_recalls, line_f1s = [], [], []
        for it, items in enumerate(tqdm(dataloader, disable=False)):
            # place data on the correct device
            for k in items:
                if type(items[k]) is torch.Tensor:
                    items[k] = items[k].to(self.device)
            with torch.no_grad():
                edge, line, _ = model(items['mask_img'], items['edge'], items['line'], masks=items['erode_mask'])

            edge_preds = edge
            line_preds = line
            precision, recall, f1 = self.metric(items['edge'] * items['erode_mask'], edge_preds * items['erode_mask'])
            edge_precisions.append(precision.item())
            edge_recalls.append(recall.item())
            edge_f1s.append(f1.item())
            precision, recall, f1 = self.metric(items['line'] * items['erode_mask'],  line_preds * items['erode_mask'])
            line_precisions.append(precision.item())
            line_recalls.append(recall.item())
            line_f1s.append(f1.item())
        return float(np.mean(edge_precisions)), float(np.mean(edge_recalls)), float(np.mean(edge_f1s)), \
               float(np.mean(line_precisions)), float(np.mean(line_recalls)), float(np.mean(line_f1s))

class TrainerForContinuousEdgeLine_video:
    def __init__(self, model, train_dataset, test_dataset, config, gpu, global_rank, iterations_per_epoch, logger=None):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.iterations_per_epoch = iterations_per_epoch
        self.config = config
        self.device = gpu

        self.model = model
        self.metric = EdgeAccuracy(threshold=0.5)
        self.global_rank = global_rank
        self.train_sampler = DistributedSampler(train_dataset, num_replicas=config.world_size,
                                                rank=global_rank, shuffle=True)
        self.logger = logger

    def save_checkpoint(self, epoch, optim, iterations, validation, edgeF1, lineF1, save_name):
        if self.global_rank == 0:  # Only save in global rank 0
            raw_model = self.model.module if hasattr(self.model, "module") else self.model
            save_url = os.path.join(self.config.ckpt_path, save_name + '.pth')
            self.logger.info("saving %s", save_url)
            torch.save({'model': raw_model.state_dict(),
                        'epoch': epoch,
                        'optimizer': optim.state_dict(),
                        'iterations': iterations,
                        'best_validation': validation,
                        'edgeF1': edgeF1,
                        'lineF1': lineF1}, save_url)

    def load_checkpoint(self, resume_path):
        if os.path.exists(resume_path):
            data = torch.load(resume_path)
            self.model.load_state_dict(data['model'])
            self.model = self.model.to(self.device)
            if self.global_rank == 0:
                self.logger.info('Finished reloading the Epoch %d model' % (data['epoch']))
            return data
        else:
            self.model = self.model.to(self.device)
            if self.global_rank == 0:
                self.logger.info('Warnning: There is no trained model found. An initialized model will be used.')
        return None

    def train(self, loaded_ckpt):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)
        writer = SummaryWriter(log_dir=self.config.ckpt_path) # tensorboard
        print(f"config.tensorboard_log_dir: {self.config.ckpt_path}") # test

        # if self.config.AMP:  ## use AMP
        #     model, optimizer = amp.initialize(model, optimizer, num_losses=1, opt_level='O1')

        previous_epoch = -1
        bestAverageF1 = 0

        if loaded_ckpt is not None:
            optimizer.load_state_dict(loaded_ckpt['optimizer'])
            self.iterations = loaded_ckpt['iterations']
            bestAverageF1 = loaded_ckpt['best_validation']
            previous_epoch = loaded_ckpt['epoch']
            if self.global_rank == 0:
                self.logger.info('Finished reloading the Epoch %d optimizer' % (loaded_ckpt['epoch']))
        else:
            if self.global_rank == 0:
                self.logger.info(
                    'Warnning: There is no previous optimizer found. An initialized optimizer will be used.')

        bz = config.batch_size // config.world_size
        self.logger.info(summary(model, [(bz, 5, 1, 240, 432), (bz, 5, 1, 240, 432), (bz, 5, 1, 240, 432),
                                         (bz, 5, 1, 240, 432), (bz, 5, 1, 240, 432), (bz, 5, 1, 240, 432)], verbose=0))

        train_loader = DataLoader(self.train_dataset, pin_memory=True,
                                  batch_size=config.batch_size // config.world_size,  # BS of each GPU
                                  num_workers=config.num_workers, sampler=self.train_sampler)
        # test_loader = DataLoader(self.test_dataset, shuffle=False, pin_memory=True,
        #                          batch_size=config.batch_size // config.world_size,
        #                          num_workers=config.num_workers)

        if loaded_ckpt is None:
            self.iterations = 0  # counter used for learning rate decay

        for epoch in range(config.max_epochs):
            # print(f"epoch: {epoch} from rank {self.global_rank}") # test
            if previous_epoch != -1 and epoch <= previous_epoch:
                continue
            if epoch == previous_epoch + 1 and self.global_rank == 0:
                self.logger.info("Resume from Epoch %d" % (epoch))

            self.train_sampler.set_epoch(epoch)  ## Shuffle each epoch

            epoch_start = time.time()

            model.train()
            loader = train_loader
            losses = []
            losses_detail = []
            not_show_tqdm = True
            for it, items in enumerate(tqdm(loader, disable=not_show_tqdm)):
                # print(f"it: {it} from rank {self.global_rank}") # test
                # place data on the correct device
                for k in items:
                    if type(items[k]) is torch.Tensor:
                        items[k] = items[k].to(self.device)

                with torch.cuda.amp.autocast(enabled=self.config.AMP):
                    edge, line, loss, losses_detail = model(items['frames'], items['edges'], items['lines'], items['edges'], items['lines'], items['masks'])
                    loss = loss.mean()
                # if self.config.AMP:
                #     with torch.cuda.amp.autocast():
                #         edge, line, loss, losses_detail = model(items['frames'], items['edges'], items['lines'], items['edges'], items['lines'], items['masks'])
                #         loss = loss.mean()
                # else:
                #     edge, line, loss, losses_detail = model(items['frames'], items['edges'], items['lines'], items['edges'], items['lines'], items['masks'])
                #     loss = loss.mean()
                # edge, line, loss, losses_detail = model(items['frames'], items['edges'], items['lines'], items['edges'], items['lines'], items['masks'])  # for video
                # loss = loss.mean()  # collapse all losses if they are scattered on multiple gpus
                losses.append(loss.item())

                edge_hole_loss, edge_valid_loss, line_hole_loss, line_valid_loss = losses_detail
                edge_hole_loss = edge_hole_loss.mean()
                edge_valid_loss = edge_valid_loss.mean()
                line_hole_loss = line_hole_loss.mean()
                line_valid_loss = line_valid_loss.mean()
                # backprop and update the parameters
                self.iterations += 1  # number of iterations processed this step (i.e. label is not -100)
                model.zero_grad()
                # if self.config.AMP:
                #     with amp.scale_loss(loss, optimizer, loss_id=0) as loss_scaled:
                #         loss_scaled.backward()
                #     torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), config.grad_norm_clip)
                # else:
                #     loss.backward()
                #     torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                optimizer.step()
                # decay the learning rate based on our progress
                if config.lr_decay:
                    if self.iterations < config.warmup_iterations:
                        # linear warmup
                        lr_mult = float(self.iterations) / float(max(1, config.warmup_iterations))
                    else:
                        # cosine learning rate decay
                        progress = float(self.iterations - config.warmup_iterations) / float(
                            max(1, config.final_iterations - config.warmup_iterations))
                        lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                    lr = config.learning_rate * lr_mult
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
                else:
                    lr = config.learning_rate

                if self.global_rank == 0:
                    if it % self.config.print_freq == 0:
                        self.logger.info(
                            f"epoch {epoch + 1} iter {it}/{self.iterations_per_epoch}: train loss {loss.item():.5f}. lr {lr:e}")

                    edge_P, edge_R, edge_F1, line_P, line_R, line_F1 = 0 , 0, 0, 0, 0, 0
                    if self.iterations % 100 == 1:
                        edge = edge.permute(1,0,2,3,4)[0]
                        line = line.permute(1,0,2,3,4)[0]
                        items_frames = items['frames'].permute(1,0,2,3,4)[0]
                        items_edges = items['edges'].permute(1,0,2,3,4)[0]
                        items_lines = items['lines'].permute(1,0,2,3,4)[0]
                        items_masks = items['masks'].permute(1,0,2,3,4)[0]

                        edge_output = edge[:4, :, :, :].squeeze(1).cpu()
                        edge_output = torch.cat(tuple(edge_output), dim=0)

                        line_output = line[:4, :, :, :].squeeze(1).cpu()
                        line_output = torch.cat(tuple(line_output), dim=0)

                        masked_edges = (items_edges[:4, ...] * (1 - items_masks[:4, ...])).squeeze(1).cpu()
                        original_edge = items_edges[:4, ...].squeeze(1).cpu()
                        masked_lines = (items_lines[:4, ...] * (1 - items_masks[:4, ...])).squeeze(1).cpu()
                        original_line = items_lines[:4, ...].squeeze(1).cpu()
                        masked_edges = torch.cat(tuple(masked_edges), dim=0)
                        original_edge = torch.cat(tuple(original_edge), dim=0)
                        masked_lines = torch.cat(tuple(masked_lines), dim=0)
                        original_line = torch.cat(tuple(original_line), dim=0)

                        # for RGB
                        # output = torch.cat([original_edge.float(), original_line.float(), masked_edges.float(),
                        #                     masked_lines.float(), edge_output.float(), line_output.float()],
                        #                 dim=-1)[:, :, None].repeat(1, 1, 3)
                        
                        # for gray scale
                        output = torch.cat([original_edge.float(), original_line.float(), masked_edges.float(),
                                            masked_lines.float(), edge_output.float(), line_output.float()],
                                        dim=-1)[:, :, None]
                        output *= 255
                        output = output.detach().numpy().astype(np.uint8)
                        # current_img = items_frames[:4, ...] * 0.5 + 0.5
                        # current_img = current_img.permute(0, 2, 3, 1) * 255
                        current_img = items_frames[:4, ...]
                        current_img = current_img.permute(0, 2, 3, 1) * 255
                        original_img = np.concatenate(current_img.cpu().numpy().astype(np.uint8), axis=0)
                        mask = items_masks[:4, ...].permute(0, 2, 3, 1)
                        current_img = (current_img * (1 - mask)).cpu().numpy().astype(np.uint8)
                        current_img = np.concatenate(current_img, axis=0)

                        output = np.concatenate([original_img, current_img, output], axis=1)
                        save_path = self.config.ckpt_path + '/samples'
                        os.makedirs(save_path, exist_ok=True)
                        cv2.imwrite(save_path + '/' + str(self.iterations).zfill(5) + '.jpg', output[:, :, ::-1])
                        torch.cuda.empty_cache()

                        '''
                        # eval
                        # model.eval()
                        edge_P, edge_R, edge_F1, line_P, line_R, line_F1 = self.val()
                        # model.train()

                        average_F1 = (edge_F1 + line_F1) / 2

                        self.logger.info("Epoch: %d, edge_P: %f, edge_R: %f, edge_F1: %f, line_P: %f, line_R: %f, "
                                        "line_F1: %f, ave_F1: %f time for 2k iter: %d seconds" %
                                        (epoch+1, edge_P, edge_R, edge_F1, line_P, line_R, line_F1, average_F1,
                                        time.time() - epoch_start))

                        # supports early stopping based on the test loss, or just save always if no test set is provided
                        good_model = self.test_dataset is None or average_F1 >= bestAverageF1
                        if self.config.ckpt_path is not None and good_model and self.global_rank == 0:  ## Validation on the global_rank==0 process
                            bestAverageF1 = average_F1
                            EdgeF1 = edge_F1
                            LineF1 = line_F1
                            self.logger.info("current best epoch is %d" % (epoch))
                            self.save_checkpoint(epoch, optimizer, self.iterations, bestAverageF1, EdgeF1, LineF1,
                                                save_name='best')

                        self.save_checkpoint(epoch, optimizer, self.iterations, average_F1, edge_F1, line_F1,
                                            save_name='latest')
                        '''

                    writer.add_scalar('Loss/total_loss', loss.item(), self.iterations) # tensorboard
                    writer.add_scalar('Loss/edge_hole_loss', edge_hole_loss.item(), self.iterations)
                    writer.add_scalar('Loss/edge_valid_loss', edge_valid_loss.item(), self.iterations)
                    writer.add_scalar('Loss/line_hole_loss', line_hole_loss.item(), self.iterations)
                    writer.add_scalar('Loss/line_valid_loss', line_valid_loss.item(), self.iterations)
                    writer.add_scalar('learning_rate', lr, self.iterations) # tensorboard
                
            # print epoch average loss
            if self.global_rank == 0:
                self.logger.info("Epoch: %d, average loss: %f" % (epoch+1, np.mean(losses)))
            
            # model eval
            edge_P, edge_R, edge_F1, line_P, line_R, line_F1 = self.val()
            average_F1 = (edge_F1 + line_F1) / 2

            self.logger.info("Epoch: %d, edge_P: %f, edge_R: %f, edge_F1: %f, line_P: %f, line_R: %f, "
                            "line_F1: %f, ave_F1: %f time for 2k iter: %d seconds" %
                            (epoch+1, edge_P, edge_R, edge_F1, line_P, line_R, line_F1, average_F1,
                            time.time() - epoch_start))
            writer.add_scalar('Evaluation/edge_precision', edge_P, self.iterations)
            writer.add_scalar('Evaluation/edge_recall', edge_R, self.iterations)
            writer.add_scalar('Evaluation/edge_F1', edge_F1, self.iterations)
            writer.add_scalar('Evaluation/line_precision', line_P, self.iterations)
            writer.add_scalar('Evaluation/line_recall', line_R, self.iterations)
            writer.add_scalar('Evaluation/line_F1', line_F1, self.iterations)

            # supports early stopping based on the test loss, or just save always if no test set is provided
            good_model = self.test_dataset is None or average_F1 >= bestAverageF1
            if self.config.ckpt_path is not None and good_model and self.global_rank == 0:  ## Validation on the global_rank==0 process
                bestAverageF1 = average_F1
                EdgeF1 = edge_F1
                LineF1 = line_F1
                self.logger.info("current best epoch is %d" % (epoch))
                self.save_checkpoint(epoch, optimizer, self.iterations, bestAverageF1, EdgeF1, LineF1,
                                    save_name='best')

            self.save_checkpoint(epoch, optimizer, self.iterations, average_F1, edge_F1, line_F1,
                                save_name='latest')

        writer.close() # tensorboard

    def val(self):
        test_loader = DataLoader(self.test_dataset, shuffle=False, pin_memory=True,
                                 batch_size=self.config.batch_size,
                                 num_workers=self.config.num_workers)
        edge_precisions, edge_recalls, edge_f1s = [], [], []
        line_precisions, line_recalls, line_f1s = [], [], []
        self.model.eval()
        with torch.no_grad():
            for items in tqdm(test_loader):
                for k in items:
                    if type(items[k]) is torch.Tensor:
                        items[k] = items[k].to(self.device)
                edge, line, _, _ = self.model(items['frames'], items['edges'], items['lines'], masks=items['masks'])
                precision, recall, f1 = self.metric(items['edges'] * items['masks'], edge * items['masks'])
                edge_precisions.append(precision.item())
                edge_recalls.append(recall.item())
                edge_f1s.append(f1.item())
                precision, recall, f1 = self.metric(items['lines'] * items['masks'], line * items['masks'])
                line_precisions.append(precision.item())
                line_recalls.append(recall.item())
                line_f1s.append(f1.item())
                torch.cuda.empty_cache()
        '''
        for it, items in enumerate(tqdm(dataloader, disable=False)):
            # place data on the correct device
            for k in items:
                if type(items[k]) is torch.Tensor:
                    items[k] = items[k].to(self.device)
            with torch.no_grad():
                edge, line, _, _ = model(items['frames'], items['edges'], items['lines'], masks=items['masks'])

            edge_preds = edge
            line_preds = line
            precision, recall, f1 = self.metric(items['edges'] * items['masks'], edge_preds * items['masks'])
            edge_precisions.append(precision.item())
            edge_recalls.append(recall.item())
            edge_f1s.append(f1.item())
            precision, recall, f1 = self.metric(items['lines'] * items['masks'],
                                                line_preds * items['masks'])
            line_precisions.append(precision.item())
            line_recalls.append(recall.item())
            line_f1s.append(f1.item())
            torch.cuda.empty_cache()
        '''
        return float(np.mean(edge_precisions)), float(np.mean(edge_recalls)), float(np.mean(edge_f1s)), \
               float(np.mean(line_precisions)), float(np.mean(line_recalls)), float(np.mean(line_f1s))

class TrainerForContinuousEdgeLine_plus:
    def __init__(self, model, train_dataset, test_dataset, config, gpu, global_rank, iterations_per_epoch, logger=None):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.iterations_per_epoch = iterations_per_epoch
        self.config = config
        self.device = gpu

        self.model = model
        self.metric = EdgeAccuracy(threshold=0.5)
        self.global_rank = global_rank
        self.train_sampler = DistributedSampler(train_dataset, num_replicas=config.world_size,
                                                rank=global_rank, shuffle=True)
        self.logger = logger

    def save_checkpoint(self, epoch, optim, iterations, validation, edgeF1, lineF1, save_name):
        if self.global_rank == 0:  # Only save in global rank 0
            raw_model = self.model.module if hasattr(self.model, "module") else self.model
            save_url = os.path.join(self.config.ckpt_path, save_name + '.pth')
            self.logger.info("saving %s", save_url)
            torch.save({'model': raw_model.state_dict(),
                        'epoch': epoch,
                        'optimizer': optim.state_dict(),
                        'iterations': iterations,
                        'best_validation': validation,
                        'edgeF1': edgeF1,
                        'lineF1': lineF1}, save_url)

    def load_checkpoint(self, resume_path):
        if os.path.exists(resume_path):
            data = torch.load(resume_path)
            self.model.load_state_dict(data['model'])
            self.model = self.model.to(self.device)
            if self.global_rank == 0:
                self.logger.info('Finished reloading the Epoch %d model' % (data['epoch']))
            return data
        else:
            self.model = self.model.to(self.device)
            if self.global_rank == 0:
                self.logger.info('Warnning: There is no trained model found. An initialized model will be used.')
        return None

    def train(self, loaded_ckpt):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)

        # if self.config.AMP:  ## use AMP
        #     model, optimizer = amp.initialize(model, optimizer, num_losses=1, opt_level='O1')

        previous_epoch = -1
        bestAverageF1 = 0

        if loaded_ckpt is not None:
            optimizer.load_state_dict(loaded_ckpt['optimizer'])
            self.iterations = loaded_ckpt['iterations']
            bestAverageF1 = loaded_ckpt['best_validation']
            previous_epoch = loaded_ckpt['epoch']
            if self.global_rank == 0:
                self.logger.info('Finished reloading the Epoch %d optimizer' % (loaded_ckpt['epoch']))
        else:
            if self.global_rank == 0:
                self.logger.info(
                    'Warnning: There is no previous optimizer found. An initialized optimizer will be used.')

        train_loader = DataLoader(self.train_dataset, pin_memory=True,
                                  batch_size=config.batch_size // config.world_size,  # BS of each GPU
                                  num_workers=config.num_workers, sampler=self.train_sampler)
        test_loader = DataLoader(self.test_dataset, shuffle=False, pin_memory=True,
                                 batch_size=config.batch_size // config.world_size,
                                 num_workers=config.num_workers)

        if loaded_ckpt is None:
            self.iterations = 0  # counter used for learning rate decay

        for epoch in range(config.max_epochs):
            if previous_epoch != -1 and epoch <= previous_epoch:
                continue
            if epoch == previous_epoch + 1 and self.global_rank == 0:
                self.logger.info("Resume from Epoch %d" % (epoch))

            self.train_sampler.set_epoch(epoch)  ## Shuffle each epoch

            epoch_start = time.time()

            model.train()
            loader = train_loader
            losses = []
            not_show_tqdm = True
            for it, items in enumerate(tqdm(loader, disable=not_show_tqdm)):
                # place data on the correct device
                for k in items:
                    if type(items[k]) is torch.Tensor:
                        items[k] = items[k].to(self.device)

                with torch.cuda.amp.autocast(enabled=self.config.AMP):
                    edge, line, loss = model(items['frames'], items['edges'], items['lines'], items['edges'], items['lines'],
                                            items['masks'])
                    loss = loss.mean()  # collapse all losses if they are scattered on multiple gpus
                losses.append(loss.item())
                # backprop and update the parameters
                self.iterations += 1  # number of iterations processed this step (i.e. label is not -100)
                model.zero_grad()
                # if self.config.AMP:
                #     with amp.scale_loss(loss, optimizer, loss_id=0) as loss_scaled:
                #         loss_scaled.backward()
                #     torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), config.grad_norm_clip)
                # else:
                #     loss.backward()
                #     torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                optimizer.step()
                # decay the learning rate based on our progress
                if config.lr_decay:
                    if self.iterations < config.warmup_iterations:
                        # linear warmup
                        lr_mult = float(self.iterations) / float(max(1, config.warmup_iterations))
                    else:
                        # cosine learning rate decay
                        progress = float(self.iterations - config.warmup_iterations) / float(
                            max(1, config.final_iterations - config.warmup_iterations))
                        lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                    lr = config.learning_rate * lr_mult
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
                else:
                    lr = config.learning_rate

                if it % self.config.print_freq == 0 and self.global_rank == 0:
                    self.logger.info(
                        f"epoch {epoch + 1} iter {it}/{self.iterations_per_epoch}: train loss {loss.item():.5f}. lr {lr:e}")

                if self.iterations % 400 == 1 and self.global_rank == 0:
                    # ------------
                    # add video process
                    edge = edge.permute(1,0,2,3,4)[0]
                    line = line.permute(1,0,2,3,4)[0]
                    items_frames = items['frames'].permute(1,0,2,3,4)[0]
                    items_edges = items['edges'].permute(1,0,2,3,4)[0]
                    items_lines = items['lines'].permute(1,0,2,3,4)[0]
                    items_masks = items['masks'].permute(1,0,2,3,4)[0]
                    # ------------
                    edge_output = edge[:4, :, :, :].squeeze(1).cpu()
                    edge_output = torch.cat(tuple(edge_output), dim=0)

                    line_output = line[:4, :, :, :].squeeze(1).cpu()
                    line_output = torch.cat(tuple(line_output), dim=0)

                    masked_edges = (items_edges[:4, ...] * (1 - items_masks[:4, ...])).squeeze(1).cpu()
                    original_edge = items_edges[:4, ...].squeeze(1).cpu()
                    masked_lines = (items_lines[:4, ...] * (1 - items_masks[:4, ...])).squeeze(1).cpu()
                    original_line = items_lines[:4, ...].squeeze(1).cpu()
                    masked_edges = torch.cat(tuple(masked_edges), dim=0)
                    original_edge = torch.cat(tuple(original_edge), dim=0)
                    masked_lines = torch.cat(tuple(masked_lines), dim=0)
                    original_line = torch.cat(tuple(original_line), dim=0)

                    output = torch.cat([original_edge.float(), original_line.float(), masked_edges.float(),
                                        masked_lines.float(), edge_output.float(), line_output.float()],
                                       dim=-1)[:, :, None].repeat(1, 1, 3)
                    output *= 255
                    output = output.detach().numpy().astype(np.uint8)
                    current_img = items_frames[:4, ...] * 0.5 + 0.5
                    current_img = current_img.permute(0, 2, 3, 1) * 255
                    original_img = np.concatenate(current_img.cpu().numpy().astype(np.uint8), axis=0)
                    mask = items_masks[:4, ...].permute(0, 2, 3, 1)
                    current_img = (current_img * (1 - mask)).cpu().numpy().astype(np.uint8)
                    current_img = np.concatenate(current_img, axis=0)

                    output = np.concatenate([original_img, current_img, output], axis=1)
                    save_path = self.config.ckpt_path + '/samples'
                    os.makedirs(save_path, exist_ok=True)
                    cv2.imwrite(save_path + '/' + str(self.iterations) + '.jpg', output[:, :, ::-1])

                    # eval
                    model.eval()
                    edge_P, edge_R, edge_F1, line_P, line_R, line_F1 = self.val(model, test_loader)
                    model.train()

                    average_F1 = (edge_F1 + line_F1) / 2

                    self.logger.info("Epoch: %d, edge_P: %f, edge_R: %f, edge_F1: %f, line_P: %f, line_R: %f, "
                                     "line_F1: %f, ave_F1: %f time for 2k iter: %d seconds" %
                                     (epoch, edge_P, edge_R, edge_F1, line_P, line_R, line_F1, average_F1,
                                      time.time() - epoch_start))
                    # supports early stopping based on the test loss, or just save always if no test set is provided
                    good_model = self.test_dataset is None or average_F1 >= bestAverageF1
                    if self.config.ckpt_path is not None and good_model and self.global_rank == 0:  ## Validation on the global_rank==0 process
                        bestAverageF1 = average_F1
                        EdgeF1 = edge_F1
                        LineF1 = line_F1
                        self.logger.info("current best epoch is %d" % (epoch))
                        self.save_checkpoint(epoch, optimizer, self.iterations, bestAverageF1, EdgeF1, LineF1,
                                             save_name='best')

                    self.save_checkpoint(epoch, optimizer, self.iterations, average_F1, edge_F1, line_F1,
                                         save_name='latest')


    def val(self, model, dataloader):
        edge_precisions, edge_recalls, edge_f1s = [], [], []
        line_precisions, line_recalls, line_f1s = [], [], []
        for it, items in enumerate(tqdm(dataloader, disable=False)):
            # place data on the correct device
            for k in items:
                if type(items[k]) is torch.Tensor:
                    items[k] = items[k].to(self.device)
            with torch.no_grad():
                edge, line, _ = model(items['frames'], items['edges'], items['lines'], masks=items['masks'])

            edge_preds = edge
            line_preds = line
            precision, recall, f1 = self.metric(items['edges'] * items['masks'], edge_preds * items['masks'])
            edge_precisions.append(precision.item())
            edge_recalls.append(recall.item())
            edge_f1s.append(f1.item())
            precision, recall, f1 = self.metric(items['lines'] * items['masks'],
                                                line_preds * items['masks'])
            line_precisions.append(precision.item())
            line_recalls.append(recall.item())
            line_f1s.append(f1.item())
        return float(np.mean(edge_precisions)), float(np.mean(edge_recalls)), float(np.mean(edge_f1s)), \
               float(np.mean(line_precisions)), float(np.mean(line_recalls)), float(np.mean(line_f1s))

class TrainerForContinuousStruct_video:
    def __init__(self, model, train_dataset, test_dataset, config, gpu, global_rank, iterations_per_epoch, logger=None, struct='edges'):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.iterations_per_epoch = iterations_per_epoch
        self.config = config
        self.device = gpu

        self.model = model
        self.metric = EdgeAccuracy(threshold=0.5)
        self.global_rank = global_rank
        self.train_sampler = DistributedSampler(train_dataset, num_replicas=config.world_size,
                                                rank=global_rank, shuffle=True)
        self.logger = logger
        self.struct = struct

    def save_checkpoint(self, epoch, optim, iterations, validation, save_name):
        if self.global_rank == 0:  # Only save in global rank 0
            raw_model = self.model.module if hasattr(self.model, "module") else self.model
            save_url = os.path.join(self.config.ckpt_path, save_name + '.pth')
            self.logger.info("saving %s", save_url)
            torch.save({'model': raw_model.state_dict(),
                        'epoch': epoch,
                        'optimizer': optim.state_dict(),
                        'iterations': iterations,
                        'best_validation': validation}, save_url)

    def load_checkpoint(self, resume_path):
        if os.path.exists(resume_path):
            data = torch.load(resume_path)
            self.model.load_state_dict(data['model'])
            self.model = self.model.to(self.device)
            if self.global_rank == 0:
                self.logger.info('Finished reloading the Epoch %d model' % (data['epoch']))
            return data
        else:
            self.model = self.model.to(self.device)
            if self.global_rank == 0:
                self.logger.info('Warnning: There is no trained model found. An initialized model will be used.')
        return None

    def train(self, loaded_ckpt):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)
        writer = SummaryWriter(log_dir=self.config.ckpt_path) # tensorboard
        print(f"config.tensorboard_log_dir: {self.config.ckpt_path}") # test

        # if self.config.AMP:  ## use AMP
        #     model, optimizer = amp.initialize(model, optimizer, num_losses=1, opt_level='O1')

        previous_epoch = -1
        bestAverageF1 = 0

        if loaded_ckpt is not None:
            optimizer.load_state_dict(loaded_ckpt['optimizer'])
            self.iterations = loaded_ckpt['iterations']
            bestAverageF1 = loaded_ckpt['best_validation']
            previous_epoch = loaded_ckpt['epoch']
            if self.global_rank == 0:
                self.logger.info('Finished reloading the Epoch %d optimizer' % (loaded_ckpt['epoch']))
        else:
            if self.global_rank == 0:
                self.logger.info(
                    'Warnning: There is no previous optimizer found. An initialized optimizer will be used.')

        train_loader = DataLoader(self.train_dataset, pin_memory=True,
                                  batch_size=config.batch_size // config.world_size,  # BS of each GPU
                                  num_workers=config.num_workers, sampler=self.train_sampler)
        test_loader = DataLoader(self.test_dataset, shuffle=False, pin_memory=True,
                                 batch_size=config.batch_size // config.world_size,
                                 num_workers=config.num_workers)

        if loaded_ckpt is None:
            self.iterations = 0  # counter used for learning rate decay

        for epoch in range(config.max_epochs):
            if previous_epoch != -1 and epoch <= previous_epoch:
                continue
            if epoch == previous_epoch + 1 and self.global_rank == 0:
                self.logger.info("Resume from Epoch %d" % (epoch))

            self.train_sampler.set_epoch(epoch)  ## Shuffle each epoch

            epoch_start = time.time()

            model.train()
            loader = train_loader
            losses = []
            losses_detail = []
            not_show_tqdm = True
            for it, items in enumerate(tqdm(loader, disable=not_show_tqdm)):
                # place data on the correct device
                for k in items:
                    if type(items[k]) is torch.Tensor:
                        items[k] = items[k].to(self.device)

                with torch.cuda.amp.autocast(enabled=self.config.AMP):
                    struct, loss, losses_detail = model(items['frames'], items[self.struct], items[self.struct], items['masks'])
                    loss = loss.mean()
                losses.append(loss.item())

                struct_hole_loss, struct_valid_loss = losses_detail
                struct_hole_loss = struct_hole_loss.mean()
                struct_valid_loss = struct_valid_loss.mean()

                # backprop and update the parameters
                self.iterations += 1  # number of iterations processed this step (i.e. label is not -100)
                model.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                optimizer.step()
                # decay the learning rate based on our progress
                if config.lr_decay:
                    if self.iterations < config.warmup_iterations:
                        # linear warmup
                        lr_mult = float(self.iterations) / float(max(1, config.warmup_iterations))
                    else:
                        # cosine learning rate decay
                        progress = float(self.iterations - config.warmup_iterations) / float(
                            max(1, config.final_iterations - config.warmup_iterations))
                        lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                    lr = config.learning_rate * lr_mult
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
                else:
                    lr = config.learning_rate

                if self.global_rank == 0:
                    if it % self.config.print_freq == 0:
                        self.logger.info(
                            f"epoch {epoch + 1} iter {it}/{self.iterations_per_epoch}: train loss {loss.item():.5f}. lr {lr:e}")

                    struct_P, struct_R, struct_F1 = 0, 0, 0
                    if self.iterations % 100 == 1:
                        struct = struct.permute(1,0,2,3,4)[0]
                        items_frames = items['frames'].permute(1,0,2,3,4)[0]
                        items_structs = items[self.struct].permute(1,0,2,3,4)[0]
                        items_masks = items['masks'].permute(1,0,2,3,4)[0]

                        struct_output = struct[:4, :, :, :].squeeze(1).cpu()
                        struct_output = torch.cat(tuple(struct_output), dim=0)

                        masked_structs = (items_structs[:4, ...] * (1 - items_masks[:4, ...])).squeeze(1).cpu()
                        original_struct = items_structs[:4, ...].squeeze(1).cpu()
                        masked_structs = torch.cat(tuple(masked_structs), dim=0)
                        original_struct = torch.cat(tuple(original_struct), dim=0)

                        # for RGB
                        # output = torch.cat([original_struct.float(), masked_structs.float(),
                        #                     struct_output.float()],
                        #                 dim=-1)[:, :, None].repeat(1, 1, 3)
                        
                        # for gray scale
                        output = torch.cat([original_struct.float(), masked_structs.float(),
                                            struct_output.float()],
                                        dim=-1)[:, :, None]
                        output *= 255
                        output = output.detach().numpy().astype(np.uint8)
                        current_img = items_frames[:4, ...] * 0.5 + 0.5
                        current_img = current_img.permute(0, 2, 3, 1) * 255
                        original_img = np.concatenate(current_img.cpu().numpy().astype(np.uint8), axis=0)
                        mask = items_masks[:4, ...].permute(0, 2, 3, 1)
                        current_img = (current_img * (1 - mask)).cpu().numpy().astype(np.uint8)
                        current_img = np.concatenate(current_img, axis=0)

                        output = np.concatenate([original_img, current_img, output], axis=1)
                        save_path = self.config.ckpt_path + '/samples'
                        os.makedirs(save_path, exist_ok=True)
                        cv2.imwrite(save_path + '/' + str(self.iterations) + '.jpg', output[:, :, ::-1])

                        # eval
                        model.eval()
                        struct_P, struct_R, struct_F1 = self.val(model, test_loader)
                        model.train()

                        average_F1 = struct_F1

                        self.logger.info("Epoch: %d, %s_P: %f, %s_R: %f, %s_F1: %f time for 2k iter: %d seconds" %
                                        (epoch+1, self.struct[:-1], struct_P, self.struct[:-1], struct_R, self.struct[:-1], struct_F1,
                                        time.time() - epoch_start))

                        # supports early stopping based on the test loss, or just save always if no test set is provided
                        good_model = self.test_dataset is None or average_F1 >= bestAverageF1
                        if self.config.ckpt_path is not None and good_model and self.global_rank == 0:  ## Validation on the global_rank==0 process
                            bestAverageF1 = average_F1
                            self.logger.info("current best epoch is %d" % (epoch))
                            self.save_checkpoint(epoch, optimizer, self.iterations, bestAverageF1,
                                                save_name='best')

                        self.save_checkpoint(epoch, optimizer, self.iterations, average_F1,
                                            save_name='latest')

                    writer.add_scalar('train_loss', loss.item(), self.iterations) # tensorboard
                    writer.add_scalar('train_struct_hole_loss', struct_hole_loss.item(), self.iterations)
                    writer.add_scalar('train_struct_valid_loss', struct_valid_loss.item(), self.iterations)
                    writer.add_scalar('learning_rate', lr, self.iterations) # tensorboard
                    writer.add_scalar('edge_precision', struct_P, self.iterations)
                    writer.add_scalar('edge_recall', struct_R, self.iterations)
                    writer.add_scalar('edge_F1', struct_F1, self.iterations)

        writer.close() # tensorboard

    def val(self, model, dataloader):
        struct_precisions, struct_recalls, struct_f1s = [], [], []
        for it, items in enumerate(tqdm(dataloader, disable=False)):
            # place data on the correct device
            for k in items:
                if type(items[k]) is torch.Tensor:
                    items[k] = items[k].to(self.device)
            with torch.no_grad():
                struct,_ ,_ = model(items['frames'], items[self.struct], masks=items['masks'])

            struct_preds = struct
            precision, recall, f1 = self.metric(items[self.struct] * items['masks'], struct_preds * items['masks'])
            struct_precisions.append(precision.item())
            struct_recalls.append(recall.item())
            struct_f1s.append(f1.item())
        return float(np.mean(struct_precisions)), float(np.mean(struct_recalls)), float(np.mean(struct_f1s))
