import time

import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import cv2

from datasets.dataset_FTR import *
from src.models.FTR_model import *
from .inpainting_metrics import *
# from .utils import Progbar, create_dir, stitch_images, SampleEdgeLineLogits, SampleEdgeLineLogits_video
from .utils import *

from skimage.metrics import structural_similarity as measure_ssim
from skimage.metrics import peak_signal_noise_ratio as measure_psnr
import lpips
from sewar.full_ref import vifp
import datetime

import torchvision.utils
from torchinfo import summary

from torch.utils.tensorboard import SummaryWriter

class LaMa:
    def __init__(self, config, gpu, rank, test=False):
        self.config = config
        self.device = gpu
        self.global_rank = rank

        self.model_name = 'inpaint'

        kwargs = dict(config.training_model)
        kwargs.pop('kind')

        self.inpaint_model = LaMaInpaintingTrainingModule(config, gpu=gpu, rank=rank, test=test, **kwargs).to(gpu)

        self.train_dataset = ImgDataset(config.TRAIN_FLIST, config.INPUT_SIZE, config.MASK_RATE, config.TRAIN_MASK_FLIST,
                                        augment=True, training=True, test_mask_path=None)
        if config.DDP:
            self.train_sampler = DistributedSampler(self.train_dataset, num_replicas=config.world_size,
                                                    rank=self.global_rank, shuffle=True)
        # else:
        #     self.train_sampler = DistributedSampler(self.train_dataset, num_replicas=1, rank=0, shuffle=True)
        self.val_dataset = ImgDataset(config.VAL_FLIST, config.INPUT_SIZE, mask_rates=None, mask_path=None, augment=False,
                                      training=False, test_mask_path=config.TEST_MASK_FLIST)
        self.sample_iterator = self.val_dataset.create_iterator(config.SAMPLE_SIZE)

        self.samples_path = os.path.join(config.PATH, 'samples')
        self.results_path = os.path.join(config.PATH, 'results')
        self.val_path = os.path.join(config.PATH, 'validation')
        create_dir(self.val_path)

        self.log_file = os.path.join(config.PATH, 'log_' + self.model_name + '.dat')

        self.best = float("inf") if self.inpaint_model.best is None else self.inpaint_model.best

    def save(self):
        if self.global_rank == 0:
            self.inpaint_model.save()

    def train(self):
        if self.config.DDP:
            train_loader = DataLoader(self.train_dataset, shuffle=False, pin_memory=True,
                                      batch_size=self.config.BATCH_SIZE // self.config.world_size,
                                      num_workers=12, sampler=self.train_sampler)
        else:
            train_loader = DataLoader(self.train_dataset, pin_memory=True,
                                      batch_size=self.config.BATCH_SIZE, num_workers=12, shuffle=True)

        epoch = 0
        keep_training = True
        max_iteration = int(float((self.config.MAX_ITERS)))
        total = len(self.train_dataset) // self.config.world_size

        if total == 0 and self.global_rank == 0:
            print('No training data was provided! Check \'TRAIN_FLIST\' value in the configuration file.')
            return

        while keep_training:
            epoch += 1
            if self.config.DDP:
                self.train_sampler.set_epoch(epoch + 1)  # Shuffle each epoch
            epoch_start = time.time()
            if self.global_rank == 0:
                print('\n\nTraining epoch: %d' % epoch)
            progbar = Progbar(total, width=20, stateful_metrics=['epoch', 'iter', 'loss_scale'],
                              verbose=1 if self.global_rank == 0 else 0)

            for _, items in enumerate(train_loader):
                self.inpaint_model.train()

                items['image'] = items['image'].to(self.device)
                items['mask'] = items['mask'].to(self.device)

                # train
                outputs, gen_loss, dis_loss, logs, batch = self.inpaint_model.process(items)
                iteration = self.inpaint_model.iteration

                if iteration >= max_iteration:
                    keep_training = False
                    break
                logs = [
                           ("epoch", epoch),
                           ("iter", iteration),
                       ] + [(i, logs[0][i]) for i in logs[0]] + [(i, logs[1][i]) for i in logs[1]]
                if self.config.No_Bar:
                    pass
                else:
                    progbar.add(len(items['image']),
                                values=logs if self.config.VERBOSE else [x for x in logs if not x[0].startswith('l_')])

                # log model at checkpoints
                if self.config.LOG_INTERVAL and iteration % self.config.LOG_INTERVAL == 1 and self.global_rank == 0:
                    self.log(logs)

                # sample model at checkpoints
                if self.config.SAMPLE_INTERVAL and iteration % self.config.SAMPLE_INTERVAL == 1 and self.global_rank == 0:
                    self.sample()

                # evaluate model at checkpoints
                if self.config.EVAL_INTERVAL and iteration % self.config.EVAL_INTERVAL == 1:
                    if self.global_rank == 0:
                        print('\nstart eval...\n')
                        print("Epoch: %d" % epoch)
                    psnr, ssim, fid = self.eval()
                    if self.best > fid and self.global_rank == 0:
                        self.best = fid
                        print("current best epoch is %d" % epoch)
                        print('\nsaving %s...\n' % self.inpaint_model.name)
                        raw_model = self.inpaint_model.generator.module if \
                            hasattr(self.inpaint_model.generator, "module") else self.inpaint_model.generator
                        torch.save({
                            'iteration': self.inpaint_model.iteration,
                            'generator': raw_model.state_dict(),
                            'best_fid': fid,
                            'ssim': ssim,
                            'psnr': psnr
                        }, os.path.join(self.config.PATH, self.inpaint_model.name + '_best_gen.pth'))
                        raw_model = self.inpaint_model.discriminator.module if \
                            hasattr(self.inpaint_model.discriminator, "module") else self.inpaint_model.discriminator
                        torch.save({
                            'discriminator': raw_model.state_dict(),
                            'best_fid': fid,
                            'ssim': ssim,
                            'psnr': psnr
                        }, os.path.join(self.config.PATH, self.inpaint_model.name + '_best_dis.pth'))

                # save model at checkpoints
                if self.config.SAVE_INTERVAL and iteration % self.config.SAVE_INTERVAL == 1 and self.global_rank == 0:
                    self.save()
            if self.global_rank == 0:
                print("Epoch: %d, time for one epoch: %d seconds" % (epoch, time.time() - epoch_start))
                logs = [('Epoch', epoch), ('time', time.time() - epoch_start)]
                self.log(logs)
        print('\nEnd training....')

    def eval(self):
        if self.config.DDP:
            val_loader = DataLoader(self.val_dataset, shuffle=False, pin_memory=True,
                                    batch_size=self.config.BATCH_SIZE // self.config.world_size,  ## BS of each GPU
                                    num_workers=12)
        else:
            val_loader = DataLoader(self.val_dataset, shuffle=False, pin_memory=True,
                                    batch_size=self.config.BATCH_SIZE, num_workers=12)

        total = len(self.val_dataset)

        self.inpaint_model.eval()

        if self.config.No_Bar:
            pass
        else:
            progbar = Progbar(total, width=20, stateful_metrics=['it'])
        iteration = 0
        with torch.no_grad():
            for items in tqdm(val_loader):
                iteration += 1
                items['image'] = items['image'].to(self.device)
                items['mask'] = items['mask'].to(self.device)
                b, _, _, _ = items['image'].size()

                # inpaint model
                # eval
                items = self.inpaint_model(items)
                outputs_merged = (items['predicted_image'] * items['mask']) + (items['image'] * (1 - items['mask']))
                # save
                outputs_merged *= 255.0
                outputs_merged = outputs_merged.permute(0, 2, 3, 1).int().cpu().numpy()
                for img_num in range(b):
                    cv2.imwrite(self.val_path + '/' + items['name'][img_num], outputs_merged[img_num, :, :, ::-1])

        our_metric = get_inpainting_metrics(self.val_path, self.config.GT_Val_FOLDER, None, fid_test=True)

        if self.global_rank == 0:
            print("iter: %d, PSNR: %f, SSIM: %f, FID: %f, LPIPS: %f" %
                  (self.inpaint_model.iteration, float(our_metric['psnr']), float(our_metric['ssim']),
                   float(our_metric['fid']), float(our_metric['lpips'])))
            logs = [('iter', self.inpaint_model.iteration), ('PSNR', float(our_metric['psnr'])),
                    ('SSIM', float(our_metric['ssim'])), ('FID', float(our_metric['fid'])), ('LPIPS', float(our_metric['lpips']))]
            self.log(logs)
        return float(our_metric['psnr']), float(our_metric['ssim']), float(our_metric['fid'])

    def sample(self, it=None):
        # do not sample when validation set is empty
        if len(self.val_dataset) == 0:
            return

        self.inpaint_model.eval()
        with torch.no_grad():
            items = next(self.sample_iterator)
            items['image'] = items['image'].to(self.device)
            items['mask'] = items['mask'].to(self.device)

            # inpaint model
            iteration = self.inpaint_model.iteration
            inputs = (items['image'] * (1 - items['mask']))
            items = self.inpaint_model(items)
            outputs_merged = (items['predicted_image'] * items['mask']) + (items['image'] * (1 - items['mask']))
            # convert to [0,1]
            outputs_merged = (outputs_merged + 1) / 2

        if it is not None:
            iteration = it

        image_per_row = 2
        if self.config.SAMPLE_SIZE <= 6:
            image_per_row = 1
        images = stitch_images(
            self.postprocess(items['image'].cpu()),
            self.postprocess(inputs.cpu()),
            self.postprocess(items['mask'].cpu()),
            self.postprocess(items['predicted_image'].cpu()),
            self.postprocess(outputs_merged.cpu()),
            img_per_row=image_per_row
        )

        path = os.path.join(self.samples_path, self.model_name)
        name = os.path.join(path, str(iteration).zfill(5) + ".png")
        create_dir(path)
        print('\nsaving sample ' + name)
        images.save(name)

    def log(self, logs):
        with open(self.log_file, 'a') as f:
            f.write('%s\n' % ' '.join([str(item[0]) + '\t' + str(item[1]) for item in logs]))

    def cuda(self, *args):
        return (item.to(self.config.DEVICE) for item in args)

    def postprocess(self, img):
        # [0, 1] => [0, 255]
        img = img * 255.0
        img = img.permute(0, 2, 3, 1)
        return img.int()


class ZITS:
    def __init__(self, config, gpu, rank, test=False, single_img_test=False):
        self.config = config
        self.device = gpu
        self.global_rank = rank

        self.model_name = 'inpaint'

        kwargs = dict(config.training_model)
        kwargs.pop('kind')

        self.inpaint_model = DefaultInpaintingTrainingModule(config, gpu=gpu, rank=rank, test=test, **kwargs).to(gpu)

        if config.min_sigma is None:
            min_sigma = 2.0
        else:
            min_sigma = config.min_sigma
        if config.max_sigma is None:
            max_sigma = 2.5
        else:
            max_sigma = config.max_sigma
        if config.round is None:
            round = 1
        else:
            round = config.round

        if not test:
            self.train_dataset = DynamicDataset(config.TRAIN_FLIST, mask_path=config.TRAIN_MASK_FLIST,
                                                batch_size=config.BATCH_SIZE // config.world_size,
                                                pos_num=config.rel_pos_num, augment=True, training=True,
                                                test_mask_path=None, train_line_path=config.train_line_path,
                                                add_pos=config.use_MPE, world_size=config.world_size,
                                                min_sigma=min_sigma, max_sigma=max_sigma, round=round)
            if config.DDP:
                self.train_sampler = DistributedSampler(self.train_dataset, num_replicas=config.world_size,
                                                        rank=self.global_rank, shuffle=True)
            else:
                self.train_sampler = DistributedSampler(self.train_dataset, num_replicas=1, rank=0, shuffle=True)

            self.samples_path = os.path.join(config.PATH, 'samples')
            self.results_path = os.path.join(config.PATH, 'results')

            self.log_file = os.path.join(config.PATH, 'log_' + self.model_name + '.dat')

            self.best = float("inf") if self.inpaint_model.best is None else self.inpaint_model.best

        if not single_img_test:
            self.val_dataset = DynamicDataset(config.VAL_FLIST, mask_path=None, pos_num=config.rel_pos_num,
                                              batch_size=config.BATCH_SIZE, augment=False, training=False,
                                              test_mask_path=config.TEST_MASK_FLIST,
                                              eval_line_path=config.eval_line_path,
                                              add_pos=config.use_MPE, input_size=config.INPUT_SIZE,
                                              min_sigma=min_sigma, max_sigma=max_sigma)
            self.sample_iterator = self.val_dataset.create_iterator(config.SAMPLE_SIZE)
            self.val_path = os.path.join(config.PATH, 'validation')
            create_dir(self.val_path)

    def save(self):
        if self.global_rank == 0:
            self.inpaint_model.save()

    def train(self):
        if self.config.DDP:
            train_loader = DataLoader(self.train_dataset, shuffle=False, pin_memory=True,
                                      batch_size=self.config.BATCH_SIZE // self.config.world_size,
                                      num_workers=12, sampler=self.train_sampler)
        else:
            train_loader = DataLoader(self.train_dataset, pin_memory=True,
                                      batch_size=self.config.BATCH_SIZE, num_workers=12,
                                      sampler=self.train_sampler)
        epoch = self.inpaint_model.iteration // len(train_loader)
        keep_training = True
        max_iteration = int(float((self.config.MAX_ITERS)))
        total = len(self.train_dataset) // self.config.world_size

        if total == 0 and self.global_rank == 0:
            print('No training data was provided! Check \'TRAIN_FLIST\' value in the configuration file.')
            return

        while keep_training:

            epoch += 1
            if self.config.DDP or self.config.DP:
                self.train_sampler.set_epoch(epoch + 1)
            if self.config.fix_256 is None or self.config.fix_256 is False:
                self.train_dataset.reset_dataset(self.train_sampler)

            epoch_start = time.time()
            if self.global_rank == 0:
                print('\n\nTraining epoch: %d' % epoch)

            progbar = Progbar(total, width=20, stateful_metrics=['epoch', 'iter', 'loss_scale',
                                                                 'g_lr', 'd_lr', 'str_lr', 'img_size'],
                              verbose=1 if self.global_rank == 0 else 0)

            for _, items in enumerate(train_loader):
                iteration = self.inpaint_model.iteration

                self.inpaint_model.train()
                for k in items:
                    if type(items[k]) is torch.Tensor:
                        items[k] = items[k].to(self.device)

                image_size = items['image'].shape[2]
                random_add_v = random.random() * 1.5 + 1.5
                random_mul_v = random.random() * 1.5 + 1.5  # [1.5~3]

                # random mix the edge and line
                if iteration > int(self.config.MIX_ITERS):
                    b, _, _, _ = items['edge'].shape
                    if int(self.config.MIX_ITERS) < iteration < int(self.config.Turning_Point):
                        pred_rate = (iteration - int(self.config.MIX_ITERS)) / \
                                    (int(self.config.Turning_Point) - int(self.config.MIX_ITERS))
                        b = np.clip(int(pred_rate * b), 2, b)
                    iteration_num_for_pred = int(random.random() * 5) + 1
                    edge_pred, line_pred = SampleEdgeLineLogits(self.inpaint_model.transformer,
                                                                context=[items['img_256'][:b, ...],
                                                                         items['edge_256'][:b, ...],
                                                                         items['line_256'][:b, ...]],
                                                                mask=items['mask_256'][:b, ...].clone(),
                                                                iterations=iteration_num_for_pred,
                                                                add_v=0.05, mul_v=4)
                    edge_pred = edge_pred.detach().to(torch.float32)
                    line_pred = line_pred.detach().to(torch.float32)
                    if self.config.fix_256 is None or self.config.fix_256 is False:
                        if image_size < 300 and random.random() < 0.5:
                            edge_pred = F.interpolate(edge_pred, size=(image_size, image_size), mode='nearest')
                            line_pred = F.interpolate(line_pred, size=(image_size, image_size), mode='nearest')
                        else:
                            edge_pred = self.inpaint_model.structure_upsample(edge_pred)[0]
                            edge_pred = torch.sigmoid((edge_pred + random_add_v) * random_mul_v)
                            edge_pred = F.interpolate(edge_pred, size=(image_size, image_size), mode='bilinear',
                                                      align_corners=False)
                            line_pred = self.inpaint_model.structure_upsample(line_pred)[0]
                            line_pred = torch.sigmoid((line_pred + random_add_v) * random_mul_v)
                            line_pred = F.interpolate(line_pred, size=(image_size, image_size), mode='bilinear',
                                                      align_corners=False)
                    items['edge'][:b, ...] = edge_pred.detach()
                    items['line'][:b, ...] = line_pred.detach()

                # train
                outputs, gen_loss, dis_loss, logs, batch = self.inpaint_model.process(items)

                if iteration >= max_iteration:
                    keep_training = False
                    break
                logs = [("epoch", epoch), ("iter", iteration)] + \
                       [(i, logs[0][i]) for i in logs[0]] + [(i, logs[1][i]) for i in logs[1]]
                logs.append(("g_lr", self.inpaint_model.g_scheduler.get_lr()[0]))
                logs.append(("d_lr", self.inpaint_model.d_scheduler.get_lr()[0]))
                logs.append(("str_lr", self.inpaint_model.str_scheduler.get_lr()[0]))
                logs.append(("img_size", batch['size_ratio'][0].item() * 256))
                progbar.add(len(items['image']),
                            values=logs if self.config.VERBOSE else [x for x in logs if not x[0].startswith('l_')])

                # log model at checkpoints
                if self.config.LOG_INTERVAL and iteration % self.config.LOG_INTERVAL == 0 and self.global_rank == 0:
                    self.log(logs)

                # sample model at checkpoints
                if self.config.SAMPLE_INTERVAL and iteration > 0 and iteration % self.config.SAMPLE_INTERVAL == 0 and self.global_rank == 0:
                    self.sample()

                # evaluate model at checkpoints
                if self.config.EVAL_INTERVAL and iteration > 0 and iteration % self.config.EVAL_INTERVAL == 0 and self.global_rank == 0:
                    print('\nstart eval...\n')
                    print("Epoch: %d" % epoch)
                    psnr, ssim, fid = self.eval()
                    if self.best > fid:
                        self.best = fid
                        print("current best epoch is %d" % epoch)
                        print('\nsaving %s...\n' % self.inpaint_model.name)
                        raw_model = self.inpaint_model.generator.module if \
                            hasattr(self.inpaint_model.generator, "module") else self.inpaint_model.generator
                        raw_encoder = self.inpaint_model.str_encoder.module if \
                            hasattr(self.inpaint_model.str_encoder, "module") else self.inpaint_model.str_encoder
                        torch.save({
                            'iteration': self.inpaint_model.iteration,
                            'generator': raw_model.state_dict(),
                            'str_encoder': raw_encoder.state_dict(),
                            'best_fid': fid,
                            'ssim': ssim,
                            'psnr': psnr
                        }, os.path.join(self.config.PATH,
                                        self.inpaint_model.name + '_best_gen_HR.pth'))
                        raw_model = self.inpaint_model.discriminator.module if \
                            hasattr(self.inpaint_model.discriminator, "module") else self.inpaint_model.discriminator
                        torch.save({
                            'discriminator': raw_model.state_dict()
                        }, os.path.join(self.config.PATH, self.inpaint_model.name + '_best_dis_HR.pth'))

                # save model at checkpoints
                if self.config.SAVE_INTERVAL and iteration > 0 and iteration % self.config.SAVE_INTERVAL == 0 and self.global_rank == 0:
                    self.save()
            if self.global_rank == 0:
                print("Epoch: %d, time for one epoch: %d seconds" % (epoch, time.time() - epoch_start))
                logs = [('Epoch', epoch), ('time', time.time() - epoch_start)]
                self.log(logs)
        print('\nEnd training....')

    def eval(self):
        val_loader = DataLoader(self.val_dataset, shuffle=False, pin_memory=True,
                                batch_size=self.config.BATCH_SIZE, num_workers=12)

        self.inpaint_model.eval()

        with torch.no_grad():
            for items in tqdm(val_loader):
                for k in items:
                    if type(items[k]) is torch.Tensor:
                        items[k] = items[k].to(self.device)
                b, _, _, _ = items['edge'].shape
                edge_pred, line_pred = SampleEdgeLineLogits(self.inpaint_model.transformer,
                                                            context=[items['img_256'][:b, ...],
                                                                     items['edge_256'][:b, ...],
                                                                     items['line_256'][:b, ...]],
                                                            mask=items['mask_256'][:b, ...].clone(),
                                                            iterations=5,
                                                            add_v=0.05, mul_v=4,
                                                            device=self.device)
                edge_pred, line_pred = edge_pred[:b, ...].detach().to(torch.float32), \
                                       line_pred[:b, ...].detach().to(torch.float32)
                if self.config.fix_256 is None or self.config.fix_256 is False:
                    edge_pred = self.inpaint_model.structure_upsample(edge_pred)[0]
                    edge_pred = torch.sigmoid((edge_pred + 2) * 2)
                    line_pred = self.inpaint_model.structure_upsample(line_pred)[0]
                    line_pred = torch.sigmoid((line_pred + 2) * 2)
                items['edge'][:b, ...] = edge_pred.detach()
                items['line'][:b, ...] = line_pred.detach()
                # eval
                items = self.inpaint_model(items)
                outputs_merged = (items['predicted_image'] * items['mask']) + (items['image'] * (1 - items['mask']))
                # save
                outputs_merged *= 255.0
                outputs_merged = outputs_merged.permute(0, 2, 3, 1).int().cpu().numpy()
                for img_num in range(b):
                    cv2.imwrite(self.val_path + '/' + items['name'][img_num], outputs_merged[img_num, :, :, ::-1])

        our_metric = get_inpainting_metrics(self.val_path, self.config.GT_Val_FOLDER, None, fid_test=True)

        if self.global_rank == 0:
            print("iter: %d, PSNR: %f, SSIM: %f, FID: %f, LPIPS: %f" %
                  (self.inpaint_model.iteration, float(our_metric['psnr']), float(our_metric['ssim']),
                   float(our_metric['fid']), float(our_metric['lpips'])))
            logs = [('iter', self.inpaint_model.iteration), ('PSNR', float(our_metric['psnr'])),
                    ('SSIM', float(our_metric['ssim'])), ('FID', float(our_metric['fid'])),
                    ('LPIPS', float(our_metric['lpips']))]
            self.log(logs)
        return float(our_metric['psnr']), float(our_metric['ssim']), float(our_metric['fid'])

    def sample(self, it=None):
        # do not sample when validation set is empty
        if len(self.val_dataset) == 0:
            return

        self.inpaint_model.eval()
        with torch.no_grad():
            items = next(self.sample_iterator)
            for k in items:
                if type(items[k]) is torch.Tensor:
                    items[k] = items[k].to(self.device)
            b, _, _, _ = items['edge'].shape
            edge_pred, line_pred = SampleEdgeLineLogits(self.inpaint_model.transformer,
                                                        context=[items['img_256'][:b, ...],
                                                                 items['edge_256'][:b, ...],
                                                                 items['line_256'][:b, ...]],
                                                        mask=items['mask_256'][:b, ...].clone(),
                                                        iterations=5,
                                                        add_v=0.05, mul_v=4,
                                                        device=self.device)
            edge_pred, line_pred = edge_pred[:b, ...].detach().to(torch.float32), \
                                   line_pred[:b, ...].detach().to(torch.float32)
            if self.config.fix_256 is None or self.config.fix_256 is False:
                edge_pred = self.inpaint_model.structure_upsample(edge_pred)[0]
                edge_pred = torch.sigmoid((edge_pred + 2) * 2)
                line_pred = self.inpaint_model.structure_upsample(line_pred)[0]
                line_pred = torch.sigmoid((line_pred + 2) * 2)
            items['edge'][:b, ...] = edge_pred.detach()
            items['line'][:b, ...] = line_pred.detach()
            # inpaint model
            iteration = self.inpaint_model.iteration
            inputs = (items['image'] * (1 - items['mask']))
            items = self.inpaint_model(items)
            outputs_merged = (items['predicted_image'] * items['mask']) + (items['image'] * (1 - items['mask']))

        if it is not None:
            iteration = it

        image_per_row = 2
        if self.config.SAMPLE_SIZE <= 6:
            image_per_row = 1
        images = stitch_images(
            self.postprocess((items['image']).cpu()),
            self.postprocess((inputs).cpu()),
            self.postprocess(items['edge'].cpu()),
            self.postprocess(items['line'].cpu()),
            self.postprocess(items['mask'].cpu()),
            self.postprocess((items['predicted_image']).cpu()),
            self.postprocess((outputs_merged).cpu()),
            img_per_row=image_per_row
        )

        path = os.path.join(self.samples_path, self.model_name)
        name = os.path.join(path, str(iteration).zfill(6) + ".jpg")
        create_dir(path)
        print('\nsaving sample ' + name)
        images.save(name)

    def log(self, logs):
        with open(self.log_file, 'a') as f:
            f.write('%s\n' % ' '.join([str(item[0]) + '\t' + str(item[1]) for item in logs]))

    def cuda(self, *args):
        return (item.to(self.config.DEVICE) for item in args)

    def postprocess(self, img):
        # [0, 1] => [0, 255]
        img = img * 255.0
        img = img.permute(0, 2, 3, 1)
        return img.int()



to_tensors = transforms.Compose([
    Stack(),
    ToTorchFormatTensor(), ])

def get_ref_index(length, sample_length):
        if random.uniform(0, 1) > 0.5:
            ref_index = random.sample(range(length), sample_length)
            ref_index.sort()
        else:
            pivot = random.randint(0, length-sample_length)
            ref_index = [pivot+i for i in range(sample_length)]
        return ref_index

# sample reference frames from the whole video 
def get_ref_index_fuseformer(f, neighbor_ids, length):
    ref_index = []
    ref_length = 10
    num_ref = -1
    if num_ref == -1: 
        for i in range(0, length, ref_length): # sample from 0 to length with interval ref_length, try to get the whole story
            if not i in neighbor_ids:
                ref_index.append(i)
    else:
        start_idx = max(0, f - ref_length * (num_ref//2))
        end_idx = min(length, f + ref_length * (num_ref//2))
        for i in range(start_idx, end_idx+1, ref_length):
            if not i in neighbor_ids:
                ref_index.append(i)
                if len(ref_index) >= num_ref:
                    break
    return ref_index

class ZITS_video:
    def __init__(self, args, config, gpu, rank, test=False, single_img_test=False, logger=None):
        self.config = config
        self.device = gpu
        self.global_rank = rank

        # self.model_name = 'inpaint' # this final RGB video inpainting model name
        self.model_name = args.model_name

        kwargs = dict(config.training_model)
        kwargs.pop('kind')

        self.inpaint_model = DefaultInpaintingTrainingModule_video(args, config, gpu=gpu, rank=rank, test=test, **kwargs).to(gpu) 
        self.input_size = args.input_size
        # self.w, self.h = args.input_size   # original is (240,432) there may be some mistake 09/22
        self.h, self.w = args.input_size   # original is (240,432) there may be some mistake 09/22 changed on 10/02
        self.neighbor_stride = args.neighbor_stride
        self.sample_length = args.ref_frame_num
        self.pos_num = config.rel_pos_num
        self.str_size = config.str_size
        self.args = args
        self.logger = logger

        if config.min_sigma is None:
            min_sigma = 2.0
        else:
            min_sigma = config.min_sigma
        if config.max_sigma is None:
            max_sigma = 2.5
        else:
            max_sigma = config.max_sigma
        if config.round is None:
            round = 1
        else:
            round = config.round

        if not test:
            # self.train_dataset = DynamicDataset(config.TRAIN_FLIST, mask_path=config.TRAIN_MASK_FLIST,
            #                                     batch_size=config.BATCH_SIZE // config.world_size,
            #                                     pos_num=config.rel_pos_num, augment=True, training=True,
            #                                     test_mask_path=None, train_line_path=config.train_line_path,
            #                                     add_pos=config.use_MPE, world_size=config.world_size,
            #                                     min_sigma=min_sigma, max_sigma=max_sigma, round=round)
            # self.train_dataset = DynamicDataset_video()
            self.train_dataset = DynamicDataset_video_v2()

            if config.DDP:
                self.train_sampler = DistributedSampler(self.train_dataset, num_replicas=config.world_size,
                                                        rank=self.global_rank, shuffle=True)
            else:
                self.train_sampler = DistributedSampler(self.train_dataset, num_replicas=1, rank=0, shuffle=True)

            self.samples_path = os.path.join(config.PATH, 'samples')
            self.results_path = os.path.join(config.PATH, 'results')

            self.log_file = os.path.join(config.PATH, 'log_' + self.model_name + '.dat')

            self.best = float("inf") if self.inpaint_model.best is None else self.inpaint_model.best

        if not single_img_test:
            # self.val_dataset = DynamicDataset(config.VAL_FLIST, mask_path=None, pos_num=config.rel_pos_num,
            #                                   batch_size=config.BATCH_SIZE, augment=False, training=False,
            #                                   test_mask_path=config.TEST_MASK_FLIST,
            #                                   eval_line_path=config.eval_line_path,
            #                                   add_pos=config.use_MPE, input_size=config.INPUT_SIZE,
            #                                   min_sigma=min_sigma, max_sigma=max_sigma)
            # self.val_dataset = DynamicDataset_video(split='valid')
            self.val_dataset = DynamicDataset_video_v2(split='valid')

            self.sample_iterator = self.val_dataset.create_iterator(config.SAMPLE_SIZE)
            self.val_path = os.path.join(config.PATH, 'validation')
            create_dir(self.val_path)


    def save(self):
        if self.global_rank == 0:
            self.inpaint_model.save()

    def train(self):
        if self.config.DDP:
            train_loader = DataLoader(self.train_dataset, shuffle=False, pin_memory=False,
                                      batch_size=self.config.BATCH_SIZE // self.config.world_size,
                                      num_workers=8, sampler=self.train_sampler)
            
        else:
            train_loader = DataLoader(self.train_dataset, pin_memory=False,
                                      batch_size=self.config.BATCH_SIZE, num_workers=8,
                                      sampler=self.train_sampler)
        self.logger.info("Start tensorboard at %s" % self.args.path)
        self.writer = SummaryWriter(log_dir=self.args.path)
        epoch = self.inpaint_model.iteration // len(train_loader)
        keep_training = True
        max_iteration = int(float((self.config.MAX_ITERS))) 
        total = len(self.train_dataset) // self.config.world_size

        if total == 0 and self.global_rank == 0:
            self.logger.error('No training data was provided! Check \'TRAIN_FLIST\' value in the configuration file.')
            return
    
        self.logger.info(summary(self.inpaint_model.generator, verbose=0))
        self.logger.info(summary(self.inpaint_model.str_encoder, verbose=0))
        self.logger.info(summary(self.inpaint_model.structure_upsample, verbose=0))

        while keep_training:

            epoch += 1
            if self.config.DDP or self.config.DP:
                self.train_sampler.set_epoch(epoch + 1)

            epoch_start = time.time()
            if self.global_rank == 0:
                self.logger.info('Training epoch: %d' % epoch)

            progbar = Progbar(total, width=20, stateful_metrics=['epoch', 'iter', 'loss_scale',
                                                                 'g_lr', 'd_lr', 'str_lr'],
                              verbose=1 if self.global_rank == 0 else 0)

            for _, items in enumerate(train_loader):
                iteration = self.inpaint_model.iteration

                self.inpaint_model.train()
                for k in items:
                    if type(items[k]) is torch.Tensor:
                        items[k] = items[k].to(self.device)

                image_size = items['frames'].shape[2]
                random_add_v = random.random() * 1.5 + 1.5
                random_mul_v = random.random() * 1.5 + 1.5  # [1.5~3]

                # random mix the edge and line
                if iteration > int(self.config.MIX_ITERS):
                    b, t, _, _, _ = items['edges'].shape  # add time dimension
                    if int(self.config.MIX_ITERS) < iteration < int(self.config.Turning_Point):
                        pred_rate = (iteration - int(self.config.MIX_ITERS)) / \
                                    (int(self.config.Turning_Point) - int(self.config.MIX_ITERS))
                        b = np.clip(int(pred_rate * b), 2, b)
                    iteration_num_for_pred = int(random.random() * 5) + 1
                    
                    # gray_frames = (items['frames'].mean(2, keepdim=True) + 1.0) / 2.0
                    edge_pred, line_pred = SampleEdgeLineLogits_video(self.inpaint_model.transformer,
                    # context=[items['frames'][:b, ...].to(torch.float16), items['edges'][:b, ...].to(torch.float16), items['lines'][:b, ...].to(torch.float16)], 
                    context=[items['grays'][:b, ...], items['edges'][:b, ...], items['lines'][:b, ...]], 
                    # masks=items['masks'][:b, ...].to(torch.float16).clone(), iterations=iteration_num_for_pred, add_v=0.05, mul_v=4, device=self.device)   
                    # masks=items['masks'][:b, ...].to(torch.float16).clone(), iterations=1, add_v=0.05, mul_v=4, device=self.device)  # modify 0925
                    masks=items['masks'][:b, ...].clone(), iterations=5, add_v=0.05, mul_v=4, device=self.device)  # modify 0925
                    edge_pred = edge_pred.detach().to(torch.float32)
                    line_pred = line_pred.detach().to(torch.float32)
                    items['edges'] = edge_pred.detach()
                    items['lines'] = line_pred.detach()
                
                # 0917 using predicted edge and line
                # # gray_frames = (items['frames'].mean(2, keepdim=True) + 1.0) / 2.0  # from -1~1 to 0~1
                # edge_pred, line_pred = SampleEdgeLineLogits_video(self.inpaint_model.transformer,
                #     context=[items['grays'], items['edges'], items['lines']],
                #     masks=items['masks'].clone(), iterations=5, device=self.device)
                # items['edges'] = edge_pred.detach()
                # items['lines'] = line_pred.detach()

                # train
                outputs, gen_loss, dis_loss, logs, batch = self.inpaint_model.process(items)

                if iteration >= max_iteration:
                    keep_training = False
                    break
                logs_bck = logs
                logs = [("epoch", epoch), ("iter", iteration)] + \
                       [(i, logs[0][i]) for i in logs[0]] + [(i, logs[1][i]) for i in logs[1]]
                logs.append(("g_lr", self.inpaint_model.g_scheduler.get_last_lr()[0]))
                logs.append(("d_lr", self.inpaint_model.d_scheduler.get_last_lr()[0]))
                logs.append(("str_lr", self.inpaint_model.str_scheduler.get_last_lr()[0]))
                # logs.append(("img_size", batch['size_ratio'][0].item() * 256))
                progbar.add(len(items['frames']),
                            values=logs if self.config.VERBOSE else [x for x in logs if not x[0].startswith('l_')])

                ## tensorboard logging
                # discriminator loss
                for i in logs_bck[0]:
                    # if i.endswith('loss'):
                    #     writer.add_scalar('Loss/%s' % i, logs_bck[0][i], iteration)
                    # else:
                    self.writer.add_scalar('Discriminator/%s' % i, logs_bck[0][i], iteration)
                # generator loss
                for i in logs_bck[1]:
                    self.writer.add_scalar('Generator/%s' % i, logs_bck[1][i], iteration)
                # total loss
                self.writer.add_scalar('Loss/gen_loss', gen_loss, iteration)
                self.writer.add_scalar('Loss/dis_loss', dis_loss, iteration)
                self.writer.add_scalar('Loss/total_loss', gen_loss + dis_loss, iteration)
                self.writer.add_scalar('Learning_rate/gen_lr', self.inpaint_model.g_scheduler.get_last_lr()[0], iteration)
                self.writer.add_scalar('Learning_rate/dis_lr', self.inpaint_model.d_scheduler.get_last_lr()[0], iteration)
                self.writer.add_scalar('Learning_rate/str_lr', self.inpaint_model.str_scheduler.get_last_lr()[0], iteration)

                # log model at checkpoints
                if self.config.LOG_INTERVAL and iteration % self.config.LOG_INTERVAL == 0 and self.global_rank == 0:
                    self.log(logs)

                # sample model at checkpoints
                if self.config.SAMPLE_INTERVAL and iteration > 0 and iteration % self.config.SAMPLE_INTERVAL == 0 and self.global_rank == 0:
                    self.sample()

                # evaluate model at checkpoints
                if self.config.EVAL_INTERVAL and iteration > 0 and iteration % self.config.EVAL_INTERVAL == 0 and self.global_rank == 0:
                    torch.cuda.empty_cache()
                    print('Start eval at epoch %d' % epoch)
                    psnr, ssim, vfid = self.eval()
                    self.writer.add_scalar('Evaluation/PSNR', psnr, iteration)
                    self.writer.add_scalar('Evaluation/SSIM', ssim, iteration)
                    self.writer.add_scalar('Evaluation/VFID', vfid, iteration)
                    if self.best > vfid:
                        self.best = vfid
                        self.logger.info("Saving current best model at epoch %d" % epoch)
                        raw_model = self.inpaint_model.generator.module if \
                            hasattr(self.inpaint_model.generator, "module") else self.inpaint_model.generator
                        raw_encoder = self.inpaint_model.str_encoder.module if \
                            hasattr(self.inpaint_model.str_encoder, "module") else self.inpaint_model.str_encoder
                        torch.save({
                            'iteration': self.inpaint_model.iteration,
                            'generator': raw_model.state_dict(),
                            'str_encoder': raw_encoder.state_dict(),
                            'best_vfid': vfid,
                            'ssim': ssim,
                            'psnr': psnr
                        }, os.path.join(self.config.PATH,
                                        self.inpaint_model.name + '_best_gen_HR.pth'))
                        raw_model = self.inpaint_model.discriminator.module if \
                            hasattr(self.inpaint_model.discriminator, "module") else self.inpaint_model.discriminator
                        torch.save({
                            'discriminator': raw_model.state_dict()
                        }, os.path.join(self.config.PATH, self.inpaint_model.name + '_best_dis_HR.pth'))
                    torch.cuda.empty_cache()

                # save model at checkpoints
                if self.config.SAVE_INTERVAL and iteration > 0 and iteration % self.config.SAVE_INTERVAL == 0 and self.global_rank == 0:
                    self.save()
            if self.global_rank == 0:
                self.logger.info("Epoch: %d, time for one epoch: %d seconds" % (epoch, time.time() - epoch_start))
                logs = [('Epoch', epoch), ('time', time.time() - epoch_start)]
                self.log(logs)
        
        print('End training....')
        self.writer.close()

    def eval(self):
        val_loader = DataLoader(self.val_dataset, shuffle=False, pin_memory=False,
                                batch_size=self.config.BATCH_SIZE, num_workers=8)

        self.inpaint_model.eval()  # set model to eval mode

        ssim_all, psnr_all, len_all = 0., 0., 0. 
        s_psnr_all = 0. 
        video_length_all = 0 
        vfid = 0.

        i3d_model = init_i3d_model('ckpt/i3d_rgb_imagenet.pt', device=self.device)
        output_i3d_activations = []
        real_i3d_activations = []
    
        with torch.no_grad(): 
            for items in tqdm(val_loader):
                for k in items:
                    if type(items[k]) is torch.Tensor:
                        items[k] = items[k].to(self.device)
                b, t, _, _, _ = items['edges'].shape
                
                if self.inpaint_model.iteration > int(self.config.MIX_ITERS):  # add 8/19 when checking the edge/line inpainting
                    # gray_frames = (items['frames'].mean(2, keepdim=True) + 1.0) / 2.0
                    # edge_pred, line_pred = SampleEdgeLineLogits_video(self.inpaint_model.transformer,
                    #     context=[gray_frames.to(torch.float16), items['edges'].to(torch.float16), items['lines'].to(torch.float16)],
                    #     masks=items['masks'].clone().to(torch.float16),
                    #     # iterations=5,
                    #     iterations=1, # test 0925
                    #     add_v=0.05, mul_v=4, device=self.device)
                    edge_pred, line_pred = SampleEdgeLineLogits_video(self.inpaint_model.transformer,
                        context=[items['grays'], items['edges'], items['lines']],
                        masks=items['masks'].clone(), device=self.device)
                    edge_pred, line_pred = edge_pred.detach().to(torch.float32), line_pred.detach().to(torch.float32)

                    items['edges'] = edge_pred # the inpainted edges
                    items['lines'] = line_pred # # the inpainted lines
                
                # 0917 using predicted edge and line
                # # gray_frames = (items['frames'].mean(2, keepdim=True) + 1.0) / 2.0
                # edge_pred, line_pred = SampleEdgeLineLogits_video(self.inpaint_model.transformer,
                #     context=[items['grays'], items['edges'], items['lines']],
                #     masks=items['masks'].clone(), iterations=5, device=self.device)
                # edge_pred, line_pred = edge_pred.detach().to(torch.float32), line_pred.detach().to(torch.float32)

                # items['edges'] = edge_pred # the inpainted edges
                # items['lines'] = line_pred # # the inpainted lines
                
                # eval
                items = self.inpaint_model(items)
                outputs_merged = (items['predicted_video'] * items['masks']) + (items['frames'] * (1 - items['masks']))

                # save
                outputs_merged = ( outputs_merged + 1 ) / 2 # [-1, 1] -> [0, 1]
                outputs_merged *= 255.0
                outputs_merged = outputs_merged.permute(0, 1, 3, 4, 2).int().cpu().numpy()
                items['frames'] = ( items['frames'] + 1 ) / 2 # [-1, 1] -> [0, 1] 
                items['frames'] *= 255
                items['frames'] = items['frames'].permute(0, 1, 3, 4, 2).int().cpu().numpy()
                ssim, s_psnr = 0., 0.
                for img_num in range(b):
                    for i in range(t):
                        pred = outputs_merged[img_num][i]
                        gt = items['frames'][img_num][i]
                        
                        pred_img = np.array(pred)
                        gt_img = np.array(gt)
                        # ssim += measure_ssim(gt_img, pred_img, data_range=255, multichannel=True, win_size=65)
                        ssim += measure_ssim(gt_img, pred_img, data_range=255, channel_axis=2)
                        s_psnr += measure_psnr(gt_img, pred_img, data_range=255)
                        
                        ''' disable saving evaluation images temporarily
                        path = os.path.join(self.val_path, items['name'][img_num])
                        if not os.path.exists(path):
                            os.makedirs(path)
                        cv2.imwrite(os.path.join(path, "pred_"+items['idxs'][i][img_num]), pred[:, :, ::-1])
                        cv2.imwrite(os.path.join(path, "gt_"+items['idxs'][i][img_num]), gt[:, :, ::-1])
                        '''
                    
                    # FVID computation
                    # get i3d activations 
                    gts = torch.from_numpy(items['frames'][img_num]).unsqueeze(0).to(self.device)
                    preds = torch.from_numpy(outputs_merged[img_num]).unsqueeze(0).to(self.device)
                    gts = gts.permute(0, 1, 4, 2, 3)
                    preds =  preds.permute(0, 1, 4, 2, 3)
                    # tranfer gts and preds to float tensor and constrain to 0~1
                    gts = gts.to(torch.float32) / 255.0
                    preds = preds.to(torch.float32) / 255.0
                    real_i3d_activations.append(get_i3d_activations(gts, i3d_model).cpu().numpy().flatten())
                    output_i3d_activations.append(get_i3d_activations(preds, i3d_model).cpu().numpy().flatten())
                        
                ssim_all += ssim
                s_psnr_all += s_psnr

        vfid_score = get_fid_score(real_i3d_activations, output_i3d_activations)
        # vfid_score = calculate_fid(real_i3d_activations, output_i3d_activations)
        # print(vfid_score)
        # vfid_score = get_fid_score(real_i3d_activations, output_i3d_activations) / self.config.BATCH_SIZE
        ssim_final = ssim_all/(len(val_loader)*self.sample_length*self.config.BATCH_SIZE)
        s_psnr_final = s_psnr_all/(len(val_loader)*self.sample_length*self.config.BATCH_SIZE)
        
        if self.global_rank == 0:
            self.logger.info("Eval iter: %d, PSNR: %f, SSIM: %f, VFID: %f" %
                    (self.inpaint_model.iteration, float(s_psnr_final), float(ssim_final),
                    float(vfid_score)))
            logs = [('iter', self.inpaint_model.iteration), ('PSNR', float(s_psnr_final)),
                    ('SSIM', float(ssim_final)), ('VFID', float(vfid_score))]
            self.log(logs)
        return float(s_psnr_final), float(ssim_final), float(vfid_score)

    def sample(self, it=None):
        # do not sample when validation set is empty
        if len(self.val_dataset) == 0:
            return

        self.inpaint_model.eval()
        with torch.no_grad():
            items = next(self.sample_iterator)
            for k in items:
                if type(items[k]) is torch.Tensor:
                    items[k] = items[k].to(self.device)
            b, t, _, _, _ = items['edges'].shape
            # gray_frames = (items['frames'].mean(2, keepdim=True) + 1.0) / 2.0
            # edges_pred, lines_pred = SampleEdgeLineLogits_video(self.inpaint_model.transformer,
            #                                             context=[gray_frames[:b, ...].to(torch.float16),
            #                                                      items['edges'][:b, ...].to(torch.float16),
            #                                                      items['lines'][:b, ...].to(torch.float16)],
            #                                             masks=items['masks'][:b, ...].clone().to(torch.float16),
            #                                             # iterations=5,
            #                                             iterations=1,
            #                                             # add_v=0.05, mul_v=4,
            #                                             device=self.device)
            edges_pred, lines_pred = SampleEdgeLineLogits_video(self.inpaint_model.transformer,
                context=[items['grays'][:b, ...], items['edges'][:b, ...], items['lines'][:b, ...]],
                masks=items['masks'][:b, ...].clone(), iterations=5, device=self.device)
            edges_pred, lines_pred = edges_pred[:b, ...].detach().to(torch.float32), \
                                   lines_pred[:b, ...].detach().to(torch.float32)
            # if self.config.fix_256 is None or self.config.fix_256 is False:
            #     edges_pred = self.inpaint_model.structure_upsample(edge_preds)[0]
            #     edges_pred = torch.sigmoid((edges_pred + 2) * 2)
            #     lines_pred = self.inpaint_model.structure_upsample(lines_pred)[0]
            #     lines_pred = torch.sigmoid((lines_pred + 2) * 2)
            items['edges'][:b, ...] = edges_pred.detach()
            items['lines'][:b, ...] = lines_pred.detach()
            # inpaint model
            iteration = self.inpaint_model.iteration
            inputs = (items['frames'] * (1 - items['masks']))
            items = self.inpaint_model(items)
            outputs_merged = (items['predicted_video'] * items['masks']) + (items['frames'] * (1 - items['masks']))
            outputs_merged = ( outputs_merged + 1 ) / 2 # [-1, 1] -> [0, 1] # test

        if it is not None:
            iteration = it

        image_per_row = 2
        if self.config.SAMPLE_SIZE <= 6:
            image_per_row = 1
            
        for b in range(items['frames'].shape[0]):            
            images = stitch_images(
                self.postprocess(((items['frames'][b,:,...] + 1) / 2).cpu()),
                self.postprocess(((inputs[b,:,...] + 1 ) / 2).cpu()),
                self.postprocess(items['edges'][b,:,...].cpu()),
                self.postprocess(items['lines'][b,:,...].cpu()),
                self.postprocess(items['masks'][b,:,...].cpu()),
                self.postprocess(((items['predicted_video'][b,:,...] + 1 ) / 2).cpu()),
                self.postprocess((outputs_merged[b,:,...]).cpu()),
                img_per_row=image_per_row
            )

            name = os.path.join(self.samples_path, str(iteration).zfill(6) + f"batch{b}.jpg")
            create_dir(self.samples_path)
            # print('\nsaving sample ' + name)
            images.save(name)

    def log(self, logs):
        with open(self.log_file, 'a') as f:
            f.write('%s\n' % ' '.join([str(item[0]) + '\t' + str(item[1]) for item in logs]))
        self.logger.info(', '.join([str(item[0]) + ': ' + ("%.5f" % item[1]) for item in logs]))

    def cuda(self, *args):
        return (item.to(self.config.DEVICE) for item in args)

    def postprocess(self, img):
        # [0, 1] => [0, 255]
        img = img * 255.0
        img = img.permute(0, 2, 3, 1)
        return img.int()

    def eval_whole_video(self, useGTLineEdge=False):
        self.inpaint_model.eval()  # set model to eval mode

        # frame_list, mask_list, edge_list, line_list = get_frame_mask_edge_line_list(self.args) # get frame and mask list
        # assert len(frame_list) == len(mask_list) == len(edge_list) == len(line_list) # check if the number of frames and masks are the same
        # video_num = len(frame_list) # number of videos

        test_dataset = TestDataset(self.args)

        ssim_all, psnr_all, len_all = 0., 0., 0. 
        s_psnr_all = 0. 
        # lpips_all, vif_all = 0., 0.
        # lpips_fn = lpips.LPIPS(net='vgg')
        count = 0
        video_length_all = 0 
        vfid = 0.
        
        i3d_model = init_i3d_model('ckpt/i3d_rgb_imagenet.pt', device=self.device)
        output_i3d_activations = []
        real_i3d_activations = []

        if self.args.useGT_LE:
            save_dir = save_result_dir = os.path.join("./results", self.args.model_name + '(GT)', self.args.input.split("/")[-1])
        else:
            save_dir = save_result_dir = os.path.join("./results", self.args.model_name, self.args.input.split("/")[-1])

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(os.path.join(save_dir, 'log.txt'))
        logger.addHandler(fh)
        logger.propagate = False

        for video_no in range(len(test_dataset)): # iterate over all videos
            video_name = test_dataset.get_name(video_no)

            idx_lst, frames_PIL, grays_PIL, edges, lines, masks = test_dataset[video_no]
            video_length = len(idx_lst)
            if(video_length == 0):
                continue

            print("[Processing video {}: {}]".format(video_no, video_name)) # print video number and name
            # logger.info("[Processing video {}: {}]".format(video_no, video_name)) # log video number and name

            # create a timestamp string for the current date
            # timestamp = datetime.datetime.now().strftime("%Y-%m-%d_")
            # input_name = self.args.input.split("/")[-1]
            # if self.args.output == "":
            #     save_result_dir = os.path.join("./results", self.args.model_name, timestamp+input_name, video_name) # save result directory
            # else:
            #     save_result_dir = os.path.join("./results", self.args.model_name, self.args.output, timestamp+input_name, video_name)
            save_result_dir = os.path.join(save_dir, video_name) # save result directory

            if not os.path.exists(save_result_dir):
                os.makedirs(save_result_dir)
            
            # frames_PIL, idx_lst = read_frame_from_videos(frame_list[video_no], self.w, self.h) # read frames from video
            # video_length = len(frames_PIL) # get video length, how many frames in this video

            # imgs = to_tensors(frames_PIL).unsqueeze(0)*2-1 # convert frames to tensors and normalize to [-1, 1]
            # # imgs = to_tensors(frames_PIL).unsqueeze(0) # convert frames to tensors and normalize to [-1, 1]
            # frames = [np.array(f).astype(np.uint8) for f in frames_PIL] # convert frames to numpy array

            # masks = read_mask(mask_list[video_no], self.w, self.h)    
            # binary_masks = [np.expand_dims((np.array(m) != 0).astype(np.uint8), 2) for m in masks] # convert masks to numpy array
            # masks = to_tensors(masks).unsqueeze(0) # convert masks to tensors

            # edges, lines = read_edge_line_PIL(edge_list[video_no], line_list[video_no], self.w, self.h)
            # edges = to_tensors(edges).unsqueeze(0)
            # lines = to_tensors(lines).unsqueeze(0)

            imgs = to_tensors(frames_PIL).unsqueeze(0) * 2.0 - 1.0  # from [0, 1] to [-1, 1]
            frames = [np.array(f).astype(np.uint8) for f in frames_PIL]
            grays = to_tensors(grays_PIL).unsqueeze(0)
            edges = to_tensors(edges).unsqueeze(0)
            lines = to_tensors(lines).unsqueeze(0)
            # binary_masks = [np.expand_dims((np.array(m) != 0).astype(np.uint8), 0) for m in masks]
            masks = to_tensors(masks).unsqueeze(0)

            comp_frames = [None]*video_length # initialize completed frames
            out_frames = [None]*video_length  # initialize output frames

            for f in tqdm(range(0, video_length, 5)): # iterate over all frames in this video, neighbor_stride=5
                '''
                # FuseFormer version
                neighbor_ids = [i for i in range(max(0, f-self.neighbor_stride), min(video_length, f+self.neighbor_stride+1))] # get neighbor frames
                ref_ids = get_ref_index_fuseformer(f, neighbor_ids, video_length) # get reference frames, approximate the whole story of the video
                len_temp = len(neighbor_ids) + len(ref_ids) # get the number of frames used for inpainting

                if f in neighbor_ids:
                    neighbor_ids.remove(f)

                if f in ref_ids:
                    ref_ids.remove(f)

                # neighbor_ids = random.sample(neighbor_ids, int(self.sample_length/2)) # randomly select neighbor frames
                # ref_ids = random.sample(ref_ids, int(self.sample_length/2))

                # revised 
                sample_length_half = int(self.sample_length/2)
                if len(neighbor_ids) >= sample_length_half and len(ref_ids) >= sample_length_half:
                    # 兩者皆大於則照原本的程式
                    neighbor_ids = random.sample(neighbor_ids, sample_length_half)  # randomly select neighbor frames
                    ref_ids = random.sample(ref_ids, sample_length_half)
                else:
                    if len(ref_ids) < len(neighbor_ids):
                        shortage = sample_length_half - len(ref_ids)
                        neighbor_ids = random.sample(neighbor_ids, sample_length_half+shortage)
                    else:
                        shortage = sample_length_half - len(neighbor_ids)
                        ref_ids = random.sample(ref_ids, sample_length_half+shortage)
                # revised 

                reference_frames = [ f ] + neighbor_ids + ref_ids # get reference frames
                reference_frames.sort() # sort reference frames
                # print(f"reference_frames: {reference_frames}") # test
                '''
                if f + 5 > video_length:
                    reference_frames = [i for i in range(video_length-5, video_length)]
                else:
                    reference_frames = [i for i in range(f, f+5)]

                selected_imgs = imgs[:1, reference_frames, :, :, :] # select frames for inpainting with neighbor frames and reference frames
                selected_grays = grays[:1, reference_frames, :, :, :]
                selected_masks = masks[:1, reference_frames, :, :, :] # select masks for inpainting 
                # print(f"selected_masks: {selected_masks[0][0]}") # test
                selected_edges = edges[:1, reference_frames, :, :, :]
                selected_lines = lines[:1, reference_frames, :, :, :]

                # ZITS version of seleting reference frames
                # neighbor_ids = get_ref_index(video_length, self.sample_length)
                # selected_imgs = imgs[:1, neighbor_ids, :, :, :] # select frames for inpainting with neighbor frames and reference frames
                # selected_masks = masks[:1, neighbor_ids, :, :, :] # select masks for inpainting 
                # selected_edges = edges[:1, neighbor_ids, :, :, :]
                # selected_lines = lines[:1, neighbor_ids, :, :, :]

                with torch.no_grad():
                    selected_imgs, selected_masks = selected_imgs.to(self.device), selected_masks.to(self.device) # move tensors to GPU
                    selected_edges, selected_lines = selected_edges.to(self.device), selected_lines.to(self.device)
                    selected_grays = selected_grays.to(self.device)

                    # selected_imgs = selected_imgs*(1-selected_masks) # add 7/17
                    # selected_edges = selected_edges*(1-selected_masks) # add 7/17
                    # selected_lines = selected_lines*(1-selected_masks) # add 7/17
                    if useGTLineEdge:
                        selected_edges = selected_edges.detach()  # old version before 0818
                        selected_lines = selected_lines.detach()  # old version before 0818
                    else: # use predicted edge and line
                        # rgb to gray using NTSC conversion
                        # gray_imgs = (selected_imgs + 1.0) / 2.0
                        # gray_imgs = 0.2989 * gray_imgs[:, :, 0, :, :] + 0.5870 * gray_imgs[:, :, 1, :, :] + 0.1140 * gray_imgs[:, :, 2, :, :]
                        # gray_imgs = gray_imgs.unsqueeze(2)  # Add the channel dimension back
                        
                        # gray_imgs = (selected_imgs.mean(2, keepdim=True) + 1.0) / 2.0  # from [-1, 1] to [0, 1]

                        gray_imgs = selected_grays
                        
                        # print(gray_imgs.shape, selected_edges.shape, selected_lines.shape, selected_masks.shape)
                        # print(min(gray_imgs), max(gray_imgs))
                        # print(min(selected_edges), max(selected_edges))
                        # print(min(selected_lines), max(selected_lines))
                        # print(min(selected_masks), max(selected_masks))
                        # time.sleep(1000000)

                        edge_pred, line_pred = SampleEdgeLineLogits_video(self.inpaint_model.transformer,
                            context=[gray_imgs, selected_edges, selected_lines], 
                            # masks=selected_masks.to(torch.float16), iterations=5, add_v=0.05, mul_v=4, device=self.device)   
                            # masks=selected_masks.to(torch.float16).clone(), iterations=1, add_v=0.05, mul_v=4, device=self.device)   # test the setting 0925
                            masks=selected_masks.clone(), mul_v=3, iterations=5)   # test the setting 0925
                        edge_pred, line_pred = edge_pred.detach().to(torch.float32), line_pred.detach().to(torch.float32)

                        
                        '''
                        # check whether the result folders are exist
                        if not os.path.exists(f"{save_result_dir}/pred_edges"):
                            os.makedirs(f"{save_result_dir}/pred_edges")

                        if not os.path.exists(f"{save_result_dir}/pred_lines"):
                            os.makedirs(f"{save_result_dir}/pred_lines")

                        if not os.path.exists(f"{save_result_dir}/gt_edges"):
                            os.makedirs(f"{save_result_dir}/gt_edges")

                        if not os.path.exists(f"{save_result_dir}/gt_lines"):
                            os.makedirs(f"{save_result_dir}/gt_lines")

                        # save the masked edge and line
                        if not os.path.exists(f"{save_result_dir}/masked_edges"):
                            os.makedirs(f"{save_result_dir}/masked_edges")

                        if not os.path.exists(f"{save_result_dir}/masked_lines"):
                            os.makedirs(f"{save_result_dir}/masked_lines")
                        
                        for i in range(5):
                            edge = selected_edges[0][i]
                            pred_edge = edge_pred[0][i]
                            line = selected_lines[0][i]
                            pred_line = line_pred[0][i]
                            mask = selected_masks[0][i]
                            # torchvision.utils.save_image(edge, f"20230925check/edge_{i}.png")
                            torchvision.utils.save_image(edge, f"{save_result_dir}/gt_edges/{idx_lst[reference_frames[i]]}")
                            torchvision.utils.save_image(pred_edge, f"{save_result_dir}/pred_edges/{idx_lst[reference_frames[i]]}")
                            torchvision.utils.save_image(line, f"{save_result_dir}/gt_lines/{idx_lst[reference_frames[i]]}")
                            torchvision.utils.save_image(pred_line, f"{save_result_dir}/pred_lines/{idx_lst[reference_frames[i]]}")

                            # compute the masked edge and line where the mask is 1, so we need to invert the mask and then multiply it with edge and line
                            mask = 1 - mask
                            mask_edge = edge * mask
                            mask_line = line * mask
                            torchvision.utils.save_image(mask_edge, f"{save_result_dir}/masked_edges/{idx_lst[reference_frames[i]]}")
                            torchvision.utils.save_image(mask_line, f"{save_result_dir}/masked_lines/{idx_lst[reference_frames[i]]}")
                        '''

                        # 0918 save concat pictures
                        if not os.path.exists(f"{save_result_dir}/struct_pred"):
                            os.makedirs(f"{save_result_dir}/struct_pred")
                        for i in range(5):
                            mask = selected_masks[0][i]
                            gray = gray_imgs[0][i]
                            mask_gray = gray * (1 - mask)
                            
                            edge = selected_edges[0][i]
                            edge = edge * (1 - mask) + mask * (1 - edge)
                            pred_edge = edge_pred[0][i]
                            pred_edge = pred_edge * (1 - mask) + mask * (1 - pred_edge)

                            line = selected_lines[0][i]
                            line = line * (1 - mask) + mask * (1 - line)
                            pred_line = line_pred[0][i]
                            pred_line = pred_line * (1 - mask) + mask * (1 - pred_line)

                            # edge = selected_edges[0][i]
                            # pred_edge = edge_pred[0][i]
                            # line = selected_lines[0][i]
                            # pred_line = line_pred[0][i]
                            # mask_edge = edge * (1 - mask)
                            # mask_line = line * (1 - mask)

                            line_cat = torch.cat([gray, line, pred_line], dim=2)
                            edge_cat = torch.cat([mask_gray, edge, pred_edge], dim=2)
                            total_cat = torch.cat([line_cat, edge_cat], dim=1)
                            # edge_cat = torch.cat([edge, pred_edge, mask_edge], dim=2)
                            # line_cat = torch.cat([line, pred_line, mask_line], dim=2)
                            # total_cat = torch.cat([edge_cat, line_cat], dim=1)

                            out = (total_cat * 255).permute(1, 2, 0).cpu().detach().numpy().astype(np.uint8)
                            # print(f"save in {save_result_dir}/struct_pred/{idx_lst[reference_frames[i]]}")
                            # print(out.shape)
                            # time.sleep(1000000)
                            cv2.imwrite(f"{save_result_dir}/struct_pred/{idx_lst[reference_frames[i]]}", out[:, :, ::-1])

                            # torchvision.utils.save_image(total_cat, f"{save_result_dir}/struct_pred/{idx_lst[reference_frames[i]]}")
                        
                        selected_edges = edge_pred # new version after 0925
                        selected_lines = line_pred # new version after 0925
                                        
                    # 1) GT
                    # selected_edges = selected_edges.detach()  # old version before 0818
                    # selected_lines = selected_lines.detach()  # old version before 0818
                    # inpainted Line Edge
                    # 2)
                    # selected_edges = edge_pred * selected_masks + selected_edges * (1 - selected_masks)  # new version after 0818
                    # selected_lines = line_pred * selected_masks + selected_lines * (1 - selected_masks)  # new version after 0818
                    # 3)
                    # selected_edges = edge_pred # new version after 0925
                    # selected_lines = line_pred # new version after 0925

                    items = dict()
                    items['frames'] = selected_imgs
                    items['masks'] = selected_masks
                    items['edges'] = selected_edges
                    items['lines'] = selected_lines
                    items['name'] = video_name
                    # batch['idxs'] = selected_idx

                    # load pos encoding
                    rel_pos_list, abs_pos_list, direct_list = [], [], []
                    for m in selected_masks[0]:
                        # transfrom mask to numpy array
                        rel_pos, abs_pos, direct = load_masked_position_encoding(np.array(m.cpu()).squeeze(0), self.input_size, self.pos_num, self.str_size)
                        # transfer tensor [4, 256, 256] to [4, 1, 1, 256, 256]
                        rel_pos = torch.from_numpy(rel_pos).unsqueeze(0)
                        abs_pos = torch.from_numpy(abs_pos).unsqueeze(0)
                        direct = torch.from_numpy(direct).unsqueeze(0)

                        rel_pos_list.append(rel_pos)
                        abs_pos_list.append(abs_pos)
                        direct_list.append(direct)

                    # concat rel_pos, abs_pos, direct individually in dimention 1
                    rel_pos_list = torch.cat(rel_pos_list, dim=0).unsqueeze(0)
                    abs_pos_list = torch.cat(abs_pos_list, dim=0).unsqueeze(0)
                    direct_list = torch.cat(direct_list, dim=0).unsqueeze(0)

                    items['rel_pos'] = rel_pos_list.clone().to(torch.long).to(self.device)
                    items['abs_pos'] = abs_pos_list.clone().to(torch.long).to(self.device)
                    items['direct'] = direct_list.clone().to(torch.long).to(self.device)

                    items = self.inpaint_model(items)
                    
                    outputs_merged = (items['predicted_video'] * items['masks']) + (items['frames'] * (1 - items['masks']))
                    outputs = items['predicted_video']

                    outputs_merged = self.postprocess(((outputs_merged + 1 )/ 2).cpu().squeeze(0)) # postprocess the inpainted frames
                    outputs = self.postprocess(((outputs + 1 )/ 2).cpu().squeeze(0)) # postprocess the inpainted frames
                    # outputs_merged = ( outputs_merged + 1 ) / 2 # [-1, 1] -> [0, 1] # test
                    # outputs_merged *= 255.0
                    # outputs_merged = outputs_merged.squeeze(0) # get rid of the batch dimension
                    # outputs_merged = outputs_merged.int().cpu().permute(0, 2, 3, 1).numpy()
                    for i in range(len(reference_frames)): # iterate over all reference frames
                        idx = reference_frames[i] # get the index of the neighbor frame
                        # img = np.array(outputs_merged[i]).astype( 
                        #     np.uint8)*binary_masks[idx] + frames[idx] * (1-binary_masks[idx]) # get the inpainted frame
                        img = np.array(outputs_merged[i]).astype(np.uint8) # get the inpainted frame
                        img2 = np.array(outputs[i]).astype(np.uint8) # get the inpainted frame
                        if comp_frames[idx] is None: # if the completed frame is None, initialize it
                            comp_frames[idx] = img
                            out_frames[idx] = img2
                        else: 
                            comp_frames[idx] = comp_frames[idx].astype(
                                np.float32)*0.5 + img.astype(np.float32)*0.5 # inpainted multiple times, and get the average result
                            out_frames[idx] = out_frames[idx].astype(np.float32)*0.5 + img2.astype(np.float32)*0.5
                        # comp_frames[idx] = img # test without average

            ssim, psnr, s_psnr = 0., 0., 0. 
            # lpip, vif = 0., 0.
            comp_PIL = []
            comp_list, gt_list = [], []
            print(f"save in {save_result_dir}")
            out_frames_tensor = to_tensors(out_frames).unsqueeze(0)
            comp_frames_tensor = to_tensors(comp_frames).unsqueeze(0)
            for f in range(video_length): # iterate over all frames in this video
                comp = comp_frames[f] # get the completed frame
                # gt = np.array(frames_PIL[f])
                comp = cv2.cvtColor(np.array(comp).astype(np.uint8), cv2.COLOR_BGR2RGB) # convert the completed frame to RGB
                # comp = 
                # print(f"save in {os.path.join(save_result_dir, idx_lst[f])}")
                cv2.imwrite(os.path.join(save_result_dir, idx_lst[f]), comp) # save the completed frame
                new_comp = cv2.imread(os.path.join(save_result_dir, idx_lst[f])) # read the saved completed frame
                new_comp = Image.fromarray(cv2.cvtColor(new_comp, cv2.COLOR_BGR2RGB)) # convert the completed frame to RGB
                # new_comp = Image.fromarray(comp) # test
                comp_PIL.append(new_comp) # append the completed frame to the list

                gt = cv2.cvtColor(np.array(frames[f]).astype(np.uint8), cv2.COLOR_BGR2RGB) # convert the ground truth frame to RGB
                # gt = np.array(frames[f]).astype(np.uint8)
                # ssim += measure_ssim(comp, gt, data_range=255, multichannel=True, win_size=65, channel_axis=2) # compute SSIM
                ssim += measure_ssim(comp, gt, data_range=255, win_size=65, channel_axis=2) # compute SSIM
                s_psnr += measure_psnr(comp, gt, data_range=255) # compute PSNR
                comp_list.append(comp)
                gt_list.append(gt)
                # comp_tensor = torch.from_numpy(comp).permute(2, 0, 1).unsqueeze(0).to(self.device).float() / 255.0 * 2.0 - 1.0
                # gt_tensor = torch.from_numpy(gt).permute(2, 0, 1).unsqueeze(0).to(self.device).float() / 255.0 * 2.0 - 1.0
                # vif += vifp(comp, gt)
                count += 1

                # save stitched images
                if not os.path.exists(f"{save_result_dir}/stitched"):
                    os.makedirs(f"{save_result_dir}/stitched")
                img = stitch_images(
                    self.postprocess(((imgs[:1, f, ...] + 1) / 2).cpu()),
                    self.postprocess((((imgs * (1 - masks))[:1, f, ...] + 1) / 2).cpu()),
                    self.postprocess(out_frames_tensor[:1, f, ...].cpu()),
                    self.postprocess(comp_frames_tensor[:1, f, ...].cpu()),
                    img_per_row=1
                )
                # save png file
                file_name = idx_lst[f].split(".")[0] + ".png"
                img.save(f"{save_result_dir}/stitched/{file_name}")

            # comp_tensor = torch.Tensor(np.array(comp_list)).permute(0, 3, 1, 2).float() / 255.0 * 2.0 - 1.0
            # gt_tensor = torch.Tensor(np.array(gt_list)).permute(0, 3, 1, 2).float() / 255.0 * 2.0 - 1.0
            # lpip = lpips_fn.forward(comp_tensor, gt_tensor).sum().item()
            ssim_all += ssim
            s_psnr_all += s_psnr
            # lpips_all += lpip
            # vif_all += vif
            video_length_all += (video_length)
            # FVID computation
            imgs = to_tensors(comp_PIL).unsqueeze(0).to(self.device)
            gts = to_tensors(frames_PIL).unsqueeze(0).to(self.device)
            out_act = get_i3d_activations(imgs, i3d_model).cpu().numpy().flatten()
            real_act = get_i3d_activations(gts, i3d_model).cpu().numpy().flatten()
            output_i3d_activations.append(out_act)
            real_i3d_activations.append(real_act)
            fid_score_tmp = get_fid_score([out_act], [real_act]) # test
            # if video_no % 50 ==1:
            #     print("video no[{}]: ssim {}, psnr {}, vfid {}".format(video_no, ssim_all/video_length_all, s_psnr_all/video_length_all, fid_score_tmp))
            # print("video no[{}]: ssim {}, psnr {}, vfid {}, lpips {}, vif {}".format(video_no, ssim/video_length, s_psnr/video_length, fid_score_tmp, lpip/video_length, vif/video_length))
            # logger.info("video no[{}]: ssim {}, psnr {}, vfid {}, lpips {}, vif {}".format(video_no, ssim/video_length, s_psnr/video_length, fid_score_tmp, lpip/video_length, vif/video_length))
            print("video no[{}]: ssim {}, psnr {}, vfid {}".format(video_no, ssim/video_length, s_psnr/video_length, fid_score_tmp))
            # logger.info("video no[{}]: ssim {}, psnr {}, vfid {}".format(video_no, ssim/video_length, s_psnr/video_length, fid_score_tmp))
            logger.info(
                f'[{video_no+1:3}/{len(test_dataset)}] Name: {str(video_name):25} | PSNR: {(s_psnr/video_length):.6f} | SSIM: {(ssim/video_length):.6f} | VFID: {fid_score_tmp:.6f}'
            )

        # ssim_final = ssim_all/(video_length_all*self.sample_length*self.config.BATCH_SIZE)
        # psnr_final = s_psnr_all/(video_length_all*self.sample_length*self.config.BATCH_SIZE)
        ssim_final = ssim_all/count
        psnr_final = s_psnr_all/count
        # lpips_final = lpips_all/count
        # vif_final = vif_all/count
        vfid_score = get_fid_score(real_i3d_activations, output_i3d_activations)
        # print("[Finish evaluating, ssim is {}, psnr is {}, vfid: {}, lpips is {}, vif is {}]".format(ssim_final, psnr_final, vfid_score, lpips_final, vif_final))
        # logger.info("[Finish evaluating, ssim: {}, psnr: {}, vfid: {}, lpips: {}, vif: {}]".format(ssim_final, psnr_final, vfid_score, lpips_final, vif_final))
        print("Finish evaluation... Average Frame PSNR / SSIM / VFID: {} / {} / {}".format(psnr_final, ssim_final, vfid_score))
        logger.info("Finish evaluation... Average Frame PSNR / SSIM / VFID: {} / {} / {}".format(psnr_final, ssim_final, vfid_score))
        