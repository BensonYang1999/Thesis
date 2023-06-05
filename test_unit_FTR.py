import os
import random
import sys
from glob import glob
import logging

import cv2
import numpy as np
import torchvision.transforms.functional as F
from skimage.color import rgb2gray
from skimage.feature import canny
from torch.utils.data import Dataset
import pickle
import skimage.draw
import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.dataloader import DataLoader

import torch.nn as nn
from src.models.TSR_model import EdgeLineGPT256RelBCE_video, EdgeLineGPTConfig
import torchvision.transforms as T

import torchvision.transforms as transforms
from src.Fuseformer.utils import create_random_shape_with_random_motion
from src.Fuseformer.utils import Stack, ToTorchFormatTensor, GroupRandomHorizontalFlip
from PIL import Image

import gc
gc.collect()
torch.cuda.empty_cache()

"""
FuseFormer imports
"""
import numpy as np
import time
import math
from functools import reduce
import torchvision.models as models
from torch.nn import functional as nnF

from src.models.fuseformer import InpaintGenerator

to_tensors = transforms.Compose([
    Stack(),
    ToTorchFormatTensor(), ])

sys.path.append('..')

logger = logging.getLogger(__name__)

def to_int(x):
    return tuple(map(int, x))

class ImgDataset_video(torch.utils.data.Dataset):

    # write a video version of __init__ function
    def __init__(self, args, sample=5, size=(432, 240), spit='train', name='YoutubeVOS', root='./datasets'):
        self.split = split
        self.sample_length = sample
        self.input_size = self.w, self.h = size
        self.args = args
        if name == 'YouTubeVOS':
            vid_lst_prefix = os.path.join(root, name, split+'_all_frames/JPEGImages')
        vid_lst = os.listdir(vid_lst_prefix)
        self.video_names = [os.path.join(vid_lst_prefix, name) for name in vid_lst]

        # don't read all the frames at once, save the video names and read them later
        # TODO: reading specific masking input

    def __len__(self): # return the number of videos
        return len(self.video_names)

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            print('Loading error in video {}'.format(self.video_names[index]))
            item = self.load_item(0)
        return item

    def load_name(self, index): # load the base name of the video with given index
        return os.path.basename(self.video_names[index])

    def load_item(self, index):
        video_name = self.video_names[index]
        all_frames = [os.path.join(video_name, name) for name in sorted(os.listdir(video_name))]
        all_masks = create_random_shape_with_random_motion(
            len(all_frames), imageHeight=self.h, imageWidth=self.w)
    
        ref_index = self.get_ref_index(len(all_frames), self.sample_length)
        frames = []
        masks = []

        for idx in ref_index:
            img = Image.open(all_frames[idx]).convert('RGB')
            img = img.resize(self.input_size)

            frames.append(img)
            masks.append(all_masks[idx])

        if self.split=='train':
            prob = random.random()
            frames = GroupRandomHorizontalFlip()(frames, prob)

        batch = dict()
        batch['frames'] = to_tensors(frames)*2.0 - 1.0 # [0, 1] => [-1, 1] normalize
        batch['masks'] = to_tensors(masks)
        batch['name'] = self.load_name(index)
        batch['ref_index'] = ref_index
        return batch


    def get_ref_index(self, length, sample_length):
        if random.uniform(0, 1) > 0.5:
            ref_index = random.sample(range(length), sample_length)
            ref_index.sort()
        else:
            pivot = random.randint(0, length-sample_length)
            ref_index = [pivot+i for i in range(sample_length)]
        return ref_index

    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True
            )

            for item in sample_loader:
                yield item

class ToLongTensorFromNumpy(object):
    """Convert numpy arrays in sample to Tensors."""
    def __call__(self, sample):
        return torch.from_numpy(sample).long()

class DynamicDataset_video(torch.utils.data.Dataset):
    def __init__(self, split='train', name='YouTubeVOS', root='./datasets', batch_size=1, add_pos=False, pos_num=128, test_mask_path=None,
                 input_size=None, default_size=(432, 240), str_size=256,
                 world_size=1,
                 round=1, sample=5): 
        # super(DynamicDataset, self).__init__()
        self.training = (split=='train' or split=='valid') # training or testing
        self.batch_size = batch_size
        self.round = round  # for places2 round is 32
        self.sample_length = sample
        self.split = split
        
        self.video_name = []

        if name == 'YouTubeVOS':
            vid_lst_prefix = os.path.join(root, name, split+'_all_frames/JPEGImages')
            edge_lst_prefix = os.path.join(root, name, split+'_all_frames/edges')
            line_lst_prefix = os.path.join(root, name, split+'_all_frames/wireframes')
            vid_lst = os.listdir(vid_lst_prefix)
            edge_lst = os.listdir(edge_lst_prefix)
            line_lst = os.listdir(line_lst_prefix)
            self.video_names = [os.path.join(vid_lst_prefix, name) for name in vid_lst]
            self.edge_names = [os.path.join(edge_lst_prefix, name) for name in edge_lst]
            self.line_names = [os.path.join(line_lst_prefix, name) for name in line_lst]

        self.video_name = [os.path.join(vid_lst_prefix, name) for name in vid_lst]

        if self.training:
            self.mask_list = None  # if training then generate the mask while loading item
        else:
            self.mask_list = glob.glob(test_mask_path + '/*')
            self.mask_list = sorted(self.mask_list, key=lambda x: x.split('/')[-1])

        self.default_size = default_size
        if input_size is None:
            self.input_size = default_size
        else:
            self.input_size = input_size
        self.str_size = str_size  # 256 fortransformer
        self.world_size = world_size

        self.add_pos = add_pos
        self.ones_filter = np.ones((3, 3), dtype=np.float32)
        self.d_filter1 = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=np.float32)
        self.d_filter2 = np.array([[0, 0, 0], [1, 1, 0], [1, 1, 0]], dtype=np.float32)
        self.d_filter3 = np.array([[0, 1, 1], [0, 1, 1], [0, 0, 0]], dtype=np.float32)
        self.d_filter4 = np.array([[0, 0, 0], [0, 1, 1], [0, 1, 1]], dtype=np.float32)
        self.pos_num = pos_num

        self._to_tensors = transforms.Compose([
            Stack(),
            ToTorchFormatTensor(), ])

        self._to_long_tensors = transforms.Compose([
            Stack(),
            ToLongTensorFromNumpy(),  # Add the custom transform here
        ])

    def reset_dataset(self, shuffled_idx): # reset the dataset for each epoch
        self.idx_map = {} # map the index to the barrel index
        barrel_idx = 0
        count = 0
        for sid in shuffled_idx: # sid is the index of the video
            self.idx_map[sid] = barrel_idx
            count += 1
            if count == self.batch_size:
                count = 0
                barrel_idx += 1
        # random img size:256~512
        # if self.training:
        #     barrel_num = int(len(self.video_name) / (self.batch_size * self.world_size))
        #     barrel_num += 2
        #     if self.round == 1:
        #         self.input_size = np.clip(np.arange(32, 65,
        #                                             step=(65 - 32) / barrel_num * 2).astype(int) * 8, 256, 512).tolist()
        #         self.input_size = self.input_size[::-1] + self.input_size
        #     else:
        #         self.input_size = []
        #         input_size = np.clip(np.arange(32, 65, step=(65 - 32) / barrel_num * 2 * self.round).astype(int) * 8,
        #                              256, 512).tolist()
        #         for _ in range(self.round + 1):
        #             self.input_size.extend(input_size[::-1])
        #             self.input_size.extend(input_size)
        # else:
        #     self.input_size = self.default_size
        self.input_size = self.default_size

    def __len__(self):
        return len(self.video_name)

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            print('Loading error in video {}'.format(self.video_names[index]))
            item = self.load_item(0)
        return item

    def load_name(self, index):
        name = self.video_name[index]
        return os.path.basename(name)

    def load_item(self, index):
        # if type(self.input_size) == list:
        #     maped_idx = self.idx_map[index]
        #     if maped_idx > len(self.input_size) - 1:
        #         size = 512
        #     else:
        #         size = self.input_size[maped_idx]
        # else:
        #     size = self.input_size
        size = self.input_size

        video_name = self.video_name[index]
        edge_name = self.edge_names[index]
        line_name = self.line_names[index]
        all_frames = [os.path.join(video_name, name) for name in sorted(os.listdir(video_name))]
        all_edges = [os.path.join(edge_name, name) for name in sorted(os.listdir(edge_name))]
        all_lines = [os.path.join(line_name, name) for name in sorted(os.listdir(line_name))]
        all_masks = create_random_shape_with_random_motion(
            len(all_frames), imageHeight=self.input_size[1], imageWidth=self.input_size[0])
        ref_index = self.get_ref_index(len(all_frames), self.sample_length)
        
        frames, edges, lines, masks = [], [], [], []
        for idx in ref_index:
            # load image
            img = Image.open(all_frames[idx]).convert('RGB')
            img = img.resize(size)
            frames.append(img)

            # # load mask
            mask = all_masks[idx]
            masks.append(mask)
            # mask = Image.open(all_masks[idx]).convert('L')
            # mask = mask.resize(self.input_size)
            # mask_small = mask.resize((self.str_size, self.str_size))
            # # mask_small[mask_256 > 0] = 255 # make sure the mask is binary 
            # masks.append(mask)
            # masks_small.append(mask_small)

            # load edge
            egde = Image.open(all_edges[idx]).convert('L')
            egde = egde.resize(size)
            edges.append(egde)

            # load line
            line = Image.open(all_lines[idx]).convert('L')
            line = line.resize(size)
            lines.append(line)

        # augment data
        if self.split == 'train':
            prob = random.random()
            frames = GroupRandomHorizontalFlip()(frames, prob)
            edges = GroupRandomHorizontalFlip()(edges, prob)
            lines = GroupRandomHorizontalFlip()(lines, prob)
            # masks = GroupRandomHorizontalFlip()(masks, prob)

        batch = dict()
        batch['frames'] = self._to_tensors(frames) # normalize to [-1, 1]
        # batch['frames_256'] = self._to_tensors(frames_256)*2.0 - 1.0 # normalize to [-1, 1]
        batch['masks'] = self._to_tensors(masks)
        # batch['masks_256'] = self._to_tensors(masks_256)
        batch['edges'] = self._to_tensors(edges)
        # batch['edges_256'] = self._to_tensors(edges_256)
        batch['lines'] = self._to_tensors(lines)
        # batch['lines_256'] = self._to_tensors(lines_256)
        batch['size_ratio'] = size[0] / self.default_size[0]

        batch['name'] = self.load_name(index)
        batch['idxs'] = [all_frames[idx].split('/')[-1] for idx in ref_index]

        # load pos encoding
        rel_pos_list, abs_pos_list, direct_list = [], [], []
        for m in masks:
            # transfrom mask to numpy array
            rel_pos, abs_pos, direct = self.load_masked_position_encoding(np.array(m))
            # transfer tensor [4, 256, 256] to [4, 1, 1, 256, 256]
            rel_pos = torch.from_numpy(rel_pos).unsqueeze(0)
            abs_pos = torch.from_numpy(abs_pos).unsqueeze(0)
            direct = torch.from_numpy(direct).unsqueeze(0)

            rel_pos_list.append(rel_pos)
            abs_pos_list.append(abs_pos)
            direct_list.append(direct)

        # concat rel_pos, abs_pos, direct individually in dimention 1
        rel_pos_list = torch.cat(rel_pos_list, dim=0)
        abs_pos_list = torch.cat(abs_pos_list, dim=0)
        direct_list = torch.cat(direct_list, dim=0)

        batch['rel_pos'] = rel_pos_list.clone().detach().to(torch.long)
        batch['abs_pos'] = abs_pos_list.clone().detach().to(torch.long)
        batch['direct'] = direct_list.clone().detach().to(torch.long)
        # batch['direct'] = batch['direct'].permute(0, 3, 1, 2)

        return batch

    def load_masked_position_encoding(self, mask):
        ori_mask = mask.copy()
        ori_h, ori_w = ori_mask.shape[0:2] # original size
        ori_mask = ori_mask / 255
        # mask = cv2.resize(mask, (self.str_size, self.str_size), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, self.input_size, interpolation=cv2.INTER_AREA)
        mask[mask > 0] = 255 # make sure the mask is binary
        h, w = mask.shape[0:2] # resized size
        mask3 = mask.copy() 
        mask3 = 1. - (mask3 / 255.0) # 0 for masked area, 1 for unmasked area
        pos = np.zeros((h, w), dtype=np.int32) # position encoding
        direct = np.zeros((h, w, 4), dtype=np.int32) # direction encoding
        i = 0
        while np.sum(1 - mask3) > 0: # while there is still unmasked area
            i += 1 # i is the index of the current mask
            mask3_ = cv2.filter2D(mask3, -1, self.ones_filter) # dilate the mask
            mask3_[mask3_ > 0] = 1 # make sure the mask is binary
            sub_mask = mask3_ - mask3 # get the newly added area
            pos[sub_mask == 1] = i # set the position encoding

            m = cv2.filter2D(mask3, -1, self.d_filter1)
            m[m > 0] = 1
            m = m - mask3
            direct[m == 1, 0] = 1

            m = cv2.filter2D(mask3, -1, self.d_filter2)
            m[m > 0] = 1
            m = m - mask3
            direct[m == 1, 1] = 1

            m = cv2.filter2D(mask3, -1, self.d_filter3)
            m[m > 0] = 1
            m = m - mask3
            direct[m == 1, 2] = 1

            m = cv2.filter2D(mask3, -1, self.d_filter4)
            m[m > 0] = 1
            m = m - mask3
            direct[m == 1, 3] = 1

            mask3 = mask3_

        abs_pos = pos.copy() # absolute position encoding
        rel_pos = pos / (self.str_size / 2)  # to 0~1 maybe larger than 1
        rel_pos = (rel_pos * self.pos_num).astype(np.int32) # to 0~pos_num
        rel_pos = np.clip(rel_pos, 0, self.pos_num - 1) # clip to 0~pos_num-1

        if ori_w != w or ori_h != h: # if the mask is resized
            rel_pos = cv2.resize(rel_pos, (ori_w, ori_h), interpolation=cv2.INTER_NEAREST)
            rel_pos[ori_mask == 0] = 0
            direct = cv2.resize(direct, (ori_w, ori_h), interpolation=cv2.INTER_NEAREST)
            direct[ori_mask == 0, :] = 0

        return rel_pos, abs_pos, direct

    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True
            )

            for item in sample_loader:
                yield item

    def get_ref_index(self, length, sample_length):
        if random.uniform(0, 1) > 0.5:
            ref_index = random.sample(range(length), sample_length)
            ref_index.sort()
        else:
            pivot = random.randint(0, length-sample_length)
            ref_index = [pivot+i for i in range(sample_length)]
        return ref_index

from src.models.upsample import *
from src.models.LaMa import *
from src.losses.adversarial import *
from src.losses.perceptual import *
from src.utils import get_lr_schedule_with_warmup

def make_optimizer(parameters, kind='adamw', **kwargs):
    if kind == 'adam':
        optimizer_class = torch.optim.Adam
    elif kind == 'adamw':
        optimizer_class = torch.optim.AdamW
    else:
        raise ValueError(f'Unknown optimizer kind {kind}')
    return optimizer_class(parameters, **kwargs)


def set_requires_grad(module, value):
    for param in module.parameters():
        param.requires_grad = value


def add_prefix_to_keys(dct, prefix):
    return {prefix + k: v for k, v in dct.items()}

class BaseInpaintingTrainingModule_video(nn.Module):
    def __init__(self, config, gpu, name, rank, args, test=False, **kwargs):
        super().__init__()
        print('BaseInpaintingTrainingModule init called')
        self.global_rank = rank
        self.config = config
        self.iteration = 0
        self.name = name
        self.test = test
        self.gen_weights_path = os.path.join(config.PATH, name + '_gen.pth')
        self.dis_weights_path = os.path.join(config.PATH, name + '_dis.pth')

        self.str_encoder = StructureEncoder_video_3D(config).cuda(gpu) # structure encoder for one image
        self.generator = ReZeroFFC_video_2D(config).cuda(gpu) # generator
        self.best = None

        print('Loading %s StructureUpsampling...' % self.name)
        self.structure_upsample = StructureUpsampling()
        data = torch.load(config.structure_upsample_path, map_location='cpu')
        self.structure_upsample.load_state_dict(data['model'])
        self.structure_upsample = self.structure_upsample.cuda(gpu).eval()
        print("Loading trained transformer...")
        model_config = EdgeLineGPTConfig(embd_pdrop=0.0, resid_pdrop=0.0, n_embd=args.n_embd, block_size=32,
                                     attn_pdrop=0.0, n_layer=args.n_layer, n_head=args.n_head, ref_frame_num=args.ref_frame_num) # video version
        self.transformer = EdgeLineGPT256RelBCE_video(model_config, args, device=gpu)
        checkpoint = torch.load(config.transformer_ckpt_path, map_location='cpu')
        if config.transformer_ckpt_path.endswith('.pt'): 
            self.transformer.load_state_dict(checkpoint) # load the line/edge inpainted model
        else:
            self.transformer.load_state_dict(checkpoint['model'])
        self.transformer.cuda(gpu).eval()  # eval mode of line/edge inpainted model
        self.transformer.half() # half precision

        if not test:
            # self.discriminator = NLayerDiscriminator_video(**self.config.discriminator).cuda(gpu) 
            self.discriminator = NLayerDiscriminator_video_2D(**self.config.discriminator).cuda(gpu) 
            # self.discriminator = net.Discriminator(
            #     in_channels=3, use_sigmoid=config['losses']['GAN_LOSS'] != 'hinge').cuda(gpu)  # video version referr to FuseFormer
            self.adversarial_loss = NonSaturatingWithR1_video_2D(**self.config.losses['adversarial'])
            # self.adversarial_loss = AdversarialLoss(type=self.config['losses']['GAN_LOSS']).cuda(gpu) # video version reference to FuseFormer
            self.generator_average = None
            self.last_generator_averaging_step = -1

            if self.config.losses.get("l1", {"weight_known": 0})['weight_known'] > 0:
                self.loss_l1 = nn.L1Loss(reduction='none')

            if self.config.losses.get("mse", {"weight": 0})['weight'] > 0:
                self.loss_mse = nn.MSELoss(reduction='none')

            assert self.config.losses['perceptual']['weight'] == 0

            if self.config.losses.get("resnet_pl", {"weight": 0})['weight'] > 0:
                self.loss_resnet_pl = ResNetPL_video(**self.config.losses['resnet_pl'])
            else:
                self.loss_resnet_pl = None
            self.gen_optimizer, self.dis_optimizer = self.configure_optimizers()
            self.str_optimizer = torch.optim.Adam(self.str_encoder.parameters(), lr=config.optimizers['generator']['lr'])
        if self.config.AMP:  # use AMP
            self.scaler = torch.cuda.amp.GradScaler()
        if not test:
            self.load_rezero()  # load pretrain model
        self.load()  # reload for restore

        # reset lr
        if not test:
            for group in self.gen_optimizer.param_groups:
                group['lr'] = config.optimizers['generator']['lr']
                group['initial_lr'] = config.optimizers['generator']['lr']
            for group in self.dis_optimizer.param_groups:
                group['lr'] = config.optimizers['discriminator']['lr']
                group['initial_lr'] = config.optimizers['discriminator']['lr']

        if self.config.DDP and not test:
            import apex
            self.generator = apex.parallel.convert_syncbn_model(self.generator)
            self.discriminator = apex.parallel.convert_syncbn_model(self.discriminator)
            self.generator = apex.parallel.DistributedDataParallel(self.generator)
            self.discriminator = apex.parallel.DistributedDataParallel(self.discriminator)

        if self.config.optimizers['decay_steps'] is not None and self.config.optimizers['decay_steps'] > 0 and not test:
            self.g_scheduler = torch.optim.lr_scheduler.StepLR(self.gen_optimizer, config.optimizers['decay_steps'],
                                                               gamma=config.optimizers['decay_rate'])
            self.d_scheduler = torch.optim.lr_scheduler.StepLR(self.dis_optimizer, config.optimizers['decay_steps'],
                                                               gamma=config.optimizers['decay_rate'])
            self.str_scheduler = get_lr_schedule_with_warmup(self.str_optimizer,
                                                             num_warmup_steps=config.optimizers['warmup_steps'],
                                                             milestone_step=config.optimizers['decay_steps'],
                                                             gamma=config.optimizers['decay_rate'])
            if self.iteration - self.config.START_ITERS > 1:
                for _ in range(self.iteration - self.config.START_ITERS):
                    self.g_scheduler.step()
                    self.d_scheduler.step()
                    self.str_scheduler.step()
        else:
            self.g_scheduler = None
            self.d_scheduler = None
            self.str_scheduler = None

    def load_rezero(self):
        if os.path.exists(self.config.gen_weights_path0):
            print('Loading %s generator...' % self.name)
            data = torch.load(self.config.gen_weights_path0, map_location='cpu')
            torch_init_model(self.generator, data, 'generator', rank=self.global_rank)
            self.gen_optimizer.load_state_dict(data['optimizer'])
            self.iteration = data['iteration']
        else:
            print('Warnning: There is no previous optimizer found. An initialized optimizer will be used.')

        # load discriminator only when training
        if (self.config.MODE == 1 or self.config.score) and os.path.exists(self.config.dis_weights_path0):
            print('Loading %s discriminator...' % self.name)
            data = torch.load(self.config.dis_weights_path0, map_location='cpu')
            torch_init_model(self.discriminator, data, 'discriminator', rank=self.global_rank)
            self.dis_optimizer.load_state_dict(data['optimizer'])
        else:
            print('Warnning: There is no previous optimizer found. An initialized optimizer will be used.')

    def load(self):
        if self.test:
            self.gen_weights_path = os.path.join(self.config.PATH, self.name + '_best_gen.pth')
            print('Loading %s generator...' % self.name)
            data = torch.load(self.gen_weights_path, map_location='cpu')
            self.generator.load_state_dict(data['generator'])
            self.str_encoder.load_state_dict(data['str_encoder'])
        if not self.test and os.path.exists(self.gen_weights_path):
            print('Loading %s generator...' % self.name)
            data = torch.load(self.gen_weights_path, map_location='cpu')
            self.generator.load_state_dict(data['generator'])
            self.str_encoder.load_state_dict(data['str_encoder'])
            self.gen_optimizer.load_state_dict(data['optimizer'])
            self.str_optimizer.load_state_dict(data['str_opt'])
            self.iteration = data['iteration']
            if self.iteration > 0:
                gen_weights_path = os.path.join(self.config.PATH, self.name + '_best_gen_HR.pth')
                data = torch.load(gen_weights_path, map_location='cpu')
                self.best = data['best_vfid']
                print('Loading best vfid...')

        else:
            print('Warnning: There is no previous optimizer found. An initialized optimizer will be used.')

        # load discriminator only when training
        if not self.test and os.path.exists(self.dis_weights_path):
            print('Loading %s discriminator...' % self.name)
            data = torch.load(self.dis_weights_path, map_location='cpu')
            self.dis_optimizer.load_state_dict(data['optimizer'])
            self.discriminator.load_state_dict(data['discriminator'])
        else:
            print('Warnning: There is no previous optimizer found. An initialized optimizer will be used.')

    def save(self):
        print('\nsaving %s...\n' % self.name)
        raw_model = self.generator.module if hasattr(self.generator, "module") else self.generator
        raw_encoder = self.str_encoder.module if hasattr(self.str_encoder, "module") else self.str_encoder
        torch.save({
            'iteration': self.iteration,
            'optimizer': self.gen_optimizer.state_dict(),
            'str_opt': self.str_optimizer.state_dict(),
            'str_encoder': raw_encoder.state_dict(),
            'generator': raw_model.state_dict()
        }, self.gen_weights_path)
        raw_model = self.discriminator.module if hasattr(self.discriminator, "module") else self.discriminator
        torch.save({
            'optimizer': self.dis_optimizer.state_dict(),
            'discriminator': raw_model.state_dict()
        }, self.dis_weights_path)

    def configure_optimizers(self):
        discriminator_params = list(self.discriminator.parameters())
        return [
            make_optimizer(self.generator.parameters(), **self.config.optimizers['generator']),
            make_optimizer(discriminator_params, **self.config.optimizers['discriminator'])
        ]


from src.losses.feature_matching import *
class DefaultInpaintingTrainingModule_video(BaseInpaintingTrainingModule_video):
    def __init__(self, args, config, gpu, rank, video_to_discriminator='predicted_video', test=False, **kwargs):
        super().__init__(args=args, config=config, gpu=gpu, name='InpaintingModel', rank=rank, test=test, **kwargs)
        self.video_to_discriminator = video_to_discriminator
        if self.config.AMP:  # use AMP
            self.scaler = torch.cuda.amp.GradScaler()

    def forward(self, batch):
        img = batch['frames'] # [B, 3, H, W] -> [B, T, 3, H, W] for video
        mask = batch['masks'] # [B, 1, H, W] -> [B, T, 1, H, W] for video
        masked_img = img * (1 - mask) + mask # [B, T, 3, H, W] for video

        masked_img = torch.cat([masked_img, mask], dim=2) # [B, T, 4, H, W] for video
        masked_str = torch.cat([batch['edges'], batch['lines'], mask], dim=2) # [B, T, 3, H, W] for video 
        if self.config.rezero_for_mpe is not None and self.config.rezero_for_mpe: # rezero for mpe
            str_feats, rel_pos_emb, direct_emb = self.str_encoder(masked_str, batch['rel_pos'], batch['direct']) # structure encoder for masked str(line/edge) and relative, direct position
            batch['predicted_video'] = self.generator(masked_img.to(torch.float32), rel_pos_emb, direct_emb, str_feats) # [B, T, 3, H, W] for video from the inpainting model
        else:
            str_feats = self.str_encoder(masked_str)  # not using masked positional encoding, so just use structure encoder for masked str(line/edge)
            batch['predicted_video'] = self.generator(masked_img.to(torch.float32), batch['rel_pos'],
                                                      batch['direct'], str_feats)
        
        batch['inpainted'] = mask * batch['predicted_video'] + (1 - mask) * batch['frames'] # [B, T, 3, H, W] for video
        batch['mask_for_losses'] = mask
        return batch

    def process(self, batch):
        self.iteration += 1 # iteration for optimizer
        self.discriminator.zero_grad() # clear gradient for discriminator
        # discriminator loss
        # print batch shape
        dis_loss, batch, dis_metric = self.discriminator_loss(batch) #  [B, T, 3, H, W] for video compute the discriminator loss
        self.dis_optimizer.step() # update discriminator
        if self.d_scheduler is not None: 
            self.d_scheduler.step() # update discriminator scheduler (learning rate decay)

        # generator loss
        self.generator.zero_grad() # clear gradient for generator
        self.str_optimizer.zero_grad() # clear gradient for structure encoder
        # generator loss
        gen_loss, gen_metric = self.generator_loss(batch) # compute the generator loss

        if self.config.AMP:  # use AMP
            self.scaler.step(self.gen_optimizer) #  update generator with scaler
            self.scaler.update()
            self.scaler.step(self.str_optimizer)
            self.scaler.update()
        else:
            self.gen_optimizer.step()
            self.str_optimizer.step()

        if self.str_scheduler is not None:
            self.str_scheduler.step()
        if self.g_scheduler is not None:
            self.g_scheduler.step()

        # create logs
        if self.config.AMP:
            gen_metric['loss_scale'] = self.scaler.get_scale()
        logs = [dis_metric, gen_metric]

        return batch['predicted_video'], gen_loss, dis_loss, logs, batch

    def generator_loss(self, batch):
        frames = batch['frames'] # Change 'image' to 'video'
        predicted_video = batch[self.video_to_discriminator] 
        original_mask = batch['masks']
        supervised_mask = batch['mask_for_losses']
        metrics = dict()
        total_loss = 0.

        # L1
        l1_value = masked_l1_loss(predicted_video, frames, supervised_mask,
                                self.config.losses['l1']['weight_known'],
                                self.config.losses['l1']['weight_missing'])

        total_loss = l1_value
        metrics[f'gen_l1'] = l1_value.item()

        # discriminator
        # adversarial_loss calls backward by itself
        self.adversarial_loss.pre_generator_step(real_batch=frames, fake_batch=predicted_video,
                                                generator=self.generator, discriminator=self.discriminator)
        discr_fake_pred, discr_fake_features = self.discriminator(predicted_video.to(torch.float32))
        adv_gen_loss, adv_metrics = self.adversarial_loss.generator_loss(discr_fake_pred=discr_fake_pred,
                                                                        mask=original_mask)
        total_loss = total_loss + adv_gen_loss
        metrics[f'gen_adv'] = adv_gen_loss.item()
        metrics.update(add_prefix_to_keys(adv_metrics, f'adv'))

        # feature matching
        if self.config.losses['feature_matching']['weight'] > 0:
            discr_real_pred, discr_real_features = self.discriminator(frames)
            need_mask_in_fm = self.config.losses['feature_matching'].get('pass_mask', False)
            mask_for_fm = supervised_mask if need_mask_in_fm else None
            fm_value = feature_matching_loss(discr_fake_features, discr_real_features,
                                            mask=mask_for_fm) * self.config.losses['feature_matching']['weight']
            total_loss += fm_value
            metrics[f'gen_fm'] = fm_value.item()

        if self.loss_resnet_pl is not None:
            resnet_pl_value = self.loss_resnet_pl(predicted_video, frames)
            total_loss += resnet_pl_value
            metrics[f'gen_resnet_pl'] = resnet_pl_value.item()

        if self.config.AMP:
            self.scaler.scale(total_loss).backward()
        else:
            total_loss.backward()
        return total_loss.item(), metrics

    def discriminator_loss(self, batch):
        self.adversarial_loss.pre_discriminator_step(real_batch=batch['frames'], fake_batch=None,
                                                     generator=self.generator, discriminator=self.discriminator)
        discr_real_pred, discr_real_features = self.discriminator(batch['frames'])
        real_loss, dis_real_loss, grad_penalty = self.adversarial_loss.discriminator_real_loss(
            real_batch=batch['frames'],
            discr_real_pred=discr_real_pred)
        real_loss.backward()
        if self.config.AMP:
            with torch.cuda.amp.autocast():
                batch = self.forward(batch)
        else:
            batch = self(batch)
        batch[self.video_to_discriminator] = batch[self.video_to_discriminator].to(torch.float32)
        predicted_video = batch[self.video_to_discriminator].detach()
        discr_fake_pred, discr_fake_features = self.discriminator(predicted_video.to(torch.float32))
        fake_loss = self.adversarial_loss.discriminator_fake_loss(discr_fake_pred=discr_fake_pred, mask=batch['masks'])
        fake_loss.backward()
        total_loss = fake_loss + real_loss
        metrics = {}
        metrics['dis_real_loss'] = dis_real_loss.mean().item()
        metrics['dis_fake_loss'] = fake_loss.item()
        metrics['grad_penalty'] = grad_penalty.mean().item()

        return total_loss.item(), batch, metrics

# uni test
import torch
import unittest
from src.config import Config
from src.models.LaMa import *
import argparse
from shutil import copyfile

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='ZITS_video_test', help='the name of this model')
parser.add_argument('--path', '--checkpoints', type=str, default="./ckpt",
                    help='model checkpoints path (default: ./checkpoints)')
parser.add_argument('--config_file', type=str, default='./config_list/config_ZITS_video.yml',
                    help='The config file of each experiment ')
parser.add_argument('--nodes', type=int, default=1, help='how many machines')
parser.add_argument('--gpus', type=int, default=1, help='how many GPUs in one node')
parser.add_argument('--GPU_ids', type=str, default='1')
parser.add_argument('--node_rank', type=int, default=0, help='the id of this machine')
parser.add_argument('--DDP', action='store_true', help='DDP')
parser.add_argument('--lama', action='store_true', help='train the lama first')
# Define the size of transformer
parser.add_argument('--ref_frame_num', type=int, default=5)
parser.add_argument('--n_layer', type=int, default=16)
parser.add_argument('--n_embd', type=int, default=256)
parser.add_argument('--n_head', type=int, default=8)
parser.add_argument('--lr', type=float, default=4.24e-4)
# AMP
parser.add_argument('--AMP', action='store_true', help='Automatic Mixed Precision')
parser.add_argument('--local_rank', type=int, default=-1, help='the id of this machine')
parser.add_argument('--loss_hole_valid_weight', type=float, nargs='+', default=[0.8, 0.2], help='the weight for computing the hole/valid part ')
parser.add_argument('--loss_edge_line_weight', type=float, nargs='+', default=[1.0, 1.0], help='the weight for computing the edge/line part ')
# add the choice to decide the loss function with l1 or mse or binary cross entropy with choice
parser.add_argument('--loss_choice', type=str, default="bce", help='the choice of loss function: l1, mse, bce')
parser.add_argument('--edge_gaussian', type=int, default=0, help='the sigma of gaussian kernel for edge')
parser.add_argument('--dataset', type=str, default="youtubevos")
parser.add_argument('--input_size', type=tuple, default=(240,432))
parser.add_argument("--neighbor_stride", type=int, default=1)
args = parser.parse_args()

class TestDefaultInpaintingTrainingModule_video(unittest.TestCase):
    def setUp(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        config = Config(f'./ckpt/{model_name}/config_ZITS_video.yml')
        config.training_model.net = args.model_name
        self.model = DefaultInpaintingTrainingModule_video(args=args, config=config, gpu=0, rank=0, test=False).to(device)

        # self.batch = {
        #     'frames': torch.rand((1, 5, 3, 64, 64)).to(device),
        #     'masks': torch.rand((1, 5, 1, 64, 64)).to(device),
        #     'edges': torch.rand((1, 5, 1, 64, 64)).to(device),
        #     'lines': torch.rand((1, 5, 1, 64, 64)).to(device),
        #     'rel_pos': torch.rand((1, 5, 1, 64, 64)).to(device),
        #     'direct': torch.rand((1, 5, 64, 64, 4)).to(device)  # four directions
        # }
        
        dataset = DynamicDataset_video()
        iterator = dataset.create_iterator(2)
        self.batch = next(iterator)
        # put all data in GPU
        self.batch['frames'] = self.batch['frames'].to(device)
        self.batch['masks'] = self.batch['masks'].to(device)
        self.batch['edges'] = self.batch['edges'].to(device) 
        self.batch['lines'] = self.batch['lines'].to(device)
        self.batch['rel_pos'] = self.batch['rel_pos'].to(device)
        self.batch['direct'] = self.batch['direct'].to(device)
        
        output = self.model.forward(self.batch)
        self.batch['predicted_video'] = output['predicted_video']
        self.batch['inpainted'] = output['inpainted']
        self.batch['mask_for_losses'] = output['mask_for_losses']

    def test_forward(self):
        output = self.model.forward(self.batch)
        self.assertTrue('predicted_video' in output)
        self.assertTrue('inpainted' in output)
        self.assertTrue('mask_for_losses' in output)

    def test_process(self):
        output = self.model.process(self.batch)
        self.assertTrue('predicted_video' in output[-1])
        self.assertTrue(len(output) == 5)

    def test_generator_loss(self):
        output = self.model.generator_loss(self.batch)
        self.assertTrue(isinstance(output[0], float))
        self.assertTrue(isinstance(output[1], dict))

    def test_discriminator_loss(self):
        output = self.model.discriminator_loss(self.batch)
        self.assertTrue(isinstance(output[0], float))
        self.assertTrue(isinstance(output[1], dict))

"""
Below is the code to test the evaluation function. Reference to FuseFormer
"""
from src.utils import create_dir, Progbar, SampleEdgeLineLogits_video, stitch_images
from tqdm import tqdm
from src.inpainting_metrics import *
from src.utils import *
import cv2

from skimage.metrics import structural_similarity as measure_ssim
from skimage.metrics import peak_signal_noise_ratio as measure_psnr

_to_tensors = transforms.Compose([
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

class ZITS_video:
    def __init__(self, args, config, gpu, rank, test=False, single_img_test=False):
        self.config = config
        self.device = gpu
        self.global_rank = rank

        # self.model_name = 'inpaint' # this final RGB video inpainting model name
        self.model_name = args.model_name

        kwargs = dict(config.training_model)
        kwargs.pop('kind')

        self.inpaint_model = DefaultInpaintingTrainingModule_video(args, config, gpu=gpu, rank=rank, test=test, **kwargs).to(gpu) 
        self.input_size = args.input_size
        self.w, self.h = args.input_size 
        self.neighbor_stride = args.neighbor_stride
        self.sample_length = args.ref_frame_num
        self.pos_num = config.rel_pos_num
        self.str_size = config.str_size

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
            self.train_dataset = DynamicDataset_video()

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
            self.val_dataset = DynamicDataset_video(split='valid')

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
                    
                    edge_pred, line_pred = SampleEdgeLineLogits_video(self.inpaint_model.transformer,
                    context=[items['frames'][:b, ...].to(torch.float16), items['edges'][:b, ...].to(torch.float16), items['lines'][:b, ...].to(torch.float16)], 
                    masks=items['masks'][:b, ...].to(torch.float16), iterations=iteration_num_for_pred, add_v=0.05, mul_v=4, device=self.device)   
                    edge_pred = edge_pred.detach().to(torch.float32)
                    line_pred = line_pred.detach().to(torch.float32)
                    items['edges'] = edge_pred.detach()
                    items['lines'] = line_pred.detach()

                # train
                outputs, gen_loss, dis_loss, logs, batch = self.inpaint_model.process(items)

                if iteration >= max_iteration:
                    keep_training = False
                    break
                logs = [("epoch", epoch), ("iter", iteration)] + \
                       [(i, logs[0][i]) for i in logs[0]] + [(i, logs[1][i]) for i in logs[1]]
                logs.append(("g_lr", self.inpaint_model.g_scheduler.get_last_lr()[0]))
                logs.append(("d_lr", self.inpaint_model.d_scheduler.get_last_lr()[0]))
                logs.append(("str_lr", self.inpaint_model.str_scheduler.get_last_lr()[0]))
                logs.append(("img_size", batch['size_ratio'][0].item() * 256))
                progbar.add(len(items['frames']),
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
                    psnr, ssim, vfid = self.eval()
                    if self.best > vfid:
                        self.best = vfid
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

        self.inpaint_model.eval()  # set model to eval mode

        ssim_all, psnr_all, len_all = 0., 0., 0. 
        s_psnr_all = 0. 
        video_length_all = 0 
        vfid = 0.
        output_i3d_activations = []
        real_i3d_activations = []
    
        with torch.no_grad(): 
            for items in tqdm(val_loader):
                for k in items:
                    if type(items[k]) is torch.Tensor:
                        items[k] = items[k].to(self.device)
                b, t, _, _, _ = items['edges'].shape
                edge_pred, line_pred = SampleEdgeLineLogits_video(self.inpaint_model.transformer,
                                                            context=[items['frames'].to(torch.float16),
                                                                        items['edges'].to(torch.float16),
                                                                        items['lines'].to(torch.float16)],
                                                            masks=items['masks'].clone().to(torch.float16),
                                                            iterations=5,
                                                            add_v=0.05, mul_v=4,
                                                            device=self.device)
                edge_pred, line_pred = edge_pred.detach().to(torch.float32), line_pred.detach().to(torch.float32)

                items['edges'] = edge_pred # the inpainted edges
                items['lines'] = line_pred # # the inpainted lines
                # eval
                items = self.inpaint_model(items)
                outputs_merged = (items['predicted_video'] * items['masks']) + (items['frames'] * (1 - items['masks']))

                # save
                outputs_merged *= 255.0
                outputs_merged = outputs_merged.permute(0, 1, 3, 4, 2).int().cpu().numpy()
                items['frames'] *= 255
                items['frames'] = items['frames'].permute(0, 1, 3, 4, 2).int().cpu().numpy()
                ssim, s_psnr = 0., 0.
                for img_num in range(b):
                    for i in range(t):
                        pred = outputs_merged[img_num][i]
                        gt = items['frames'][img_num][i]
                        
                        pred_img = np.array(pred)
                        gt_img = np.array(gt)
                        ssim += measure_ssim(gt_img, pred_img, data_range=255, multichannel=True, win_size=65)
                        s_psnr += measure_psnr(gt_img, pred_img, data_range=255)
                        
                        path = os.path.join(self.val_path, items['name'][img_num])
                        if not os.path.exists(path):
                            os.makedirs(path)
                        cv2.imwrite(os.path.join(path, "pred_"+items['idxs'][i][img_num]), pred[:, :, ::-1])
                        cv2.imwrite(os.path.join(path, "gt_"+items['idxs'][i][img_num]), gt[:, :, ::-1])
                    
                    # FVID computation
                    # get i3d activations 
                    gts = torch.from_numpy(items['frames'][img_num]).unsqueeze(0).to(self.device)
                    preds = torch.from_numpy(outputs_merged[img_num]).unsqueeze(0).to(self.device)
                    gts = gts.permute(0, 1, 4, 2, 3)
                    preds =  preds.permute(0, 1, 4, 2, 3)
                    # tranfer gts and preds to float tensor and constrain to 0~1
                    gts = gts.to(torch.float32) / 255.0
                    preds = preds.to(torch.float32) / 255.0
                    real_i3d_activations.append(get_i3d_activations(gts, device=self.device).cpu().numpy().flatten())
                    output_i3d_activations.append(get_i3d_activations(preds, device=self.device).cpu().numpy().flatten())
                        
                ssim_all += ssim
                s_psnr_all += s_psnr

        vfid_score = get_fid_score(real_i3d_activations, output_i3d_activations)
        ssim_final = ssim_all/(len(val_loader)*args.ref_frame_num*self.config.BATCH_SIZE)
        s_psnr_final = s_psnr_all/(len(val_loader)*args.ref_frame_num*self.config.BATCH_SIZE)
        
        if self.global_rank == 0:
            print("iter: %d, PSNR: %f, SSIM: %f, VFID: %f" %
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
            edges_pred, lines_pred = SampleEdgeLineLogits_video(self.inpaint_model.transformer,
                                                        context=[items['frames'][:b, ...].to(torch.float16),
                                                                 items['edges'][:b, ...].to(torch.float16),
                                                                 items['lines'][:b, ...].to(torch.float16)],
                                                        masks=items['masks'][:b, ...].clone().to(torch.float16),
                                                        iterations=5,
                                                        add_v=0.05, mul_v=4,
                                                        device=self.device)
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

        if it is not None:
            iteration = it

        image_per_row = 2
        if self.config.SAMPLE_SIZE <= 6:
            image_per_row = 1
            
        for b in range(items['frames'].shape[0]):            
            images = stitch_images(
                self.postprocess((items['frames'][b,:,...]).cpu()),
                self.postprocess((inputs[b,:,...]).cpu()),
                self.postprocess(items['edges'][b,:,...].cpu()),
                self.postprocess(items['lines'][b,:,...].cpu()),
                self.postprocess(items['masks'][b,:,...].cpu()),
                self.postprocess((items['predicted_video'][b,:,...]).cpu()),
                self.postprocess((outputs_merged[b,:,...]).cpu()),
                img_per_row=image_per_row
            )

            name = os.path.join(self.samples_path, str(iteration).zfill(6) + f"batch{b}.jpg")
            create_dir(self.samples_path)
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

if __name__ == '__main__':
    # test the ZITS_video train function
    # from src.inpainting_metrics import get_inpainting_metrics_video
    # print(get_inpainting_metrics_video(src='datasets/YouTubeVOS/test_all_frames/JPEGImages/', tgt='datasets/YouTubeVOS/test_all_frames/JPEGImages/', logger=None, device="cuda"))
    args.path = os.path.join(args.path, args.model_name)
    config_path = os.path.join(args.path, 'config.yml')

    # create checkpoints path if does't exist
    os.makedirs(args.path, exist_ok=True)

    # copy config template if does't exist
    if not os.path.exists(config_path):
        copyfile(args.config_file, config_path)  ## Training, always copy

    args.config_path = config_path
    config = Config(args.config_path, args.model_name)
    # config_path = './config_list/config_ZITS_video.yml'
    # config = Config(config_path)
    gpu = "cuda:1"
    rank = 0 
    args.world_size = 1
    config.world_size = 1
    
    args.DDP = True
    model = ZITS_video(args, config, gpu, rank)
    model.train()

# if __name__ == '__main__':
#     unittest.main()

# main
# if __name__ == '__main__':
#     split = 'train'

#     dataset = DynamicDataset_video()
#     # test loading the dataset
#     iterator = dataset.create_iterator(4)
#     for i in range(10):
#         batch = next(iterator)
#         print(batch['name'])
#         print(batch['size_ratio'])
#         print(batch['frames'].shape)
#         print(batch['masks'].shape)
#         print(batch['edges'].shape)
#         print(batch['lines'].shape)
#         print(batch['rel_pos'].shape)
#         print(batch['abs_pos'].shape)
#         print(batch['direct'].shape)
#         print()
        
#         # combine above image, mask, edge, line, re_pls, abs_pos, direct into one image
#         # and then save the image in the folder
#         image = batch['frames']
#         mask = batch['masks']
#         edge = batch['edges']
#         line = batch['lines']
#         rel_pos = batch['rel_pos']
#         abs_pos = batch['abs_pos']
#         direct = batch['direct']
#         # turn image to numpy array
#         from torchvision.utils import make_grid, save_image
#         combined = torch.cat(tuple(image), dim=2)
#         combined_mask = torch.cat(tuple(mask), dim=2)
#         combined_edge = torch.cat(tuple(edge), dim=2)
#         combined_line = torch.cat(tuple(line), dim=2)
#         rel_pos.unsqueeze_(dim=2)  # add one dimension to the tensor
#         abs_pos.unsqueeze_(dim=2)
#         combined_rel = torch.cat(tuple(rel_pos), dim=2)
#         combined_abs = torch.cat(tuple(abs_pos), dim=2)
#         # rel_pos is torch.Size([4, 5, 256, 256]) image combined them with 4 rows and 5 columns
#         # combined_rel = torch.cat(rel_pos.unbind(dim=2), dim=1)  # unbind along the third dimension (dim=2), then concatenate along the second dimension (dim=1)
#         # combined_abs = torch.cat(abs_pos.unbind(dim=2), dim=1)
        
#         save_image(combined, f'combined_image_{i}.png')
#         save_image(combined_mask, f'combined_mask_{i}.png')
#         save_image(combined_edge, f'combined_edge_{i}.png')
#         save_image(combined_line, f'combined_line_{i}.png')
#         combined_rel = combined_rel.float()  # Convert to float tensor
#         combined_rel = (combined_rel - combined_rel.min()) / (combined_rel.max() - combined_rel.min())  # Normalize to [0, 1]
#         save_image(combined_rel, f'combined_rel_{i}.png')
#         combined_abs = combined_abs.float()  # Convert to float tensor
#         combined_abs = (combined_abs - combined_abs.min()) / (combined_abs.max() - combined_abs.min())  # Normalize to [0, 1]
#         save_image(combined_abs, f'combined_abs_{i}.png')
        
#         # Let's assume tensor is your 4D tensor with shape [256, 256, 5, 4]
#         direct = direct[0].permute(0, 3, 1, 2)  # Change the shape to [5, 4, 256, 256]
#         direct = direct.reshape(-1, 240, 432)  # Flatten the first two dimensions, new shape is [20, 256, 256]

#         # Now, we need to add a dimension for channels, because make_grid and save_image expect tensors in the shape (C, H, W) or (B, C, H, W)
#         direct = direct.unsqueeze(1)  # New shape is [20, 1, 256, 256]

#         grid = make_grid(direct, nrow=4)  # Arrange images into a grid with 4 images per row

#         # Normalize to [0, 1] and save
#         grid = (grid - grid.min()) / (grid.max() - grid.min())
#         save_image(grid, f'grid_{i}.png')
        

# #     #  test DefaultInpaintingTrainingModule_video
    
