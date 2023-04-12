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
from src.models.TSR_model import EdgeLineGPT256RelBCE, EdgeLineGPTConfig
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

sys.path.append('..')

logger = logging.getLogger(__name__)

def to_int(x):
    return tuple(map(int, x))

""" 
My utility
"""
from torchvision.utils import save_image


class ContinuousEdgeLineDatasetMask_video(Dataset):

    def __init__(self, pt_dataset, mask_path=None, test_mask_path=None, is_train=False, mask_rates=None,
                 frame_size=256, line_path=None):

        self.is_train = is_train
        self.pt_dataset = pt_dataset # image dataset (a .txt file that indicates the directory of each images)

        self.video_id_list = []  # create 2D arrays [video, frames]
        n_video = 0
        frames = []
        with open(self.pt_dataset) as f:
            for line in f:
                if "video " in line:  # a new video case
                    if n_video != 0:
                        self.video_id_list.append(frames)
                        frames = []
                    n_video += 1
                else: frames.append(line.strip())  # 從指定的training image txt讀入所有要訓練的image的路徑，此list裡面每一個element就是一張RGB圖片
            self.video_id_list.append(frames)  # append the last

        if is_train:
            # training mask TYPE1: irregular mask
            self.irregular_mask_list = []
            with open(mask_path[0]) as f:
                for line in f:
                    self.irregular_mask_list.append(line.strip())
            self.irregular_mask_list = sorted(self.irregular_mask_list, key=lambda x: x.split('/')[-1])
            # training mask TYPE2: segmentation mask
            # self.segment_mask_list = []
            # with open(mask_path[1]) as f:
            #     for line in f:
            #         self.segment_mask_list.append(line.strip())
            # self.segment_mask_list = sorted(self.segment_mask_list, key=lambda x: x.split('/')[-1])
        else:  # TBD: change to video version
            self.mask_list = glob(test_mask_path + '/*')  # 在測試時，mask的路徑預設為一個資料夾，因此在建立mask list時要將參數給定的mask路徑下所有個圖片都讀入，glob為取得所有的檔案的路徑
            self.mask_list = sorted(self.mask_list, key=lambda x: x.split('/')[-1])

        self.frame_size = frame_size  # 設定圖片大小：預設訓練的圖片大小為固定
        self.training = is_train  # 是否為訓練模式
        # self.mask_rates = mask_rates  # 設定mask的比例, 'irregular rate, coco rate, addition rate': 0.4, 0.8, 1.0
        self.mask_rates = [1.0, 0., 0.]
        self.line_path = line_path  # 設定預先使用wireframe偵測儲存下來的圖片
        self.wireframe_th = 0.85

    def __len__(self):
        return len(self.video_id_list)  # 有多少組訓練影片

    def resize(self, img, height, width, center_crop=False):  # resize成正方形
        imgh, imgw = img.shape[0:2]

        if center_crop and imgh != imgw:
            # center crop
            side = np.minimum(imgh, imgw)
            j = (imgh - side) // 2
            i = (imgw - side) // 2
            img = img[j:j + side, i:i + side, ...]

        if imgh > height and imgw > width:
            inter = cv2.INTER_AREA
        else:
            inter = cv2.INTER_LINEAR
        img = cv2.resize(img, (height, width), interpolation=inter)
        return img

    def load_mask(self, img, video_idx, frame_idx):
        imgh, imgw = img.shape[0:2]

        # test mode: load mask non random
        if self.training is False:
            mask = cv2.imread(self.mask_list[video_idx][frame_idx], cv2.IMREAD_GRAYSCALE)  # 以灰階的模式取得mask的路徑
            mask = cv2.resize(mask, (imgw, imgh), interpolation=cv2.INTER_NEAREST)
            mask = (mask > 127).astype(np.uint8) * 255
            return mask
        else:  # train mode: 40% mask with random brush, 40% mask with coco mask, 20% with additions
            rdv = random.random()
            if rdv < self.mask_rates[0]:
                mask_index = random.randint(0, len(self.irregular_mask_list) - 1)
                mask = cv2.imread(self.irregular_mask_list[mask_index],
                                  cv2.IMREAD_GRAYSCALE)
            elif rdv < self.mask_rates[1]:
                mask_index = random.randint(0, len(self.segment_mask_list) - 1)
                mask = cv2.imread(self.segment_mask_list[mask_index],
                                  cv2.IMREAD_GRAYSCALE)
            else:
                mask_index1 = random.randint(0, len(self.segment_mask_list) - 1)
                mask_index2 = random.randint(0, len(self.irregular_mask_list) - 1)
                mask1 = cv2.imread(self.segment_mask_list[mask_index1],
                                   cv2.IMREAD_GRAYSCALE).astype(np.float)
                mask2 = cv2.imread(self.irregular_mask_list[mask_index2],
                                   cv2.IMREAD_GRAYSCALE).astype(np.float)
                mask = np.clip(mask1 + mask2, 0, 255).astype(np.uint8)  # 混合irregular mask和segmentation mask兩種

            if mask.shape[0] != imgh or mask.shape[1] != imgw:
                mask = cv2.resize(mask, (imgw, imgh), interpolation=cv2.INTER_NEAREST)
            mask = (mask > 127).astype(np.uint8) * 255  # threshold due to interpolation
            return mask

    def load_irregular_mask(self, img, irr_dx):
        imgh, imgw = img.shape[0:2]

        mask = cv2.imread(self.irregular_mask_list[irr_dx], cv2.IMREAD_GRAYSCALE)
        if mask.shape[0] != imgh or mask.shape[1] != imgw:
            mask = cv2.resize(mask, (imgw, imgh), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype(np.uint8) * 255  # threshold due to interpolation
        return mask

    def to_tensor(self, img, norm=False):
        # img = Image.fromarray(img)
        img_t = F.to_tensor(img).float()
        if norm:
            img_t = F.normalize(img_t, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        return img_t

    def load_edge(self, img):
        return canny(img, sigma=2, mask=None).astype(np.float)

    def load_wireframe(self, video_idx, frame_idx, size):
        selected_video_frame_name = self.video_id_list[video_idx][frame_idx]
        video_case, frame_no = selected_video_frame_name.split("/")[-2:]
        line_name = os.path.join(self.line_path, video_case, frame_no).replace('.png', '.pkl').replace('.jpg', '.pkl')
        wf = pickle.load(open(line_name, 'rb'))  # 讀入對應的wireframe檔
        lmap = np.zeros((size, size))
        for i in range(len(wf['scores'])):  # 所有偵測到的線條
            if wf['scores'][i] > self.wireframe_th:  # 依序檢查，有超過threshold的線條才會拿來使用
                line = wf['lines'][i].copy()
                line[0] = line[0] * size
                line[1] = line[1] * size
                line[2] = line[2] * size
                line[3] = line[3] * size
                rr, cc, value = skimage.draw.line_aa(*to_int(line[0:2]), *to_int(line[2:4]))
                lmap[rr, cc] = np.maximum(lmap[rr, cc], value)
        return lmap

    def __getitem__(self, idx):
        # selected_img_name = self.image_id_list[idx]  # 目前訓練的image case名字
        video_name, frame_no = self.video_id_list[idx][0].split("/")[-2:]
        selected_video = self.video_id_list[idx]  # 目前訓練的image case名字
        frame_list = []  # 最後用np.stact(frame_list) 把所有的三維frame(H, W, C)疊成 video(t H, W, C)
        edge_list = []
        line_list = []
        mask_list = []
        irr_mask_index = random.randint(0, len(self.irregular_mask_list) - 1)  # 只適用在假設每個影片的frame都被遮擋同樣的固定區塊

        for frame_idx, frame_name in enumerate(selected_video):
            frame = cv2.imread(frame_name)  # 讀取此image的rgb版本
            while frame is None:
                print('Bad image {}...'.format(frame_name))
                idx = random.randint(0, len(selected_video) - 1)
                frame = cv2.imread(selected_video[idx])
            frame = frame[:, :, ::-1]  # RGB轉成BGR

            frame = self.resize(frame, self.frame_size, self.frame_size, center_crop=False)  # 切割成正方形

            frame_gray = rgb2gray(frame)
            edge = self.load_edge(frame_gray)  # canny edge
            line = self.load_wireframe(video_idx=idx, frame_idx=frame_idx, size=self.frame_size)
            # load mask
            # mask = self.load_mask(img=img, video_idx=idx, frame_idx=frame_idx)
            mask = self.load_irregular_mask(img=frame, irr_dx=irr_mask_index)

            # augment data -> 左右反射
            # if self.training is True:
            #     if random.random() < 0.5:
            #         img = img[:, ::-1, ...].copy()
            #         edge = edge[:, ::-1].copy()
            #         line = line[:, ::-1].copy()
            #     if random.random() < 0.5:
            #         mask = mask[:, ::-1, ...].copy()
            #     if random.random() < 0.5:
            #         mask = mask[::-1, :, ...].copy()

            frame = self.to_tensor(frame.copy(), norm=True)  # 不加copy會錯，不知道為何
            edge = self.to_tensor(edge.copy())
            line = self.to_tensor(line.copy())
            mask = self.to_tensor(mask.copy())
            frame_list.append(frame)
            edge_list.append(edge)
            line_list.append(line)
            mask_list.append(mask)

        meta = {'frames': torch.stack(frame_list), 'masks': torch.stack(mask_list), 'edges': torch.stack(edge_list), 'lines': torch.stack(line_list),
                'name': os.path.join(video_name, frame_no)}
        return meta
    
class Dataset(torch.utils.data.Dataset):
    def __init__(self, sample=5, size=(432,240), split='train', name='YouTubeVOS', root='./datasets'):
        self.split = split
        self.sample_length = sample
        self.size = self.w, self.h = size

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

        self._to_tensors = transforms.Compose([
            Stack(),
            ToTorchFormatTensor(), ])

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            print('Loading error in video {}'.format(self.video_names[index]))
            item = self.load_item(0)
        return item

    def load_item(self, index):
        video_name = self.video_names[index]
        edge_name = self.edge_names[index]
        line_name = self.line_names[index]
        all_frames = [os.path.join(video_name, name) for name in sorted(os.listdir(video_name))]
        all_edges = [os.path.join(edge_name, name) for name in sorted(os.listdir(edge_name))]
        all_lines = [os.path.join(line_name, name) for name in sorted(os.listdir(line_name))]
        all_masks = create_random_shape_with_random_motion(
            len(all_frames), imageHeight=self.h, imageWidth=self.w)
        ref_index = self.get_ref_index(len(all_frames), self.sample_length)

        frames = []
        edges = []
        lines = []
        masks = []
        for idx in ref_index:
            img = Image.open(all_frames[idx]).convert('RGB')
            img = img.resize(self.size)
            frames.append(img)
            
            edge = Image.open(all_edges[idx]).convert('L')
            edge = edge.resize(self.size)
            edges.append(edge)

            line = Image.open(all_lines[idx]).convert('L')
            line = line.resize(self.size)
            lines.append(line)

            masks.append(all_masks[idx])
        if self.split == 'train':
            prob = random.random()
            frames = GroupRandomHorizontalFlip()(frames, prob)
            edges = GroupRandomHorizontalFlip()(edges, prob)
            lines = GroupRandomHorizontalFlip()(lines, prob)

        # To tensors
        frame_tensors = self._to_tensors(frames)*2.0 - 1.0
        edge_tensors = self._to_tensors(edges)
        line_tensors = self._to_tensors(lines)
        mask_tensors = self._to_tensors(masks)
        return [frame_tensors, edge_tensors, line_tensors, mask_tensors]

    def get_ref_index(self, length, sample_length):
        if random.uniform(0, 1) > 0.5:
            ref_index = random.sample(range(length), sample_length)
            ref_index.sort()
        else:
            pivot = random.randint(0, length-sample_length)
            ref_index = [pivot+i for i in range(sample_length)]
        return ref_index

class EdgeLineGPT256RelBCE_video(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config, device):
        super().__init__()

        self.pad1 = nn.ReflectionPad2d(3) # square -> bigger square (extend 3 for each side)
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=64, kernel_size=7, padding=0) # downsample input
        self.act = nn.ReLU(True)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)  # downsample 1

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1) # downsample 2

        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1) # downsample 3

        self.pos_emb = nn.Parameter(torch.zeros(1, 1024, 256))  # special tensor that automatically add into parameter list
        self.drop = nn.Dropout(config.embd_pdrop)

        # decoder, input: 32*32*config.n_embd
        self.ln_f = nn.LayerNorm(256)

        self.convt1 = nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1) # upsample 1

        self.convt2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1) # upsample 2

        self.convt3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1) # upsample 3

        self.padt = nn.ReflectionPad2d(3)
        self.convt4 = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=7, padding=0) # upsample and ouput only edge/line

        self.act_last = nn.Sigmoid()

        # Feature Fusion (6 channels to 3 channels)
        self.fuse_channel = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=3, padding='same'),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding='same'),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 16, kernel_size=3, padding='same'),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 3, kernel_size=3, padding='same'),
        )

        self.fuseformerBlock = InpaintGenerator()
        self.fuseformerBlock = self.fuseformerBlock.to(device)

        self.config = config

        self.apply(self._init_weights)  # initialize the weights (multiple layer initialization)

        self.resize_tensor = T.Resize(size = (240, 432))
        # self.resize_tensor = T.Resize(size = (64, 64))

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def get_block_size(self):
        return self.block_size
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv2d, nn.ConvTranspose2d)):  # if one of the type
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()  
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):  # some parameter need weight decay to avoid overfitting
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d, torch.nn.ConvTranspose2d) # need weight decay
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')
        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert len(
            param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params),)
        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def forward(self, img_idx, edge_idx, line_idx, edge_targets=None, line_targets=None, masks=None):
        img_idx = img_idx * (1 - masks)  # create masked image
        edge_idx = edge_idx * (1 - masks) # create masked edge
        line_idx = line_idx * (1 - masks) # create masked line

        # for b in range(img_idx.shape[0]):
        #     print(f"Process batch: {b}...")
        #     for t in range(img_idx.shape[1]):
        #         save_image(img_idx[b][t], f'tensor_img_b{b}_t{t}.png')

        # [b, t, c, w, h]
        x = torch.cat((img_idx, edge_idx, line_idx, masks), dim=2)  # concat method NEED checking (maybe is channel-wise)

        # Encoder: downsample
        # x = self.pad1(x)  # reflection padding
        # x = self.conv1(x)  # downsample input layer
        # x = self.act(x)  # activate with ReLU

        # x = self.conv2(x)  # downsample 1 
        # x = self.act(x)

        # x = self.conv3(x)  # downsample 2 
        # x = self.act(x)

        # x = self.conv4(x)  # downsample 3 
        # x = self.act(x)

        [b, t, c, h, w] = x.shape  # before here, the video data is still with Height x Width -> [50, 256, 32, 32] -> [t, c, h, w]
        # print(f"shape after concat: {x.shape}")  # test
        # x = x.view(t, c, h * w).transpose(1, 2).contiguous() # image 2D -> 1D (flatten) and change image and color channel
        # make the data into shape like -> [batch size, image(1D), channels(RGB, edge, line, mask)]


        # Transformer blocks
        # input [50, 256, 32, 32] -> original ZITS
        # input [1, 5, 3, 240, 432] -> original fuseformer
        print(f"shape before FuseFormer: {x.shape}")  # test
        x = self.fuseformerBlock(x)
        print(f"shape after FuseFormer: {x.shape}")  # test

        print(f"x shape before upsample: {x.shape}")
        # Decoder: upsample
        # x = self.convt1(x) # upsample 1
        # x = self.act(x)

        # x = self.convt2(x) # upsample 2
        # x = self.act(x)

        # x = self.convt3(x) # upsample 3
        # x = self.act(x)

        # x = self.padt(x)  # padding back
        # x = self.convt4(x)  # upsample output as the original image shape
        
        edge, line = torch.split(x, [1, 1], dim=1)  # seperate the TSR outputs
        print(f"edge shape(after transformer): {edge.shape}")  # test
        print(f"line shape(after transformer): {line.shape}")  # test

        print(f"edge_targets shape(after transformer): {edge_targets.shape}")  # test
        print(f"line_targets shape(after transformer): {line_targets.shape}")  # test
        edge_targets = edge_targets.view(b * t, 1, h, w)
        line_targets = line_targets.view(b * t, 1, h, w)
        masks = masks.view(b * t, 1, h, w)
        # Loss computing
        if edge_targets is not None and line_targets is not None:
            print(f"masks shape: {masks.shape}")  # test
            print(f"edge shape: {edge.shape}")  # test
            print(f"edge_targets shape: {edge_targets.shape}")  # test
            print(f"edge.permute(0, 2, 3, 1).contiguous().view(-1, 1) shape:{edge.permute(0, 2, 3, 1).contiguous().view(-1, 1).shape}")
            # edge loss
            loss = nnF.binary_cross_entropy_with_logits(edge.permute(0, 2, 3, 1).contiguous().view(-1, 1),
                                                      edge_targets.permute(0, 2, 3, 1).contiguous().view(-1, 1),
                                                      reduction='none')
            # line loss
            loss = loss + nnF.binary_cross_entropy_with_logits(line.permute(0, 2, 3, 1).contiguous().view(-1, 1),
                                                             line_targets.permute(0, 2, 3, 1).contiguous().view(-1, 1),
                                                             reduction='none')
            masks_ = masks.permute(0, 2, 3, 1).contiguous().view(-1, 1) # only compute the loss in the masked region
            print(f"mask reshape: {masks_.shape}")  # test

            loss *= masks_
            loss = torch.mean(loss)
        else:
            loss = 0

        edge, line = self.act_last(edge), self.act_last(line)  # sigmoid activate

        return edge, line, loss
    
    def forward_with_logits(self, img_idx, edge_idx, line_idx, masks=None):
        img_idx = img_idx * (1 - masks)  # create masked image
        edge_idx = edge_idx * (1 - masks) # create masked edge
        line_idx = line_idx * (1 - masks) # create masked line
        img_idx, edge_idx, line_idx, masks = img_idx.squeeze(dim=0), edge_idx.squeeze(dim=0), line_idx.squeeze(dim=0), masks.squeeze(dim=0)  # 把batch的維度拿掉(video每次只處理1個)
        edge_targets, line_targets = edge_targets.squeeze(dim=0), line_targets.squeeze(dim=0)

        x = torch.cat((img_idx, edge_idx, line_idx, masks), dim=1)  # concat method NEED checking (maybe is channel-wise)

        # Encoder: downsample
        x = self.pad1(x)  # reflection padding
        x = self.conv1(x)  # downsample input layer
        x = self.act(x)  # activate with ReLU

        x = self.conv2(x)  # downsample 1 
        x = self.act(x)

        x = self.conv3(x)  # downsample 2 
        x = self.act(x)

        x = self.conv4(x)  # downsample 3 
        x = self.act(x)

        [t, c, h, w] = x.shape  # before here, the video data is still with Height x Width -> [50, 256, 32, 32] -> [t, c, h, w]
        x = x.view(t, c, h * w).transpose(1, 2).contiguous() # image 2D -> 1D (flatten) and change image and color channel
        # make the data into shape like -> [batch size, image(1D), channels(RGB, edge, line, mask)]

        position_embeddings = self.pos_emb[:, :h * w, :]  # each position maps to a (learnable) vector
        x = self.drop(x + position_embeddings)  # [b,hw,c]  # add positional embeddings, but dropping to make some position missing pos-emb
        x = x.permute(0, 2, 1).reshape(t, c, h, w)  # swap the image and channel back to [b, c, h*w] then reshape to [b,c,h,w]

        # Transformer blocks
        # input [50, 256, 32, 32]
        # print(f"shape before FuseFormer: {x.shape}")
        x = self.fuseformerBlock(x)
        # print(f"shape after FuseFormer: {tmp.shape}")

        # print(f"x shape: {x.shape}")
        # Decoder: upsample
        x = self.convt1(x) # upsample 1
        x = self.act(x)

        x = self.convt2(x) # upsample 2
        x = self.act(x)

        x = self.convt3(x) # upsample 3
        x = self.act(x)

        x = self.padt(x)  # padding back
        x = self.convt4(x)  # upsample output as the original image shape
        
        edge, line = torch.split(x, [1, 1], dim=1)  # seperate the TSR outputs

        return edge, line

if __name__=="__main__":

    stride = 5

    train_dataset = Dataset()

    train_sampler = None
    train_loader = DataLoader(
            train_dataset,
            batch_size= 2,
            shuffle=(train_sampler is None), 
            num_workers=8, 
            sampler=train_sampler)

    model_config = EdgeLineGPTConfig(embd_pdrop=0.0, resid_pdrop=0.0, n_embd=256, block_size=32,
                                     attn_pdrop=0.0, n_layer=16, n_head=8)
    IGPT_model = EdgeLineGPT256RelBCE_video(model_config, "cuda")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    IGPT_model.to(device)
    IGPT_model.train()

    # # items = next(iter(train_loader))
    for i, items in enumerate(train_loader):

        img, edge, line, mask = items
        img, edge, line, mask = img.cuda(), edge.cuda(), line.cuda(), mask.cuda()
        # print(f"type img: {(img.shape)}")
        # print(f"type mask: {(mask.shape)}")
        # print(f"type edge: {(edge.shape)}")
        # print(f"type line: {(line.shape)}")
        # feed all frames in one process
        edge, line, loss = IGPT_model(img, edge, line, edge, line, mask)

        print(f"loss: {loss}")

        # stride 5                 
        # original shape: [b, t, c, h, w]       
        # frame_len = items['frames'].shape[-4]  
        # edge_list, line_list, loss_list = [], [], []
        # for idx in range(0, frame_len, stride): 
        #     select_frames = items['frames'][:,idx:idx+stride,:,:,:].to(device)
        #     select_edges = items['edges'][:,idx:idx+stride,:,:,:].to(device)
        #     select_lines = items['lines'][:,idx:idx+stride,:,:,:].to(device)
        #     select_masks = items['masks'][:,idx:idx+stride,:,:,:].to(device)
        #     # edge, line, loss = IGPT_model(select_frames, select_edges, select_lines, select_edges, select_lines, select_masks)
        #     _, _, loss = IGPT_model(select_frames, select_edges, select_lines, select_edges, select_lines, select_masks)
            
        #     # edge_list.append(edge)
        #     # line_list.append(line)
        #     loss_list.append(loss)

        #     print(f"\nvideo no.: {i}, idx: {idx}")
        #     # print(f"edge shape: {edge.shape}")
        #     # print(f"line shape: {line.shape}")
        #     print(f"average loss= {sum(loss_list)/len(loss_list)}")

        # print(f"\n final loss: {sum(loss_list)/len(loss_list)}")
        # IGPT_model.zero_grad()
        # if i == 0: break


