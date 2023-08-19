import os
import random
import sys
from glob import glob

import cv2
import numpy as np
import torchvision.transforms.functional as F
from skimage.color import rgb2gray
from skimage.feature import canny
from torch.utils.data import Dataset
import pickle
import skimage.draw
import torch

"""
FuseFormer imports
"""
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from src.Fuseformer.utils import create_random_shape_with_random_motion
from src.Fuseformer.utils import Stack, ToTorchFormatTensor, GroupRandomHorizontalFlip

sys.path.append('..')


def to_int(x):
    return tuple(map(int, x))


class ContinuousEdgeLineDatasetMask(Dataset):

    def __init__(self, pt_dataset, mask_path=None, test_mask_path=None, is_train=False, mask_rates=None,
                 image_size=256, line_path=None):

        self.is_train = is_train
        self.pt_dataset = pt_dataset

        self.image_id_list = []
        with open(self.pt_dataset) as f:
            for line in f:
                self.image_id_list.append(line.strip())  # 從指定的training image txt讀入所有要訓練的image的路徑，此list裡面每一個element就是一張RGB圖片

        if is_train:
            # training mask TYPE1: irregular mask
            self.irregular_mask_list = []
            with open(mask_path[0]) as f:
                for line in f:
                    self.irregular_mask_list.append(line.strip())
            self.irregular_mask_list = sorted(self.irregular_mask_list, key=lambda x: x.split('/')[-1])
            # training mask TYPE2: segmentation mask
            self.segment_mask_list = []
            with open(mask_path[1]) as f:
                for line in f:
                    self.segment_mask_list.append(line.strip())
            self.segment_mask_list = sorted(self.segment_mask_list, key=lambda x: x.split('/')[-1])
        else:
            self.mask_list = glob(test_mask_path + '/*')  # 在測試時，mask的路徑預設為一個資料夾，因此在建立mask list時要將參數給定的mask路徑下所有個圖片都讀入，glob為取得所有的檔案的路徑
            self.mask_list = sorted(self.mask_list, key=lambda x: x.split('/')[-1])
            print(f"self.mask_list: {self.mask_list}")  # test

        self.image_size = image_size  # 設定圖片大小：預設訓練的圖片大小為固定
        self.training = is_train  # 是否為訓練模式
        self.mask_rates = mask_rates  # 設定mask的比例, 'irregular rate, coco rate, addition rate': 0.4, 0.8, 1.0
        self.line_path = line_path  # 設定預先使用wireframe偵測儲存下來的圖片
        self.wireframe_th = 0.85

    def __len__(self):
        return len(self.image_id_list)  # 有多少張訓練圖片

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

    def load_mask(self, img, index):
        imgh, imgw = img.shape[0:2]

        # test mode: load mask non random
        if self.training is False:
            mask = cv2.imread(self.mask_list[index], cv2.IMREAD_GRAYSCALE)  # 以灰階的模式取得mask的路徑
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

    def to_tensor(self, img, norm=False):
        # img = Image.fromarray(img)
        img_t = F.to_tensor(img).float()
        if norm:
            img_t = F.normalize(img_t, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        return img_t

    def load_edge(self, img):
        return canny(img, sigma=2, mask=None).astype(np.float)

    def load_wireframe(self, idx, size):
        selected_img_name = self.image_id_list[idx]
        line_name = self.line_path + '/' + os.path.basename(selected_img_name).replace('.png', '.pkl').replace('.jpg', '.pkl')  # 從訓練的index知道目前的訓練image的名稱是什麼，而wireframe的檔名會與image相同但副檔名不同
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
        selected_img_name = self.image_id_list[idx]  # 目前訓練的image case名字
        img = cv2.imread(selected_img_name)  # 讀取此image的rgb版本
        while img is None:
            print('Bad image {}...'.format(selected_img_name))
            idx = random.randint(0, len(self.image_id_list) - 1)
            img = cv2.imread(self.image_id_list[idx])
        img = img[:, :, ::-1]  # RGB轉成BGR

        img = self.resize(img, self.image_size, self.image_size, center_crop=False)  # 切割成正方形
        img_gray = rgb2gray(img)
        edge = self.load_edge(img_gray)
        line = self.load_wireframe(idx, self.image_size)
        # load mask
        mask = self.load_mask(img, idx)
        # augment data
        if self.training is True:
            if random.random() < 0.5:
                img = img[:, ::-1, ...].copy()
                edge = edge[:, ::-1].copy()
                line = line[:, ::-1].copy()
            if random.random() < 0.5:
                mask = mask[:, ::-1, ...].copy()
            if random.random() < 0.5:
                mask = mask[::-1, :, ...].copy()

        img = self.to_tensor(img, norm=True)
        edge = self.to_tensor(edge)
        line = self.to_tensor(line)
        mask = self.to_tensor(mask)
        meta = {'img': img, 'mask': mask, 'edge': edge, 'line': line,
                'name': os.path.basename(selected_img_name)}
        return meta


class ContinuousEdgeLineDatasetMaskFinetune(ContinuousEdgeLineDatasetMask):

    def __init__(self, pt_dataset, mask_path=None, test_mask_path=None,
                 is_train=False, mask_rates=None, image_size=256, line_path=None):
        super().__init__(pt_dataset, mask_path, test_mask_path, is_train, mask_rates, image_size, line_path)

    def __getitem__(self, idx):
        selected_img_name = self.image_id_list[idx]
        img = cv2.imread(selected_img_name)
        while img is None:
            print('Bad image {}...'.format(selected_img_name))
            idx = random.randint(0, len(self.image_id_list) - 1)
            img = cv2.imread(self.image_id_list[idx])
        img = img[:, :, ::-1]

        img = self.resize(img, self.image_size, self.image_size, center_crop=False)
        img_gray = rgb2gray(img)
        edge = self.load_edge(img_gray)
        line = self.load_wireframe(idx, self.image_size)
        # load mask
        mask = self.load_mask(img, idx)
        # augment data
        if self.training is True:
            if random.random() < 0.5:
                img = img[:, ::-1, ...].copy()
                edge = edge[:, ::-1].copy()
                line = line[:, ::-1].copy()
            if random.random() < 0.5:
                mask = mask[:, ::-1, ...].copy()
            if random.random() < 0.5:
                mask = mask[::-1, :, ...].copy()

        erode = mask
        img = self.to_tensor(img, norm=True)
        edge = self.to_tensor(edge)
        line = self.to_tensor(line)
        mask = self.to_tensor(mask)
        mask_img = img * (1 - mask)

        # aug for mask-predict
        while True:
            if random.random() > 0.5:
                erode = self.to_tensor(erode)
                break
            k_size = random.randint(5, 25)
            erode2 = cv2.erode(erode // 255, np.ones((k_size, k_size), np.uint8), iterations=1)
            if np.sum(erode2) > 0:
                erode = self.to_tensor(erode2 * 255)
                break

        meta = {'img': img, 'mask_img': mask_img, 'mask': mask, 'erode_mask': erode, 'edge': edge, 'line': line,
                'name': os.path.basename(selected_img_name)}

        return meta

class SetNonZeroToOne:
    def __call__(self, tensor):
        tensor[tensor != 0] = 1
        return tensor

class MinMaxNormalize(torch.nn.Module):
    def __init__(self, eps=1e-8):
        super(MinMaxNormalize, self).__init__()
        self.eps = eps

    def forward(self, image_tensor):
        max_value = image_tensor.max()
        min_value = image_tensor.min()
        return (image_tensor - min_value) / (max_value - min_value + self.eps)

class StandardizeNormalize(torch.nn.Module):
    def __init__(self, mean, std, eps=1e-8):
        super(StandardizeNormalize, self).__init__()
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)
        self.eps = eps

    def forward(self, image_tensor):
        # Ensure the input tensors are in the range [0, 1] before standardizing
        image_tensor = torch.clamp(image_tensor, 0, 1)

        # Expand dimensions of mean and std tensors to match the dimensions of the input tensor
        mean = self.mean.view(-1, 1, 1)
        std = self.std.view(-1, 1, 1)

        # Perform standardization
        standardized_image = (image_tensor - mean) / (std + self.eps)
        return standardized_image


class ContinuousEdgeLineDatasetMask_video(Dataset):  # mostly refer to FuseFormer
    def __init__(self, opts, sample=5, size=(432,240), split='train', name='YouTubeVOS', root='./datasets'):
        self.split = split
        self.sample_length = sample
        self.size = self.w, self.h = size
        self.opts = opts

        if name == 'YouTubeVOS':
            vid_lst_prefix = os.path.join(root, name, split+'_all_frames/JPEGImages')
            if self.opts.edge_gaussian == 0:
                edge_lst_prefix = os.path.join(root, name, split+'_all_frames/edges_old')
            else:
                edge_lst_prefix = os.path.join(root, name, split+'_all_frames/edges')
            line_lst_prefix = os.path.join(root, name, split+'_all_frames/wireframes')
            vid_lst = os.listdir(vid_lst_prefix)
            edge_lst = os.listdir(edge_lst_prefix)
            line_lst = os.listdir(line_lst_prefix)
            self.video_names = [os.path.join(vid_lst_prefix, name) for name in vid_lst]
            self.edge_names = [os.path.join(edge_lst_prefix, name) for name in edge_lst]
            self.line_names = [os.path.join(line_lst_prefix, name) for name in line_lst]

            if not split == 'train':
                mask_lst_prefix = os.path.join(root, name, split+'_all_frames/mask_random')
                mask_lst = os.listdir(mask_lst_prefix)
                self.mask_names = [os.path.join(mask_lst_prefix, name) for name in mask_lst]

        self._to_tensors = transforms.Compose([
            Stack(),
            ToTorchFormatTensor(), ])

        self._to_tensors_minMaxNorm = transforms.Compose([
            Stack(),
            ToTorchFormatTensor(), 
            MinMaxNormalize(),])
            # SetNonZeroToOne(),])

        mean = [0.485, 0.456, 0.406]  # imagenet
        std = [0.229, 0.224, 0.225] # imagenet
        self._to_tensors_stdNorm = transforms.Compose([
            Stack(),
            ToTorchFormatTensor(),
            StandardizeNormalize(mean, std),
        ])


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
        if self.split=="train":
            all_masks = create_random_shape_with_random_motion(
                len(all_frames), imageHeight=self.h, imageWidth=self.w)
        else:
            mask_name = self.mask_names[index]
            all_masks = [os.path.join(mask_name, name) for name in sorted(os.listdir(mask_name))]

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

            if self.split == 'train':
                masks.append(all_masks[idx])
            else:
                mask = Image.open(all_masks[idx]).convert('L')
                # make sure the value of mask is either 0 or 255
                mask = np.array(mask)
                mask[mask > 0] = 255
                mask = Image.fromarray(mask)
                mask = mask.resize(self.size)
                masks.append(mask)
            # masks.append(all_masks[idx])

        if self.split == 'train':
            prob = random.random()
            frames = GroupRandomHorizontalFlip()(frames, prob)
            edges = GroupRandomHorizontalFlip()(edges, prob)
            lines = GroupRandomHorizontalFlip()(lines, prob)

        # To tensors
        frame_tensors = self._to_tensors(frames)*2.0 - 1.0 # normalize RGB to [-1, 1] -> from fuseformer
        # frame_tensors = self._to_tensors_stdNorm(frames) 
        # edge_tensors = self._to_tensors(edges)
        edge_tensors = self._to_tensors_minMaxNorm(edges)  # try to normalize
        # line_tensors = self._to_tensors(lines)
        line_tensors = self._to_tensors_minMaxNorm(lines) # try to normalize
        mask_tensors = self._to_tensors(masks)
        meta = {'frames': frame_tensors, 'masks': mask_tensors, 'edges': edge_tensors, 'lines': line_tensors, 
                'name': video_name.split('/')[-1], 'idxs': [all_frames[idx].split('/')[-1] for idx in ref_index]}
        return meta

    def get_ref_index(self, length, sample_length):
        if random.uniform(0, 1) > 0.5:
            ref_index = random.sample(range(length), sample_length)
            ref_index.sort()
        else:
            pivot = random.randint(0, length-sample_length)
            ref_index = [pivot+i for i in range(sample_length)]
        return ref_index