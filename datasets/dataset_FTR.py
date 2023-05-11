import glob
import os
import pickle
import random

import cv2
import numpy as np
import skimage.draw
import torch
import torchvision.transforms.functional as F
from skimage.color import rgb2gray
from skimage.feature import canny
from torch.utils.data import DataLoader

""" FuseFormer imports """
import torchvision.transforms as transforms
from src.Fuseformer.utils import create_random_shape_with_random_motion
from src.Fuseformer.utils import Stack, ToTorchFormatTensor

sys.path.append('..')

def to_int(x):
    return tuple(map(int, x))

to_tensors = transforms.Compose([
    Stack(),
    ToTorchFormatTensor(), ])

to_tensors_minMaxNorm = transforms.Compose([
    Stack(),
    ToTorchFormatTensor(), 
    MinMaxNormalize(),])
    # SetNonZeroToOne(),])


class ImgDataset(torch.utils.data.Dataset):
    def __init__(self, flist, input_size, mask_rates=None, mask_path=None, augment=True, training=True, test_mask_path=None):
        super(ImgDataset, self).__init__()
        self.augment = augment
        self.training = training

        self.data = []
        f = open(flist, 'r')
        for i in f.readlines():
            i = i.strip()
            self.data.append(i)
        f.close()

        if training:
            self.irregular_mask_list = []
            with open(mask_path[0]) as f:
                for line in f:
                    self.irregular_mask_list.append(line.strip())
            self.irregular_mask_list = sorted(self.irregular_mask_list, key=lambda x: x.split('/')[-1])
            self.segment_mask_list = []
            with open(mask_path[1]) as f:
                for line in f:
                    self.segment_mask_list.append(line.strip())
            self.segment_mask_list = sorted(self.segment_mask_list, key=lambda x: x.split('/')[-1])
        else:
            self.mask_list = glob.glob(test_mask_path + '/*')
            self.mask_list = sorted(self.mask_list, key=lambda x: x.split('/')[-1])
        self.input_size = input_size
        self.mask_rates = mask_rates

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.load_item(index)
        return item

    def load_name(self, index):
        name = self.data[index]
        return os.path.basename(name)

    def load_item(self, index):
        size = self.input_size
        # load image
        img = cv2.imread(self.data[index])
        while img is None:
            print('Bad image {}...'.format(self.data[index]))
            idx = random.randint(0, len(self.data) - 1)
            img = cv2.imread(self.data[idx])
        img = img[:, :, ::-1]
        # resize/crop if needed
        img = self.resize(img, size, size)
        # load mask
        mask = self.load_mask(img, index)
        # augment data
        if self.augment and random.random() > 0.5 and self.training:
            img = img[:, ::-1, ...].copy()
        if self.augment and random.random() > 0.5 and self.training:
            mask = mask[:, ::-1, ...].copy()
        if self.augment and random.random() > 0.5 and self.training:
            mask = mask[::-1, :, ...].copy()

        batch = dict()
        batch['image'] = self.to_tensor(img)
        batch['mask'] = self.to_tensor(mask)
        batch['name'] = self.load_name(index)
        return batch

    def load_mask(self, img, index):
        imgh, imgw = img.shape[0:2]
        if self.training is False:
            mask = cv2.imread(self.mask_list[index], cv2.IMREAD_GRAYSCALE)
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
                mask = np.clip(mask1 + mask2, 0, 255).astype(np.uint8)

            if mask.shape[0] != imgh or mask.shape[1] != imgw:
                mask = cv2.resize(mask, (imgw, imgh), interpolation=cv2.INTER_NEAREST)
            mask = (mask > 127).astype(np.uint8) * 255  # threshold due to interpolation
            return mask

    def to_tensor(self, img):
        # img = Image.fromarray(img)
        img_t = F.to_tensor(img).float()
        return img_t

    def resize(self, img, height, width, center_crop=False):
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

    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True
            )

            for item in sample_loader:
                yield item

class ImgDataset_video(torch.utils.data.Dataset):

    # write a video version of __init__ function
    def __init__(self, opts, sample=5, size=(432, 240), spit='train', name='YoutubeVOS', root='./datasets'):
        self.split = split
        self.sample_length = sample
        self.input_size = self.w, self.h = size
        self.opts = opts
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

class DynamicDataset(torch.utils.data.Dataset):
    def __init__(self, flist, batch_size,
                 mask_path=None, add_pos=False, pos_num=128,
                 train_line_path=None, eval_line_path=None, wireframe_th=0.85,
                 augment=True, training=True, test_mask_path=None,
                 input_size=None, load_path=False, default_size=256, str_size=256,
                 world_size=1, mask_rate=[0.4, 0.8, 1.0], min_sigma=2.0, max_sigma=2.5,
                 round=1):
        super(DynamicDataset, self).__init__()
        self.augment = augment
        self.training = training
        self.train_line_path = train_line_path
        self.eval_line_path = eval_line_path
        self.wireframe_th = wireframe_th
        self.batch_size = batch_size
        self.mask_rate = mask_rate
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.round = round  # for places2 round is 32

        self.data = []
        if load_path:
            self.data = glob.glob(flist + '/*')
            self.data = sorted(self.data, key=lambda x: x.split('/')[-1])
        else:
            f = open(flist, 'r')
            for i in f.readlines():
                i = i.strip()
                self.data.append(i)
            f.close()

        if training:
            self.irregular_mask_list = []
            with open(mask_path[0]) as f:
                for line in f:
                    self.irregular_mask_list.append(line.strip())
            self.irregular_mask_list = sorted(self.irregular_mask_list, key=lambda x: x.split('/')[-1])
            self.segment_mask_list = []
            with open(mask_path[1]) as f:
                for line in f:
                    self.segment_mask_list.append(line.strip())
            self.segment_mask_list = sorted(self.segment_mask_list, key=lambda x: x.split('/')[-1])
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

    def reset_dataset(self, shuffled_idx):
        self.idx_map = {}
        barrel_idx = 0
        count = 0
        for sid in shuffled_idx:
            self.idx_map[sid] = barrel_idx
            count += 1
            if count == self.batch_size:
                count = 0
                barrel_idx += 1
        # random img size:256~512
        if self.training:
            barrel_num = int(len(self.data) / (self.batch_size * self.world_size))
            barrel_num += 2
            if self.round == 1:
                self.input_size = np.clip(np.arange(32, 65,
                                                    step=(65 - 32) / barrel_num * 2).astype(int) * 8, 256, 512).tolist()
                self.input_size = self.input_size[::-1] + self.input_size
            else:
                self.input_size = []
                input_size = np.clip(np.arange(32, 65, step=(65 - 32) / barrel_num * 2 * self.round).astype(int) * 8,
                                     256, 512).tolist()
                for _ in range(self.round + 1):
                    self.input_size.extend(input_size[::-1])
                    self.input_size.extend(input_size)
        else:
            self.input_size = self.default_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.load_item(index)
        return item

    def load_name(self, index):
        name = self.data[index]
        return os.path.basename(name)

    def load_item(self, index):
        if type(self.input_size) == list:
            maped_idx = self.idx_map[index]
            if maped_idx > len(self.input_size) - 1:
                size = 512
            else:
                size = self.input_size[maped_idx]
        else:
            size = self.input_size

        # load image
        img = cv2.imread(self.data[index])
        while img is None:
            print('Bad image {}...'.format(self.data[index]))
            idx = random.randint(0, len(self.data) - 1)
            img = cv2.imread(self.data[idx])
        img = img[:, :, ::-1]
        # resize/crop if needed
        img = self.resize(img, size, size)
        img_256 = self.resize(img, self.str_size, self.str_size)

        # load mask
        mask = self.load_mask(img, index)
        mask_256 = cv2.resize(mask, (self.str_size, self.str_size), interpolation=cv2.INTER_AREA)
        mask_256[mask_256 > 0] = 255

        gray = rgb2gray(img)
        edge = self.load_edge(gray, sigma=self.min_sigma + ((size - 256) / 256 * (self.max_sigma - self.min_sigma)))
        gray_256 = rgb2gray(img_256)
        edge_256 = self.load_edge(gray_256, sigma=self.min_sigma + (
                (self.str_size - 256) / 256 * (self.max_sigma - self.min_sigma)))

        # load line
        line = self.load_wireframe(index, size)
        line_256 = self.load_wireframe(index, self.str_size)

        # augment data
        if self.augment and random.random() > 0.5 and self.training:
            img = img[:, ::-1, ...].copy()
            img_256 = img_256[:, ::-1, ...].copy()
            edge = edge[:, ::-1, ...].copy()
            edge_256 = edge_256[:, ::-1, ...].copy()
            line = line[:, ::-1, ...].copy()
            line_256 = line_256[:, ::-1, ...].copy()
        if self.augment and random.random() > 0.5 and self.training:
            mask = mask[:, ::-1, ...].copy()
            mask_256 = mask_256[:, ::-1, ...].copy()
        if self.augment and random.random() > 0.5 and self.training:
            mask = mask[::-1, :, ...].copy()
            mask_256 = mask_256[::-1, :, ...].copy()

        batch = dict()
        batch['image'] = self.to_tensor(img)
        batch['img_256'] = self.to_tensor(img_256, norm=True)
        batch['mask'] = self.to_tensor(mask)
        batch['mask_256'] = self.to_tensor(mask_256)
        batch['edge'] = self.to_tensor(edge)
        batch['edge_256'] = self.to_tensor(edge_256)
        batch['line'] = self.to_tensor(line)
        batch['line_256'] = self.to_tensor(line_256)
        batch['size_ratio'] = size / self.default_size

        batch['name'] = self.load_name(index)

        # load pos encoding
        rel_pos, abs_pos, direct = self.load_masked_position_encoding(mask)
        batch['rel_pos'] = torch.LongTensor(rel_pos)
        batch['abs_pos'] = torch.LongTensor(abs_pos)
        batch['direct'] = torch.LongTensor(direct)

        return batch

    def load_masked_position_encoding(self, mask):
        ori_mask = mask.copy()
        ori_h, ori_w = ori_mask.shape[0:2]
        ori_mask = ori_mask / 255
        mask = cv2.resize(mask, (self.str_size, self.str_size), interpolation=cv2.INTER_AREA)
        mask[mask > 0] = 255
        h, w = mask.shape[0:2]
        mask3 = mask.copy()
        mask3 = 1. - (mask3 / 255.0)
        pos = np.zeros((h, w), dtype=np.int32)
        direct = np.zeros((h, w, 4), dtype=np.int32)
        i = 0
        while np.sum(1 - mask3) > 0:
            i += 1
            mask3_ = cv2.filter2D(mask3, -1, self.ones_filter)
            mask3_[mask3_ > 0] = 1
            sub_mask = mask3_ - mask3
            pos[sub_mask == 1] = i

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

        abs_pos = pos.copy()
        rel_pos = pos / (self.str_size / 2)  # to 0~1 maybe larger than 1
        rel_pos = (rel_pos * self.pos_num).astype(np.int32)
        rel_pos = np.clip(rel_pos, 0, self.pos_num - 1)

        if ori_w != w or ori_h != h:
            rel_pos = cv2.resize(rel_pos, (ori_w, ori_h), interpolation=cv2.INTER_NEAREST)
            rel_pos[ori_mask == 0] = 0
            direct = cv2.resize(direct, (ori_w, ori_h), interpolation=cv2.INTER_NEAREST)
            direct[ori_mask == 0, :] = 0

        return rel_pos, abs_pos, direct

    def load_mask(self, img, index):
        imgh, imgw = img.shape[0:2]
        if self.training is False:
            mask = cv2.imread(self.mask_list[index % len(self.mask_list)], cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (imgw, imgh), interpolation=cv2.INTER_NEAREST)
            mask = (mask > 127).astype(np.uint8) * 255
            return mask
        else:  # train mode: 40% mask with random brush, 40% mask with coco mask, 20% with additions
            rdv = random.random()
            if rdv < self.mask_rate[0]:
                mask_index = random.randint(0, len(self.irregular_mask_list) - 1)
                mask = cv2.imread(self.irregular_mask_list[mask_index],
                                  cv2.IMREAD_GRAYSCALE)
            elif rdv < self.mask_rate[1]:
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
                mask = np.clip(mask1 + mask2, 0, 255).astype(np.uint8)

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

    def resize(self, img, height, width, center_crop=False):
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

    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True
            )

            for item in sample_loader:
                yield item

    def load_edge(self, img, sigma=2):
        return canny(img, sigma=sigma, mask=None).astype(np.float)

    def load_wireframe(self, idx, size):
        selected_img_name = self.data[idx]
        line_name = os.path.basename(selected_img_name)
        if self.training is False:
            line_name = self.eval_line_path + '/' + line_name
        else:
            line_name = self.train_line_path + '/' + line_name
        line_name = line_name.replace('.png', '.pkl').replace('.jpg', '.pkl')

        wf = pickle.load(open(line_name, 'rb'))
        lmap = np.zeros((size, size))
        for i in range(len(wf['scores'])):
            if wf['scores'][i] > self.wireframe_th:
                line = wf['lines'][i].copy()
                line[0] = line[0] * size
                line[1] = line[1] * size
                line[2] = line[2] * size
                line[3] = line[3] * size
                rr, cc, value = skimage.draw.line_aa(*to_int(line[0:2]), *to_int(line[2:4]))
                lmap[rr, cc] = np.maximum(lmap[rr, cc], value)
        return lmap

class ToLongTensorFromNumpy(object):
    """Convert numpy arrays in sample to Tensors."""
    def __call__(self, sample):
        return torch.from_numpy(sample).long()

class DynamicDataset_video(torch.utils.data.Dataset):

    def __init__(self, split='train', name='YouTubeVOS', root='./datasets', batch_size=1, add_pos=False, pos_num=128, test_mask_path=None,
                 input_size=None, default_size=256, str_size=256,
                 world_size=1,
                 round=1, sample=5): 
        # super(DynamicDataset, self).__init__()
        self.training = (split=='train') # training or testing
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
        if self.training:
            barrel_num = int(len(self.video_name) / (self.batch_size * self.world_size))
            barrel_num += 2
            if self.round == 1:
                self.input_size = np.clip(np.arange(32, 65,
                                                    step=(65 - 32) / barrel_num * 2).astype(int) * 8, 256, 512).tolist()
                self.input_size = self.input_size[::-1] + self.input_size
            else:
                self.input_size = []
                input_size = np.clip(np.arange(32, 65, step=(65 - 32) / barrel_num * 2 * self.round).astype(int) * 8,
                                     256, 512).tolist()
                for _ in range(self.round + 1):
                    self.input_size.extend(input_size[::-1])
                    self.input_size.extend(input_size)
        else:
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
        if type(self.input_size) == list:
            maped_idx = self.idx_map[index]
            if maped_idx > len(self.input_size) - 1:
                size = 512
            else:
                size = self.input_size[maped_idx]
        else:
            size = self.input_size

        video_name = self.video_name[index]
        edge_name = self.edge_names[index]
        line_name = self.line_names[index]
        all_frames = [os.path.join(video_name, name) for name in sorted(os.listdir(video_name))]
        all_edges = [os.path.join(edge_name, name) for name in sorted(os.listdir(edge_name))]
        all_lines = [os.path.join(line_name, name) for name in sorted(os.listdir(line_name))]
        all_masks = create_random_shape_with_random_motion(
            len(all_frames), imageHeight=size, imageWidth=size)
        ref_index = self.get_ref_index(len(all_frames), self.sample_length)
        
        frames, frames_256 = [], []
        edges, edges_256 = [], []
        lines, lines_256 = [], []
        masks, masks_256 = [], []
        for idx in ref_index:
            # load image
            img = Image.open(all_frames[idx]).convert('RGB')
            img = img.resize((size, size))
            img_256 = img.resize((self.str_size, self.str_size))
            frames.append(img)
            frames_256.append(img_256)

            # # load mask
            mask = all_masks[idx]
            mask_256 = mask.resize((self.str_size, self.str_size))
            masks.append(mask)
            masks_256.append(mask_256)
            # mask = Image.open(all_masks[idx]).convert('L')
            # mask = mask.resize(self.input_size)
            # mask_small = mask.resize((self.str_size, self.str_size))
            # # mask_small[mask_256 > 0] = 255 # make sure the mask is binary 
            # masks.append(mask)
            # masks_small.append(mask_small)

            # load edge
            egde = Image.open(all_edges[idx]).convert('L')
            egde = egde.resize((size, size))
            egde_256 = egde.resize((self.str_size, self.str_size))
            edges.append(egde)
            edges_256.append(egde_256)

            # load line
            line = Image.open(all_lines[idx]).convert('L')
            line = line.resize((size, size))
            line_256 = line.resize((self.str_size, self.str_size))
            lines.append(line)
            lines_256.append(line_256)

        # augment data
        if self.split == 'train':
            prob = random.random()
            frames = GroupRandomHorizontalFlip()(frames, prob)
            edges = GroupRandomHorizontalFlip()(edges, prob)
            lines = GroupRandomHorizontalFlip()(lines, prob)
            # masks = GroupRandomHorizontalFlip()(masks, prob)

        batch = dict()
        batch['image'] = self._to_tensors(frames)
        batch['img_256'] = self._to_tensors(frames_256)*2.0 - 1.0 # normalize to [-1, 1]
        batch['mask'] = self._to_tensors(masks)
        batch['mask_256'] = self._to_tensors(masks_256)
        batch['edge'] = self._to_tensors(edges)
        batch['edge_256'] = self._to_tensors(edges_256)
        batch['line'] = self._to_tensors(lines)
        batch['line_256'] = self._to_tensors(lines_256)
        batch['size_ratio'] = size / self.default_size

        batch['name'] = self.load_name(index)

        # load pos encoding
        rel_pos_list, abs_pos_list, direct_list = [], [], []
        for m in masks:
            # transfrom mask to numpy array
            rel_pos, abs_pos, direct = self.load_masked_position_encoding(np.array(m))
            rel_pos_list.append(rel_pos)
            abs_pos_list.append(abs_pos)
            direct_list.append(direct)
        
        batch['rel_pos'] = self._to_long_tensors(rel_pos_list) 
        batch['abs_pos'] = self._to_long_tensors(abs_pos_list) 
        batch['direct'] = self._to_long_tensors(direct_list)

        return batch

    def load_masked_position_encoding(self, mask):
        ori_mask = mask.copy()
        ori_h, ori_w = ori_mask.shape[0:2] # original size
        ori_mask = ori_mask / 255
        mask = cv2.resize(mask, (self.str_size, self.str_size), interpolation=cv2.INTER_AREA)
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