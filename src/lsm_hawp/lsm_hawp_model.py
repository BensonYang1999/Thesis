import torch
from .detector import WireframeDetector
from tqdm import tqdm
import torchvision.transforms as transforms
import os
import numpy as np
from skimage import io
from torchvision.transforms import functional as F
from skimage.transform import resize
import pickle

from PIL import Image as im
import skimage.draw
from skimage.feature import canny
from skimage.color import rgb2gray
import cv2

class ResizeImage(object):
    def __init__(self, image_height, image_width):
        self.image_height = image_height
        self.image_width = image_width

    def __call__(self, image):
        image = resize(image, (self.image_height, self.image_width))
        image = np.array(image, dtype=np.float32) / 255.0
        return image


class ToTensor(object):
    def __call__(self, image):
        return F.to_tensor(image)


class Normalize(object):
    def __init__(self, mean, std, to_255=True):
        self.mean = mean
        self.std = std
        self.to_255 = to_255

    def __call__(self, image):
        if self.to_255:
            image *= 255.0
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image


def to_device(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    if isinstance(data, dict):
        for key in data:
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].to(device)
        return data
    if isinstance(data, list):
        return [to_device(d, device) for d in data]


class LSM_HAWP:
    def __init__(self, threshold=0.6, size=512):
        self.lsm_hawp = WireframeDetector(is_cuda=True).cuda()
        self.transform = transforms.Compose([ResizeImage(size, size), ToTensor(),
                                             Normalize(mean=[109.730, 103.832, 98.681],
                                                       std=[22.275, 22.124, 23.229],
                                                       to_255=True)])
        self.threshold = threshold

    def wireframe_detect(self, img_paths, output_path):
        os.makedirs(output_path, exist_ok=True)
        self.lsm_hawp.eval()
        with torch.no_grad():
            for img_path in tqdm(img_paths):
                image = io.imread(img_path).astype(float)
                if len(image.shape) == 3:
                    image = image[:, :, :3]
                else:
                    image = image[:, :, None]
                    image = np.tile(image, [1, 1, 3])
                image = self.transform(image).unsqueeze(0).cuda()
                output = self.lsm_hawp(image)
                output = to_device(output, 'cpu')
                lines = []
                scores = []
                if output['num_proposals'] > 0:
                    lines_tmp = output['lines_pred'].numpy()
                    scores_tmp = output['lines_score'].tolist()
                    for line, score in zip(lines_tmp, scores_tmp):
                        if score > self.threshold:
                            # y1, x1, y2, x2
                            lines.append([line[1], line[0], line[3], line[2]])
                            scores.append(score)
                wireframe_info = {'lines': lines, 'scores': scores}
                with open(os.path.join(output_path, img_path.split('/')[-1].split('.')[0] + '.pkl'), 'wb') as w:
                    pickle.dump(wireframe_info, w)
    
    def wireframe_detect_visualize(self, img_paths, output_wire_pkl_path, output_wire_path, output_edge_path, size):
        def to_int(x):
            return tuple(map(int, x))
        
        self.lsm_hawp.eval()
        with torch.no_grad():
            for img_path in tqdm(img_paths):
                image = io.imread(img_path).astype(float)
                if len(image.shape) == 3:
                    image = image[:, :, :3]
                else:
                    image = image[:, :, None]
                    image = np.tile(image, [1, 1, 3])
                image = self.transform(image).unsqueeze(0).cuda()
                output = self.lsm_hawp(image)
                output = to_device(output, 'cpu')
                lines = []
                scores = []
                if output['num_proposals'] > 0:
                    lines_tmp = output['lines_pred'].numpy()
                    scores_tmp = output['lines_score'].tolist()
                    for line, score in zip(lines_tmp, scores_tmp):
                        if score > self.threshold:
                            # y1, x1, y2, x2
                            lines.append([line[1], line[0], line[3], line[2]])
                            scores.append(score)
                wireframe_info = {'lines': lines, 'scores': scores}

                # Draw line image (.png)
                lmap = np.zeros((size, size))
                for i in range(len(wireframe_info['scores'])):  # 所有偵測到的線條
                    if wireframe_info['scores'][i] > 0.8:  # 依序檢查，有超過threshold的線條才會拿來使用
                        line = wireframe_info['lines'][i].copy()
                        line[0] = line[0] * size
                        line[1] = line[1] * size
                        line[2] = line[2] * size
                        line[3] = line[3] * size
                        rr, cc, value = skimage.draw.line_aa(*to_int(line[0:2]), *to_int(line[2:4]))
                        # lmap[rr, cc] = np.maximum(lmap[rr, cc], value) 
                        lmap[rr, cc] = 255

                img = im.fromarray(lmap)
                if img.mode != 'L':
                    img = img.convert('L')
                img.save(os.path.join(output_wire_path, img_path.split('/')[-1].split('.')[0] + '.png'))

                # Draw edge image (.png)
                img_rgb = cv2.imread(img_path)
                edge = canny(rgb2gray(img_rgb), sigma=2, mask=None).astype(np.float)*255
                img_edge = im.fromarray(edge)
                if img_edge.mode != 'L':
                    img_edge = img_edge.convert('L')
                img_edge.save(os.path.join(output_edge_path, img_path.split('/')[-1].split('.')[0] + '.png'))

                # dump line (.pkl)
                with open(os.path.join(output_wire_pkl_path, img_path.split('/')[-1].split('.')[0] + '.pkl'), 'wb') as w:
                    pickle.dump(wireframe_info, w)

    def wireframe_places2_detect(self, img_paths, output_path):
        os.makedirs(output_path, exist_ok=True)
        self.lsm_hawp.eval()
        with torch.no_grad():
            for img_path in tqdm(img_paths):
                sub_paths = img_path.split('/')
                idx = sub_paths.index('data_large')
                new_output = output_path + '/'.join(sub_paths[idx + 1:-1])
                os.makedirs(new_output, exist_ok=True)
                new_output = os.path.join(new_output, img_path.split('/')[-1].split('.')[0] + '.pkl')
                if os.path.exists(new_output):
                    continue
                try:
                    image = io.imread(img_path).astype(float)
                except:
                    print('error to load', img_path)
                    continue
                if len(image.shape) == 3:
                    image = image[:, :, :3]
                else:
                    image = image[:, :, None]
                    image = np.tile(image, [1, 1, 3])
                image = self.transform(image).unsqueeze(0).cuda()
                output = self.lsm_hawp(image)
                output = to_device(output, 'cpu')
                lines = []
                scores = []
                if output['num_proposals'] > 0:
                    lines_tmp = output['lines_pred'].numpy()
                    scores_tmp = output['lines_score'].tolist()
                    for line, score in zip(lines_tmp, scores_tmp):
                        if score > self.threshold:
                            # y1, x1, y2, x2
                            lines.append([line[1], line[0], line[3], line[2]])
                            scores.append(score)
                wireframe_info = {'lines': lines, 'scores': scores}
                with open(new_output, 'wb') as w:
                    pickle.dump(wireframe_info, w)
