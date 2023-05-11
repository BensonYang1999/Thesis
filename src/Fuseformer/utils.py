import os
import cv2
import random
import numpy as np
from PIL import Image, ImageOps

import torch
import matplotlib
import matplotlib.patches as patches
from matplotlib.path import Path
from matplotlib import pyplot as plt
matplotlib.use('agg')

import yaml
from torch.optim.lr_scheduler import LambdaLR
import math
import sys
import time
import torch.nn.functional as F
import torchvision.transforms.functional as FF
from torchvision import transforms

from skimage.color import gray2rgb
from skimage.color import rgb2gray
from skimage.feature import canny
# ###########################################################################
# ###########################################################################


class GroupRandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """

    def __init__(self, is_flow=False):
        self.is_flow = is_flow

    def __call__(self, img_group, random_seed, is_flow=False):
        # v = random.random()
        v = random_seed
        if v < 0.5:
            ret = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in img_group]
            if self.is_flow:
                for i in range(0, len(ret), 2):
                    # invert flow pixel values when flipping
                    ret[i] = ImageOps.invert(ret[i])
            return ret
        else:
            return img_group


class Stack(object):
    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):
        if isinstance(img_group[0], np.ndarray):
            if self.roll and img_group[0].shape[-1] == 3:
                return np.stack([x[:, :, ::-1] for x in img_group], axis=2)
            else:
                return np.stack(img_group, axis=2)
        else:
            mode = img_group[0].mode
            if mode == '1':
                img_group = [img.convert('L') for img in img_group]
                mode = 'L'
            if mode == 'L':
                return np.stack([np.expand_dims(x, 2) for x in img_group], axis=2)
            elif mode == 'RGB':
                if self.roll:
                    return np.stack([np.array(x)[:, :, ::-1] for x in img_group], axis=2)
                else:
                    return np.stack([np.array(x) for x in img_group], axis=2)
            else:
                raise NotImplementedError(f"Image mode {mode}")



class ToTorchFormatTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """

    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # numpy img: [L, C, H, W]
            img = torch.from_numpy(pic).permute(2, 3, 0, 1).contiguous()
        else:
            # handle PIL Image
            img = torch.ByteTensor(
                torch.ByteStorage.from_buffer(pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            # put it from HWC to CHW format
            # yikes, this transpose takes 80% of the loading time/CPU
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        img = img.float().div(255) if self.div else img.float()
        return img


# ##########################################
# ##########################################

def create_random_shape_with_random_motion(video_length, imageHeight=240, imageWidth=432):
    # get a random shape
    height = random.randint(imageHeight//3, imageHeight-1)
    width = random.randint(imageWidth//3, imageWidth-1)
    edge_num = random.randint(6, 8)
    ratio = random.randint(6, 8)/10
    region = get_random_shape(
        edge_num=edge_num, ratio=ratio, height=height, width=width)
    region_width, region_height = region.size
    # get random position
    x, y = random.randint(
        0, imageHeight-region_height), random.randint(0, imageWidth-region_width)
    velocity = get_random_velocity(max_speed=3)
    m = Image.fromarray(np.zeros((imageHeight, imageWidth)).astype(np.uint8))
    m.paste(region, (y, x, y+region.size[0], x+region.size[1]))
    masks = [m.convert('L')]
    # return fixed masks
    if random.uniform(0, 1) > 0.5:
        return masks*video_length
    # return moving masks
    for _ in range(video_length-1):
        x, y, velocity = random_move_control_points(
            x, y, imageHeight, imageWidth, velocity, region.size, maxLineAcceleration=(3, 0.5), maxInitSpeed=3)
        m = Image.fromarray(
            np.zeros((imageHeight, imageWidth)).astype(np.uint8))
        m.paste(region, (y, x, y+region.size[0], x+region.size[1]))
        masks.append(m.convert('L'))
    return masks


def get_random_shape(edge_num=9, ratio=0.7, width=432, height=240):
    '''
      There is the initial point and 3 points per cubic bezier curve. 
      Thus, the curve will only pass though n points, which will be the sharp edges.
      The other 2 modify the shape of the bezier curve.
      edge_num, Number of possibly sharp edges
      points_num, number of points in the Path
      ratio, (0, 1) magnitude of the perturbation from the unit circle, 
    '''
    points_num = edge_num*3 + 1
    angles = np.linspace(0, 2*np.pi, points_num)
    codes = np.full(points_num, Path.CURVE4)
    codes[0] = Path.MOVETO
    # Using this instad of Path.CLOSEPOLY avoids an innecessary straight line
    verts = np.stack((np.cos(angles), np.sin(angles))).T * \
        (2*ratio*np.random.random(points_num)+1-ratio)[:, None]
    verts[-1, :] = verts[0, :]
    path = Path(verts, codes)
    # draw paths into images
    fig = plt.figure()
    ax = fig.add_subplot(111)
    patch = patches.PathPatch(path, facecolor='black', lw=2)
    ax.add_patch(patch)
    ax.set_xlim(np.min(verts)*1.1, np.max(verts)*1.1)
    ax.set_ylim(np.min(verts)*1.1, np.max(verts)*1.1)
    ax.axis('off')  # removes the axis to leave only the shape
    fig.canvas.draw()
    # convert plt images into numpy images
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape((fig.canvas.get_width_height()[::-1] + (3,)))
    plt.close(fig)
    # postprocess
    data = cv2.resize(data, (width, height))[:, :, 0]
    data = (1 - np.array(data > 0).astype(np.uint8))*255
    corrdinates = np.where(data > 0)
    xmin, xmax, ymin, ymax = np.min(corrdinates[0]), np.max(
        corrdinates[0]), np.min(corrdinates[1]), np.max(corrdinates[1])
    region = Image.fromarray(data).crop((ymin, xmin, ymax, xmax))
    return region


def random_accelerate(velocity, maxAcceleration, dist='uniform'):
    speed, angle = velocity
    d_speed, d_angle = maxAcceleration
    if dist == 'uniform':
        speed += np.random.uniform(-d_speed, d_speed)
        angle += np.random.uniform(-d_angle, d_angle)
    elif dist == 'guassian':
        speed += np.random.normal(0, d_speed / 2)
        angle += np.random.normal(0, d_angle / 2)
    else:
        raise NotImplementedError(
            f'Distribution type {dist} is not supported.')
    return (speed, angle)


def get_random_velocity(max_speed=3, dist='uniform'):
    if dist == 'uniform':
        speed = np.random.uniform(max_speed)
    elif dist == 'guassian':
        speed = np.abs(np.random.normal(0, max_speed / 2))
    else:
        raise NotImplementedError(
            f'Distribution type {dist} is not supported.')
    angle = np.random.uniform(0, 2 * np.pi)
    return (speed, angle)


def random_move_control_points(X, Y, imageHeight, imageWidth, lineVelocity, region_size, maxLineAcceleration=(3, 0.5), maxInitSpeed=3):
    region_width, region_height = region_size
    speed, angle = lineVelocity
    X += int(speed * np.cos(angle))
    Y += int(speed * np.sin(angle))
    lineVelocity = random_accelerate(
        lineVelocity, maxLineAcceleration, dist='guassian')
    if ((X > imageHeight - region_height) or (X < 0) or (Y > imageWidth - region_width) or (Y < 0)):
        lineVelocity = get_random_velocity(maxInitSpeed, dist='guassian')
    new_X = np.clip(X, 0, imageHeight - region_height)
    new_Y = np.clip(Y, 0, imageWidth - region_width)
    return new_X, new_Y, lineVelocity


# ##############################################
# utils.py from MST model
# ##############################################
import torch.nn.functional as F
from torchvision.transforms.functional import rgb_to_grayscale
import torchvision.transforms as T

class Config(object):
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self._yaml = f.read()
            self._dict = yaml.load(self._yaml, Loader=yaml.SafeLoader)
            self._dict['path'] = os.path.dirname(config_path)

    def __getattr__(self, name):
        if self._dict.get(name) is not None:
            return self._dict[name]

        return None

    def print(self):
        print('Model configurations:')
        print('---------------------------------')
        print(self._yaml)
        print('')
        print('---------------------------------')
        print('')
        

def resize(img, height, width, centerCrop=False):
    img = img.unsqueeze(0)  # tensor version
    img = postprocess(img).cpu().numpy().astype('uint8')
    img = img.squeeze(0)  # tensor version
        
    # imgh, imgw = img.size[0:2]  # original
    imgh, imgw = img.shape[-2:]  # my adjust tensor version

    if centerCrop and imgh != imgw:
        # center crop
        side = np.minimum(imgh, imgw)
        j = (imgh - side) // 2
        i = (imgw - side) // 2
        img = img[j:j + side, i:i + side, ...]

    if height < imgh:
        img = cv2.resize(img, (height, width), interpolation=cv2.INTER_AREA)
    else:
        img = cv2.resize(img, (height, width))
    
    
    return img
     

def process_img(frames, img_size, hawp_size):
    origin_shape = frames.shape[-3:] # keep the (channel, width, height)

    item_lists = [[0 for _ in range(frames.shape[1])] for _ in range(frames.shape[0])]
    for batch_i in range(frames.shape[0]):
        for time_j in range(frames.shape[1]):
            origin_img = frames[batch_i, time_j, ...].clone()  # tensor format
            img = frames[batch_i, time_j, ...].clone()
            
            if origin_shape[0] == 1:  # only one color channel
                img = gray2rgb(img)
            hawp_img = resize(img, hawp_size, hawp_size)
            img = resize(img, img_size, img_size, centerCrop=False)
            img_gray = rgb2gray(img)
            edge = canny(img_gray, sigma=2).astype(np.float)

            meta = dict()
            img = to_tensor(img)
            edge = to_tensor(edge)
            hawp_img = to_tensor(hawp_img)
            img_gray = to_tensor(img_gray)
            meta['img'] = img
            meta['edges'] = edge
            meta['hawp_img'] = hawp_img
            meta['img_gray'] = img_gray
            meta['origin_shape'] = origin_shape
            meta['origin_img'] = origin_img
            item_lists[batch_i][time_j] = meta
            
    return item_lists


def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def stitch_images(inputs, outputs, img_per_row=2):
    gap = 5
    columns = len(outputs) + 1

    width, height = inputs[0][:, :, 0].shape
    img = Image.new('RGB', (width * img_per_row * columns + gap * (img_per_row - 1),
                            height * int(len(inputs) / img_per_row)))
    images = [inputs, *outputs]

    for ix in range(len(inputs)):
        xoffset = int(ix % img_per_row) * width * columns + int(ix % img_per_row) * gap
        yoffset = int(ix / img_per_row) * height

        for cat in range(len(images)):
            im = np.array((images[cat][ix]).cpu()).astype(np.uint8).squeeze()
            im = Image.fromarray(im)
            img.paste(im, (xoffset + cat * width, yoffset))

    return img


def torch_show_all_params(model, rank=0):
    params = list(model.parameters())
    k = 0
    for i in params:
        l = 1
        for j in i.size():
            l *= j
        k = k + l
    return k


def get_lr_schedule_with_steps(decay_type,
                               optimizer,
                               drop_steps=None,
                               gamma=None,
                               total_steps=None):
    def lr_lambda(current_step):
        if decay_type == 'fix':
            return 1.0
        elif decay_type == 'linear':
            return 1.0 * (current_step / total_steps)
        elif decay_type == 'cos':
            return 1.0 * (math.cos(
                (current_step / total_steps) * math.pi) + 1) / 2
        elif decay_type == 'milestone':
            return 1.0 * math.pow(gamma, int(current_step / drop_steps))
        else:
            raise NotImplementedError

    return LambdaLR(optimizer, lr_lambda)


def to_cuda(meta, device):
    for k in meta:
        if type(meta[k]) is torch.Tensor:
            meta[k] = meta[k].to(device)
    return meta


class Progbar(object):
    """Displays a progress bar.

    Arguments:
        target: Total number of steps expected, None if unknown.
        width: Progress bar width on screen.
        verbose: Verbosity mode, 0 (silent), 1 (verbose), 2 (semi-verbose)
        stateful_metrics: Iterable of string names of metrics that
            should *not* be averaged over time. Metrics in this list
            will be displayed as-is. All others will be averaged
            by the progbar before display.
        interval: Minimum visual progress update interval (in seconds).
    """

    def __init__(self,
                 target,
                 max_iters=None,
                 width=25,
                 verbose=1,
                 interval=0.05,
                 stateful_metrics=None):
        self.target = target
        self.width = width
        self.verbose = verbose
        self.interval = interval
        self.max_iters = max_iters
        self.iters = 0
        if stateful_metrics:
            self.stateful_metrics = set(stateful_metrics)
        else:
            self.stateful_metrics = set()

        self._dynamic_display = ((hasattr(sys.stdout, 'isatty')
                                  and sys.stdout.isatty())
                                 or 'ipykernel' in sys.modules
                                 or 'posix' in sys.modules)
        self._total_width = 0
        self._seen_so_far = 0
        # We use a dict + list to avoid garbage collection
        # issues found in OrderedDict
        self._values = {}
        self._values_order = []
        self._start = time.time()
        self._last_update = 0

    def update(self, current, values=None):
        """Updates the progress bar.

        Arguments:
            current: Index of current step.
            values: List of tuples:
                `(name, value_for_last_step)`.
                If `name` is in `stateful_metrics`,
                `value_for_last_step` will be displayed as-is.
                Else, an average of the metric over time will be displayed.
        """
        values = values or []
        for k, v in values:
            if k not in self._values_order:
                self._values_order.append(k)
            if k not in self.stateful_metrics:
                if k not in self._values:
                    self._values[k] = [
                        v * (current - self._seen_so_far),
                        current - self._seen_so_far
                    ]
                else:
                    self._values[k][0] += v * (current - self._seen_so_far)
                    self._values[k][1] += (current - self._seen_so_far)
            else:
                self._values[k] = v
        self._seen_so_far = current

        now = time.time()
        info = ' - %.0fs' % (now - self._start)
        if self.verbose == 1:
            if (now - self._last_update < self.interval
                    and self.target is not None and current < self.target):
                return

            prev_total_width = self._total_width
            if self._dynamic_display:
                sys.stdout.write('\b' * prev_total_width)
                sys.stdout.write('\r')
            else:
                sys.stdout.write('\n')

            if self.target is not None:
                numdigits = int(np.floor(np.log10(self.target))) + 1
                barstr = '%%%dd/%d [' % (numdigits, self.target)
                bar = barstr % current
                prog = float(current) / self.target
                prog_width = int(self.width * prog)
                if prog_width > 0:
                    bar += ('=' * (prog_width - 1))
                    if current < self.target:
                        bar += '>'
                    else:
                        bar += '='
                bar += ('.' * (self.width - prog_width))
                bar += ']'
            else:
                bar = '%7d/Unknown' % current

            self._total_width = len(bar)
            sys.stdout.write(bar)

            if current:
                time_per_unit = (now - self._start) / current
            else:
                time_per_unit = 0
            # if self.target is not None and current < self.target:
            if self.max_iters is None or self.iters < self.max_iters:
                eta = time_per_unit * (self.target - current)
                if eta > 3600:
                    eta_format = '%d:%02d:%02d' % (eta // 3600,
                                                   (eta % 3600) // 60,
                                                   eta % 60)
                elif eta > 60:
                    eta_format = '%d:%02d' % (eta // 60, eta % 60)
                else:
                    eta_format = '%ds' % eta

                info = ' - ETA: %s' % eta_format
            else:
                if time_per_unit >= 1:
                    info += ' %.0fs/step' % time_per_unit
                elif time_per_unit >= 1e-3:
                    info += ' %.0fms/step' % (time_per_unit * 1e3)
                else:
                    info += ' %.0fus/step' % (time_per_unit * 1e6)

            for k in self._values_order:
                info += ' - %s:' % k
                if isinstance(self._values[k], list):
                    avg = np.mean(self._values[k][0] /
                                  max(1, self._values[k][1]))
                    if abs(avg) > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                else:
                    info += ' %s' % self._values[k]

            self._total_width += len(info)
            if prev_total_width > self._total_width:
                info += (' ' * (prev_total_width - self._total_width))

            if self.target is not None and current >= self.target:
                info += '\n'

            sys.stdout.write(info)
            sys.stdout.flush()

        elif self.verbose == 2:
            if self.target is None or current >= self.target:
                for k in self._values_order:
                    info += ' - %s:' % k
                    avg = np.mean(self._values[k][0] /
                                  max(1, self._values[k][1]))
                    if avg > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                info += '\n'

                sys.stdout.write(info)
                sys.stdout.flush()

        self._last_update = now

    def add(self, n, values=None):
        self.iters += 1
        self.update(self._seen_so_far + n, values)


def postprocess(img, norm=False, simple_norm=False):
    # [-1, 1] => [0, 255]
    if img.shape[2] < 256:
        img = F.interpolate(img, (256, 256))
    if norm:  # to [-1~1]
        img = (img - img.min()) / (img.max() - img.min() + 1e-7)  # [0~1]
        img = img * 2 - 1  # [-1~1]
    if simple_norm:
        img = img * 2 - 1
    img = (img + 1) / 2 * 255.0
    img = img.permute(0, 2, 3, 1)
    return img.int()


def to_tensor(img):
    img = Image.fromarray(img)
    img_t = FF.to_tensor(img).float()
    return img_t


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

def imsave(img, path):
    im = Image.fromarray(img.cpu().numpy().astype(np.uint8).squeeze())
    im.save(path)


# ##############################################
# ##############################################

if __name__ == '__main__':

    trials = 10
    for _ in range(trials):
        video_length = 10
        # The returned masks are either stationary (50%) or moving (50%)
        masks = create_random_shape_with_random_motion(
            video_length, imageHeight=240, imageWidth=432)

        for m in masks:
            cv2.imshow('mask', np.array(m))
            cv2.waitKey(500)

