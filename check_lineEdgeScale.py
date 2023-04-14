import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.util import random_noise
from skimage import feature
from skimage.color import rgb2gray
from PIL import Image as im
from PIL import Image
import torchvision.transforms as transforms
from src.Fuseformer.utils import Stack, ToTorchFormatTensor

import cv2
import os
from torch import bincount
import torch
import torchvision.transforms as T

edge_oath = "./datasets/YouTubeVOS/train_all_frames/edges/0a7a2514aa/00000.jpg"
line_oath = "./datasets/YouTubeVOS/train_all_frames/wireframes/0a7a2514aa/00000.png"

_to_tensors = transforms.Compose([
            ToTorchFormatTensor(), ])

transform = T.ToPILImage()

# minmaxscaler
def minmaxscaler(x):
    x = x - x.min()
    x = x / x.max()
    return x

def to0_1(x):
    x = torch.where(x > 0.0, 1.0, 0.0)
    return x

edge = Image.open(edge_oath).convert('L')
edge = edge.resize((432,240))
edge = _to_tensors(edge)
print(f"edge: {edge}")
print(f"edge max: {edge.max()}")
print(f"edge min: {edge.min()}")
print(f"edge mean: {edge.mean()}")
print(f"edge median: {edge.median()}")
print(f"edge bincount: {bincount(edge.long().flatten())}")

# edge_new = torch.where(_to_tensors(edge) > 0.0, 1.0, 0.0)
edge_new = minmaxscaler(edge)
# edge_new = to0_1(edge)
print(f"edge_new max: {edge_new.max()}")
print(f"edge_new min: {edge_new.min()}")
print(f"edge_new mean: {edge_new.mean()}")
print(f"edge_new median: {edge_new.median()}")
print(f"edge_new bincount: {bincount(edge_new.long().flatten())}")
print(f"edge_new type: {type(edge_new)}")
# save the edge_new by the TO_PIL_IMAGE

line = Image.open(line_oath).convert('L')
line = line.resize((432,240))
line = _to_tensors(line)
print(f"line max: {line.max()}")
print(f"line min: {line.min()}")
print(f"line mean: {line.mean()}")
print(f"line median: {line.median()}")
print(f"line bincount: {bincount(line.long().flatten())}")
line_new = minmaxscaler(line)
# line_new = to0_1(line)
print(f"line_new max: {line_new.max()}")
print(f"line_new min: {line_new.min()}")
print(f"line_new mean: {line_new.mean()}")
print(f"line_new median: {line_new.median()}")
print(f"line_new bincount: {bincount(line_new.long().flatten())}")
line_new = transform(line_new)
edge_new = transform(edge_new)
# edge_new.save("01_edge_00000.jpg")
# line_new.save("01_line_00000.jpg")

line = transform(line)
edge = transform(edge)
edge.save("origin_edge_00000.jpg")
line.save("origin_line_00000.jpg")