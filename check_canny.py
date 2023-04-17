import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.util import random_noise
from skimage import feature
from skimage.color import rgb2gray
from PIL import Image as im
from PIL import Image

import cv2
import os
from skimage.feature import canny
import torchvision.transforms as transforms
from src.Fuseformer.utils import Stack, ToTorchFormatTensor, GroupRandomHorizontalFlip

# Generate noisy image of a square
# image = np.zeros((128, 128), dtype=float)
# image[32:-32, 32:-32] = 1
image_origin = cv2.imread("./test_imgs/00000.jpg")
# image = cv2.cvtColor(image_origin, cv2.COLOR_BGR2GRAY) 
image = rgb2gray(image_origin)

image_origin = cv2.cvtColor(image_origin, cv2.COLOR_BGR2RGB) 
# print(f"shape: {image.shape}")

image = ndi.gaussian_filter(image, 2)
# image = random_noise(image, mode='speckle', mean=0.1)

# Compute the Canny filter for two values of sigma
# edges1 = feature.canny(image)
# edges2 = feature.canny(image, sigma=2)
# edges3 = feature.canny(image, sigma=3)
# edges1 = feature.canny(image, sigma=1, mask=None).astype(np.float)
# edges2 = feature.canny(image, sigma=2, mask=None).astype(np.float)
# edges3 = feature.canny(image, sigma=3, mask=None).astype(np.float)
edges1 = feature.canny(image, sigma=1, mask=None)
edges2 = feature.canny(image, sigma=2, mask=None)
edges3 = feature.canny(image, sigma=3, mask=None)

print(f"edge: {edges2}")

# edges = [edges1, edges2, edges3]
# for edge in edges:
#     edge = im.fromarray(edge)
#     if edge.mode != 'L':
#         edge = edge.convert('L')
# display results
fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(30, 5))

ax[0].imshow(image_origin) # type: ignore
ax[0].set_title('RGB image', fontsize=10)

ax[1].imshow(image, cmap='gray')
ax[1].set_title('preprocessed image', fontsize=10)

ax[2].imshow(edges1, cmap='gray')
ax[2].set_title(r'Canny filter, $\sigma=1$', fontsize=10)

ax[3].imshow(edges2, cmap='gray')
ax[3].set_title(r'Canny filter, $\sigma=2$', fontsize=10)

ax[4].imshow(edges3, cmap='gray')
ax[4].set_title(r'Canny filter, $\sigma=3$', fontsize=10)

for a in ax:
    a.axis('off')

fig.tight_layout()
plt.savefig("canny.png")

def origin_edge(img_path):
    img_rgb = cv2.imread(img_path)
    edge = canny(rgb2gray(img_rgb), sigma=2, mask=None).astype(np.float)*255
    img_edge = im.fromarray(edge)
    if img_edge.mode != 'L':
        img_edge = img_edge.convert('L')
    img_edge.save('origin_edge.png')
    return edge

def new_edge(img_path):
    image = rgb2gray(cv2.imread(img_path))
    # edge = feature.canny(ndi.gaussian_filter(image, 2), sigma=2).astype(float)  # gaussian_filter
    edge = feature.canny(image, sigma=2).astype(float)  # canny_filter
    cv2.imwrite('new_edge.png', edge * 255)
    return edge

_to_tensors = transforms.Compose([
        Stack(),
        ToTorchFormatTensor(), ])

if __name__=="__main__":
    img_path = './datasets/YouTubeVOS/train_all_frames/JPEGImages/0a8c467cc3/00000.jpg'
    origin = origin_edge(img_path)
    new = new_edge(img_path)
    print(f"origin: {origin.shape}, new: {new.shape}")
    print(f"origin: {origin}, new: {new}")
    print(f"origin min: {origin.min()}, new: {new.min()}")
    print(f"origin max: {origin.max()}, new: {new.max()}")

    origin_edge = Image.open('origin_edge.png').convert('L')
    origin_edge = origin_edge.resize((432,240))
    origin_edge = _to_tensors([origin_edge])
    new_edge = Image.open('new_edge.png').convert('L')
    new_edge = new_edge.resize((432,240))
    new_edge = _to_tensors([new_edge])

    # 計算最大值和最小值
    max_value = new_edge.max()
    min_value = new_edge.min()

    # 最小值-最大值正規化
    normalized_image_tensor = (new_edge - min_value) / (max_value - min_value)
    print(f"After read")
    print(f"origin: {origin_edge.shape}, new: {new_edge.shape}")
    print(f"origin: {origin_edge}, new: {new_edge}")
    print(f"origin min: {origin_edge.min()}, new: {new_edge.min()}")
    print(f"origin max: {origin_edge.max()}, new: {new_edge.max()}")
    print(f"normalized_image_tensor: {normalized_image_tensor.shape}")
    print(f"normalized_image_tensor min: {normalized_image_tensor.min()}")
    print(f"normalized_image_tensor max: {normalized_image_tensor.max()}")