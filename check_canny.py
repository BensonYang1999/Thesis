import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.util import random_noise
from skimage import feature
from skimage.color import rgb2gray
from PIL import Image as im

import cv2
import os

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

# image = cv2.imread(image_origin)
# image = cv2.cvtColor(image_origin, cv2.COLOR_BGR2GRAY) 

# image = ndi.gaussian_filter(image, 2)

# edge = feature.canny(image, sigma=2, mask=None).astype(np.float)
print(f"edge: {edges2}")
print(f"edge shape: {type(edges2)}")
# edge = cv2.normalize(edge, dst=None, alpha=0, beta=255,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
# print(f"edge: {edge}")
# cv2.imwrite(os.path.join(output_edge_path, img.split('/')[-1]), edge)
edges2 = np.where(edges2 > 0.0, 1.0, 0.0)
cv2.imwrite("canny_tmp.jpg", edges2*255)