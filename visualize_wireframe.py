import os
import skimage.draw
import numpy as np
import pickle
from PIL import Image as im
from skimage.feature import canny
import cv2
from skimage.color import rgb2gray

size = 1024

def to_int(x):
    return tuple(map(int, x))

img_rgb = cv2.imread("./my_testing/YoutubeVOS_case2/00025.jpg")
# img_rgb = cv2.imread("./test_imgs/img1.png")

# selected_img_name = "./my_testing/TSR_input/input512.jpg"
# line_name = "./my_testing/TSR_wire" + '/' + os.path.basename(selected_img_name).replace('.png', '.pkl').replace('.jpg', '.pkl')  # 從訓練的index知道目前的訓練image的名稱是什麼，而wireframe的檔名會與image相同但副檔名不同
wf = pickle.load(open("./wireframes/YoutubeVOS_case2/00025.pkl", 'rb'))  # 讀入對應的wireframe檔
# wf = pickle.load(open("./wireframes/img1.pkl", 'rb'))  # 讀入對應的wireframe檔
print(f"wf: {wf}")
lmap = np.zeros((size, size))
for i in range(len(wf['scores'])):  # 所有偵測到的線條
    if wf['scores'][i] > 0.8:  # 依序檢查，有超過threshold的線條才會拿來使用
        line = wf['lines'][i].copy()
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
img.save("test_wireframe.jpg")

edge = canny(rgb2gray(img_rgb), sigma=2, mask=None).astype(np.float)*255
print(f"edge: {type(edge)}")
img_edge = im.fromarray(edge)
if img_edge.mode != 'L':
    img_edge = img_edge.convert('L')
img_edge.save("test_edge.jpg")