from os import walk
from os.path import join
import os
from PIL import Image

img_folder = "./datasets/DAVIS_small/test_set_1/line_25percent/JPEGImages"

width, height = 512, 512

im = Image.new('L', (width, height))

for root, dirs, files in walk(img_folder):
    video_name = root.split("/")[-1]

    files.sort()
    for f in files:
        fullpath = join(root, f)
        if ".jpg" in fullpath or ".png" in fullpath:
            wire_fullpath = fullpath.replace("JPEGImages", "wireframes").replace("jpg", "png")
            # wire_fullpath = fullpath.replace("Full-Resolution", "Full-Resolution_wireframes").replace("jpg", "png")
            if not os.path.isfile(wire_fullpath):
                print(f"file: {wire_fullpath} not exist!!")
                im.save(wire_fullpath)