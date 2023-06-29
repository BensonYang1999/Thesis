from src.lsm_hawp.lsm_hawp_model import LSM_HAWP
from glob import glob
import torch
import os
import argparse
import csv
from tqdm import tqdm
import time

ckpt_path = "ckpt/best_lsm_hawp.pth"
input_size = 512

parser = argparse.ArgumentParser()
parser.add_argument("-r", "--root", type=str, default='./datasets/YouTubeVOS/valid_all_frames/JPEGImages', required=False)
parser.add_argument("-t", "--threshold", type=int, default=50, required=False)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

if __name__=="__main__":
    model = LSM_HAWP(threshold=0.8, size=512)
    model.lsm_hawp.load_state_dict(torch.load(ckpt_path)['model'])

    # set the output csv file path with the name of the root folder
    csv_path = os.path.join(*args.root.split('/')[:-1], 'line_count.csv')
    count_larger_path = os.path.join(*args.root.split('/')[:-1], f'count_larger_than_{args.threshold}.txt')

    # get the list of all the images in the root folder
    video_list = glob(args.root+"/*")
    for video_path in tqdm(video_list, desc=f"Processing [{args.root}]..."):
        print(f"Processing [{video_path}]...")
        video_name = video_path.split("/")[-1]
        input_path = video_path

        img_paths = glob(input_path + '/*')
        counts = model.wireframe_count(img_paths)

        # with open(info_csv, 'a', newline='') as csvfile:
        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            tmp = counts.copy()
            tmp.insert(0, video_name)
            writer.writerow(tmp)

        # count the average of counts
        average = sum(counts)/len(counts)
        print(f"Average count of {video_name}: {average}")

        # count the number of images with counts larger than threshold save the path into a txt file
        if average > args.threshold:
            with open(count_larger_path, 'a') as f:
                f.write(input_path + '\n')
    