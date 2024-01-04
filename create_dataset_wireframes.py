from src.lsm_hawp.lsm_hawp_model import LSM_HAWP
from glob import glob
import torch
import os
import argparse
import csv
from tqdm import tqdm
import time

"""
Two datasets should be processed:
1. DAVIS
2. YoutubeVOS
Need these folders:
1. wireframes (./(cases)/(frames).pkl)
2. visulize wireframes (.jpg) 512*512
3. visulize edges (.jpg) 512*512
"""

ckpt_path = "ckpt/best_lsm_hawp.pth"
# dataset_roots = {'DAVIS': './datasets/DAVIS/JPEGImages/Full-Resolution', 'YouTubeVOS': './datasets/YouTubeVOS/train_all_frames/JPEGImages'}
# wire_pkl_roots = {'DAVIS': './datasets/DAVIS/JPEGImages/Full-Resolution_wireframes_pkl', 'YouTubeVOS': './datasets/YouTubeVOS/train_all_frames/wireframes_pkl'}
# wire_roots = {'DAVIS': './datasets/DAVIS/JPEGImages/Full-Resolution_wireframes', 'YouTubeVOS': './datasets/YouTubeVOS/train_all_frames/wireframes'}
# edge_roots = {'DAVIS': './datasets/DAVIS/JPEGImages/Full-Resolution_edges', 'YouTubeVOS': './datasets/YouTubeVOS/train_all_frames/edges'}
# dataset_roots = {'YouTubeVOS': './datasets/YouTubeVOS/test_all_frames/JPEGImages'}
# wire_pkl_roots = {'YouTubeVOS': './datasets/YouTubeVOS/test_all_frames/wireframes_pkl'}
# wire_roots = {'YouTubeVOS': './datasets/YouTubeVOS/test_all_frames/wireframes'}
# edge_roots = {'YouTubeVOS': './datasets/YouTubeVOS/test_all_frames/edges'}
# dataset_roots = {'YouTubeVOS': './datasets/YouTubeVOS/train_all_frames/JPEGImages'}
# wire_pkl_roots = {'YouTubeVOS': './datasets/YouTubeVOS/train_all_frames/wireframes_pkl'}
# wire_roots = {'YouTubeVOS': './datasets/YouTubeVOS/train_all_frames/wireframes'}
# edge_roots = {'YouTubeVOS': './datasets/YouTubeVOS/train_all_frames/edges'}
dataset_roots = {'YouTubeVOS': './datasets/YouTubeVOS/test_all_frames/JPEGImages'}
wire_pkl_roots = {'YouTubeVOS': './datasets/YouTubeVOS/test_all_frames/wireframes_pkl'}
wire_roots = {'YouTubeVOS': './datasets/YouTubeVOS/test_all_frames/wireframes'}
edge_roots = {'YouTubeVOS': './datasets/YouTubeVOS/test_all_frames/edges'}

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


if __name__ == '__main__':

    model = LSM_HAWP(threshold=0.8, size=512)
    model.lsm_hawp.load_state_dict(torch.load(ckpt_path)['model'])

    time_costs = {}

    for dataset, root in dataset_roots.items():
        info_csv = os.path.join("./datasets/", dataset, "valid_all_frames_video_wireCounts.csv")
        if os.path.exists(info_csv):
            os.remove(info_csv)

        start_t = time.time()
        video_list = glob(root+"/*")
        for video_path in tqdm(video_list, desc=f"Processing [{dataset}]..."):
            video_name = video_path.split("/")[-1]
            input_path = video_path
            output_wire_pkl_path = os.path.join(wire_pkl_roots[dataset], video_name)
            output_wire_path = os.path.join(wire_roots[dataset], video_name)
            output_edge_path = os.path.join(edge_roots[dataset], video_name)
            os.makedirs(output_wire_pkl_path, exist_ok=True)
            os.makedirs(output_wire_path, exist_ok=True)
            os.makedirs(output_edge_path, exist_ok=True)

            img_paths = glob(input_path + '/*')
            model.wireframe_detect_visualize(img_paths, output_wire_pkl_path, output_wire_path, output_edge_path, size=512)

            # with open(info_csv, 'a', newline='') as csvfile:
            #     writer = csv.writer(csvfile)
            #     count.insert(0, video_name)
            #     writer.writerow(count)

        end_t = time.time()
        time_costs[dataset] = round((end_t-start_t), 4)

    print(f"[Time costs]:\n")
    for dataset, time in time_costs.items():
        print(f"{dataset} -> {time} (sec.)")