import argparse
from sewar.full_ref import vifp
import cv2
import os
import glob
import logging

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-path', type=str, default='/mnt/SSD1/Benson/Thesis/results/0805_FTR_con/youtubevos', help='path to the reasults')

    args = parser.parse_args()

    data_dir = '/mnt/SSD1/Benson/Thesis/datasets/YouTubeVOS/test_all_frames/JPEGImages'
    output_dir = args.path
    vid_lst = sorted([f for f in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir,f))])

    ### logging
    logger = logging.getLogger()
    file_handler = logging.FileHandler(os.path.join(output_dir, 'vif.txt'))
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())

    total = 0
    frame_cnt = 0

    for vid in vid_lst:
        vid_total = 0
        gt_lst = sorted(glob.glob(os.path.join(data_dir, vid, '*.jpg')))
        out_lst = sorted(glob.glob(os.path.join(output_dir, vid, '*.jpg')))
        if len(out_lst) == 0:
            out_lst = sorted(glob.glob(os.path.join(output_dir, vid, '*.png')))
        
        if len(gt_lst) != len(out_lst):
            print(f'Length mismatch: {vid} gt: {len(gt_lst)} out: {len(out_lst)}')
            continue

        for i in range(len(gt_lst)):
            gt = cv2.imread(gt_lst[i])
            out = cv2.imread(out_lst[i])
            gt = cv2.resize(gt, (out.shape[1], out.shape[0]))

            val = vifp(gt, out)
            vid_total += val
        
        # print(f'{vid}: {vid_total / len(gt_lst)}')
        # print(output_dir, end='/')
        logger.info("%s, %f", vid, vid_total / len(gt_lst))
        total += vid_total
        frame_cnt += len(gt_lst)
    
    logger.info(f'\nTest case: {output_dir}')
    logger.info(f'Average VIF: {total / frame_cnt}')
        