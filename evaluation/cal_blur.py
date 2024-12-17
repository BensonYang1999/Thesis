import argparse
import cv2
import os
import logging
from multiprocessing import Pool
import pandas as pd
import numpy as np
import glob

def cal_blur(data_dir, output_dir, vid, logger, mask):
    total = 0
    out_lst = sorted(glob.glob(os.path.join(output_dir, vid, '*.jpg')))
    if len(out_lst) == 0:
        out_lst = sorted(glob.glob(os.path.join(output_dir, vid, '*.png')))

    if mask:
        mask_lst = sorted(glob.glob(os.path.join(data_dir.replace('JPEGImages_432_240', 'test_masks'), vid, '*.png')))

    for i in range(len(out_lst)):
        out = cv2.imread(out_lst[i])

        if mask:
            mask_ = cv2.imread(mask_lst[i], cv2.IMREAD_GRAYSCALE) # type: ignore
            mask_ = np.repeat(mask_[:, :, np.newaxis], 3, axis=2)
            assert mask_.shape == out.shape
            out = cv2.bitwise_and(out, mask_)

        gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()

        total += variance
    
    logger.info("%s, %f, %d", vid, total / len(out_lst), len(out_lst))

    return vid, total, len(out_lst)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset', type=str, default='youtubevos', help='dataset name')
    parser.add_argument('--path', type=str, default='/mnt/SSD1/Benson/Thesis/results/0805_FTR_con/youtubevos', help='path to the reasults')
    parser.add_argument('--mask', action='store_true')

    args = parser.parse_args()

    if 'davis' in args.path:
        data_dir = '/mnt/SSD1/Benson/datasets/DAVIS/JPEGImages/JPEGImages_432_240'
    elif 'youtube' in args.path:
        data_dir = '/mnt/SSD1/Benson/datasets/YouTubeVOS/test_all_frames/JPEGImages_432_240'
    else:
        raise ValueError('Invalid dataset name')
    
    mask = False
    if args.mask:
        mask = True
    
    output_dir = args.path
    vid_lst = sorted([f for f in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir,f))])
    print(f'Number of videos: {len(vid_lst)}')
    # print(vid_lst)

    ### logging
    logger = logging.getLogger()
    if mask:
        logging_path = os.path.join(output_dir, 'blur_mask.txt')
    else:
        logging_path = os.path.join(output_dir, 'blur.txt')
    file_handler = logging.FileHandler(logging_path)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())

    table = []
    blur_total, frame_cnt = 0, 0

    pool = Pool()
    arg = [(data_dir, output_dir, vid, logger, mask) for vid in vid_lst]
    results = pool.starmap(cal_blur, arg)

    for res in results:
        vid_name = res[0]
        blur_total += res[1]
        frame_cnt += res[2]

        table.append([res[0], res[1]/res[2], res[2]])
    
    print(f'Test case: {output_dir}')
    logger.info(f'Average blur: {blur_total / frame_cnt}')

    df = pd.DataFrame(table, columns=['video', 'blur', 'frames'])
    df = df.sort_values('video')

    if mask:
        df.to_csv(os.path.join(output_dir, 'blur_mask.csv'), index=False)
    else:
        df.to_csv(os.path.join(output_dir, 'blur.csv'), index=False)
