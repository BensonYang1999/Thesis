import argparse
import lpips
import torch
import torchvision.transforms as transforms
import cv2
import os
import glob
import logging
from multiprocessing import Pool

def cal_lpips(data_dir, output_dir, vid, logger):
    vid_total = 0
    gt_lst = sorted(glob.glob(os.path.join(data_dir, vid, '*.jpg')))
    out_lst = sorted(glob.glob(os.path.join(output_dir, vid, '*.jpg')))
    if len(out_lst) == 0:
        out_lst = sorted(glob.glob(os.path.join(output_dir, vid, '*.png')))
    
    if len(gt_lst) != len(out_lst):
        print(f'Length mismatch: {vid} gt: {len(gt_lst)} out: {len(out_lst)}')
        return 0, 0

    for i in range(len(gt_lst)):
        gt = cv2.imread(gt_lst[i]) / 255 # type: ignore
        out = cv2.imread(out_lst[i]) / 255 # type: ignore
        gt = cv2.resize(gt, (out.shape[1], out.shape[0]))
        
        gt = transf(gt).to(torch.float32)
        out = transf(out).to(torch.float32)

        val = lpips_fn(gt, out).item()
        vid_total += val
    
    logger.info("%s, %f, %d, %f", vid, vid_total / len(gt_lst), len(gt_lst), vid_total)

    return vid_total, len(gt_lst)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='youtubevos', help='dataset name')
    parser.add_argument('--path', type=str, default='/mnt/SSD1/Benson/Thesis/results/0805_FTR_con/youtubevos', help='path to the reasults')

    args = parser.parse_args()

    lpips_fn = lpips.LPIPS(net='vgg')
    transf = transforms.ToTensor()

    if args.dataset == 'youtubevos':
        data_dir = '/mnt/SSD1/Benson/datasets/YouTubeVOS/test_all_frames/JPEGImages_432_240'
    elif args.dataset == 'davis':
        data_dir = '/mnt/SSD1/Benson/datasets/DAVIS/JPEGImages/JPEGImages_432_240'
    else:
        raise ValueError('Invalid dataset name')
    output_dir = args.path
    vid_lst = sorted([f for f in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir,f))])

    ### logging
    logger = logging.getLogger()
    file_handler = logging.FileHandler(os.path.join(output_dir, 'lpips.txt'))
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())

    pool = Pool()
    arg = [(data_dir, output_dir, vid, logger) for vid in vid_lst]
    results = pool.starmap(cal_lpips, arg)

    total = 0
    cnt = 0
    for (val, num) in results:
        total += val
        cnt += num
    print(f'Test case: {output_dir}')
    logger.info(f'Average LPIPS: {total / cnt}\n')
        