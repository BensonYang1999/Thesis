import argparse
from sewar.full_ref import vifp
import cv2
import os
import glob
import logging
from multiprocessing import Pool

def cal_vif(data_dir, output_dir, vid, logger):
    vif_total = 0
    gt_lst = sorted(glob.glob(os.path.join(data_dir, vid, '*.jpg')))
    out_lst = sorted(glob.glob(os.path.join(output_dir, vid, '*.jpg')))
    if len(out_lst) == 0:
        out_lst = sorted(glob.glob(os.path.join(output_dir, vid, '*.png')))
    
    if len(gt_lst) != len(out_lst):
        print(f'Length mismatch: {vid} gt: {len(gt_lst)} out: {len(out_lst)}')
        return 0, 0

    for i in range(len(gt_lst)):
        gt = cv2.imread(gt_lst[i])
        out = cv2.imread(out_lst[i])
        # gt = cv2.resize(gt, (out.shape[1], out.shape[0]))

        val = vifp(gt, out)
        vif_total += val
    
    logger.info("%s, %f, %d, %f", vid, vif_total / len(gt_lst), len(gt_lst), vif_total)

    return vif_total, len(gt_lst)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='youtubevos', help='dataset name')
    parser.add_argument('--path', type=str, default='/mnt/SSD1/Benson/Thesis/results/0805_FTR_con/youtubevos', help='path to the reasults')

    args = parser.parse_args()

    if args.dataset == 'youtubevos':
        data_dir = '/mnt/SSD1/Benson/datasets/YouTubeVOS/test_all_frames/JPEGImages_432_240/'
    elif args.dataset == 'davis':
        data_dir = '/mnt/SSD1/Benson/datasets/DAVIS/JPEGImages/JPEGImages_432_240'
    else:
        raise ValueError('Invalid dataset name')
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

    pool = Pool()
    arg = [(data_dir, output_dir, vid, logger) for vid in vid_lst]
    results = pool.starmap(cal_vif, arg)

    # logger.info(f'\nTest case: {output_dir}')
    # logger.info(f'Average VIF: {total / frame_cnt}')

    total = 0
    cnt = 0
    for (val, num) in results:
        total += val
        cnt += num
    print(f'Test case: {output_dir}')
    logger.info(f'Average VIF: {total / cnt}\n')