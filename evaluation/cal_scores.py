import argparse
import cv2
import os
import logging
from multiprocessing import Pool
import pandas as pd

# psnr
import numpy as np

# ssim
from skimage.metrics import structural_similarity as compare_ssim

# lpips
import lpips
import torch
import torchvision.transforms as transforms

# vif
from sewar.full_ref import vifp

# vfid
from PIL import Image
import sys
# sys.path.insert(0, '../')
sys.path.append('../')
from src.inpainting_metrics import *

def cal_psnr(img1, img2):
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))

to_tensors = transforms.Compose([
    Stack(),
    ToTorchFormatTensor()])

def cal_scores(data_dir, output_dir, vid, i3d_model, logger, mask):
    psnr_total = 0
    ssim_total = 0
    vif_total = 0
    lpips_total = 0
    gt_lst = sorted(glob(os.path.join(data_dir, vid, '*.jpg')))
    out_lst = sorted(glob(os.path.join(output_dir, vid, '*.jpg')))
    if len(out_lst) == 0:
        out_lst = sorted(glob(os.path.join(output_dir, vid, '*.png')))
    if mask:
        mask_lst = sorted(glob(os.path.join(data_dir.replace('JPEGImages_432_240', 'test_masks'), vid, '*.png')))
    
    if len(gt_lst) != len(out_lst):
        raise ValueError(f'Length mismatch: {vid}, gt: {len(gt_lst)}, out: {len(out_lst)}')

    frames, gts = [], []
    
    for i in range(len(gt_lst)):
        gt = cv2.imread(gt_lst[i])
        out = cv2.imread(out_lst[i])
        # gt = cv2.resize(gt, (out.shape[1], out.shape[0]))
        if mask:
            mask_ = cv2.imread(mask_lst[i], cv2.IMREAD_GRAYSCALE) # type: ignore
            mask_ = np.repeat(mask_[:, :, np.newaxis], 3, axis=2)
            assert mask_.shape == out.shape
            out = cv2.bitwise_and(out, mask_)
            gt = cv2.bitwise_and(gt, mask_)

        frames.append(Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB)))
        gts.append(Image.fromarray(cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)))

        # psnr
        psnr_total += cal_psnr(gt, out)

        # ssim
        ssim_total += compare_ssim(gt, out, data_range=255, channel_axis=2, win_size=65)
        
        # vif
        vif_total += vifp(gt, out)

        # lpips
        gt_1 = gt / 255 # type: ignore
        out_1 = out / 255 # type: ignore
        # gt = cv2.resize(gt, (out.shape[1], out.shape[0]))
        
        gt_1 = transf(gt_1).to(torch.float32)
        out_1 = transf(out_1).to(torch.float32)

        lpips_total += lpips_fn(gt_1, out_1).item()
    
    if len(frames) == 0:
        raise ValueError(f'No frames found for {vid}')
    frames = to_tensors(frames).unsqueeze(0) # type: ignore
    gts = to_tensors(gts).unsqueeze(0) # type: ignore
    out_acts = get_i3d_activations(frames, i3d_model).numpy().flatten()
    gt_acts = get_i3d_activations(gts, i3d_model).numpy().flatten()
    vifd = get_fid_score([out_acts], [gt_acts])

    logger.info("%s, %f, %f, %f, %f, %f, %d", vid, psnr_total / len(gt_lst), ssim_total / len(gt_lst), lpips_total / len(gt_lst), vif_total / len(gt_lst), vifd, len(gt_lst))

    return vid, psnr_total, ssim_total, lpips_total, vif_total, vifd, out_acts, gt_acts, len(gt_lst)

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

    # lpips
    lpips_fn = lpips.LPIPS(net='vgg')
    transf = transforms.ToTensor()

    # vfid
    i3d_model = init_i3d_model('../ckpt/i3d_rgb_imagenet.pt', 'cpu')

    ### logging
    logger = logging.getLogger()
    if mask:
        logging_path = os.path.join(output_dir, 'score_mask.txt')
    else:
        logging_path = os.path.join(output_dir, 'score.txt')
    file_handler = logging.FileHandler(logging_path)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())

    table = []
    psnr_total, ssim_total, lpips_total, vif_total, frame_cnt = 0, 0, 0, 0, 0
    out_act, gt_act = [], []

    pool = Pool()
    arg = [(data_dir, output_dir, vid, i3d_model, logger, mask) for vid in vid_lst]
    results = pool.starmap(cal_scores, arg)

    for res in results:
        vid_name = res[0]
        psnr_total += res[1]
        ssim_total += res[2]
        lpips_total += res[3]
        vif_total += res[4]
        frame_cnt += res[8]

        out_act.append(res[6])
        gt_act.append(res[7])

        table.append([res[0], res[1]/res[8], res[2]/res[8], res[3]/res[8], res[4]/res[8], res[5], res[8]])
    
    vfid_score = get_fid_score(out_act, gt_act)
    print(f'Test case: {output_dir}')
    logger.info(f'Average PSNR: {psnr_total / frame_cnt}')
    logger.info(f'Average SSIM: {ssim_total / frame_cnt}')
    logger.info(f'Average LPIPS: {lpips_total / frame_cnt}')
    logger.info(f'Average VIF: {vif_total / frame_cnt}')
    logger.info(f'Average VFID: {vfid_score}')

    df = pd.DataFrame(table, columns=['video', 'psnr', 'ssim', 'lpips', 'vif', 'vfid', 'frames'])
    df = df.sort_values('video')

    if mask:
        df.to_csv(os.path.join(output_dir, 'score_mask.csv'), index=False)
    else:
        df.to_csv(os.path.join(output_dir, 'score.csv'), index=False)
