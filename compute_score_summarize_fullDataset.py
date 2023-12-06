from skimage.metrics import structural_similarity as measure_ssim
from skimage.metrics import peak_signal_noise_ratio as measure_psnr
from src.inpainting_metrics import get_fid_score, get_i3d_activations
import tqdm
import os
import cv2
import numpy as np
from torchmetrics.image.fid import FrechetInceptionDistance
import torch
from PIL import Image
import itertools
import argparse
import lpips

from src.utils import read_frame_from_videos, read_mask, compute_vfid, compute_lpips

# from sewar.full_ref import uqi, mse, rmse, scc, rase, sam, msssim, vifp, psnrb
from sewar.full_ref import uqi, mse, vifp


def compute_fid(gt, pred):
    #  conver gt, pred to torch tensor
    gt = torch.from_numpy(gt).permute(2, 0, 1).unsqueeze(0).to(torch.uint8)
    pred = torch.from_numpy(pred).permute(2, 0, 1).unsqueeze(0).to(torch.uint8)
    # put the images to GPU
    gt = gt.cuda()
    pred = pred.cuda()
    fid_fn.update(gt, real=True)
    fid_fn.update(pred, real=False)
    fid_score = fid_fn.compute().item()
    return fid_score


# define a function to compute the desire metric for each video
def compute_metrics(pred_dir, evalOnlyMask=False, split="valid"):
    vname = pred_dir.split('/')[-1]
    gt_dir = os.path.join(f"./datasets/YouTubeVOS/{split}_all_frames/JPEGImages", vname)
    mask_dir = os.path.join(f"./datasets/YouTubeVOS/{split}_all_frames/mask_random", vname)
    # read all the images with no prefix
    gt_images = read_frame_from_videos(gt_dir)
    # read all the images with no prefix
    pred_images = read_frame_from_videos(pred_dir)
    # read all the masks with no prefix
    mask_images = read_mask(mask_dir)

    # compute the PSNR for each pair of images
    psnr_scores, ssim_scores, lpips_scores = [], [], []
    vif_scores, uqi_scores, mse_scores = [], [], []
    for gt_image, pred_image, mask_image in zip(gt_images, pred_images, mask_images):
        gt_img = np.array(gt_image)
        pred_img = np.array(pred_image)
        mask_img = np.array(mask_image)

        if evalOnlyMask:
            mask_img = mask_img.astype(np.bool)

            # Apply mask to each channel
            pred_img = pred_img * np.stack([mask_img, mask_img, mask_img], axis=-1)
            gt_img = gt_img * np.stack([mask_img, mask_img, mask_img], axis=-1)

        psnr_score = measure_psnr(gt_img, pred_img, data_range=255)
        ssim_score = measure_ssim(gt_img, pred_img, data_range=255, multichannel=True, win_size=65)
        lpips_score = compute_lpips(gt_img, pred_img)
        vif_score = vifp(gt_img, pred_img)
        uqi_score = uqi(gt_img, pred_img)
        mse_score = mse(gt_img, pred_img)
        # fid_score = compute_fid(gt_img, pred_img)

        psnr_scores.append(psnr_score)
        ssim_scores.append(ssim_score)
        lpips_scores.append(lpips_score)
        vif_scores.append(vif_score)
        uqi_scores.append(uqi_score)
        mse_scores.append(mse_score)
        # fid_scores.append(fid_score)

    return psnr_scores, ssim_scores, lpips_scores, vif_scores, uqi_scores, mse_scores


def write_scores(video, csv_path, scores):
    with open(csv_path, "a") as f:
        f.write(video + ",")
        for score in scores:
            f.write(str(score) + ",")
        f.write("\n")

# process all videos and save the results
def process_videos(input_dir, evalOnlyMask=False, split="valid"):
    # get all the subfolders in the input_dir
    subfolders = [f.path for f in os.scandir(input_dir) if f.is_dir()]

    # loop through each subfolder
    for subfolder in tqdm.tqdm(subfolders):
        # compute the metrics for each subfolder
        psnr_scores, ssim_scores, lpips_scores, vif_scores, uqi_scores, mse_scores = compute_metrics(subfolder, evalOnlyMask)
        print(f"Video: {subfolder.split('/')[-1]}, PSNR: {np.mean(psnr_scores)}, SSIM: {np.mean(ssim_scores)}, LPIPS: {np.mean(lpips_scores)}, VIF: {np.mean(vif_scores)}, UQI: {np.mean(uqi_scores)}, MSE: {np.mean(mse_scores)}")

        # save the results
        # get the video id
        video_id = subfolder.split("/")[-1]
        psnr_csv_path = os.path.join(input_dir, f"PSNR_onlyMask_{evalOnlyMask}.csv")
        ssim_csv_path = os.path.join(input_dir, f"SSIM_onlyMask_{evalOnlyMask}.csv")
        lpips_csv_path = os.path.join(input_dir, f"LPIPS_onlyMask_{evalOnlyMask}.csv")
        vif_csv_path = os.path.join(input_dir, f"VIF_onlyMask_{evalOnlyMask}.csv")
        uqi_csv_path = os.path.join(input_dir, f"UQI_onlyMask_{evalOnlyMask}.csv")
        mse_csv_path = os.path.join(input_dir, f"MSE_onlyMask_{evalOnlyMask}.csv")

        # save PSNR scores
        write_scores(video_id, psnr_csv_path, psnr_scores)
        # save SSIM scores
        write_scores(video_id, ssim_csv_path, ssim_scores)
        # save LPIPS scores
        write_scores(video_id, lpips_csv_path, lpips_scores)
        # save VIF scores
        write_scores(video_id, vif_csv_path, vif_scores)
        # save UQI scores
        write_scores(video_id, uqi_csv_path, uqi_scores)
        # save MSE scores
        write_scores(video_id, mse_csv_path, mse_scores)

def compute_avg(input_dir):
    # each line refers to a video, and the first element is the video id, and the rest are the scores
    with open(input_dir, "r") as f:
        lines = f.readlines()
    # loop through each line
    avg_scores = []
    video_ids = []
    for line in lines:
        # split the line
        scores = line.split(",")
        # convert the scores to float
        video_ids.append(scores[0])
        scores = [float(score) for score in scores[1:-1]]
        # compute the average score
        avg_score = np.mean(scores)
        avg_score = round(avg_score, 3)
        avg_scores.append(avg_score)

    # compute the average score for all videos
    # avg_score = np.mean(avg_scores)
    # get round with 3
    return video_ids, avg_scores
        

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='./results/0530_ZITS_video_YoutubeVOS_max500k_mix458k_turn470k/2023-07-07_youtubevos')
    parser.add_argument('--onlyMask', action='store_true', default=False)
    parser.add_argument('--split', type=str, default="test")
    args = parser.parse_args()

    # show whethre only mask is evaluated
    print(f"Only Mask: {args.onlyMask}")

    psnr_dir = os.path.join(args.input_dir, f"PSNR_onlyMask_{args.onlyMask}.csv")
    ssim_dir = os.path.join(args.input_dir, f"SSIM_onlyMask_{args.onlyMask}.csv")
    lpips_dir = os.path.join(args.input_dir, f"LPIPS_onlyMask_{args.onlyMask}.csv")
    vif_dir = os.path.join(args.input_dir, f"VIF_onlyMask_{args.onlyMask}.csv")
    uqi_dir = os.path.join(args.input_dir, f"UQI_onlyMask_{args.onlyMask}.csv")
    mse_dir = os.path.join(args.input_dir, f"MSE_onlyMask_{args.onlyMask}.csv")

    # if os.path.exists(psnr_dir):
    #     os.remove(psnr_dir)

    # if os.path.exists(ssim_dir):
    #     os.remove(ssim_dir)
    
    # if os.path.exists(lpips_dir):
    #     os.remove(lpips_dir)

    # if os.path.exists(vif_dir):
    #     os.remove(vif_dir)

    # if os.path.exists(uqi_dir):
    #     os.remove(uqi_dir)

    # if os.path.exists(mse_dir):
    #     os.remove(mse_dir)

    # # process all videos
    # lpips_loss_fn = lpips.LPIPS(net='vgg')
    # # put the model to GPU
    # lpips_loss_fn.cuda()

    # fid_fn = FrechetInceptionDistance(feature=64)
    # fid_fn.cuda()
    # process_videos(args.input_dir, evalOnlyMask=args.onlyMask, split=args.split)

    import pandas as pd
    # create a dataframe to store the results of different thresholds
    df = pd.DataFrame(columns=["video_id", "psnr", "ssim", "lpips", "vif", "uqi", "mse"])
    psnr_id, psnr_avg = compute_avg(psnr_dir)
    ssim_id, ssim_avg = compute_avg(ssim_dir)
    lpips_id, lpips_avg = compute_avg(lpips_dir)
    vif_id, vif_avg = compute_avg(vif_dir)
    uqi_id, uqi_avg = compute_avg(uqi_dir)
    mse_id, mse_avg = compute_avg(mse_dir)

    # check if the video ids are the same
    assert psnr_id == ssim_id == lpips_id == vif_id == uqi_id == mse_id
    # add the results to the dataframe
    df["video_id"] = psnr_id
    df["psnr"] = psnr_avg
    df["ssim"] = ssim_avg
    df["lpips"] = lpips_avg
    df["vif"] = vif_avg
    df["uqi"] = uqi_avg
    df["mse"] = mse_avg

    # save the dataframe
    df.to_csv(os.path.join(args.input_dir, f"metrics_onlyMask_{args.onlyMask}.csv"), index=False)
    # save the dataframe description
    df.describe().to_csv(os.path.join(args.input_dir, f"metrics_onlyMask_{args.onlyMask}_description.csv"))

    print(f"df description: \n{df.describe()}")
