from skimage.metrics import structural_similarity as measure_ssim
from skimage.metrics import peak_signal_noise_ratio as measure_psnr
from src.inpainting_metrics import get_fid_score as get_vfid_score
from src.inpainting_metrics import get_i3d_activations
import tqdm
import os
import cv2
import numpy as np
# from torchmetrics.image.fid import FrechetInceptionDistance
import torch
from PIL import Image
import itertools
import argparse
import lpips
from sewar.full_ref import uqi, mse, vifp
from torchvision import transforms
from src.utils import Stack, ToTorchFormatTensor
import pandas as pd
from pathlib import Path

from src.utils import read_frame_from_videos, read_mask
from src.inpainting_metrics import compute_vfid, compute_lpips

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, default="./results")
# parser.add_argument('--model', type=str, default="0710_ZITS_video_YoutubeVOS_max500k_mix458k_turn470k_frame-1_1_ReFFC_removed_last")
parser.add_argument('--date', type=str, default="")
parser.add_argument('--onlyMask', action='store_true', default=False)
parser.add_argument('--split', type=str, default="test")
parser.add_argument('--cuda', action='store_true', default=False)
parser.add_argument('--use_DAVIS', action='store_true', default=False)
args = parser.parse_args()

lpips_loss_fn = lpips.LPIPS(net='vgg')
# put the model to GPU
if args.cuda:
    lpips_loss_fn.cuda()

# fid_fn = FrechetInceptionDistance(feature=64)
# if args.cuda:
#     fid_fn.cuda()

_to_tensors = transforms.Compose([
    Stack(),
    ToTorchFormatTensor()])


import numpy as np
import os

def read_images_from_dir(directory, prefix=""):
    # Function to read images should be defined elsewhere
    # This is just a placeholder for actual implementation
    return read_frame_from_videos(directory, prefix)

def compute_score_per_frame(gt_img, pred_img, mask_img, eval_mask):
    if eval_mask:
        mask = mask_img.astype(np.bool)
        gt_img = gt_img * np.stack([mask] * 3, axis=-1)
        pred_img = pred_img * np.stack([mask] * 3, axis=-1)

    return {
        'psnr': measure_psnr(gt_img, pred_img, data_range=255),
        'ssim': measure_ssim(gt_img, pred_img, data_range=255, multichannel=True, win_size=65),
        'lpips': compute_lpips(gt_img, pred_img),
        'vif': vifp(gt_img, pred_img)
    }

def compute_feature_scores(pred_images, gt_images, use_cuda):
    tensors_option = {'unsqueeze': 0, 'cuda': use_cuda}
    imgs_tensor = _to_tensors(pred_images, **tensors_option)
    gts_tensor = _to_tensors(gt_images, **tensors_option)

    img_i3d_feature = get_i3d_activations(imgs_tensor).cpu().numpy().flatten()
    gt_i3d_feature = get_i3d_activations(gts_tensor).cpu().numpy().flatten()
    
    return img_i3d_feature, gt_i3d_feature

def get_directory_paths(pred_dir, use_davis, split):
    vname = os.path.basename(pred_dir)
    if use_davis:
        base_path = "./datasets/DAVIS/JPEGImages"
        return {
            'gt': os.path.join(base_path, "Full-Resolution", vname),
            'mask': os.path.join(base_path, "mask_random", vname)
        }
    else:
        base_path = f"./datasets/YouTubeVOS/{split}_all_frames/JPEGImages"
        return {
            'gt': os.path.join(base_path, vname),
            'mask': os.path.join(base_path, "mask_random", vname)
        }

def compute_video_quality_metrics(pred_dir, use_davis=False, eval_only_mask=False, split="test", use_cuda=False):
    paths = get_directory_paths(pred_dir, use_davis, split)
    gt_images = read_images_from_dir(paths['gt'])
    pred_images = read_images_from_dir(pred_dir)
    mask_images = read_images_from_dir(paths['mask'])

    scores = {
        'psnr': [],
        'ssim': [],
        'lpips': [],
        'vif': []
    }

    for gt_img, pred_img, mask_img in zip(gt_images, pred_images, mask_images):
        frame_scores = compute_score_per_frame(np.array(gt_img), np.array(pred_img), np.array(mask_img), eval_only_mask)
        for key in scores:
            scores[key].append(frame_scores[key])

    feature_scores = compute_feature_scores(pred_images, gt_images, use_cuda)

    return scores, feature_scores

def write_scores(video, csv_path, scores):
    with open(csv_path, "a") as f:
        f.write(video + ",")
        for score in scores:
            f.write(str(score) + ",")
        f.write("\n")

def process_videos(input_dir, eval_only_mask=False, split="test"):
    subfolders = [f.path for f in os.scandir(input_dir) if f.is_dir()]

    # Initialize lists to collect i3d features for all videos
    imgs_i3d_feature_all, gts_i3d_feature_all = [], []
    
    for subfolder in tqdm.tqdm(subfolders):
        metrics = process_videos_and_compute_metrics(subfolder, eval_only_mask, split)
        video_id = os.path.basename(subfolder)
        display_and_save_results(metrics, video_id, input_dir, eval_only_mask)
        
        imgs_i3d_feature, gts_i3d_feature = metrics[-1]
        imgs_i3d_feature_all.append(imgs_i3d_feature)
        gts_i3d_feature_all.append(gts_i3d_feature)

    return imgs_i3d_feature_all, gts_i3d_feature_all

def display_and_save_results(metrics, video_id, input_dir, eval_only_mask):
    psnr_scores, ssim_scores, lpips_scores, vif_scores = metrics[:4]
    print_result(video_id, psnr_scores, ssim_scores, lpips_scores, vif_scores)
    
    for score_name, scores in zip(('PSNR', 'SSIM', 'LPIPS', 'VIF'), metrics[:4]):
        csv_path = os.path.join(input_dir, f"{score_name}_onlyMask_{eval_only_mask}.csv")
        save_scores(video_id, csv_path, scores)

def print_result(video_id, psnr_scores, ssim_scores, lpips_scores, vif_scores):
    print(f"Video: {video_id}, "
          f"PSNR: {np.mean(psnr_scores):.4f}, "
          f"SSIM: {np.mean(ssim_scores):.4f}, "
          f"LPIPS: {np.mean(lpips_scores):.4f}, "
          f"VIF: {np.mean(vif_scores):.4f}")

def save_scores(video_id, csv_path, scores):
    with open(csv_path, 'a') as file:
        file.write(f"{video_id},{','.join(map(str, scores))}\n")

def compute_avg(score_file_path):
    try:
        with open(score_file_path, "r") as f:
            lines = f.readlines()

        avg_scores = [np.mean(list(map(float, line.split(",")[1:]))) for line in lines]
        overall_avg_score = np.mean(avg_scores)
        return round(overall_avg_score, 4)
    
    except FileNotFoundError:
        print(f"File not found: {score_file_path}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
        

def remove_file_if_exists(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)

def process_videos_and_compute_metrics(input_dir, eval_only_mask, split):
    imgs_i3d_feature_all, gts_i3d_feature_all = process_videos(input_dir, eval_only_mask=eval_only_mask, split=split)
    metrics = {
        'psnr': compute_avg(f"{input_dir}/PSNR_onlyMask_{eval_only_mask}.csv"),
        'ssim': compute_avg(f"{input_dir}/SSIM_onlyMask_{eval_only_mask}.csv"),
        'lpips': compute_avg(f"{input_dir}/LPIPS_onlyMask_{eval_only_mask}.csv"),
        'vif': compute_avg(f"{input_dir}/VIF_onlyMask_{eval_only_mask}.csv"),
        'vfid': get_vfid_score(gts_i3d_feature_all, imgs_i3d_feature_all)
    }
    return imgs_i3d_feature_all, gts_i3d_feature_all, metrics

def append_to_dataframe(df, condition, metrics):
    return df.append(
        {"condition": condition, **metrics},
        ignore_index=True
    )

def main(args):
    print(f"Only Mask: {args.onlyMask}")

    results_df = pd.DataFrame(columns=["condition", "psnr", "ssim", "lpips", "vfid", "vif"])
    real_i3d_whole, output_i3d_whole = [], []

    for type in ["line", "edge"]:
        for th in [25, 50, 75, 100]:
            condition_dir = f"{args.date}_{type}_{th}percent" if args.date else f"{type}_{th}percent"
            condition_path = Path(args.root) / condition_dir

            # Remove existing metric files
            for metric in ["PSNR", "SSIM", "LPIPS", "VIF"]:
                remove_file_if_exists(condition_path / f"{metric}_onlyMask_{args.onlyMask}.csv")

            # Process videos and compute metrics
            imgs_i3d, gts_i3d, metrics = process_videos_and_compute_metrics(condition_path, args.onlyMask, args.split)

            real_i3d_whole += gts_i3d
            output_i3d_whole += imgs_i3d

            # Append to DataFrame
            results_df = append_to_dataframe(results_df, f"{type}_{th}", metrics)

    # Compute the final VFID score and append the average metrics
    final_vfid = get_vfid_score(real_i3d_whole, output_i3d_whole)
    avg_metrics = {
        "psnr": round(results_df["psnr"].mean(), 2),
        "ssim": round(results_df["ssim"].mean(), 3),
        "lpips": round(results_df["lpips"].mean(), 3),
        "vif": round(results_df["vif"].mean(), 3),
        "vfid": round(final_vfid, 3)
    }
    results_df = append_to_dataframe(results_df, "Average", avg_metrics)

    # Save to CSV
    summary_filename = f"metrics_summary_{args.date}_onlyMask_{args.onlyMask}.csv" if args.date else f"metrics_summary_onlyMask_{args.onlyMask}.csv"
    results_df.to_csv(Path(args.root) / summary_filename, index=False)

if __name__ == "__main__":
    # Assume args is already defined
    main(args)