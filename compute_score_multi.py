from skimage.metrics import structural_similarity as measure_ssim
from skimage.metrics import peak_signal_noise_ratio as measure_psnr
from src.inpainting_metrics import get_fid_score, get_i3d_activations
import tqdm
import os
import cv2
import numpy as np
import torch
from PIL import Image
import argparse
import lpips
from sewar.full_ref import vifp
from torchvision import transforms
from src.utils import Stack, ToTorchFormatTensor
import concurrent.futures  # Import for parallel processing
import pandas as pd
import itertools

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, default="./results")
parser.add_argument('--date', type=str, default="2023-07-13")
parser.add_argument('--onlyMask', action='store_true', default=False)
parser.add_argument('--split', type=str, default="test")
parser.add_argument('--cuda', action='store_true', default=False)
args = parser.parse_args()

lpips_loss_fn = lpips.LPIPS(net='vgg')
if args.cuda:
    lpips_loss_fn.cuda()

_to_tensors = transforms.Compose([
    Stack(),
    ToTorchFormatTensor()])

def read_frame_from_videos(vname , prefix:str):
    lst = os.listdir(vname)
    lst.sort()
    fr_lst = [vname+'/'+name for name in lst if name.startswith(prefix)]
    frames = []
    for fr in fr_lst:
        image = cv2.imread(fr)
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # resize image to 432*240
        image = image.resize((432, 240))
        frames.append(image)
    return frames 

def read_mask_from_video(vname , prefix:str):
    lst = os.listdir(vname)
    lst.sort()
    mk_lst = [vname+'/'+name for name in lst if name.startswith(prefix)]
    masks = []
    for mk in mk_lst:
        image = cv2.imread(mk, cv2.IMREAD_GRAYSCALE)
        image = Image.fromarray(image)
        # resize image to 432*240
        image = image.resize((432, 240))
        # convert to boolean
        image = np.array(image) > 0
        masks.append(image)
    return masks

def compute_lpips(gt, pred):
    # conver gt, pred to torch tensor
    gt = torch.from_numpy(gt).permute(2, 0, 1).unsqueeze(0).to(torch.uint8)
    pred = torch.from_numpy(pred).permute(2, 0, 1).unsqueeze(0).to(torch.uint8)
    # put the images to GPU
    if args.cuda:
        gt = gt.cuda()
        pred = pred.cuda()

    lpips_score = lpips_loss_fn.forward(gt, pred).item()
    return lpips_score

def compute_metrics(pred_dir, evalOnlyMask=False, split="test"):
    vname = pred_dir.split('/')[-1]
    gt_dir = os.path.join(f"./datasets/YouTubeVOS/{split}_all_frames/JPEGImages", vname)
    mask_dir = os.path.join(f"./datasets/YouTubeVOS/{split}_all_frames/mask_random", vname)
    # read all the images with no prefix
    gt_images = read_frame_from_videos(gt_dir, "")
    # read all the images with no prefix
    pred_images = read_frame_from_videos(pred_dir, "")
    # read all the masks with no prefix
    mask_images = read_mask_from_video(mask_dir, "")

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
        ssim_score = measure_ssim(gt_img, pred_img, data_range=255, multichannel=True)
        # ssim_score = measure_ssim(gt_img, pred_img, data_range=255, multichannel=True, win_size=65)
        lpips_score = compute_lpips(gt_img, pred_img)
        vif_score = vifp(gt_img, pred_img)
        psnr_scores.append(psnr_score)
        ssim_scores.append(ssim_score)
        lpips_scores.append(lpips_score)
        vif_scores.append(vif_score)

    # vfid computation
    imgs = _to_tensors(pred_images).unsqueeze(0).cuda() if args.cuda else _to_tensors(pred_images).unsqueeze(0)
    gts = _to_tensors(gt_images).unsqueeze(0).cuda() if args.cuda else _to_tensors(gt_images).unsqueeze(0)
    img_i3d_feature = get_i3d_activations(imgs).cpu().numpy().flatten()
    gt_i3d_feature = get_i3d_activations(gts).cpu().numpy().flatten()

    return psnr_scores, ssim_scores, lpips_scores, vif_scores, [img_i3d_feature, gt_i3d_feature]

def write_scores(video, csv_path, scores):
    with open(csv_path, "a") as f:
        f.write(video + ",")
        for score in scores:
            f.write(str(score) + ",")
        f.write("\n")

def process_videos(input_dir, evalOnlyMask=False, split="test"):
    # get all the subfolders in the input_dir
    subfolders = [f.path for f in os.scandir(input_dir) if f.is_dir()]

    imgs_i3d_feature_all, gts_i3d_feature_all = [], []
    # loop through each subfolder
    for subfolder in tqdm.tqdm(subfolders):
        # compute the metrics for each subfolder
        psnr_scores, ssim_scores, lpips_scores, vif_scores, [imgs_i3d_feature, gts_i3d_feature] = compute_metrics(subfolder, evalOnlyMask, split=split)
        print(f"Video: {subfolder.split('/')[-1]}, PSNR: {np.mean(psnr_scores)}, SSIM: {np.mean(ssim_scores)}, LPIPS: {np.mean(lpips_scores)}, VIF: {np.mean(vif_scores)}")

        # save the results
        # get the video id
        video_id = subfolder.split("/")[-1]
        psnr_csv_path = os.path.join(input_dir, f"PSNR_onlyMask_{evalOnlyMask}.csv")
        ssim_csv_path = os.path.join(input_dir, f"SSIM_onlyMask_{evalOnlyMask}.csv")
        lpips_csv_path = os.path.join(input_dir, f"LPIPS_onlyMask_{evalOnlyMask}.csv")
        vif_csv_path = os.path.join(input_dir, f"VIF_onlyMask_{evalOnlyMask}.csv")

        # save PSNR scores
        write_scores(video_id, psnr_csv_path, psnr_scores)
        # save SSIM scores
        write_scores(video_id, ssim_csv_path, ssim_scores)
        # save LPIPS scores
        write_scores(video_id, lpips_csv_path, lpips_scores)
        # save VIF scores
        write_scores(video_id, vif_csv_path, vif_scores)
        imgs_i3d_feature_all.append(imgs_i3d_feature)
        gts_i3d_feature_all.append(gts_i3d_feature)
        
    return imgs_i3d_feature_all, gts_i3d_feature_all

if __name__ == "__main__":
    # ... (same as before)
    print(f"Only Mask: {args.onlyMask}")

    # Process all videos using parallel processing
    with concurrent.futures.ThreadPoolExecutor() as executor:
        video_dirs = [f.path for f in os.scandir(args.root) if f.is_dir()]
        results = list(executor.map(process_videos, video_dirs, itertools.repeat(args)))

    imgs_i3d_feature_all, gts_i3d_feature_all = zip(*results)

    df = pd.DataFrame(columns=["condition", "psnr", "ssim", "lpips", "vif"])

    real_i3d_whole, output_i3d_whole = [], []
    for type in ["line", "edge"]:
        for th in [25, 50, 75, 100]:
            in_psnr_dir = os.path.join(args.root, f"{args.date}_{type}_{th}percent", f"PSNR_onlyMask_{args.onlyMask}.csv")
            in_ssim_dir = os.path.join(args.root, f"{args.date}_{type}_{th}percent", f"SSIM_onlyMask_{args.onlyMask}.csv")
            in_lpips_dir = os.path.join(args.root, f"{args.date}_{type}_{th}percent", f"LPIPS_onlyMask_{args.onlyMask}.csv")
            in_vif_dir = os.path.join(args.root, f"{args.date}_{type}_{th}percent", f"VIF_onlyMask_{args.onlyMask}.csv")

            if os.path.exists(in_psnr_dir):
                os.remove(in_psnr_dir)

            if os.path.exists(in_ssim_dir):
                os.remove(in_ssim_dir)

            if os.path.exists(in_lpips_dir):
                os.remove(in_lpips_dir)

            if os.path.exists(in_vif_dir):
                os.remove(in_vif_dir)

            imgs_i3d_feature_all, gts_i3d_feature_all = process_videos(os.path.join(args.root, f"{args.date}_{type}_{th}percent"), evalOnlyMask=args.onlyMask, split=args.split)
            real_i3d_whole.append(gts_i3d_feature_all)
            output_i3d_whole.append(imgs_i3d_feature_all)

            psnr_avg = compute_avg(in_psnr_dir)
            ssim_avg = compute_avg(in_ssim_dir)
            lpips_avg = compute_avg(in_lpips_dir)
            vif_avg = compute_avg(in_vif_dir)
            vfid = compute_fid(gts_i3d_feature_all, imgs_i3d_feature_all)

            df = df.append({"condition": f"{type}_{th}", "psnr": psnr_avg, "ssim": ssim_avg, "lpips": lpips_avg, "vfid": vfid, "vif": vif_avg}, ignore_index=True)

    final_vfid = get_fid_score(real_i3d_whole, output_i3d_whole)
    df = df.append({"condition": "Average", "psnr": df["psnr"].mean(), "ssim": df["ssim"].mean(), "lpips": df["lpips"].mean(), "vfid": final_vfid, "vif": df["vif"].mean()}, ignore_index=True)
    df.to_csv(os.path.join(args.root, f"metrics_summary_{args.date}_onlyMask_{args.onlyMask}.csv"), index=False)

    # Print the results
    print(df)
