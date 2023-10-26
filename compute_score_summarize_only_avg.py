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

def read_frame_from_videos(vname , prefix:str):
    lst = os.listdir(vname)
    lst.sort()
    fr_lst = [vname+'/'+name for name in lst if name.startswith(prefix) and name.endswith(".jpg")]
    frames = []
    # print(f"fr_lst: {fr_lst}") # test
    for fr in fr_lst:
        image = cv2.imread(fr)
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # resize image to 432*240
        image = image.resize((432, 240))
        frames.append(image)
    return frames, fr_lst[0].endswith(".png")

# define a function call read_mask_from_video which is similar to read_frame_from_videos but read the image with gray scale and resize to 432*240 and convert to boolean
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

def compute_fid(gt, pred):
    #  conver gt, pred to torch tensor
    gt = torch.from_numpy(gt).permute(2, 0, 1).unsqueeze(0).to(torch.uint8)
    pred = torch.from_numpy(pred).permute(2, 0, 1).unsqueeze(0).to(torch.uint8)
    # put the images to GPU
    if args.cuda:
        gt = gt.cuda()
        pred = pred.cuda()

    fid_fn.update(gt, real=True)
    fid_fn.update(pred, real=False)
    fid_score = fid_fn.compute().item()
    return fid_score


# define a function to compute the desire metric for each video
def compute_metrics(pred_dir, evalOnlyMask=False, split="test"):
    vname = pred_dir.split('/')[-1]
    if args.use_DAVIS:
        gt_dir = os.path.join(f"./datasets/DAVIS/JPEGImages/Full-Resolution", vname)
        mask_dir = os.path.join(f"./datasets/DAVIS/JPEGImages/mask_random", vname)
    else:
        gt_dir = os.path.join(f"./datasets/YouTubeVOS/{split}_all_frames/JPEGImages", vname)
        mask_dir = os.path.join(f"./datasets/YouTubeVOS/{split}_all_frames/mask_random", vname)
    # read all the images with no prefix
    gt_images, _ = read_frame_from_videos(gt_dir, "")
    # read all the images with no prefix
    pred_images, use_PNG = read_frame_from_videos(pred_dir, "")
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
        # if use_PNG:
        #     ssim_score = measure_ssim(gt_img, pred_img, data_range=255, multichannel=True)
        # else:
        #     ssim_score = measure_ssim(gt_img, pred_img, data_range=255, channel_axis=-1)
        ssim_score = measure_ssim(gt_img, pred_img, data_range=255, multichannel=True, win_size=65)
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

# process all videos and save the results
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

def compute_avg(input_dir):
    # each line refers to a video, and the first element is the video id, and the rest are the scores
    with open(input_dir, "r") as f:
        lines = f.readlines()
    # loop through each line
    avg_scores = []
    for line in lines:
        # split the line
        scores = line.split(",")
        # convert the scores to float
        scores = [float(score) for score in scores[1:-1]]
        # compute the average score
        avg_score = np.mean(scores)
        avg_scores.append(avg_score)

    # compute the average score for all videos
    avg_score = np.mean(avg_scores)
    # get round with 3
    avg_score = round(avg_score, 3)
    return avg_score
        

if __name__ == "__main__":
    # get the input_dir
    # input_dir = "./ckpt/0530_ZITS_video_YoutubeVOS_max500k_mix458k_turn470k/validation"
    # input_dir = "../FuseFormer/2023-06-30_gen_00050_scratch_youtube_results"
    # input_dir = "../FuseFormer/gen_00050_scratch_youtube_results"
    # input_dir = "../FuseFormer/2023-07-03_gen_00050_scratch_line_gt0_sample10_results"

    # show whethre only mask is evaluated
    print(f"Only Mask: {args.onlyMask}")

    # psnr_dir = os.path.join(args.root, args.model, f"{args.date}_{type}_{th}percent", f"PSNR_onlyMask_{args.onlyMask}.csv")
    # ssim_dir = os.path.join(args.root, args.model, f"{args.date}_{type}_{th}percent", f"SSIM_onlyMask_{args.onlyMask}.csv")
    # lpips_dir = os.path.join(args.root, args.model, f"{args.date}_{type}_{th}percent", f"LPIPS_onlyMask_{args.onlyMask}.csv")

    # if os.path.exists(psnr_dir):
    #     os.remove(psnr_dir)

    # if os.path.exists(ssim_dir):
    #     os.remove(ssim_dir)
    
    # if os.path.exists(lpips_dir):
    #     os.remove(lpips_dir)

    # # process all videos
    # process_videos(args.root, args.model, f"{args.date}_{type}_{th}percent", evalOnlyMask=args.onlyMask)

    import pandas as pd
    # create a dataframe to store the results of different thresholds
    df = pd.DataFrame(columns=["condition", "psnr", "ssim", "lpips", "vfid", "vif"])

    # line part
    # for type in ["line"]:
    #     for th in [25]:
    real_i3d_whole, output_i3d_whole = [], []
    for type in ["line", "edge"]:
        for th in [25, 50, 75, 100]:
            if args.date != "":
                in_psnr_dir = os.path.join(args.root, f"{args.date}_{type}_{th}percent", f"PSNR_onlyMask_{args.onlyMask}.csv")
                in_ssim_dir = os.path.join(args.root, f"{args.date}_{type}_{th}percent", f"SSIM_onlyMask_{args.onlyMask}.csv")
                in_lpips_dir = os.path.join(args.root, f"{args.date}_{type}_{th}percent", f"LPIPS_onlyMask_{args.onlyMask}.csv")
                in_vif_dir = os.path.join(args.root, f"{args.date}_{type}_{th}percent", f"VIF_onlyMask_{args.onlyMask}.csv")
            else:
                in_psnr_dir = os.path.join(args.root, f"{type}_{th}percent", f"PSNR_onlyMask_{args.onlyMask}.csv")
                in_ssim_dir = os.path.join(args.root, f"{type}_{th}percent", f"SSIM_onlyMask_{args.onlyMask}.csv")
                in_lpips_dir = os.path.join(args.root, f"{type}_{th}percent", f"LPIPS_onlyMask_{args.onlyMask}.csv")
                in_vif_dir = os.path.join(args.root, f"{type}_{th}percent", f"VIF_onlyMask_{args.onlyMask}.csv")

            # if os.path.exists(in_psnr_dir):
            #     os.remove(in_psnr_dir)

            # if os.path.exists(in_ssim_dir):
            #     os.remove(in_ssim_dir)
            
            # if os.path.exists(in_lpips_dir):
            #     os.remove(in_lpips_dir)

            # if os.path.exists(in_vif_dir):
            #     os.remove(in_vif_dir)

            # # process all videos
            # if args.date != "":
            #     imgs_i3d_feature_all, gts_i3d_feature_all = process_videos(os.path.join(args.root, f"{args.date}_{type}_{th}percent"), evalOnlyMask=args.onlyMask, split=args.split)
            # else:
            #     imgs_i3d_feature_all, gts_i3d_feature_all = process_videos(os.path.join(args.root, f"{type}_{th}percent"), evalOnlyMask=args.onlyMask, split=args.split)

            # real_i3d_whole += gts_i3d_feature_all
            # output_i3d_whole += imgs_i3d_feature_all

            psnr_avg = compute_avg(in_psnr_dir)
            ssim_avg = compute_avg(in_ssim_dir)
            lpips_avg = compute_avg(in_lpips_dir)
            vif_avg = compute_avg(in_vif_dir)
            # vfid = get_vfid_score(gts_i3d_feature_all, imgs_i3d_feature_all)
            vfid = 0

            # check is "edge" or "line" is in the input_dir
            df = df.append({"condition": f"{type}_{th}", "psnr": psnr_avg, "ssim": ssim_avg, "lpips": lpips_avg, "vfid": vfid, "vif": vif_avg}, ignore_index=True)

    # save the results with the header
    final_vfid = get_vfid_score(real_i3d_whole, output_i3d_whole)
    # df = df.append({"condition": "Average", "psnr": round(df["psnr"].mean(), 2), "ssim": round(df["ssim"].mean(), 3), "lpips": round(df["lpips"].mean(), 3), "vfid": round(final_vfid, 3), "vif": round(df["vif"].mean(), 3)}, ignore_index=True)
    # use concate to add the average row, with value round to 3
    df = pd.concat([df, pd.DataFrame([["Average", round(df["psnr"].mean(), 2), round(df["ssim"].mean(), 3), round(df["lpips"].mean(), 3), round(final_vfid, 3), round(df["vif"].mean(), 3)]], columns=["condition", "psnr", "ssim", "lpips", "vfid", "vif"])], ignore_index=True)

    
    if args.date != "":
        df.to_csv(os.path.join(args.root, f"metrics_summary_tmp_{args.date}_onlyMask_{args.onlyMask}.csv"), index=False)
    else:
        df.to_csv(os.path.join(args.root, f"metrics_summary_tmp_onlyMask_{args.onlyMask}.csv"), index=False)
        
    # print the results
    # print(df)