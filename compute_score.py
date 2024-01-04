# there is a folder contains multiple subfolders (videos), each subfolder contains multiple jpg files (frames for each video).
# in each sufolders, there are two kind of prefix for image file, "gt_" and "pred_"
# write a function that input the path of the folder, and read all the jpg files in the subfolders
# compute the PSNR, SSIM, VFID score for each pair of "gt_" and "pred_" images
# the results should be save in three csv files with name "PSNR.csv", "SSIM.csv", "VFID.csv" under the input_dir's parent folder
# in each csv file, the first column is the video id, and the rest of the columns are the score for each frame

# write a function to read all images in a folder with a given prefix
# input: folder path (for a video)
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

from sewar.full_ref import uqi, mse, rmse, scc, rase, sam, msssim, vifp, psnrb

lpips_loss_fn = lpips.LPIPS(net='vgg')
# put the model to GPU
# lpips_loss_fn.cuda()

fid_fn = FrechetInceptionDistance(feature=64)
# fid_fn.cuda()

# def read_images_in_folder(input_dir:str, prefix:str):
#     output = []

#     # get all the jpg files in the folder
#     jpg_files = [f.path for f in os.scandir(input_dir) if f.is_file() and f.name.startswith(prefix)]
#     # sort
#     jpg_files.sort()
#     # loop through each jpg file
#     for jpg_file in jpg_files:
#         # read the image
#         img = cv2.imread(jpg_file)
#         # append the image to the output
#         output.append(img)

#     return output

def read_frame_from_videos(vname , prefix:str):
    # lst = os.listdir(vname)
    lst = [f for f in os.listdir(vname) if f.endswith(".jpg")]
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

def compute_vfid(gt, pred):
    # conver list to numpy array
    gt = np.array(gt)
    pred = np.array(pred)

    # convert the images to torch tensors
    print(gt.shape)
    gt = torch.from_numpy(gt).permute(0, 3, 1, 2).unsqueeze(0).float()
    pred = torch.from_numpy(pred).permute(0, 3, 1, 2).unsqueeze(0).float()

    # put the images to GPU
    # gt = gt.cuda()
    # pred = pred.cuda()

    real_i3d_activation = get_i3d_activations(gt).cpu().numpy().flatten()
    output_i3d_activation = get_i3d_activations(pred).cpu().numpy().flatten()

    print("real i3d", real_i3d_activation.shape)
    print("output i3d", output_i3d_activation.shape)

    return fid_score

def compute_lpips(gt, pred):
    # conver gt, pred to torch tensor
    gt = torch.from_numpy(gt).permute(2, 0, 1).unsqueeze(0).to(torch.uint8)
    pred = torch.from_numpy(pred).permute(2, 0, 1).unsqueeze(0).to(torch.uint8)
    # put the images to GPU
    # gt = gt.cuda()
    # pred = pred.cuda()

    lpips_score = lpips_loss_fn.forward(gt, pred).item()
    return lpips_score

def compute_fid(gt, pred):
    #  conver gt, pred to torch tensor
    gt = torch.from_numpy(gt).permute(2, 0, 1).unsqueeze(0).to(torch.uint8)
    pred = torch.from_numpy(pred).permute(2, 0, 1).unsqueeze(0).to(torch.uint8)
    # put the images to GPU
    # gt = gt.cuda()
    # pred = pred.cuda()
    fid_fn.update(gt, real=True)
    fid_fn.update(pred, real=False)
    fid_score = fid_fn.compute().item()
    return fid_score

# define a function to compute the desire metric for each video
# def compute_metrics(input_dir):
#     # read all the images with prefix "gt_"
#     gt_images = read_frame_from_videos(input_dir, "gt_")
#     # read all the images with prefix "pred_"
#     pred_images = read_frame_from_videos(input_dir, "pred_")

#     # compute the PSNR for each pair of images
#     psnr_scores, ssim_scores = [], []
#     for gt_image, pred_image in zip(gt_images, pred_images):
#         gt_img = np.array(gt_image)
#         pred_img = np.array(pred_image)

#         psnr_score = measure_psnr(gt_img, pred_img, data_range=255)
#         ssim_score = measure_ssim(gt_img, pred_img, data_range=255, multichannel=True, win_size=65)
#         psnr_scores.append(psnr_score)
#         ssim_scores.append(ssim_score)

#     return psnr_scores, ssim_scores

# define a function to compute the desire metric for each video
def compute_metrics(pred_dir, evalOnlyMask=False, split="test"):
    # print("pred_dir", pred_dir)
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
    vif_scores, uqi_scores, mse_scores, rmse_scores, scc_scores, rase_scores, sam_scores, msssim_scores, psnrb_scores = [], [], [], [], [], [], [], [], []
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
        # ssim_score = measure_ssim(gt_img, pred_img, data_range=255, multichannel=True, win_size=65)
        ssim_score = measure_ssim(gt_img, pred_img, data_range=255, channel_axis=2)
        lpips_score = compute_lpips(gt_img, pred_img)
        vif_score = vifp(gt_img, pred_img)
        uqi_score = uqi(gt_img, pred_img)
        mse_score = mse(gt_img, pred_img)
        rmse_score = rmse(gt_img, pred_img)
        scc_score = scc(gt_img, pred_img)
        rase_score = rase(gt_img, pred_img)
        sam_score = sam(gt_img, pred_img)
        msssim_score = msssim(gt_img, pred_img)
        psnrb_score = psnrb(gt_img, pred_img)

        # fid_score = compute_fid(gt_img, pred_img)
        psnr_scores.append(psnr_score)
        ssim_scores.append(ssim_score)
        lpips_scores.append(lpips_score)
        vif_scores.append(vif_score)
        uqi_scores.append(uqi_score)
        mse_scores.append(mse_score)
        rmse_scores.append(rmse_score)
        scc_scores.append(scc_score)
        rase_scores.append(rase_score)
        sam_scores.append(sam_score)
        msssim_scores.append(msssim_score)
        psnrb_scores.append(psnrb_score)
        # fid_scores.append(fid_score)

    return psnr_scores, ssim_scores, lpips_scores, vif_scores, uqi_scores, mse_scores, rmse_scores, scc_scores, rase_scores, sam_scores, msssim_scores, psnrb_scores

def write_scores(csv_path, scores, video_id):
    with open(csv_path, "a") as f:
        f.write(video_id + ",")
        for score in scores:
            f.write(str(score) + ",")
        f.write("\n")
        
# process all videos and save the results
def process_videos(input_dir, evalOnlyMask=False):
    # get all the subfolders in the input_dir
    subfolders = [f.path for f in os.scandir(input_dir) if f.is_dir()]

    # loop through each subfolder
    for subfolder in tqdm.tqdm(subfolders):
        # compute the metrics for each subfolder
        psnr_scores, ssim_scores, lpips_scores, vif_scores, uqi_scores, mse_scores, rmse_scores, scc_scores, rase_scores, sam_scores, msssim_scores, psnrb_scores = compute_metrics(subfolder, evalOnlyMask, args.split)
        print(f"Video: {subfolder.split('/')[-1]}, PSNR: {np.mean(psnr_scores)}, SSIM: {np.mean(ssim_scores)}, LPIPS: {np.mean(lpips_scores)}, VIF: {np.mean(vif_scores)}, UQI: {np.mean(uqi_scores)}, MSE: {np.mean(mse_scores)}, RMSE: {np.mean(rmse_scores)}, SCC: {np.mean(scc_scores)}, RASE: {np.mean(rase_scores)}, SAM: {np.mean(sam_scores)}, MSSSIM: {np.mean(msssim_scores)}, PSNRB: {np.mean(psnrb_scores)}")

        # save the results
        # get the video id
        video_id = subfolder.split("/")[-1]
        psnr_csv_path = os.path.join(input_dir, f"PSNR_onlyMask_{evalOnlyMask}.csv")
        ssim_csv_path = os.path.join(input_dir, f"SSIM_onlyMask_{evalOnlyMask}.csv")
        lpips_csv_path = os.path.join(input_dir, f"LPIPS_onlyMask_{evalOnlyMask}.csv")
        vif_csv_path = os.path.join(input_dir, f"VIF_onlyMask_{evalOnlyMask}.csv")
        uqi_csv_path = os.path.join(input_dir, f"UQI_onlyMask_{evalOnlyMask}.csv")
        mse_csv_path = os.path.join(input_dir, f"MSE_onlyMask_{evalOnlyMask}.csv")
        rmse_csv_path = os.path.join(input_dir, f"RMSE_onlyMask_{evalOnlyMask}.csv")
        scc_csv_path = os.path.join(input_dir, f"SCC_onlyMask_{evalOnlyMask}.csv")
        rase_csv_path = os.path.join(input_dir, f"RASE_onlyMask_{evalOnlyMask}.csv")
        sam_csv_path = os.path.join(input_dir, f"SAM_onlyMask_{evalOnlyMask}.csv")
        msssim_csv_path = os.path.join(input_dir, f"MSSSIM_onlyMask_{evalOnlyMask}.csv")
        psnrb_csv_path = os.path.join(input_dir, f"PSNRB_onlyMask_{evalOnlyMask}.csv")

        
        # save PSNR scores
        write_scores(psnr_csv_path, psnr_scores, video_id)
        # save SSIM scores
        write_scores(ssim_csv_path, ssim_scores, video_id)
        # save LPIPS scores
        write_scores(lpips_csv_path, lpips_scores, video_id)
        # save VIF scores
        write_scores(vif_csv_path, vif_scores, video_id)
        # save UQI scores
        write_scores(uqi_csv_path, uqi_scores, video_id)
        # save MSE scores
        write_scores(mse_csv_path, mse_scores, video_id)
        # save RMSE scores
        write_scores(rmse_csv_path, rmse_scores, video_id)
        # save SCC scores
        write_scores(scc_csv_path, scc_scores, video_id)
        # save RASE scores
        write_scores(rase_csv_path, rase_scores, video_id)
        # save SAM scores
        write_scores(sam_csv_path, sam_scores, video_id)
        # save MSSSIM scores
        write_scores(msssim_csv_path, msssim_scores, video_id)
        # save PSNRB scores
        write_scores(psnrb_csv_path, psnrb_scores, video_id)
        

if __name__ == "__main__":
    # get the input_dir
    # input_dir = "./ckpt/0530_ZITS_video_YoutubeVOS_max500k_mix458k_turn470k/validation"
    # input_dir = "../FuseFormer/2023-06-30_gen_00050_scratch_youtube_results"
    # input_dir = "../FuseFormer/gen_00050_scratch_youtube_results"
    # input_dir = "../FuseFormer/2023-07-03_gen_00050_scratch_line_gt0_sample10_results"

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default="../FuseFormer/2023-07-03_gen_00050_scratch_line_gt0_sample10_results")
    parser.add_argument('--onlyMask', action='store_true', default=False)
    parser.add_argument('--runAll', action='store_true', default=False)
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
    rmse_dir = os.path.join(args.input_dir, f"RMSE_onlyMask_{args.onlyMask}.csv")
    scc_dir = os.path.join(args.input_dir, f"SCC_onlyMask_{args.onlyMask}.csv")
    rase_dir = os.path.join(args.input_dir, f"RASE_onlyMask_{args.onlyMask}.csv")
    sam_dir = os.path.join(args.input_dir, f"SAM_onlyMask_{args.onlyMask}.csv")
    msssim_dir = os.path.join(args.input_dir, f"MSSSIM_onlyMask_{args.onlyMask}.csv")
    psnrb_dir = os.path.join(args.input_dir, f"PSNRB_onlyMask_{args.onlyMask}.csv")


    if os.path.exists(psnr_dir):
        os.remove(psnr_dir)

    if os.path.exists(ssim_dir):
        os.remove(ssim_dir)
    
    if os.path.exists(lpips_dir):
        os.remove(lpips_dir)

    if os.path.exists(vif_dir):
        os.remove(vif_dir)

    if os.path.exists(uqi_dir):
        os.remove(uqi_dir)

    if os.path.exists(mse_dir):
        os.remove(mse_dir)

    if os.path.exists(rmse_dir):
        os.remove(rmse_dir)

    if os.path.exists(scc_dir):
        os.remove(scc_dir)

    if os.path.exists(rase_dir):
        os.remove(rase_dir)

    if os.path.exists(sam_dir):
        os.remove(sam_dir)

    if os.path.exists(msssim_dir):
        os.remove(msssim_dir)

    if os.path.exists(psnrb_dir):
        os.remove(psnrb_dir)

    # process all videos
    process_videos(args.input_dir, evalOnlyMask=args.onlyMask)

    # import calculate_avereage_with_th from select_video_with_given_portion
    from select_video_with_given_portion import calculate_avereage_with_th

    # line_th = [0, 50, 100, 150, 200]
    # edge_th = [0, 2, 4, 6, 8, 10]
    line_th = [0, 1.251, 6.703, 31.697]
    edge_th = [0, 1.879, 3.521, 5.755]

    import pandas as pd
    # create a dataframe to store the results of different thresholds
    df = pd.DataFrame(columns=["line_th", "edge_th", "psnr", "ssim", "lpips", "vif", "uqi", "mse", "rmse", "scc", "rase", "sam", "msssim", "psnrb"])

    # go through each combination of thresholds
    for (line_th, edge_th) in itertools.product(line_th, edge_th):
        psnr_avg = calculate_avereage_with_th(psnr_dir, line_th, edge_th)
        ssim_avg = calculate_avereage_with_th(ssim_dir, line_th, edge_th)
        lpips_avg = calculate_avereage_with_th(lpips_dir, line_th, edge_th)
        vif_avg = calculate_avereage_with_th(vif_dir, line_th, edge_th)
        uqi_avg = calculate_avereage_with_th(uqi_dir, line_th, edge_th)
        mse_avg = calculate_avereage_with_th(mse_dir, line_th, edge_th)
        rmse_avg = calculate_avereage_with_th(rmse_dir, line_th, edge_th)
        scc_avg = calculate_avereage_with_th(scc_dir, line_th, edge_th)
        rase_avg = calculate_avereage_with_th(rase_dir, line_th, edge_th)
        sam_avg = calculate_avereage_with_th(sam_dir, line_th, edge_th)
        msssim_avg = calculate_avereage_with_th(msssim_dir, line_th, edge_th)
        psnrb_avg = calculate_avereage_with_th(psnrb_dir, line_th, edge_th)

        df = df.append({"line_th": line_th, "edge_th": edge_th, "psnr": psnr_avg, "ssim": ssim_avg, "lpips": lpips_avg, "vif": vif_avg, "uqi": uqi_avg}, ignore_index=True)

    # save the results with the header
    df.to_csv(os.path.join(args.input_dir, f"metrics_all_onlyMask_{args.onlyMask}.csv"), index=False)

    # print the results
    # print(df)