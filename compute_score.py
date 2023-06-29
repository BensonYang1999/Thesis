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

def compute_vfid(gt, pred):
    # conver list to numpy array
    gt = np.array(gt)
    pred = np.array(pred)

    # convert the images to torch tensors
    print(gt.shape)
    gt = torch.from_numpy(gt).permute(0, 3, 1, 2).unsqueeze(0).float()
    pred = torch.from_numpy(pred).permute(0, 3, 1, 2).unsqueeze(0).float()

    # put the images to GPU
    gt = gt.cuda()
    pred = pred.cuda()

    real_i3d_activation = get_i3d_activations(gt).cpu().numpy().flatten()
    output_i3d_activation = get_i3d_activations(pred).cpu().numpy().flatten()

    print("real i3d", real_i3d_activation.shape)
    print("output i3d", output_i3d_activation.shape)

    return fid_score

# define a function to compute the desire metric for each video
def compute_metrics(input_dir):
    # read all the images with prefix "gt_"
    gt_images = read_frame_from_videos(input_dir, "gt_")
    # read all the images with prefix "pred_"
    pred_images = read_frame_from_videos(input_dir, "pred_")

    # compute the PSNR for each pair of images
    psnr_scores, ssim_scores = [], []
    for gt_image, pred_image in zip(gt_images, pred_images):
        gt_img = np.array(gt_image)
        pred_img = np.array(pred_image)

        psnr_score = measure_psnr(gt_img, pred_img, data_range=255)
        ssim_score = measure_ssim(gt_img, pred_img, data_range=255, multichannel=True, win_size=65)
        psnr_scores.append(psnr_score)
        ssim_scores.append(ssim_score)

    return psnr_scores, ssim_scores

# define a function to compute the desire metric for each video
def compute_metrics_FuseFormer(pred_dir):
    vname = pred_dir.split('/')[-1]
    gt_dir = os.path.join("./datasets/YouTubeVOS/valid_all_frames/JPEGImages", vname)
    # read all the images with prefix "gt_"
    gt_images = read_frame_from_videos(gt_dir, "")
    # read all the images with prefix "pred_"
    pred_images = read_frame_from_videos(pred_dir, "")

    # compute the PSNR for each pair of images
    psnr_scores, ssim_scores = [], []
    for gt_image, pred_image in zip(gt_images, pred_images):
        gt_img = np.array(gt_image)
        pred_img = np.array(pred_image)

        psnr_score = measure_psnr(gt_img, pred_img, data_range=255)
        ssim_score = measure_ssim(gt_img, pred_img, data_range=255, multichannel=True, win_size=65)
        psnr_scores.append(psnr_score)
        ssim_scores.append(ssim_score)

    return psnr_scores, ssim_scores

# process all videos and save the results
def process_videos(input_dir, isSota=False):
    # get all the subfolders in the input_dir
    subfolders = [f.path for f in os.scandir(input_dir) if f.is_dir()]

    # loop through each subfolder
    for subfolder in tqdm.tqdm(subfolders):
        # compute the metrics for each subfolder
        if isSota:
            psnr_scores, ssim_scores = compute_metrics_FuseFormer(subfolder)
        else:
            psnr_scores, ssim_scores = compute_metrics(subfolder)
        print(f"Video: {subfolder.split('/')[-1]}, PSNR: {np.mean(psnr_scores)}, SSIM: {np.mean(ssim_scores)}")

        # save the results
        # get the video id
        video_id = subfolder.split("/")[-1]
        # save the PSNR scores
        with open(os.path.join(input_dir, "PSNR.csv"), "a") as f:
            f.write(video_id + ",")
            for psnr_score in psnr_scores:
                f.write(str(psnr_score) + ",")
            f.write("\n")
        # save the SSIM scores
        with open(os.path.join(input_dir, "SSIM.csv"), "a") as f:
            f.write(video_id + ",")
            for ssim_score in ssim_scores:
                f.write(str(ssim_score) + ",")
            f.write("\n")
        # save the VFID scores
        # with open(os.path.join(input_dir, "VFID.csv"), "a") as f:
        #     f.write(video_id + ",")
        #     for vfid_score in vfid_scores:
        #         f.write(str(vfid_score) + ",")
        #     f.write("\n")

if __name__ == "__main__":
    # get the input_dir
    # input_dir = "./ckpt/0530_ZITS_video_YoutubeVOS_max500k_mix458k_turn470k/validation"
    input_dir = "../FuseFormer/gen_00050_scratch_youtube_results"

    # process all videos
    process_videos(input_dir, isSota=True)