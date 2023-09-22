import cv2 
import numpy as np
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import lpips
import torch

lpips_loss_fn = lpips.LPIPS(net='vgg')
# put the model to GPU
lpips_loss_fn.cuda()

gt_dir = "./datasets/YouTubeVOS/test_all_frames/JPEGImages/0b6db1c6fd/00000.jpg"
pd_dir = "./results/0530_ZITS_video_YoutubeVOS_max500k_mix458k_turn470k/2023-07-07_youtubevos/0b6db1c6fd/00000.jpg"
fuse_dir = "../FuseFormer/fuseformer_5frames_youtube_results/2023-07-07_gen_00050_youtubevos_split_test/0b6db1c6fd/0000.png"

# define a function that read the image and resize it to 432*240, then compute the gradient
def compute_gradient(img_dir):
    img = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (432, 240))
    img = img.astype(np.float32)
    img_grad = np.gradient(img)
    img_grad = np.array(img_grad)
    return img_grad

# compute the gradient of rgb image
def compute_rgb_gradient(img_dir):
    img = cv2.imread(img_dir)
    img = cv2.resize(img, (432, 240))
    img = img.astype(np.float32)
    img_grad = np.gradient(img)
    img_grad = np.array(img_grad)
    return img_grad

def normalize(image):
    image = (image - np.min(image)) / (np.max(image) - np.min(image))  # normalize to [0, 1]
    return image

def compute_lpips(gt, pred):
    # conver gt, pred to torch tensor
    gt = torch.from_numpy(gt).unsqueeze(0).to(torch.uint8)
    pred = torch.from_numpy(pred).unsqueeze(0).to(torch.uint8)
    # put the images to GPU
    print(f"gt shape: {gt.shape}")
    print(f"pred shape: {pred.shape}")
    gt = gt.cuda()
    pred = pred.cuda()

    lpips_score = lpips_loss_fn.forward(gt, pred).item()
    return lpips_score

def compute_edge(self, img):
    return canny(img, sigma=2, mask=None).astype(np.float)

def read_img2gray(dir):
    img = cv2.imread(dir)
    img = cv2.resize(img, (432, 240))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


# read gt image the convert to gray scale
gt_gray = read_img2gray(gt_dir)
pd_gray = read_img2gray(pd_dir)
fuse_gray = read_img2gray(fuse_dir)

print(f"min: {gt_gray.min()}, max: {gt_gray.max()}")
print(f"min: {pd_gray.min()}, max: {pd_gray.max()}")
print(f"min: {fuse_gray.min()}, max: {fuse_gray.max()}")

# compute the mse, psnr, ssim
mse_pd = mse(gt_gray, pd_gray)
mse_fuse = mse(gt_gray, fuse_gray)
psnr_pd = psnr(gt_gray, pd_gray, data_range=1)
psnr_fuse = psnr(gt_gray, fuse_gray, data_range=1)
ssim_pd = ssim(gt_gray, pd_gray, multichannel=True, data_range=1)
ssim_fuse = ssim(gt_gray, fuse_gray, multichannel=True, data_range=1)
lpips_pd = compute_lpips(gt_gray, pd_gray)
lpips_fuse = compute_lpips(gt_gray, fuse_gray)

# round above result to 4 decimal places
mse_pd = round(mse_pd, 4)
mse_fuse = round(mse_fuse, 4)
psnr_pd = round(psnr_pd, 4)
psnr_fuse = round(psnr_fuse, 4)
ssim_pd = round(ssim_pd, 4)
ssim_fuse = round(ssim_fuse, 4)
lpips_pd = round(lpips_pd, 4)
lpips_fuse = round(lpips_fuse, 4)


# show the results
print("mse_pd: ", mse_pd)
print("mse_fuse: ", mse_fuse)
print("psnr_pd: ", psnr_pd)
print("psnr_fuse: ", psnr_fuse)
print("ssim_pd: ", ssim_pd)
print("ssim_fuse: ", ssim_fuse)
print("lpips_pd: ", lpips_pd)
print("lpips_fuse: ", lpips_fuse)
