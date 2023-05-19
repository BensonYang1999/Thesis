import os
from glob import glob

import cv2
import numpy as np
import torch
from scipy import linalg
from skimage.color import rgb2gray
from skimage.measure import compare_ssim
from torch.autograd import Variable
from torch.nn.functional import adaptive_avg_pool2d
from tqdm import tqdm

from src.models.inception import InceptionV3
import torch.nn as nn
import lpips


def get_activations(images, model, batch_size=64, dims=2048,
                    cuda=False, verbose=False):
    """Calculates the activations of the pool_3 layer for all images.
    Params:
    -- images      : Numpy array of dimension (n_images, 3, hi, wi). The values
                     must lie between 0 and 1.
    -- model       : Instance of inception model
    -- batch_size  : the images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size depends
                     on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the number
                     of calculated batches is reported.
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    d0 = images.shape[0]
    if batch_size > d0:
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = d0

    n_batches = d0 // batch_size
    if d0 % batch_size != 0:
        n_batches += 1
    n_used_imgs = d0

    pred_arr = np.empty((n_used_imgs, dims))
    with torch.no_grad():
        for i in tqdm(range(n_batches)):
            if verbose:
                print('\rPropagating batch %d/%d' % (i + 1, n_batches),
                      end='', flush=True)
            start = i * batch_size
            end = min(start + batch_size, d0)

            batch = torch.from_numpy(images[start:end]).type(torch.FloatTensor)
            batch = Variable(batch)
            if cuda:
                batch = batch.cuda()

            pred = model(batch)[0]

            # If model output is not scalar, apply global spatial average pooling.
            # This happens if you choose a dimensionality not equal 2048.
            if pred.shape[2] != 1 or pred.shape[3] != 1:
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

            pred_arr[start:end] = pred.cpu().data.numpy().reshape(end - start, -1)

        if verbose:
            print(' done')

    return pred_arr


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representive data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representive data set.
    Returns:
    --   : The Frechet Distance.
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            # raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def calculate_activation_statistics(images, model, batch_size=64,
                                    dims=2048, cuda=False, verbose=False):
    """Calculation of the statistics used by the FID.
    Params:
    -- images      : Numpy array of dimension (n_images, 3, hi, wi). The values
                     must lie between 0 and 1.
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the
                     number of calculated batches is reported.
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act = get_activations(images, model, batch_size, dims, cuda, verbose)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def _compute_statistics_of_path(path, model, batch_size, dims, cuda):
    npz_file = os.path.join(path, 'statistics.npz')
    if os.path.exists(npz_file):
        f = np.load(npz_file)
        m, s = f['mu'][:], f['sigma'][:]
        f.close()
    else:
        # path = pathlib.Path(path)
        # files = list(path.glob('*.jpg')) + list(path.glob('*.png'))
        files = list(glob(path + '/*.jpg')) + list(glob(path + '/*.png'))
        files = sorted(files, key=lambda x: x.split('/')[-1])

        imgs = []
        for fn in tqdm(files):
            imgs.append(cv2.imread(str(fn)).astype(np.float32)[:, :, ::-1])
        imgs = np.array(imgs)

        # Bring images to shape (B, 3, H, W)
        imgs = imgs.transpose((0, 3, 1, 2))

        # Rescale images to be between 0 and 1
        imgs /= 255

        m, s = calculate_activation_statistics(imgs, model, batch_size, dims, cuda)
        # np.savez(npz_file, mu=m, sigma=s)

    return m, s


def calculate_fid_given_paths(paths, batch_size, cuda, dims):
    """Calculates the FID of two paths"""

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx])
    if cuda:
        model.cuda()

    print('calculate path1 statistics...')
    m1, s1 = _compute_statistics_of_path(paths[0], model, batch_size, dims, cuda)
    print('calculate path2 statistics...')
    m2, s2 = _compute_statistics_of_path(paths[1], model, batch_size, dims, cuda)
    print('calculate frechet distance...')
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value


def get_inpainting_metrics(src, tgt, logger, fid_test=True):
    input_paths = sorted(glob(src + '/*'), key=lambda x: x.split('/')[-1]) # input -> ground truth
    output_paths = sorted(glob(tgt + '/*'), key=lambda x: x.split('/')[-1]) # output -> inpainted

    assert len(input_paths) == len(output_paths), (len(input_paths), len(output_paths)) # make sure the number of images is the same

    # PSNR and SSIM
    psnrs = []
    ssims = []
    maes = []
    mses = []
    max_value = 1.0 # 255.0
    for p1, p2 in tqdm(zip(input_paths, output_paths)): # p1: input, p2: output
        img1 = cv2.imread(p1) # GT
        if img1 is None:
            print(p1, 'is bad image!')
        img2 = cv2.imread(p2) # inpainted
        if img2 is None:
            print(p2, 'is bad image!')

        mse_ = np.mean((img1 / 255.0 - img2 / 255.0) ** 2) # MSE
        mae_ = np.mean(abs(img1 / 255.0 - img2 / 255.0)) # MAE
        psnr_ = max_value - 10 * np.log(mse_ + 1e-7) / np.log(10) # PSNR
        ssim_ = compare_ssim(rgb2gray(img1), rgb2gray(img2)) # SSIM
        psnrs.append(psnr_)
        ssims.append(ssim_)
        mses.append(mse_)
        maes.append(mae_)

    psnr = np.mean(psnrs) # mean PSNR
    ssim = np.mean(ssims) # mean SSIM
    mse = np.mean(mses) # mean MSE
    mae = np.mean(maes) # mean MAE

    loss_fn_alex = lpips.LPIPS(net='alex').cuda() # LPIPS

    with torch.no_grad():
        ds = [] # LPIPS
        for im1, im2 in tqdm(zip(input_paths, output_paths)): # p1: input, p2: output
            img1 = lpips.im2tensor(lpips.load_image(im1)).cuda() # GT
            img2 = lpips.im2tensor(lpips.load_image(im2)).cuda() # inpainted
            img2 = torch.nn.functional.interpolate(img2, size=(img1.shape[2], img1.shape[3]), mode='area') # resize
            d = loss_fn_alex(img1, img2) # LPIPS
            ds.append(d) # LPIPS

        ds = torch.stack(ds) # LPIPS
        ds = torch.mean(ds) # LPIPS

    # FID
    if fid_test:
        fid = calculate_fid_given_paths([src, tgt], batch_size=16, cuda=True, dims=2048)
        if logger is None:
            print('\nPSNR:{0:.3f}, SSIM:{1:.3f}, MSE:{2:.6f}, MAE:{3:.6f}, FID:{4:.3f}, LPIPS:{5:.3f}\n'.format(psnr,
                                                                                                                ssim,
                                                                                                                mse,
                                                                                                                mae,
                                                                                                                fid,
                                                                                                                ds))
        else:
            logger.info(
                '\nPSNR:{0:.3f}, SSIM:{1:.3f}, MSE:{2:.6f}, MAE:{3:.6f}, FID:{4:.3f}, LPIPS:{5:.3f}\n'.format(psnr,
                                                                                                              ssim, mse,
                                                                                                              mae,
                                                                                                              fid, ds))
        return {'psnr': psnr, 'ssim': ssim, 'mse': mse, 'mae': mae, 'fid': fid, 'lpips': ds}
    else:
        if logger is None:
            print('\nPSNR:{0:.3f}, SSIM:{1:.3f}, MSE:{2:.6f}, MAE:{3:.6f}, LPIPS:{4:.3f}\n'.format(psnr, ssim, mse, mae,
                                                                                                   ds))
        else:
            logger.info(
                '\nPSNR:{0:.3f}, SSIM:{1:.3f}, MSE:{2:.6f}, MAE:{3:.6f}, LPIPS:{4:.3f}\n'.format(psnr, ssim, mse, mae,
                                                                                                 ds))
        return {'psnr': psnr, 'ssim': ssim, 'mse': mse, 'mae': mae, 'lpips': ds}

"""
Below is for video version evaluation
"""
def get_pred_gt_frame_list(pred_path, gt_path):
    pred_folder = sorted(os.listdir(pred_path))
    pred_list = [os.path.join(pred_path, name) for name in pred_folder]
    gt_folder = sorted(os.listdir(gt_path))
    gt_list = [os.path.join(gt_path, name) for name in gt_folder]

    print(f"[Finish building creating original video and inpainted result list {origin_path}, {result_path}]")
    return pred_list, gt_list

def read_frame_from_videos(vname):
    lst = os.listdir(vname)
    lst.sort()
    fr_lst = [vname+'/'+name for name in lst]
    frames = []
    for fr in fr_lst:
        image = cv2.imread(fr)
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return frames  

_to_tensors = transforms.Compose([
    Stack(),
    ToTorchFormatTensor()])

def get_i3d_activations(batched_video, target_endpoint='Logits', flatten=True, grad_enabled=False):
    """
    Get features from i3d model and flatten them to 1d feature,
    valid target endpoints are defined in InceptionI3d.VALID_ENDPOINTS
    VALID_ENDPOINTS = (
        'Conv3d_1a_7x7',
        'MaxPool3d_2a_3x3',
        'Conv3d_2b_1x1',
        'Conv3d_2c_3x3',
        'MaxPool3d_3a_3x3',
        'Mixed_3b',
        'Mixed_3c',
        'MaxPool3d_4a_3x3',
        'Mixed_4b',
        'Mixed_4c',
        'Mixed_4d',
        'Mixed_4e',
        'Mixed_4f',
        'MaxPool3d_5a_2x2',
        'Mixed_5b',
        'Mixed_5c',
        'Logits',
        'Predictions',
    )
    """
    init_i3d_model()
    with torch.set_grad_enabled(grad_enabled):
        feat = i3d_model.extract_features(batched_video.transpose(1, 2), target_endpoint)
    if flatten:
        feat = feat.view(feat.size(0), -1)

    return feat

def get_inpainting_metrics_video(src, tgt, logger, fid_test=True):
    pred_list, gt_list = get_origin_inpainted_frame_list(src, tgt) # input -> ground truth

    assert len(result_path) == len(gt_list), (len(result_path), len(gt_list)) # make sure the number of images is the same
    
    video_num = len(frame_list) # number of videos

    ssim_all, psnr_all, len_all = 0., 0., 0. 
    s_psnr_all = 0. 
    video_length_all = 0 
    vfid = 0.
    output_i3d_activations = []
    real_i3d_activations = []

    for video_no in range(video_num):
        print("[Processing: {}]".format(pred_list[video_no].split("/")[-1])) # print video name
        gt_PIL = read_frame_from_videos(pred_list[video_no]) # read GT
        pred_PIL = read_frame_from_videos(gt_list[video_no]) # read inpainted
        video_length = len(gt_PIL) # length of the video

        ssim, psnr, s_psnr = 0., 0., 0. 
        for gt_img, pred_img in zip(gt_PIL, pred_PIL):
            gt_img = np.array(gt_img)
            pred_img = np.array(pred_img)
            print(f"gt_img: {gt_img}") # test
            print(f"pred_img: {pred_img}") # test
            ssim += measure.compare_ssim(gt_img, pred_img, data_range=255, multichannel=True, win_size=65)
            s_psnr += measure.compare_psnr(gt_img, pred_img, data_range=255)

        ssim_all += ssim
        s_psnr_all += s_psnr
        video_length_all += (video_length)
        if video_no % 50 ==1:
            print("ssim {}, psnr {}".format(ssim_all/video_length_all, s_psnr_all/video_length_all))
        # FVID computation
        gts = _to_tensors(gt_PIL).unsqueeze(0).to(device)
        preds = _to_tensors(pred_PIL).unsqueeze(0).to(device)
        real_i3d_activations.append(get_i3d_activations(gts).cpu().numpy().flatten())
        output_i3d_activations.append(get_i3d_activations(preds).cpu().numpy().flatten())
    fid_score = get_fid_score(real_i3d_activations, output_i3d_activations)
    print("[Finish evaluating, ssim is {}, psnr is {}]".format(ssim_all/video_length_all, s_psnr_all/video_length_all))
    print("[vfid score is {}]".format(fid_score))

    return {'psnr': s_psnr_all/video_length_all, 'ssim': ssim_all/video_length_all, 'vfid': fid_score}


if __name__ == "__main__":
    tgt = 'GT' # ground truth
    src1 = 'results' # inpainted results

    one = get_inpainting_metrics(src1, tgt, None, fid_test=True) # fid_test=True

    print('\nMean PSNR:{0:.3f},Mean SSIM:{1:.3f},Mean FID:{2:.3f}\n'.format(one['psnr'], one['ssim'], one['fid'])) # fid_test=True


