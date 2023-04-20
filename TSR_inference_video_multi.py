import argparse
import os
import time

import cv2
import numpy as np
import torch
from tqdm import tqdm

from datasets.dataset_TSR import ContinuousEdgeLineDatasetMask_video
from src.models.TSR_model import EdgeLineGPTConfig, EdgeLineGPT256RelBCE_video
from src.utils import set_seed, SampleEdgeLineLogits_video

if __name__ == '__main__':
    set_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument('--GPU_ids', type=str, default='0')
    parser.add_argument('--ckpt_path', type=str, default='./ckpt/places2_continous_edgeline/best.pth')
    parser.add_argument('--n_layer', type=int, default=16)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--n_embd', type=int, default=256)
    parser.add_argument('--save_url', type=str, default=None, help='save the output results')
    parser.add_argument('--iterations', type=int, default=5)
    parser.add_argument('--dataset_root', type=str, default="./datasets", help='Indicate where is the root of training set folder')
    parser.add_argument('--dataset_name', type=str, default="YouTubeVOS", help='Indicate which training set')
    parser.add_argument('--ref_frame_num', type=int, default=5)
    parser.add_argument('--loss_choice', type=str, default="bce", help='the choice of loss function: l1, mse, bce')
    parser.add_argument('--edge_gaussian', type=int, default=0, help='the sigma of gaussian kernel for edge')

    opts = parser.parse_args()

    device = torch.device(f"cuda:{opts.GPU_ids}" if torch.cuda.is_available() else "cpu")

    s_time = time.time()
    model_config = EdgeLineGPTConfig(embd_pdrop=0.0, resid_pdrop=0.0, n_embd=opts.n_embd, block_size=32,
                                     attn_pdrop=0.0, n_layer=opts.n_layer, n_head=opts.n_head, ref_frame_num=opts.ref_frame_num)
    IGPT_model = EdgeLineGPT256RelBCE_video(model_config, opts, device=device)
    checkpoint = torch.load(opts.ckpt_path)
    IGPT_model.load_state_dict(checkpoint if opts.ckpt_path.endswith('.pt') else checkpoint['model'])
    IGPT_model.to("cpu")

    test_dataset = ContinuousEdgeLineDatasetMask_video(opts, sample=opts.ref_frame_num, size=(432, 240), split='test', name=opts.dataset_name, root=opts.dataset_root)

    for it in tqdm(range(test_dataset.__len__())):

        items = test_dataset.__getitem__(it)

        edge_pred, line_pred = IGPT_model.forward_with_logits(img_idx=items['frames'],
                                                              edge_idx=items['edges'], line_idx=items['lines'], masks=items['masks'])

        edge_output = (edge_pred.cpu() * items['masks'] + items['edges'] * (1 - items['masks'])).repeat(1, 3, 1, 1).permute(0, 2, 3, 1)
        line_output = (line_pred.cpu() * items['masks'] + items['lines'] * (1 - items['masks'])).repeat(1, 3, 1, 1).permute(0, 2, 3, 1)
        edge_output = (edge_output * 255).detach().numpy().astype(np.uint8)
        line_output = (line_output * 255).detach().numpy().astype(np.uint8)

        # Get the original image and invert the masked area
        original_edge = items['edges']*(1-items['masks']) + items['masks']*(1-items['edges'])
        original_line = items['lines']*(1-items['masks']) + items['masks']*(1-items['lines'])
        original_edge = original_edge.repeat(1, 3, 1, 1).permute(0, 2, 3, 1)
        original_line = original_line.repeat(1, 3, 1, 1).permute(0, 2, 3, 1)
        original_edge = (original_edge * 255).detach().numpy().astype(np.uint8)
        original_line = (original_line * 255).detach().numpy().astype(np.uint8)

        # Concatenate the original image with the inverted masked area and the outputs
        edge_output = np.concatenate((original_edge, edge_output), axis=2)
        line_output = np.concatenate((original_line, line_output), axis=2)

        edge_folder = os.path.join(opts.save_url, "edges", items['name'])
        line_folder = os.path.join(opts.save_url, "lines", items['name'])
        for folder in [edge_folder, line_folder]:
            os.makedirs(folder, exist_ok=True)

        for idx in range(opts.ref_frame_num):
            cv2.imwrite(os.path.join(edge_folder, items['idxs'][idx]), edge_output[idx, :, :, ::-1])
            cv2.imwrite(os.path.join(line_folder, items['idxs'][idx]), line_output[idx, :, :, ::-1])


    e_time = time.time()
    print("This inference totally costs %.5f seconds" % (e_time - s_time))
