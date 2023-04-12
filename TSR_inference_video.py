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
    # parser.add_argument('--image_url', type=str, default=None, help='the folder of image')
    # parser.add_argument('--mask_url', type=str, default=None)
    # parser.add_argument('--test_line_path', type=str, default='', help='Indicate where is the wireframes of test set')
    # parser.add_argument('--image_size', type=int, default=256, help='input sequence length: image_size*image_size')
    parser.add_argument('--n_layer', type=int, default=16)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--n_embd', type=int, default=256)
    parser.add_argument('--save_url', type=str, default=None, help='save the output results')
    parser.add_argument('--iterations', type=int, default=5)
    # video
    parser.add_argument('--dataset_root', type=str, default="./datasets", help='Indicate where is the root of training set folder')
    parser.add_argument('--dataset_name', type=str, default="YouTubeVOS", help='Indicate which training set')
    parser.add_argument('--ref_frame_num', type=int, default=5)

    opts = parser.parse_args()

    gpu = f"cuda:{opts.GPU_ids}"
    device = torch.device(gpu if torch.cuda.is_available() else "cpu")

    s_time = time.time()
    model_config = EdgeLineGPTConfig(embd_pdrop=0.0, resid_pdrop=0.0, n_embd=opts.n_embd, block_size=32,
                                     attn_pdrop=0.0, n_layer=opts.n_layer, n_head=opts.n_head, ref_frame_num=opts.ref_frame_num)
    # Load model
    IGPT_model = EdgeLineGPT256RelBCE_video(model_config, device=gpu)
    checkpoint = torch.load(opts.ckpt_path)

    if opts.ckpt_path.endswith('.pt'):
        IGPT_model.load_state_dict(checkpoint)
    else:
        IGPT_model.load_state_dict(checkpoint['model'])

    IGPT_model.cuda()

    test_dataset = ContinuousEdgeLineDatasetMask_video(sample=opts.ref_frame_num, size=(432,240), split='test', name=opts.dataset_name, root=opts.dataset_root)

    for it in tqdm(range(test_dataset.__len__())):

        items = test_dataset.__getitem__(it)

        edge_pred, line_pred = SampleEdgeLineLogits_video(IGPT_model, context=[items['frames'],
                                                   items['edges'], items['lines']],
                              mask=items['masks'], iterations=opts.iterations)
        # save separately
        # edge_output = edge_pred.cpu() * items['masks'] + items['edges'] * (1 - items['masks'])
        edge_output = edge_pred.cpu()
        edge_output = edge_output.repeat(1, 3, 1, 1).permute(0, 2, 3, 1)   # gray -> rgb
        
        # line_output = line_pred.cpu() * items['masks'] + items['lines'] * (1 - items['masks'])
        line_output = line_pred.cpu()
        line_output = line_output.repeat(1, 3, 1, 1).permute(0, 2, 3, 1)  # gray -> rgb

        edge_output = (edge_output * 255).detach().numpy().astype(np.uint8)
        line_output = (line_output * 255).detach().numpy().astype(np.uint8)

        edge_foler = os.path.join(opts.save_url, "edges", items['name'])
        line_foler = os.path.join(opts.save_url, "lines", items['name'])
        for folder in [edge_foler, line_foler]:
            if not os.path.exists(folder):
                os.makedirs(folder)

        for idx in range(opts.ref_frame_num):
            cv2.imwrite(os.path.join(edge_foler, items['idxs'][idx]), edge_output[idx, :, :, ::-1])
            cv2.imwrite(os.path.join(line_foler, items['idxs'][idx]), line_output[idx, :, :, ::-1])

    e_time = time.time()
    print("This inference totally costs %.5f seconds" % (e_time - s_time))
