import argparse
import os
import time

import cv2
import numpy as np
import torch
from tqdm import tqdm

from datasets.dataset_TSR import ContinuousEdgeLineDatasetMask_video, EdgeLineDataset_v2
from src.models.TSR_model import EdgeLineGPTConfig, EdgeLineGPT256RelBCE_video, StructGPT256RelBCE_video, EdgeLine_CNN
from src.utils import set_seed, SampleEdgeLineLogits_video, SampleEdgeLineLogits

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
    parser.add_argument("--model", type=str, default='fuseformer')
    # parser.add_argument("-v", "--video", type=str, required=True)
    parser.add_argument("--exp", action="store_true", help="use the exp version of the model")
    parser.add_argument('--cnn', action="store_true", help="use the cnn version of the model")

    opts = parser.parse_args()

    device = torch.device(f"cuda:{opts.GPU_ids}" if torch.cuda.is_available() else "cpu")

    s_time = time.time()
    model_config = EdgeLineGPTConfig(embd_pdrop=0.0, resid_pdrop=0.0, n_embd=opts.n_embd, block_size=32,
                                     attn_pdrop=0.0, n_layer=opts.n_layer, n_head=opts.n_head, ref_frame_num=opts.ref_frame_num)
    if opts.exp:
        IGPT_model = StructGPT256RelBCE_video(model_config, opts, device=device)
    elif opts.cnn:
        IGPT_model = EdgeLine_CNN()
    else:
        IGPT_model = EdgeLineGPT256RelBCE_video(model_config, opts, device=device)
    checkpoint = torch.load(opts.ckpt_path)
    IGPT_model.load_state_dict(checkpoint if opts.ckpt_path.endswith('.pt') else checkpoint['model'])
    IGPT_model.to(device)

    # test_dataset = ContinuousEdgeLineDatasetMask_video(opts, sample=opts.ref_frame_num, size=(432, 240), split='test', name=opts.dataset_name, root=opts.dataset_root)
    test_dataset = EdgeLineDataset_v2(opts, sample=opts.ref_frame_num, size=(432, 240), split='test', name=opts.dataset_name, root=opts.dataset_root)
    

    for it in tqdm(range(test_dataset.__len__())):

        items = test_dataset.__getitem__(it)
        if not items:
            continue
        edge_folder = os.path.join(opts.save_url, "edges", items['name'])
        line_folder = os.path.join(opts.save_url, "lines", items['name'])
        concat_folder = os.path.join(opts.save_url, "concat", items['name'])
        for folder in [edge_folder, line_folder, concat_folder]:
            os.makedirs(folder, exist_ok=True)
        
        edge_pred, line_pred = SampleEdgeLineLogits_video(IGPT_model, context=[items['frames'].unsqueeze(0),
                                                items['edges'].unsqueeze(0), items['lines'].unsqueeze(0)],
                            masks=items['masks'].unsqueeze(0), iterations=opts.iterations, device=device)

        # denormalize the result of edge_pred and line_pred with the above min-max normalization
        # edge_pred = edge_pred * (edge_pred.max() - edge_pred.min()) + edge_pred.min()
        # line_pred = line_pred * (line_pred.max() - line_pred.min()) + line_pred.min()

        for i, ref_idx in enumerate(items['idxs']):
            # save separately
            edge_output = edge_pred[0, i, ...].cpu() * items['masks'][i] + items['edges'][i] * (1 - items['masks'][i])
            edge_output = edge_output.repeat(3, 1, 1).permute(1, 2, 0)
            line_output = line_pred[0, i, ...].cpu() * items['masks'][i] + items['lines'][i] * (1 - items['masks'][i])
            line_output = line_output.repeat(3, 1, 1).permute(1, 2, 0)

            edge_output = (edge_output * 255).detach().numpy().astype(np.uint8)
            line_output = (line_output * 255).detach().numpy().astype(np.uint8)

            cv2.imwrite(os.path.join(edge_folder, ref_idx), edge_output[:, :, ::-1])
            cv2.imwrite(os.path.join(line_folder, ref_idx), line_output[:, :, ::-1])

            # combine the result in one figure for better visualization
            # Get the original RGB image
            masked_image = items['frames'][i] * (1-items['masks'][i])
            masked_image = ((masked_image)).permute(1, 2, 0)
            masked_image = (masked_image * 255).detach().numpy().astype(np.uint8)
            original_image = ((items['frames'][i])).permute(1, 2, 0)
            original_image = (original_image * 255).detach().numpy().astype(np.uint8)
            
            # Get the original image and invert the masked area
            original_edge = items['edges'][i]*(1-items['masks'][i]) + items['masks'][i]*(1-items['edges'][i])
            original_line = items['lines'][i]*(1-items['masks'][i]) + items['masks'][i]*(1-items['lines'][i])
            # original_edge = original_edge.repeat(3, 1, 1).permute(1, 2, 0)
            # original_line = original_line.repeat(3, 1, 1).permute(1, 2, 0)
            original_edge = original_edge.permute(1, 2, 0)
            original_line = original_line.permute(1, 2, 0)
            original_edge = (original_edge * 255).detach().numpy().astype(np.uint8)
            original_line = (original_line * 255).detach().numpy().astype(np.uint8)
            
            output_edge = (edge_pred[0][i].cpu() * (1-items['masks'][i])) + ((1-edge_pred[0][i].cpu()) * items['masks'][i])
            # output_edge = output_edge.repeat(3, 1, 1).permute(1, 2, 0)
            output_edge = output_edge.permute(1, 2, 0)
            output_edge = (output_edge * 255).detach().numpy().astype(np.uint8)
            output_line = (line_pred[0][i].cpu() * (1-items['masks'][i])) + ((1-line_pred[0][i].cpu()) * items['masks'][i])
            # output_line = output_line.repeat(3, 1, 1).permute(1, 2, 0)
            output_line = output_line.permute(1, 2, 0)
            output_line = (output_line * 255).detach().numpy().astype(np.uint8)
            line_concat = np.concatenate((original_image, original_line, output_line), axis=1)
            edge_concat = np.concatenate((masked_image, original_edge, output_edge), axis=1)
            concat = np.concatenate((line_concat, edge_concat), axis=0)
            cv2.imwrite(os.path.join(concat_folder, ref_idx), concat[:, :, ::-1])

    e_time = time.time()
    print("This inference totally costs %.5f seconds" % (e_time - s_time))
