import argparse
import os
import time

import cv2
import numpy as np
import torch
from tqdm import tqdm

from datasets.dataset_TSR import ContinuousEdgeLineDatasetMask_video
from src.models.TSR_model import EdgeLineGPTConfig, EdgeLineGPT256RelBCE_video
from src.utils import set_seed, SampleEdgeLineLogits_video, SampleEdgeLineLogits

from torchvision import transforms
from src.utils import Stack, ToTorchFormatTensor
from PIL import Image

_to_tensors = transforms.Compose([
    Stack(),
    ToTorchFormatTensor()])

# read frame-wise masks 
def read_mask(mpath):
    masks = []
    mnames = os.listdir(mpath)
    mnames.sort()
    for m in mnames: 
        m = Image.open(os.path.join(mpath, m))
        m = m.resize((w, h), Image.NEAREST)
        m = np.array(m.convert('L'))
        m = np.array(m > 0).astype(np.uint8)
        m = cv2.dilate(m, cv2.getStructuringElement(
            cv2.MORPH_CROSS, (3, 3)), iterations=4)
        masks.append(Image.fromarray(m*255))
    return masks

def read_line(lpath):
    lines = []
    lnames = os.listdir(lpath)
    lnames.sort()
    for l in lnames:
        l = Image.open(os.path.join(lpath, l))
        l = l.resize((w, h), Image.NEAREST)
        l = np.array(l.convert('L'))
        lines.append(Image.fromarray((l*255).astype(np.uint8)))
    return lines

def read_edge(epath):
    edges = []
    enames = os.listdir(epath)
    enames.sort()
    for e in enames:
        e = Image.open(os.path.join(epath, e))
        e = e.resize((w, h), Image.NEAREST)
        e = np.array(e.convert('L'))
        edges.append(Image.fromarray((e*255).astype(np.uint8)))
    return edges

#  read frames from video 
def read_frame_from_videos(args):
    vname = args.video
    frames = []
    if args.use_mp4:
        vidcap = cv2.VideoCapture(vname)
        success, image = vidcap.read()
        count = 0
        while success:
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            frames.append(image.resize((w,h)))
            success, image = vidcap.read()
            count += 1
    else:
        lst = os.listdir(vname)
        lst.sort()
        fr_lst = [vname+'/'+name for name in lst]
        for fr in fr_lst:
            image = cv2.imread(fr)
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            frames.append(image.resize((w,h)))
    return frames, lst   

# sample reference frames from the whole video 
def get_ref_index(f, neighbor_ids, length):
    ref_index = []
    if num_ref == -1:
        for i in range(0, length, ref_length):
            if not i in neighbor_ids:
                ref_index.append(i)
    else:
        start_idx = max(0, f - ref_length * (num_ref//2))
        end_idx = min(length, f + ref_length * (num_ref//2))
        for i in range(start_idx, end_idx+1, ref_length):
            if not i in neighbor_ids:
                if len(ref_index) > num_ref:
                #if len(ref_index) >= 5-len(neighbor_ids):
                    break
                ref_index.append(i)
    return ref_index

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
    # fuseformer
    parser.add_argument("--model", type=str, default='fuseformer')
    parser.add_argument("-v", "--video", type=str, required=True)
    # parser.add_argument("-m", "--mask",   type=str, required=True)
    # parser.add_argument("-l", "--line",   type=str, required=True)
    # parser.add_argument("-e", "--edge", type=str, required=True)
    parser.add_argument("--width", type=int, default=432)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--outw", type=int, default=432)
    parser.add_argument("--outh", type=int, default=240)
    parser.add_argument("--step", type=int, default=10)
    parser.add_argument("--num_ref", type=int, default=-1)
    parser.add_argument("--neighbor_stride", type=int, default=5)
    parser.add_argument("--savefps", type=int, default=24)
    parser.add_argument("--use_mp4", action='store_true')

    args = parser.parse_args()

    w, h = args.width, args.height
    ref_length = args.step  # ref_step
    num_ref = args.num_ref
    neighbor_stride = args.neighbor_stride
    default_fps = args.savefps

    args.mask = args.video.replace('JPEGImages', 'mask_random')
    args.line = args.video.replace('JPEGImages', 'wireframes')
    args.edge = args.video.replace('JPEGImages', 'edges_old')
    video_name = args.video.split('/')[-1].split('.')[0]

    device = torch.device(f"cuda:{args.GPU_ids}" if torch.cuda.is_available() else "cpu")

    s_time = time.time()
    model_config = EdgeLineGPTConfig(embd_pdrop=0.0, resid_pdrop=0.0, n_embd=args.n_embd, block_size=32,
                                     attn_pdrop=0.0, n_layer=args.n_layer, n_head=args.n_head, ref_frame_num=args.ref_frame_num)
    IGPT_model = EdgeLineGPT256RelBCE_video(model_config, args, device=device)
    checkpoint = torch.load(args.ckpt_path)
    IGPT_model.load_state_dict(checkpoint if args.ckpt_path.endswith('.pt') else checkpoint['model'])
    IGPT_model.to("cuda")

    # items = test_dataset.__getitem__(it)
    edge_folder = os.path.join(args.save_url, "edges", video_name)
    line_folder = os.path.join(args.save_url, "lines", video_name)
    concat_folder = os.path.join(args.save_url, "concat", video_name)
    for folder in [edge_folder, line_folder, concat_folder]:
        os.makedirs(folder, exist_ok=True)
    
    # show the information of the args.video args.mask, args.line, args.edge with proper visualization
    print(f"video: {args.video}, mask: {args.mask}, line: {args.line}, edge: {args.edge}")
    # prepare datset, encode all frames into deep space 
    frames, frame_name = read_frame_from_videos(args)
    frames_all = [(np.array(f)).astype(np.uint8) for f in frames]
    video_length = len(frames)
    imgs = _to_tensors(frames).unsqueeze(0)*2-1
    # frames = [np.array(f).astype(np.uint8) for f in frames]

    masks = read_mask(args.mask)
    binary_masks = [np.expand_dims((np.array(m) != 0).astype(np.uint8), 2) for m in masks]
    masks = _to_tensors(masks).unsqueeze(0)

    lines = read_line(args.line)
    lines_all = [(np.expand_dims(np.array(l), 2)).astype(np.uint8) for l in lines]
    lines = _to_tensors(lines).unsqueeze(0)

    edges = read_edge(args.edge)
    edges_all = [(np.expand_dims(np.array(e), 2)).astype(np.uint8) for e in edges]
    edges = _to_tensors(edges).unsqueeze(0)

    print(f"Edge max: {np.max(edges_all[0])}") # test
    print(f"Edge min: {np.min(edges_all[0])}") # test

    # imgs, masks = imgs.to(device), masks.to(device)
    # lines, edges = lines.to(device), edges.to(device)
    
    # edge_pred, line_pred = SampleEdgeLineLogits_video(IGPT_model, context=[items['frames'].unsqueeze(0),
    #                                         items['edges'].unsqueeze(0), items['lines'].unsqueeze(0)],
    #                     masks=items['masks'].unsqueeze(0), iterations=args.iterations)
    comp_lines = [None]*video_length
    comp_edges = [None]*video_length

    # completing holes by spatial-temporal transformers
    # use tqdm to show the progress bar with the video name and current frame
    for f in tqdm(range(0, video_length, neighbor_stride)):
        neighbor_ids = [i for i in range(max(0, f-neighbor_stride), min(video_length, f+neighbor_stride+1))]
        ref_ids = get_ref_index(f, neighbor_ids, video_length)
        print(f, len(neighbor_ids), len(ref_ids))
        len_temp = len(neighbor_ids) + len(ref_ids)
        selected_imgs = imgs[:1, neighbor_ids+ref_ids, :, :, :]
        selected_masks = masks[:1, neighbor_ids+ref_ids, :, :, :]
        selected_edges = edges[:1, neighbor_ids+ref_ids, :, :, :]
        selected_lines = lines[:1, neighbor_ids+ref_ids, :, :, :]
        
        selected_imgs = selected_imgs.to(device)
        selected_masks = selected_masks.to(device)
        selected_edges = selected_edges.to(device)
        selected_lines = selected_lines.to(device)

        with torch.no_grad():
            masked_imgs = selected_imgs*(1-selected_masks)
            masked_edges = selected_edges*(1-selected_masks)
            masked_lines = selected_lines*(1-selected_masks)

            edge_pred, line_pred = SampleEdgeLineLogits_video(IGPT_model, context=[masked_imgs,
                                            masked_edges, masked_lines],
                        masks=selected_masks, iterations=args.iterations, add_v=0.05, mul_v=4)

            edge_pred = edge_pred.detach().to(torch.float32)
            line_pred = line_pred.detach().to(torch.float32)

            print(f"edge_pred: {edge_pred.shape}, line_pred: {line_pred.shape}")
            # print the max and min value of the edge_pred and line_pred
            print(f"edge_pred max, min: {torch.max(edge_pred)}, {torch.min(edge_pred)}")
            print(f"line_pred max, min: {torch.max(line_pred)}, {torch.min(line_pred)}")


            edge_pred = edge_pred.cpu().squeeze(0).permute(0, 2, 3, 1).numpy() * 255
            line_pred = line_pred.cpu().squeeze(0).permute(0, 2, 3, 1).numpy() * 255
            # # clear the masked_imgs, masked_edges, masked_lines from gpu memory
            del selected_imgs, selected_masks, selected_edges, selected_lines
            torch.cuda.empty_cache()

            # save the completed edges and lines
            # print(f"edge_pred: {edge_pred.shape}, line_pred: {line_pred.shape}")
            for i, e in enumerate(edge_pred):
                # save under the save_url/complete_edges/video_name
                cv2.imwrite(os.path.join(edge_folder, f"{frame_name[f]}"), e)

            for i, l in enumerate(line_pred):
                # save under the save_url/complete_lines/video_name
                cv2.imwrite(os.path.join(line_folder, f"{frame_name[f]}"), l)
            
            break
            for i in range(len(neighbor_ids)):
                idx = neighbor_ids[i]
                
                edge = np.array(edge_pred[i]).astype(
                    np.uint8)*binary_masks[idx] + edges_all[idx] * (1-binary_masks[idx])
                if comp_edges[idx] is None:
                    comp_edges[idx] = edge
                else:
                    comp_edges[idx] = comp_edges[idx].astype(
                        np.float32)*0.5 + edge.astype(np.float32)*0.5

                line = np.array(line_pred[i]).astype(
                    np.uint8)*binary_masks[idx] + lines_all[idx] * (1-binary_masks[idx])
                if comp_lines[idx] is None:
                    comp_lines[idx] = line
                else:
                    comp_lines[idx] = comp_lines[idx].astype(
                        np.float32)*0.5 + line.astype(np.float32)*0.5



    # for line_output, edge_output, img, mask, line, edge, fname in zip(comp_lines, comp_edges, frames_all, binary_masks, lines_all, edges_all, frame_name):
    #     # edge_output = edge_pred[0, i, ...].cpu() * mask + edge * (1 - mask)
    #     # edge_output = edge_output.repeat(3, 1, 1).permute(1, 2, 0)
    #     # line_output = line_pred[0, i, ...].cpu() * mask + line * (1 - mask)
    #     # line_output = line_output.repeat(3, 1, 1).permute(1, 2, 0)
    #     print(f"edge output: {edge_output.shape}")
    #     print(f"line output: {line_output.shape}")
    #     print(f"img: {img.shape}")
    #     print(f"mask: {mask.shape}")
    #     print(f"line: {line.shape}")

    #     edge_output = edge_output.astype(np.uint8)
    #     line_output = line_output.astype(np.uint8)

    #     cv2.imwrite(os.path.join(edge_folder, fname), edge_output[:, :, ::-1])
    #     cv2.imwrite(os.path.join(line_folder, fname), line_output[:, :, ::-1])

    #     # combine the result in one figure for better visualization
    #     # Get the original RGB image
    #     masked_image = img * (1-mask)
    #     # masked_image = ((masked_image + 1) * 0.5)
    #     # masked_image = (masked_image * 255).astype(np.uint8)
    #     # original_image = ((img + 1) * 0.5)
    #     # original_image = (original_image * 255).astype(np.uint8)
    #     original_image = img

    #     # Get the original image and invert the masked area
    #     original_edge = edge*(1-mask) + mask*(1-edge)
    #     original_line = line*(1-mask) + mask*(1-line)
    #     original_edge = original_edge.repeat(3, axis=2)
    #     original_line = original_line.repeat(3, axis=2)
    #     # original_edge = (original_edge * 255).astype(np.uint8)
    #     # original_line = (original_line * 255).astype(np.uint8)
    #     original_edge = (original_edge).astype(np.uint8)
    #     original_line = (original_line).astype(np.uint8)

    #     output_edge = (edge_output * (1-mask)) + ((1-edge_output) * mask)
    #     output_edge = output_edge.repeat(3, axis=2)
    #     # output_edge = (output_edge * 255).astype(np.uint8)
    #     output_edge = (output_edge).astype(np.uint8)
    #     output_line = (line_output * (1-mask)) + ((1-line_output) * mask)
    #     output_line = output_line.repeat(3, axis=2)
    #     output_line = (output_line).astype(np.uint8)
    #     # output_line = (output_line * 255).astype(np.uint8)
    #     line_concat = np.concatenate((original_image, original_line, output_line), axis=1)
    #     edge_concat = np.concatenate((masked_image, original_edge, output_edge), axis=1)
    #     concat = np.concatenate((line_concat, edge_concat), axis=0)
    #     cv2.imwrite(os.path.join(concat_folder, fname), concat)

    e_time = time.time()
    print("This inference totally costs %.5f seconds" % (e_time - s_time))