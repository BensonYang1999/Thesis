import os
from src.utils import read_mask, read_edge_line_PIL, read_frame_from_videos
import numpy as np
import torch
from src.models.TSR_model import EdgeLineGPTConfig, EdgeLineGPT256RelBCE_video
import torchvision.utils
import argparse
from torchvision import transforms
from src.utils import Stack, ToTorchFormatTensor, SampleEdgeLineLogits_video

to_tensors = transforms.Compose([
    Stack(),
    ToTorchFormatTensor(), ])


parser = argparse.ArgumentParser()
parser.add_argument('--path', '--checkpoints', type=str, default=None,
                    help='model checkpoints path (default: ./checkpoints)')
parser.add_argument('--config_file', type=str, default='./config_list/config_LAMA_MPE_HR.yml',
                    help='The config file of each experiment ')
parser.add_argument('--nodes', type=int, default=1, help='how many machines')
parser.add_argument('--gpus', type=int, default=1, help='how many GPUs in one node')
parser.add_argument('--GPU_ids', type=str, default='0')
parser.add_argument('--node_rank', type=int, default=0, help='the id of this machine')
parser.add_argument('--DDP', action='store_true', help='DDP')
parser.add_argument('--lama', action='store_true', help='train the lama first')
parser.add_argument('--n_layer', type=int, default=16)
parser.add_argument('--n_embd', type=int, default=256)
parser.add_argument('--n_head', type=int, default=8)
# parser.add_argument('--loss_item', type=str, nargs='+', default=["hole", "valid"], help='the id of this machine')
parser.add_argument('--loss_hole_valid_weight', type=float, nargs='+', default=[0.8, 0.2], help='the weight for computing the hole/valid part ')
parser.add_argument('--loss_edge_line_weight', type=float, nargs='+', default=[1.0, 1.0], help='the weight for computing the edge/line part ')
# add the choice to decide the loss function with l1 or mse or binary cross entropy with choice
parser.add_argument('--loss_choice', type=str, default="bce", help='the choice of loss function: l1, mse, bce')
# parser.add_argument('--edge_gaussian', type=int, default=0, help='the sigma of gaussian kernel for edge')

args = parser.parse_args()
# config_path = os.path.join(args.path, 'config.yml')

args.config_path = "./config_list/config_ZITS_video.yml"

gpu = "cpu"

model_config = EdgeLineGPTConfig(embd_pdrop=0.0, resid_pdrop=0.0, n_embd=args.n_embd, block_size=32,attn_pdrop=0.0, n_layer=args.n_layer, n_head=args.n_head, ref_frame_num=5) # video version
                                     
transformer = EdgeLineGPT256RelBCE_video(model_config, args, device=gpu)


width = 240
height = 432

# frame_dir = "./datasets/YouTubeVOS_small/test/set_1/line_25percent/JPEGImages/4445dc0af5"
# mask_dir = "./datasets/YouTubeVOS_small/test/set_1/line_25percent/mask_random/4445dc0af5"
# edge_dir = "./datasets/YouTubeVOS_small/test/set_1/line_25percent/edges/4445dc0af5"
# line_dir = "./datasets/YouTubeVOS_small/test/set_1/line_25percent/wireframes/4445dc0af5"
frame_dir = "./datasets/YouTubeVOS_small/test/set_1/line_25percent/JPEGImages/cc6c653874"
mask_dir = "./datasets/YouTubeVOS_small/test/set_1/line_25percent/mask_random/cc6c653874"
edge_dir = "./datasets/YouTubeVOS_small/test/set_1/line_25percent/edges/cc6c653874"
line_dir = "./datasets/YouTubeVOS_small/test/set_1/line_25percent/wireframes/cc6c653874"

frames_PIL, idx_lst = read_frame_from_videos(frame_dir, width, height) # read frames from video

imgs = to_tensors(frames_PIL).unsqueeze(0)*2-1 # convert frames to tensors and normalize to [-1, 1]
masks = read_mask(mask_dir, width, height) 
binary_masks = [np.expand_dims((np.array(m) != 0).astype(np.uint8), 2) for m in masks] # convert masks to numpy array
masks = to_tensors(masks).unsqueeze(0) # convert masks to tensors

edges, lines = read_edge_line_PIL(edge_dir, line_dir, width, height)
edges = to_tensors(edges).unsqueeze(0)
lines = to_tensors(lines).unsqueeze(0)

selected_imgs = imgs[:1, :5, :, :, :] # select frames for inpainting with neighbor frames and reference frames
selected_masks = masks[:1, :5, :, :, :] # select masks for inpainting 
# print(f"selected_masks: {selected_masks[0][0]}") # test
selected_edges = edges[:1, :5, :, :, :]
selected_lines = lines[:1, :5, :, :, :]

# print the value range of selected_mask
print(f"mask max: {masks.max()}")
print(f"mask min: {masks.min()}")
print(f"selected_masks max: {selected_masks.max()}")
print(f"selected_masks min: {selected_masks.min()}")

edge_pred, line_pred = SampleEdgeLineLogits_video(transformer,
                    context=[selected_imgs, selected_edges, selected_lines], 
                    # masks=selected_masks.clone(), iterations=5, add_v=0.05, mul_v=4, device=gpu)  
                    masks=selected_masks.clone(), iterations=1, add_v=0.05, mul_v=16, device=gpu)  
# edge_pred, line_pred = SampleEdgeLineLogits_video(transformer,
#                     context=[selected_imgs.to(torch.float16), selected_edges.to(torch.float16), selected_lines.to(torch.float16)], 
#                     masks=selected_masks.to(torch.float16), iterations=5, add_v=0.05, mul_v=4, device=gpu)  
edge_pred, line_pred = edge_pred.detach().to(torch.float32), line_pred.detach().to(torch.float32)

edge_pred, line_pred = edge_pred - selected_masks, line_pred - selected_masks # test

# selected_edges = selected_edges * (1 - selected_masks) + selected_masks
# selected_lines = selected_lines * (1 - selected_masks) + selected_masks

selected_edges = edge_pred * selected_masks + selected_edges * (1 - selected_masks)  # new version after 0818
selected_lines = line_pred * selected_masks + selected_lines * (1 - selected_masks)  # new version after 0818



# save the inpainted edges and lines under the folder "0922_check_sample_line_edge"
save_dir = "./0922_check_sample_line_edge"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
for i in range(5):
    # save 
    torchvision.utils.save_image(selected_edges[0][i], os.path.join(save_dir, f"edge_{i}.png"))
    torchvision.utils.save_image(selected_lines[0][i], os.path.join(save_dir, f"line_{i}.png"))
    # save the mask
    torchvision.utils.save_image(selected_masks[0][i], os.path.join(save_dir, f"mask_{i}.png"))

