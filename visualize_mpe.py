# write a code that input a RGB image and a binary mask and store the masked image in a folder

import os
import cv2
import numpy as np

mask_path = './datasets/YouTubeVOS_small/valid/edge_50percent/mask_random/d1dd586cfd/00010.jpg'
# the input path is to substitude the mask_random to JPEGImages
input_path = './datasets/YouTubeVOS_small/valid/edge_50percent/JPEGImages/d1dd586cfd/00010.jpg'
# the edge path is to substitude the mask_random to edges_old
edge_path = './datasets/YouTubeVOS_small/valid/edge_50percent/edges_old/d1dd586cfd/00010.png'
# the line path is to substitude the mask_random to wireframes
line_path = './datasets/YouTubeVOS_small/valid/edge_50percent/wireframes/d1dd586cfd/00010.png'

def load_masked_position_encoding(mask):
    ones_filter = np.ones((3, 3), dtype=np.float32)
    d_filter1 = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=np.float32)
    d_filter2 = np.array([[0, 0, 0], [1, 1, 0], [1, 1, 0]], dtype=np.float32)
    d_filter3 = np.array([[0, 1, 1], [0, 1, 1], [0, 0, 0]], dtype=np.float32)
    d_filter4 = np.array([[0, 0, 0], [0, 1, 1], [0, 1, 1]], dtype=np.float32)
    pos_num = 128
    str_size = 256
        
    ori_mask = mask.copy()
    ori_h, ori_w = ori_mask.shape[0:2] # original size
    ori_mask = ori_mask / 255
    # mask = cv2.resize(mask, (self.str_size, self.str_size), interpolation=cv2.INTER_AREA)
    # mask = cv2.resize(mask, 256, interpolation=cv2.INTER_AREA)
    new_height = int(mask.shape[0] * (256.0 / mask.shape[1]))
    mask = cv2.resize(mask, (256, new_height), interpolation=cv2.INTER_AREA)

    mask[mask > 0] = 255 # make sure the mask is binary
    h, w = mask.shape[0:2] # resized size
    mask3 = mask.copy() 
    mask3 = 1. - (mask3 / 255.0) # 0 for masked area, 1 for unmasked area
    pos = np.zeros((h, w), dtype=np.int32) # position encoding
    direct = np.zeros((h, w, 4), dtype=np.int32) # direction encoding
    i = 0
    while np.sum(1 - mask3) > 0: # while there is still unmasked area
        i += 1 # i is the index of the current mask
        mask3_ = cv2.filter2D(mask3, -1, ones_filter) # dilate the mask
        mask3_[mask3_ > 0] = 1 # make sure the mask is binary
        sub_mask = mask3_ - mask3 # get the newly added area
        pos[sub_mask == 1] = i # set the position encoding

        m = cv2.filter2D(mask3, -1, d_filter1)
        m[m > 0] = 1
        m = m - mask3
        direct[m == 1, 0] = 1

        m = cv2.filter2D(mask3, -1, d_filter2)
        m[m > 0] = 1
        m = m - mask3
        direct[m == 1, 1] = 1

        m = cv2.filter2D(mask3, -1, d_filter3)
        m[m > 0] = 1
        m = m - mask3
        direct[m == 1, 2] = 1

        m = cv2.filter2D(mask3, -1, d_filter4)
        m[m > 0] = 1
        m = m - mask3
        direct[m == 1, 3] = 1

        mask3 = mask3_

    abs_pos = pos.copy() # absolute position encoding
    print(f"max pos: {np.max(abs_pos)}")
    print(f"min pos: {np.min(abs_pos)}")
    heatmap = cv2.applyColorMap(np.uint8(abs_pos)*10, cv2.COLORMAP_VIRIDIS)
    cv2.imwrite('mask3.jpg', heatmap)
    rel_pos = pos / (str_size / 2)  # to 0~1 maybe larger than 1
    rel_pos = (rel_pos * pos_num).astype(np.int32) # to 0~pos_num
    rel_pos = np.clip(rel_pos, 0, pos_num - 1) # clip to 0~pos_num-1

    if ori_w != w or ori_h != h: # if the mask is resized
        rel_pos = cv2.resize(rel_pos, (ori_w, ori_h), interpolation=cv2.INTER_NEAREST)
        rel_pos[ori_mask == 0] = 0
        direct = cv2.resize(direct, (ori_w, ori_h), interpolation=cv2.INTER_NEAREST)
        direct[ori_mask == 0, :] = 0

    return rel_pos, abs_pos, direct

# save the masked image and masked edge and masked line in the folder and name them as masked_00010.jpg
save_folder = './thesis_pic'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# read the mask
mask = cv2.imread(mask_path)
mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
mask = mask.astype(np.float32) / 255.0
mask = np.expand_dims(mask, axis=2)

# read the input
input = cv2.imread(input_path)
input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
input = input.astype(np.float32) / 255.0

# read the edge
edge = cv2.imread(edge_path)
edge = cv2.cvtColor(edge, cv2.COLOR_BGR2GRAY)
edge = edge.astype(np.float32) / 255.0
edge = np.expand_dims(edge, axis=2)

# read the line
line = cv2.imread(line_path)
line = cv2.cvtColor(line, cv2.COLOR_BGR2GRAY)
line = line.astype(np.float32) / 255.0
line = np.expand_dims(line, axis=2)

# reshape the mask, line, edge to the same size as input
mask = cv2.resize(mask, (input.shape[1], input.shape[0]))
edge = cv2.resize(edge, (input.shape[1], input.shape[0]))
line = cv2.resize(line, (input.shape[1], input.shape[0]))

# mask the input
# convert mask (720, 1280) to (720, 1280, 3) expand the channel
mask_3 = np.stack([mask, mask, mask], axis=-1)
masked_input = input * (1 - mask_3)

# mask the edge
masked_edge = edge * (1 - mask)

# mask the line
masked_line = line * (1 - mask)

# save the masked input
masked_input = cv2.cvtColor(masked_input, cv2.COLOR_RGB2BGR)
cv2.imwrite(os.path.join(save_folder, 'masked_input_00010.jpg'), masked_input*255)
cv2.imwrite(os.path.join(save_folder, 'masked_edge_00010.jpg'), masked_edge*255)
cv2.imwrite(os.path.join(save_folder, 'masked_line_00010.jpg'), masked_line*255)

# get the mask encoding
mask = cv2.imread(mask_path)
mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
print(f"mask shape: {mask.shape}")
rel_pos, abs_pos, direct = load_masked_position_encoding(mask)
print(rel_pos.shape)
print(abs_pos.shape)
print(direct.shape)