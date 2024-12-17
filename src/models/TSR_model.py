import logging
from logging import config

import torch
import torch.nn as nn
from torch.nn import functional as F

from .transformer import BlockAxial, my_Block_2

"""
FuseFormer imports
"""
import numpy as np
import time
import math
from functools import reduce
import torchvision.models as models
from torch.nn import functional as nnF

from src.models.fuseformer import InpaintGenerator
import torchvision.transforms as T

logger = logging.getLogger(__name__)


class EdgeLineGPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
            # embd_pdrop=0.0, resid_pdrop=0.0, n_embd=opts.n_embd, block_size=32, attn_pdrop=0.0, n_layer=opts.n_layer, n_head=opts.n_head


class EdgeLineGPT256RelBCE(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()

        self.pad1 = nn.ReflectionPad2d(3) # square -> bigger square (extend 3 for each side)
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=64, kernel_size=7, padding=0) # downsample input
        self.act = nn.ReLU(True)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)  # downsample 1

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1) # downsample 2

        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1) # downsample 3

        self.pos_emb = nn.Parameter(torch.zeros(1, 1024, 256))  # special tensor that automatically add into parameter list
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer, input: 32*32*config.n_embd
        self.blocks = []
        for _ in range(config.n_layer // 2):
            self.blocks.append(BlockAxial(config))
            self.blocks.append(my_Block_2(config))
        self.blocks = nn.Sequential(*self.blocks)
        # decoder, input: 32*32*config.n_embd
        self.ln_f = nn.LayerNorm(256)

        self.convt1 = nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1) # upsample 1

        self.convt2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1) # upsample 2

        self.convt3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1) # upsample 3

        self.padt = nn.ReflectionPad2d(3)
        self.convt4 = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=7, padding=0) # upsample and ouput only edge/line

        self.act_last = nn.Sigmoid()

        self.block_size = 32  # transformer input size
        self.config = config

        self.apply(self._init_weights)  # initialize the weights (multiple layer initialization)

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv2d, nn.ConvTranspose2d)):  # if one of the type
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()  
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):  # some parameter need weight decay to avoid overfitting
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d, torch.nn.ConvTranspose2d) # need weight decay
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')
        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert len(
            param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params),)
        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def forward(self, img_idx, edge_idx, line_idx, edge_targets=None, line_targets=None, masks=None):
        img_idx = img_idx * (1 - masks)  # create masked image
        edge_idx = edge_idx * (1 - masks) # create masked edge
        line_idx = line_idx * (1 - masks) # create masked line
        x = torch.cat((img_idx, edge_idx, line_idx, masks), dim=1)  # concat method NEED checking (maybe is channel-wise)
        x = self.pad1(x)  # reflection padding
        x = self.conv1(x)  # downsample input layer
        x = self.act(x)  # activate with ReLU

        x = self.conv2(x)  # downsample 1 
        x = self.act(x)

        x = self.conv3(x)  # downsample 2 
        x = self.act(x)

        x = self.conv4(x)  # downsample 3 
        x = self.act(x)

        [b, c, h, w] = x.shape  # before here, the image data is stil with Height x Width
        x = x.view(b, c, h * w).transpose(1, 2).contiguous() # image 2D -> 1D (flatten) and change image and color channel
        # make the data into shape like -> [batch size, image(1D), channels(RGB, edge, line, mask)]

        position_embeddings = self.pos_emb[:, :h * w, :]  # each position maps to a (learnable) vector
        x = self.drop(x + position_embeddings)  # [b,hw,c]  # add positional embeddings, but dropping to make some position missing pos-emb
        x = x.permute(0, 2, 1).reshape(b, c, h, w)  # swap the image and channel back to [b, c, h*w] then reshape to [b,c,h,w]

        # Transformer Input: [b,c,h,w]
        x = self.blocks(x)
        x = x.permute(0, 2, 3, 1)  # swap to [b, h, w, c]
        x = self.ln_f(x).permute(0, 3, 1, 2).contiguous()  # layer norm then swap back (以batch中的instance為單位normalize)

        x = self.convt1(x) # upsample 1
        x = self.act(x)

        x = self.convt2(x) # upsample 2
        x = self.act(x)

        x = self.convt3(x) # upsample 3
        x = self.act(x)

        x = self.padt(x)  # padding back
        x = self.convt4(x)  # upsample output as the original image shape

        edge, line = torch.split(x, [1, 1], dim=1)  # seperate the TSR outputs

        if edge_targets is not None and line_targets is not None:
            loss = F.binary_cross_entropy_with_logits(edge.permute(0, 2, 3, 1).contiguous().view(-1, 1),
                                                      edge_targets.permute(0, 2, 3, 1).contiguous().view(-1, 1),
                                                      reduction='none')
            loss = loss + F.binary_cross_entropy_with_logits(line.permute(0, 2, 3, 1).contiguous().view(-1, 1),
                                                             line_targets.permute(0, 2, 3, 1).contiguous().view(-1, 1),
                                                             reduction='none')
            masks_ = masks.view(-1, 1)  # only compute the loss in the masked region

            loss *= masks_
            loss = torch.mean(loss)
        else:
            loss = 0
        edge, line = self.act_last(edge), self.act_last(line)  # sigmoid activate
        return edge, line, loss  # edge/line is in shape [b, c, h, w]

    def forward_with_logits(self, img_idx, edge_idx, line_idx, masks=None):  # for inference, no loss computing
        img_idx = img_idx * (1 - masks)
        edge_idx = edge_idx * (1 - masks)
        line_idx = line_idx * (1 - masks)
        x = torch.cat((img_idx, edge_idx, line_idx, masks), dim=1)
        x = self.pad1(x)
        x = self.conv1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.act(x)

        x = self.conv3(x)
        x = self.act(x)

        x = self.conv4(x)
        x = self.act(x)

        [b, c, h, w] = x.shape
        x = x.view(b, c, h * w).transpose(1, 2).contiguous()

        position_embeddings = self.pos_emb[:, :h * w, :]  # each position maps to a (learnable) vector
        x = self.drop(x + position_embeddings)  # [b,hw,c]
        x = x.permute(0, 2, 1).reshape(b, c, h, w)

        x = self.blocks(x)
        x = x.permute(0, 2, 3, 1)
        x = self.ln_f(x).permute(0, 3, 1, 2).contiguous()

        x = self.convt1(x)
        x = self.act(x)

        x = self.convt2(x)
        x = self.act(x)

        x = self.convt3(x)
        x = self.act(x)

        x = self.padt(x)
        x = self.convt4(x)

        edge, line = torch.split(x, [1, 1], dim=1)

        return edge, line
    
class EdgeLineGPT256RelBCE_video(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config, opts, device):
        super().__init__()

        '''
        self.pad1 = nn.ReflectionPad2d(3) # square -> bigger square (extend 3 for each side)
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=64, kernel_size=7, padding=0) # downsample input
        self.act = nn.ReLU(True)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)  # downsample 1

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1) # downsample 2

        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1) # downsample 3
        '''
        self.pos_emb = nn.Parameter(torch.zeros(1, 1024, 256))  # special tensor that automatically add into parameter list
        '''
        self.drop = nn.Dropout(config.embd_pdrop)

        # decoder, input: 32*32*config.n_embd
        self.ln_f = nn.LayerNorm(256)

        self.convt1 = nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1) # upsample 1

        self.convt2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1) # upsample 2

        self.convt3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1) # upsample 3

        self.padt = nn.ReflectionPad2d(3)
        self.convt4 = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=7, padding=0) # upsample and ouput only edge/line

        '''
        self.act_last = nn.Sigmoid()  # original in ZITS
        # self.act_last = nn.LeakyReLU()  # original in For ZITS_video

        '''
        self.loss_functions = {"l1": nn.L1Loss(), "bce": nn.BCELoss(), "mse": nn.MSELoss()}
        self.loss_function = self.loss_functions[opts.loss_choice]

        # Feature Fusion (6 channels to 3 channels)
        self.fuse_channel = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=3, padding='same'),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding='same'),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 16, kernel_size=3, padding='same'),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 3, kernel_size=3, padding='same'),
        )
        '''

        # self.fuseformerBlock = InpaintGenerator(input_ch=6, ref_frames=5)
        self.fuseformerBlock = InpaintGenerator(input_ch=4, out_ch=2, ref_frames=5)
        self.fuseformerBlock = self.fuseformerBlock.to(device)

        '''
        self.fuseFramesToLineEdge = nn.Sequential(
            nn.Conv3d(config.ref_frame_num, 1, (config.ref_frame_num, 3, 3),stride=1, padding='same'), # transfer fuseformer output to edge/line
        )
        '''

        self.l1_loss = nn.L1Loss()

        self.config = config
        self.opts = opts

        self.apply(self._init_weights)  # initialize the weights (multiple layer initialization)

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def get_block_size(self):
        return self.block_size
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv2d, nn.ConvTranspose2d)):  # if one of the type
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()  
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):  # some parameter need weight decay to avoid overfitting
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        # whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d, torch.nn.ConvTranspose2d) # need weight decay
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d, torch.nn.ConvTranspose2d, torch.nn.Conv3d, torch.nn.MultiheadAttention) # need weight decay
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')
        no_decay.add('fuseformerBlock.add_pos_emb.pos_emb')
        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert len(
            param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params),)
        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def forward(self, img_idx, edge_idx, line_idx, edge_targets=None, line_targets=None, masks=None):
        img_idx = img_idx * (1 - masks)  # create masked image
        edge_idx = edge_idx * (1 - masks) # create masked edge
        line_idx = line_idx * (1 - masks) # create masked line

        # for b in range(img_idx.shape[0]):
        #     print(f"Process batch: {b}...")
        #     for t in range(img_idx.shape[1]):
        #         save_image(img_idx[b][t], f'tensor_img_b{b}_t{t}.png')

        # [b, t, c, w, h]
        x = torch.cat((img_idx, edge_idx, line_idx, masks), dim=2)  # concat method NEED checking (maybe is channel-wise)

        # Encoder: downsample
        # x = self.pad1(x)  # reflection padding
        # x = self.conv1(x)  # downsample input layer
        # x = self.act(x)  # activate with ReLU

        # x = self.conv2(x)  # downsample 1 
        # x = self.act(x)

        # x = self.conv3(x)  # downsample 2 
        # x = self.act(x)

        # x = self.conv4(x)  # downsample 3 
        # x = self.act(x)

        [b, t, c, h, w] = x.shape  # before here, the video data is still with Height x Width -> [50, 256, 32, 32] -> [t, c, h, w]
        # print(f"shape after concat: {x.shape}")  # test
        # x = x.view(t, c, h * w).transpose(1, 2).contiguous() # image 2D -> 1D (flatten) and change image and color channel
        # make the data into shape like -> [batch size, image(1D), channels(RGB, edge, line, mask)]


        # Transformer blocks
        # input [50, 256, 32, 32] -> original ZITS
        # input [1, 5, 3, 240, 432] -> original fuseformer
        # print(f"shape before FuseFormer: {x.shape}")  # test
        # add the learned resize layer for x
        
        x = self.fuseformerBlock(x)
        # print(f"shape after FuseFormer: {x.shape}")  # test

        # Fuseformer feature -> decode as line edge
        # _, c_, h_, w_ = x.size()
        # x = x.view(b, t, c_, h_, w_)  # test
        # x = self.fuseFramesToLineEdge(x)
        # x = torch.squeeze(x)

        
        # Decoder: upsample
        # x = self.convt1(x) # upsample 1
        # x = self.act(x)

        # x = self.convt2(x) # upsample 2
        # x = self.act(x)

        # x = self.convt3(x) # upsample 3
        # x = self.act(x)

        # x = self.padt(x)  # padding back
        # x = self.convt4(x)  # upsample output as the original image shape
        
        edge, line = torch.split(x, [1, 1], dim=1)  # seperate the TSR outputs

        eps = 1e-8
        loss, edge_hole_loss, edge_valid_loss, line_hole_loss, line_valid_loss = 0., 0., 0., 0., 0.
        
        # Loss computing
        if edge_targets is not None and line_targets is not None:
            edge_targets = edge_targets.view(b * t, 1, h, w)
            line_targets = line_targets.view(b * t, 1, h, w)
            masks = masks.view(b * t, 1, h, w)

            # print(f"edge_targets: {edge_targets}")
            # print(f"edge: {edge}")

            # hole loss
            # hole_weight, valid_weight = self.opts.loss_hole_valid_weight
            # edge_weight, line_weight = self.opts.loss_edge_line_weight

            # masks_mean = torch.mean(masks)
            # one_minus_masks_mean = torch.mean(1 - masks)

            # edge_hole_loss = self.loss_function(edge * masks, edge_targets * masks)
            # edge_hole_loss = edge_hole_loss / (masks_mean + eps)

            # line_hole_loss = self.loss_function(line * masks, line_targets * masks)
            # line_hole_loss = line_hole_loss / (masks_mean + eps)

            # # total loss
            # loss += (edge_hole_loss * edge_weight + line_hole_loss * line_weight) * hole_weight

            # # valid loss
            # edge_valid_loss = self.loss_function(edge * (1 - masks), edge_targets * (1 - masks))
            # edge_valid_loss = edge_valid_loss / (one_minus_masks_mean + eps)

            # line_valid_loss = self.loss_function(line * (1 - masks), line_targets * (1 - masks))
            # line_valid_loss = line_valid_loss / (one_minus_masks_mean + eps)

            # # total loss
            # loss += (edge_valid_loss * edge_weight + line_valid_loss * line_weight) * valid_weight

            
            # ZITS loss computation
            # edge loss
            loss_edge = nnF.binary_cross_entropy_with_logits(edge.permute(0, 2, 3, 1).contiguous().view(-1, 1),
                                                      edge_targets.permute(0, 2, 3, 1).contiguous().view(-1, 1),
                                                      reduction='none')
            # line loss
            loss_line = nnF.binary_cross_entropy_with_logits(line.permute(0, 2, 3, 1).contiguous().view(-1, 1),
                                                             line_targets.permute(0, 2, 3, 1).contiguous().view(-1, 1),
                                                             reduction='none')
            

            masks_ = masks.permute(0, 2, 3, 1).contiguous().view(-1, 1) # only compute the loss in the masked region
            loss = ((loss_edge+loss_line) * masks_)*self.opts.loss_hole_valid_weight[0] + \
                   ((loss_edge+loss_line) * (1-masks_))*self.opts.loss_hole_valid_weight[1]
            # loss = (loss_edge*4+loss_line) * masks_
            # print(f"loss shape: {loss.size()}") # test
            loss = torch.mean(loss)
            
            edge_hole_loss = (loss_edge*masks_)*self.opts.loss_hole_valid_weight[0]
            edge_valid_loss = (loss_edge*(1-masks_))*self.opts.loss_hole_valid_weight[1]
            line_hole_loss = (loss_line*masks_)*self.opts.loss_hole_valid_weight[0]
            line_valid_loss = (loss_line*(1-masks_))*self.opts.loss_hole_valid_weight[1]
        else:
            loss = 0

        edge, line = edge.view(b, t, 1, h, w), line.view(b, t, 1, h, w)
        edge, line = self.act_last(edge), self.act_last(line)  # sigmoid activate 

        loss_detail = [edge_hole_loss, edge_valid_loss, line_hole_loss, line_valid_loss]

        return edge, line, loss, loss_detail
    
    def forward_with_logits(self, img_idx, edge_idx, line_idx, masks=None):
        # if the shape of input is not [b, t, c, w, h], then add a dimension
        if len(img_idx.shape) != 5:
            img_idx = img_idx.unsqueeze(0)
        if len(edge_idx.shape) != 5:
            edge_idx = edge_idx.unsqueeze(0)
        if len(line_idx.shape) != 5:
            line_idx = line_idx.unsqueeze(0)
        if masks is not None and len(masks.shape) != 5:
            masks = masks.unsqueeze(0)

        img_idx = img_idx * (1 - masks)  # create masked image
        edge_idx = edge_idx * (1 - masks) # create masked edge
        line_idx = line_idx * (1 - masks) # create masked line

        # [b, t, c, w, h]
        x = torch.cat((img_idx, edge_idx, line_idx, masks), dim=2)  # concat method NEED checking (maybe is channel-wise)

        [b, t, c, h, w] = x.shape  # before here, the video data is still with Height x Width -> [50, 256, 32, 32] -> [t, c, h, w]

        # Transformer blocks
        x = self.fuseformerBlock(x)
        
        edge, line = torch.split(x, [1, 1], dim=1)  # seperate the TSR outputs
        # edge, line = edge.squeeze(0), line.squeeze(0)
        edge, line = edge.view(b, t, 1, h, w), line.view(b, t, 1, h, w)

        return edge, line
    
# -------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=(3, 3), stride=stride, padding='same')
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=stride, padding='same')
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.bn1(self.conv1(x))
        out = self.relu(out)
        out = self.bn2(self.conv2(out))
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

from einops.layers.torch import Rearrange
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, emb_size, img_size):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        # self.projection = nn.Sequential(
        #     nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
        #     Rearrange('b e (h) (w) -> b (h w) e'),
        #     # Rearrange('b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=patch_size, s2=patch_size),
        # )
        self.projection = nn.Unfold(kernel_size=(patch_size, patch_size), stride=(patch_size, patch_size))
        c_in = in_channels * patch_size * patch_size
        self.embedding = nn.Linear(c_in, emb_size)
        self.dropout = nn.Dropout(0.1)

        num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
        self.positions = nn.Parameter(torch.randn(1, num_patches, emb_size))

    def forward(self, x):
        x = self.projection(x)
        x = x.permute(0, 2, 1)
        x = self.embedding(x)
        x = self.dropout(x)
        x += self.positions
        return x

class PatchDecoder(nn.Module):
    def __init__(self, out_channels, patch_size, emb_size, img_size):
        super(PatchDecoder, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        # self.deprojection = nn.Sequential(
        #     Rearrange('b (h w) e -> b e (h) (w)', h=img_size[0]//patch_size, w=img_size[1]//patch_size),
        #     nn.ConvTranspose2d(emb_size, out_channels, kernel_size=patch_size, stride=patch_size)
        # )
        c_out = out_channels * patch_size * patch_size
        self.embedding = nn.Linear(emb_size, c_out)
        self.projection = nn.Fold(output_size=img_size, kernel_size=(patch_size, patch_size), stride=(patch_size, patch_size))

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x = self.projection(x)
        return x

class EdgeLine_CNN(nn.Module):
    def __init__(self):
        super(EdgeLine_CNN, self).__init__()
        # actication function
        self.act = nn.ReLU(True)
        self.act_last = nn.Sigmoid()

        # 3D convolutional layers (Encoder)
        # self.conv1 = nn.Conv3d(4, 64, kernel_size=(3, 7, 7), padding=(1, 0, 0))
        # self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 4, 4))
        # self.conv3 = nn.Conv3d(128, 256, kernel_size=(3, 4, 4))

        # self.conv1 = nn.Conv3d(4, 32, kernel_size=(3, 7, 7), stride=(1, 4, 4), padding=(1, 2, 2))
        # self.conv2 = nn.Conv3d(32, 64, kernel_size=(3, 4, 4), stride=(1, 3, 3), padding=(0, 1, 1))
        # self.conv3 = nn.Conv3d(64, 128, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1))

        self.conv1 = nn.Conv3d(4, 64, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1))
        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
        self.conv3 = nn.Conv3d(128, 256, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1))

        # dilated 2D convolutional layers (resnet block)
        # resnet_block = []
        # for _ in range(8):
        #     resnet_block.append(nn.Conv2d(256, 256, kernel_size=(3, 3), padding='same', dilation=2))
        #     resnet_block.append(nn.BatchNorm2d(256))
        # self.resnet = nn.Sequential(*resnet_block)
        resnet_block = []
        for _ in range(4):
            resnet_block.append(ResidualBlock(256, 256))
        self.resnet = nn.Sequential(*resnet_block)

        # transformer
        # self.patch_embed = PatchEmbedding(256, 16, 768, (60, 108))
        # encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=4)
        # self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=8)
        # self.patch_decoder = PatchDecoder(256, 16, 768, (60, 108))

        # 3D transposed convolutional layers (Decoder)
        # self.convt1 = nn.ConvTranspose3d(256, 128, kernel_size=(3, 4, 4))
        # self.convt2 = nn.ConvTranspose3d(128, 64, kernel_size=(3, 4, 4))
        # self.convt3 = nn.ConvTranspose3d(64, 2, kernel_size=(3, 7, 7), padding=(1, 0, 0))

        # self.convt1 = nn.ConvTranspose3d(128, 64, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1))
        # self.convt2 = nn.ConvTranspose3d(64, 32, kernel_size=(3, 3, 3), stride=(1, 3, 3), padding=(0, 0, 0))
        # self.convt3 = nn.ConvTranspose3d(32, 2, kernel_size=(3, 6, 6), stride=(1, 4, 4), padding=(1, 1, 1))

        self.convt1 = nn.ConvTranspose3d(256, 128, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1))
        self.convt2 = nn.ConvTranspose3d(128, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
        self.convt3 = nn.ConvTranspose3d(64, 2, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1))
    
    def configure_optimizers(self, train_config):
        # optimizer = torch.optim.Adam(self.parameters(), lr=train_config.learning_rate, betas=train_config.betas)
        optimizer = torch.optim.Adam(self.parameters(), lr=train_config.learning_rate)
        return optimizer

    def forward(self, img_idx, edge_idx, line_idx, edge_targets=None, line_targets=None, masks=None):
        img_idx = img_idx * (1 - masks)  # create masked image
        edge_idx = edge_idx * (1 - masks) # create masked edge
        line_idx = line_idx * (1 - masks) # create masked line
        x = torch.cat((img_idx, edge_idx, line_idx, masks), dim=2) # [b, t, c, h, w]
        x = x.permute(0, 2, 1, 3, 4) # [b, c, t, h, w]
        x_shape = x.size()

        # Encoder: downsample
        x = self.conv1(x) # batch*4*5*240*432 -> batch*64*5*234*426
        x = self.act(x)
        x = self.conv2(x) # batch*64*5*234*426 -> batch*128*3*231*423
        x = self.act(x)
        x = self.conv3(x) # batch*128*3*231*423 -> batch*256*1*228*420
        x = self.act(x)

        # solution 1: resnet block
        # Resnet block
        b, c, t, h, w = x.size()
        x = x.view(b, c*t, h, w) # [b, c*t, h, w]
        x = self.resnet(x)
        x = x.view(b, c, t, h, w)

        # # solution 2: vision transformer
        # x = torch.squeeze(x, 2)  # [b, c, t, h, w] -> [b, c, h, w]
        # enc_feat = x
        # x = self.patch_embed(x)
        # x = self.transformer(x)
        # x = self.patch_decoder(x)
        # x = x + enc_feat
        # x = torch.unsqueeze(x, 2)  # [b, c, h, w] -> [b, c, t, h, w]


        # Decoder: upsample
        x = self.convt1(x)
        x = self.act(x)
        x = self.convt2(x)
        x = self.act(x)
        x = self.convt3(x)

        x = x.permute(0, 2, 1, 3, 4) # [b, t, c, h, w]
        edge, line = torch.split(x, [1, 1], dim=2) # [b, t, 1, h, w]

        # loss computing
        # loss, loss_edge_hole, loss_edge_valid, loss_line_hole, loss_line_valid = 0., 0., 0., 0., 0.
        # if edge_targets is not None and line_targets is not None:
        #     # [b, 1, t, h, w] -> [b, t, 1, h, w] -> [b*t, 1, h, w]
        #     edge = edge.permute(0, 2, 1, 3, 4).contiguous().view(x_shape[0]*x_shape[2], 1, x_shape[3], x_shape[4])
        #     line = line.permute(0, 2, 1, 3, 4).contiguous().view(x_shape[0]*x_shape[2], 1, x_shape[3], x_shape[4])
        #     # [b, t, 1, h, w] -> [b*t, 1, h, w]
        #     edge_targets = edge_targets.view(x_shape[0]*x_shape[2], 1, x_shape[3], x_shape[4])
        #     line_targets = line_targets.view(x_shape[0]*x_shape[2], 1, x_shape[3], x_shape[4])

        #     loss_edge = F.binary_cross_entropy_with_logits(edge.permute(0, 2, 3, 1).contiguous().view(-1, 1),
        #                                               edge_targets.permute(0, 2, 3, 1).contiguous().view(-1, 1),
        #                                               reduction='none')
        #     loss_line = F.binary_cross_entropy_with_logits(line.permute(0, 2, 3, 1).contiguous().view(-1, 1),
        #                                                line_targets.permute(0, 2, 3, 1).contiguous().view(-1, 1),
        #                                                reduction='none')
        #     masks_ = masks.view(x_shape[0]*x_shape[2], 1, x_shape[3], x_shape[4]) # [b*t, 1, h, w]
        #     masks_ = masks_.permute(0, 2, 3, 1).contiguous().view(-1, 1)
        #     loss = (loss_edge+loss_line) * masks_
        #     # loss *= masks_
        #     loss = torch.mean(loss)

        #     loss_edge_hole = loss_edge * masks_
        #     loss_edge_valid = loss_edge * 0
        #     loss_line_hole = loss_line * masks_
        #     loss_line_valid = loss_line * 0
        # # else:
        # #     loss = 0

        total_loss, loss, loss_edge_hole, loss_edge_valid, loss_line_hole, loss_line_valid = 0., 0., 0., 0., 0., 0.

        if edge_targets is not None and line_targets is not None:
            masks_ = masks.view(x_shape[0], x_shape[2], 1, x_shape[3], x_shape[4]) # [b, t, 1, h, w]
            edge_sig = torch.sigmoid(edge)
            line_sig = torch.sigmoid(line)
            
            # BCE loss
            # criterion = nn.BCEWithLogitsLoss(reduction='none')
            # loss_edge = criterion(edge, edge_targets)
            # loss_line = criterion(line, line_targets)
            criterion = nn.BCELoss(reduction='none')
            loss_edge = criterion(edge_sig, edge_targets)
            loss_line = criterion(line_sig, line_targets)

            loss = (loss_edge + loss_line) * masks_
            # loss = loss_edge + loss_line
            loss = torch.mean(loss)
            total_loss += loss

            loss_edge_hole = loss_edge * masks_
            loss_edge_valid = loss_edge * (1-masks_)
            loss_line_hole = loss_line * masks_
            loss_line_valid = loss_line * (1-masks_)


            # MSE loss
            # criterion = nn.MSELoss(reduction='none')
            # loss_edge = criterion(edge_sig, edge_targets)
            # loss_line = criterion(line_sig, line_targets)

            # loss = (loss_edge + loss_line) * masks_
            # loss = torch.mean(loss)
            # total_loss = loss

            # loss_edge_hole += loss_edge * masks_
            # loss_edge_valid += loss_edge * (1-masks_)
            # loss_line_hole += loss_line * masks_
            # loss_line_valid += loss_line * (1-masks_)
        
        # edge, line = edge.view(x_shape[0], x_shape[2], 1, x_shape[3], x_shape[4]), line.view(x_shape[0], x_shape[2], 1, x_shape[3], x_shape[4])
        # edge, line = self.act_last(edge), self.act_last(line)  # sigmoid activate

        loss_detail = [loss_edge_hole, loss_edge_valid, loss_line_hole, loss_line_valid]
        # return self.act_last(edge), self.act_last(line), loss, loss_detail
        return self.act_last(edge), self.act_last(line), total_loss, loss_detail
    
    def forward_with_logits(self, img_idx, edge_idx, line_idx, masks):  # for inference, no loss computing
        img_idx = img_idx * (1 - masks)
        edge_idx = edge_idx * (1 - masks)
        line_idx = line_idx * (1 - masks)
        x = torch.cat((img_idx, edge_idx, line_idx, masks), dim=2)
        x = x.permute(0, 2, 1, 3, 4)
        x_shape = x.size()

        # Encoder: downsample
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.conv3(x)
        x = self.act(x)

        # Resnet block
        b, c, t, h, w = x.size()
        x = x.view(b, c*t, h, w)
        x = self.resnet(x)
        x = x.view(b, c, t, h, w)
        
        # # transformer block
        # x = torch.squeeze(x, 2)  # [b, c, t, h, w] -> [b, c, h, w]
        # enc_feat = x
        # x = self.patch_embed(x)
        # x = self.transformer(x)
        # x = self.patch_decoder(x)
        # x = x + enc_feat
        # x = torch.unsqueeze(x, 2)  # [b, c, h, w] -> [b, c, t, h, w]

        # Decoder: upsample
        x = self.convt1(x)
        x = self.act(x)
        x = self.convt2(x)
        x = self.act(x)
        x = self.convt3(x)
        
        x = x.permute(0, 2, 1, 3, 4)
        edge, line = torch.split(x, [1, 1], dim=2) # [b, t, 1, h, w]
        # edge, line = edge.permute(0, 2, 1, 3, 4), line.permute(0, 2, 1, 3, 4) # [b, t, 1, h, w]
        return edge, line
    
# -------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------
    
class StructGPT256RelBCE_video(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config, opts, device):
        super().__init__()

        self.act_last = nn.Sigmoid()

        self.l1_loss = nn.L1Loss()
        
        # self.fuseformerBlock = InpaintGenerator(input_ch=6, ref_frames=5)
        self.fuseformerBlock = InpaintGenerator(input_ch=3, out_ch=1, ref_frames=5)
        self.fuseformerBlock = self.fuseformerBlock.to(device)

        self.config = config
        self.opts = opts

        self.apply(self._init_weights)  # initialize the weights (multiple layer initialization)

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def get_block_size(self):
        return self.block_size
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv2d, nn.ConvTranspose2d)):  # if one of the type
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()  
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):  # some parameter need weight decay to avoid overfitting
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        # whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d, torch.nn.ConvTranspose2d) # need weight decay
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d, torch.nn.ConvTranspose2d, torch.nn.Conv3d, torch.nn.MultiheadAttention) # need weight decay
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
        # special case the position embedding parameter in the root GPT module as not decayed
        # no_decay.add('pos_emb')
        no_decay.add('fuseformerBlock.add_pos_emb.pos_emb')
        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert len(
            param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params),)
        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def forward(self, img_idx, struct_idx, struct_targets=None, masks=None):
        img_idx = img_idx * (1 - masks)  # create masked image
        struct_idx = struct_idx * (1 - masks) # create masked structure

        # [b, t, c, w, h]
        x = torch.cat((img_idx, struct_idx, masks), dim=2)  # concat method NEED checking (maybe is channel-wise)

        [b, t, c, h, w] = x.shape  # before here, the video data is still with Height x Width -> [50, 256, 32, 32] -> [t, c, h, w]
        
        x = self.fuseformerBlock(x)
        
        # edge, line = torch.split(x, [1, 1], dim=1)  # seperate the TSR outputs
        struct = x

        eps = 1e-8
        loss, struct_hole_loss, struct_valid_loss = 0., 0., 0.
        
        # Loss computing
        if struct_targets is not None :
            struct_targets = struct_targets.view(b * t, 1, h, w)
            masks = masks.view(b * t, 1, h, w)

            # struct loss (BCE loss)
            # loss_struct = nnF.binary_cross_entropy_with_logits(struct.permute(0, 2, 3, 1).contiguous().view(-1, 1),
            #                                           struct_targets.permute(0, 2, 3, 1).contiguous().view(-1, 1),
            #                                           reduction='none')

            # masks_ = masks.permute(0, 2, 3, 1).contiguous().view(-1, 1) # only compute the loss in the masked region
            # loss = (loss_struct * masks_)*self.opts.loss_hole_valid_weight[0] + \
            #        (loss_struct * (1-masks_))*self.opts.loss_hole_valid_weight[1]
            # loss = torch.mean(loss)
            
            # struct_hole_loss = (loss_struct*masks_)*self.opts.loss_hole_valid_weight[0]
            # struct_valid_loss = (loss_struct*(1-masks_))*self.opts.loss_hole_valid_weight[1]

            # struct loss (L1 loss)
            struct_hole_loss = self.l1_loss(struct * masks, struct_targets * masks)
            struct_hole_loss = struct_hole_loss / torch.mean(masks)

            struct_valid_loss = self.l1_loss(struct * (1 - masks), struct_targets * (1 - masks))
            struct_valid_loss = struct_valid_loss / torch.mean(1 - masks)
            
            loss = struct_hole_loss + struct_valid_loss

        else:
            loss = 0

        struct = struct.view(b, t, 1, h, w)
        # struct = self.act_last(struct)  # sigmoid activate 

        loss_detail = [struct_hole_loss, struct_valid_loss]

        return struct, loss, loss_detail
    
    def forward_with_logits(self, img_idx, struct_idx, masks=None):
        # if the shape of input is not [b, t, c, w, h], then add a dimension
        if len(img_idx.shape) != 5:
            img_idx = img_idx.unsqueeze(0)
        if len(struct_idx.shape) != 5:
            struct_idx = struct_idx.unsqueeze(0)
        if masks is not None and len(masks.shape) != 5:
            masks = masks.unsqueeze(0)

        img_idx = img_idx * (1 - masks)  # create masked image
        struct_idx = struct_idx * (1 - masks) # create masked structure

        # [b, t, c, w, h]
        x = torch.cat((img_idx, struct_idx, masks), dim=2)  # concat method NEED checking (maybe is channel-wise)

        [b, t, c, h, w] = x.shape  # before here, the video data is still with Height x Width -> [50, 256, 32, 32] -> [t, c, h, w]

        # Transformer blocks
        x = self.fuseformerBlock(x)
        
        # edge, line = torch.split(x, [1, 1], dim=1)  # seperate the TSR outputs
        struct = x
        # edge, line = edge.squeeze(0), line.squeeze(0)
        struct = struct.view(b, t, 1, h, w)

        return struct