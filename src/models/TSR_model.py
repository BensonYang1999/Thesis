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

        self.pad1 = nn.ReflectionPad2d(3) # square -> bigger square (extend 3 for each side)
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=64, kernel_size=7, padding=0) # downsample input
        self.act = nn.ReLU(True)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)  # downsample 1

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1) # downsample 2

        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1) # downsample 3

        self.pos_emb = nn.Parameter(torch.zeros(1, 1024, 256))  # special tensor that automatically add into parameter list
        self.drop = nn.Dropout(config.embd_pdrop)

        # decoder, input: 32*32*config.n_embd
        self.ln_f = nn.LayerNorm(256)

        self.convt1 = nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1) # upsample 1

        self.convt2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1) # upsample 2

        self.convt3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1) # upsample 3

        self.padt = nn.ReflectionPad2d(3)
        self.convt4 = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=7, padding=0) # upsample and ouput only edge/line

        self.act_last = nn.Sigmoid()  # original in ZITS
        # self.act_last = nn.LeakyReLU()  # original in For ZITS_video

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

        self.fuseformerBlock = InpaintGenerator(input_ch=6, ref_frames=5)
        self.fuseformerBlock = self.fuseformerBlock.to(device)

        self.fuseFramesToLineEdge = nn.Sequential(
            nn.Conv3d(config.ref_frame_num, 1, (config.ref_frame_num, 3, 3),stride=1, padding='same'), # transfer fuseformer output to edge/line
        )

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
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d, torch.nn.ConvTranspose2d, torch.nn.Conv3d) # need weight decay
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

        # Loss computing
        if edge_targets is not None and line_targets is not None:
            edge_targets = edge_targets.view(b * t, 1, h, w)
            line_targets = line_targets.view(b * t, 1, h, w)
            masks = masks.view(b * t, 1, h, w)

            loss = 0
            # hole loss
            if "hole" in self.opts.loss_item:
                edge_hole_loss = self.l1_loss(edge*masks, edge_targets*masks)
                edge_hole_loss = edge_hole_loss / torch.mean(masks)

                line_hole_loss = self.l1_loss(line*masks, line_targets*masks)
                line_hole_loss = line_hole_loss / torch.mean(masks)
                # total loss
                loss += edge_hole_loss + line_hole_loss

            # valid loss
            if "valid" in self.opts.loss_item:
                edge_valid_loss = self.l1_loss(edge*(1-masks), edge_targets*(1-masks))
                edge_valid_loss = edge_valid_loss / torch.mean(1-masks)

                line_valid_loss = self.l1_loss(line*(1-masks), line_targets*(1-masks))
                line_valid_loss = line_valid_loss / torch.mean(1-masks)
                # total loss
                loss += edge_valid_loss + line_valid_loss

            
            # ZITS loss computation
            # edge loss
            # loss = nnF.binary_cross_entropy_with_logits(edge.permute(0, 2, 3, 1).contiguous().view(-1, 1),
            #                                           edge_targets.permute(0, 2, 3, 1).contiguous().view(-1, 1),
            #                                           reduction='none')
            # line loss
            # loss = loss + nnF.binary_cross_entropy_with_logits(line.permute(0, 2, 3, 1).contiguous().view(-1, 1),
            #                                                  line_targets.permute(0, 2, 3, 1).contiguous().view(-1, 1),
            #                                                  reduction='none')
            
            # masks_ = masks.permute(0, 2, 3, 1).contiguous().view(-1, 1) # only compute the loss in the masked region

            # loss *= masks_
            # print(f"loss shape: {loss.size()}") # test
            # loss = torch.mean(loss)
        else:
            loss = 0

        edge, line = self.act_last(edge), self.act_last(line)  # sigmoid activate
        edge, line = edge.view(b, t, 1, h, w), line.view(b, t, 1, h, w)

        return edge, line, loss
    
    def forward_with_logits(self, img_idx, edge_idx, line_idx, masks=None):
        img_idx, edge_idx, line_idx, masks = [
            var.unsqueeze(0) if len(var.size()) != 5 else var
            for var in (img_idx, edge_idx, line_idx, masks)
        ]

        img_idx = img_idx * (1 - masks)  # create masked image
        edge_idx = edge_idx * (1 - masks) # create masked edge
        line_idx = line_idx * (1 - masks) # create masked line

        # [b, t, c, w, h]
        x = torch.cat((img_idx, edge_idx, line_idx, masks), dim=2)  # concat method NEED checking (maybe is channel-wise)

        [b, t, c, h, w] = x.shape  # before here, the video data is still with Height x Width -> [50, 256, 32, 32] -> [t, c, h, w]

        # Transformer blocks
        x = self.fuseformerBlock(x)
        
        edge, line = torch.split(x, [1, 1], dim=1)  # seperate the TSR outputs

        return edge, line