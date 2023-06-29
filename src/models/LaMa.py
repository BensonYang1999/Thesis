import numpy as np

from .ffc import *
from .layers import *


class ResnetBlock_remove_IN(nn.Module):
    def __init__(self, dim, dilation=1):
        super(ResnetBlock_remove_IN, self).__init__()

        self.ffc1 = FFC_BN_ACT(dim, dim, 3, 0.75, 0.75, stride=1, padding=1, dilation=dilation, groups=1, bias=False,
                               norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU, enable_lfu=False)

        self.ffc2 = FFC_BN_ACT(dim, dim, 3, 0.75, 0.75, stride=1, padding=1, dilation=1, groups=1, bias=False,
                               norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU, enable_lfu=False)

    def forward(self, x):
        output = x
        _, c, _, _ = output.shape
        output = torch.split(output, [c - int(c * 0.75), int(c * 0.75)], dim=1)
        x_l, x_g = self.ffc1(output)
        output = self.ffc2((x_l, x_g))
        output = torch.cat(output, dim=1)
        output = x + output

        return output


class MaskedSinusoidalPositionalEmbedding(nn.Embedding):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(num_embeddings, embedding_dim)
        self.weight = self._init_weight(self.weight)

    @staticmethod
    def _init_weight(out: nn.Parameter):
        """
        Identical to the XLM create_sinusoidal_embeddings except features are not interleaved. The cos features are in
        the 2nd half of the vector. [dim // 2:]
        """
        n_pos, dim = out.shape
        position_enc = np.array(
            [[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)]
        )
        out.requires_grad = False  # set early to avoid an error in pytorch-1.8+
        sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
        out[:, 0:sentinel] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        out[:, sentinel:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        out.detach_()
        return out

    @torch.no_grad()
    def forward(self, input_ids):
        """`input_ids` is expected to be [bsz x seqlen]."""
        return super().forward(input_ids)


class MultiLabelEmbedding(nn.Module):
    def __init__(self, num_positions: int, embedding_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(num_positions, embedding_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight)

    def forward(self, input_ids):
        # input_ids:[B,HW,4](onehot)
        out = torch.matmul(input_ids, self.weight)  # [B,HW,dim]
        return out


class LaMa_model(nn.Module):
    def __init__(self):
        super().__init__()

        self.pad1 = nn.ReflectionPad2d(3)
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=64, kernel_size=7, padding=0)
        self.bn1 = nn.BatchNorm2d(64)
        self.act = nn.ReLU(True)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(512)

        blocks = []
        ### resnet blocks
        for i in range(9):
            cur_resblock = ResnetBlock_remove_IN(512, 1)
            blocks.append(cur_resblock)

        self.middle = nn.Sequential(*blocks)

        self.convt1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bnt1 = nn.BatchNorm2d(256)

        self.convt2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bnt2 = nn.BatchNorm2d(128)

        self.convt3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bnt3 = nn.BatchNorm2d(64)

        self.padt = nn.ReflectionPad2d(3)
        self.convt4 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, padding=0)
        self.act_last = nn.Tanh()

    def forward(self, x, rel_pos_emb=None, direct_emb=None, str_feats=None):
        x = self.pad1(x)
        x = self.conv1(x)
        x = self.bn1(x.to(torch.float32))
        x = self.act(x)

        x = self.conv2(x)
        x = self.bn2(x.to(torch.float32))
        x = self.act(x)

        x = self.conv3(x)
        x = self.bn3(x.to(torch.float32))
        x = self.act(x)

        x = self.conv4(x)
        x = self.bn4(x.to(torch.float32))
        x = self.act(x)

        x = self.middle(x)

        x = self.convt1(x)
        x = self.bnt1(x.to(torch.float32))
        x = self.act(x)

        x = self.convt2(x)
        x = self.bnt2(x.to(torch.float32))
        x = self.act(x)

        x = self.convt3(x)
        x = self.bnt3(x.to(torch.float32))
        x = self.act(x)

        x = self.padt(x)
        x = self.convt4(x)
        x = self.act_last(x)
        x = (x + 1) / 2
        return x

class LaMa_model_video_2D(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.channel1 = config.generator['ngf']
        self.channel2 = self.channel1 * 2
        self.channel3 = self.channel2 * 2
        self.channel4 = self.channel3 * 2

        self.pad1 = nn.ReflectionPad2d(3)
        # self.conv1 = nn.Conv2d(in_channels=4, out_channels=64, kernel_size=7, padding=0)
        # self.bn1 = nn.BatchNorm2d(64)
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=self.channel1, kernel_size=7, padding=0)
        self.bn1 = nn.BatchNorm2d(self.channel1)
        self.act = nn.ReLU(True)

        # self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        # self.bn2 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(in_channels=self.channel1, out_channels=self.channel2, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(self.channel2)

        # self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        # self.bn3 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(in_channels=self.channel2, out_channels=self.channel3, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(self.channel3)

        # self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1)
        # self.bn4 = nn.BatchNorm2d(512)
        self.conv4 = nn.Conv2d(in_channels=self.channel3, out_channels=self.channel4, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(self.channel4)

        blocks = []
        ### resnet blocks
        for i in range(9):
            # cur_resblock = ResnetBlock_remove_IN(512, 1)
            cur_resblock = ResnetBlock_remove_IN(self.channel4, 1)
            blocks.append(cur_resblock)

        self.middle = nn.Sequential(*blocks)

        # self.convt1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        # self.bnt1 = nn.BatchNorm2d(256)
        self.convt1 = nn.ConvTranspose2d(self.channel4, self.channel3, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bnt1 = nn.BatchNorm2d(self.channel3)

        # self.convt2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        # self.bnt2 = nn.BatchNorm2d(128)
        self.convt2 = nn.ConvTranspose2d(self.channel3, self.channel2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bnt2 = nn.BatchNorm2d(self.channel2)

        # self.convt3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        # self.bnt3 = nn.BatchNorm2d(64)
        self.convt3 = nn.ConvTranspose2d(self.channel2, self.channel1, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bnt3 = nn.BatchNorm2d(self.channel1)

        self.padt = nn.ReflectionPad2d(3)
        # self.convt4 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, padding=0)
        self.convt4 = nn.Conv2d(in_channels=self.channel1, out_channels=3, kernel_size=7, padding=0)
        self.act_last = nn.Tanh()

    def forward(self, x, rel_pos_emb=None, direct_emb=None, str_feats=None):
        b, t, c, h, w = x.shape
        x = x.view(b*t, c, h, w)
        x = self.pad1(x)
        x = self.conv1(x)
        x = self.bn1(x.to(torch.float32))
        x = self.act(x)

        x = self.conv2(x)
        x = self.bn2(x.to(torch.float32))
        x = self.act(x)

        x = self.conv3(x)
        x = self.bn3(x.to(torch.float32))
        x = self.act(x)

        x = self.conv4(x)
        x = self.bn4(x.to(torch.float32))
        x = self.act(x)

        x = self.middle(x)

        x = self.convt1(x)
        x = self.bnt1(x.to(torch.float32))
        x = self.act(x)

        x = self.convt2(x)
        x = self.bnt2(x.to(torch.float32))
        x = self.act(x)

        x = self.convt3(x)
        x = self.bnt3(x.to(torch.float32))
        x = self.act(x)

        x = self.padt(x)
        x = self.convt4(x)
        x = self.act_last(x)
        x = (x + 1) / 2
        return x

class LaMa_model_video_3D(nn.Module):
    def __init__(self):
        super().__init__()

        self.pad1 = nn.ReplicationPad3d(3)
        self.conv1 = nn.Conv3d(in_channels=4, out_channels=64, kernel_size=7, padding=0)
        self.bn1 = nn.BatchNorm3d(64)
        self.act = nn.ReLU(True)

        self.conv2 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm3d(128)

        self.conv3 = nn.Conv3d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm3d(256)

        self.conv4 = nn.Conv3d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm3d(512)

        blocks = []
        ### resnet blocks
        for i in range(9):
            cur_resblock = ResnetBlock_remove_IN(512, 1)
            blocks.append(cur_resblock)

        self.middle = nn.Sequential(*blocks)

        self.convt1 = nn.ConvTranspose3d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bnt1 = nn.BatchNorm3d(256)

        self.convt2 = nn.ConvTranspose3d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bnt2 = nn.BatchNorm3d(128)

        self.convt3 = nn.ConvTranspose3d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bnt3 = nn.BatchNorm3d(64)

        self.padt = nn.ReplicationPad3d(3)
        self.convt4 = nn.Conv3d(in_channels=64, out_channels=3, kernel_size=7, padding=0)
        self.act_last = nn.Tanh()

    def forward(self, x, rel_pos_emb=None, direct_emb=None, str_feats=None):
        x = self.pad1(x)
        x = self.conv1(x)
        x = self.bn1(x.to(torch.float32))
        x = self.act(x)

        x = self.conv2(x)
        x = self.bn2(x.to(torch.float32))
        x = self.act(x)

        x = self.conv3(x)
        x = self.bn3(x.to(torch.float32))
        x = self.act(x)

        x = self.conv4(x)
        x = self.bn4(x.to(torch.float32))
        x = self.act(x)

        x = self.middle(x)

        x = self.convt1(x)
        x = self.bnt1(x.to(torch.float32))
        x = self.act(x)

        x = self.convt2(x)
        x = self.bnt2(x.to(torch.float32))
        x = self.act(x)

        x = self.convt3(x)
        x = self.bnt3(x.to(torch.float32))
        x = self.act(x)

        x = self.padt(x)
        x = self.convt4(x)
        x = self.act_last(x)
        x = (x + 1) / 2
        return x

class ReZeroFFC(LaMa_model):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, x, rel_pos_emb=None, direct_emb=None, str_feats=None):
        x = self.pad1(x)
        x = self.conv1(x)
        if self.config.use_MPE:
            inp = x.to(torch.float32) + rel_pos_emb + direct_emb
        else:
            inp = x.to(torch.float32)
        x = self.bn1(inp)
        x = self.act(x)

        x = self.conv2(x + str_feats[0])
        x = self.bn2(x.to(torch.float32))
        x = self.act(x)

        x = self.conv3(x + str_feats[1])
        x = self.bn3(x.to(torch.float32))
        x = self.act(x)

        x = self.conv4(x + str_feats[2])
        x = self.bn4(x.to(torch.float32))
        x = self.act(x)

        x = self.middle(x + str_feats[3])

        x = self.convt1(x)
        x = self.bnt1(x.to(torch.float32))
        x = self.act(x)

        x = self.convt2(x)
        x = self.bnt2(x.to(torch.float32))
        x = self.act(x)

        x = self.convt3(x)
        x = self.bnt3(x.to(torch.float32))
        x = self.act(x)

        x = self.padt(x)
        x = self.convt4(x)
        x = self.act_last(x)
        x = (x + 1) / 2
        return x

class ReZeroFFC_video_2D(LaMa_model_video_2D):
    def __init__(self, config):
        super().__init__(config)
        # self.config = config

    def forward(self, x, rel_pos_emb=None, direct_emb=None, str_feats=None):
        b, t, c, h, w = x.shape
        rel_pos_emb =  rel_pos_emb.reshape(b*t, -1, h, w)
        direct_emb = direct_emb.reshape(b*t, -1, h, w) # four directions
        str_feats = [feat.reshape(-1, *feat.shape[2:]) for feat in str_feats]
        
        x = x.view(b*t, c, h, w)
        x = self.pad1(x)
        x = self.conv1(x)
        if self.config.use_MPE:
            inp = x.to(torch.float32) + rel_pos_emb + direct_emb
        else:
            inp = x.to(torch.float32)
        x = self.bn1(inp)
        x = self.act(x) 

        x = self.conv2(x + str_feats[0])
        x = self.bn2(x.to(torch.float32))
        x = self.act(x)

        x = self.conv3(x + str_feats[1])
        x = self.bn3(x.to(torch.float32))
        x = self.act(x)

        x = self.conv4(x + str_feats[2])
        x = self.bn4(x.to(torch.float32))
        x = self.act(x)

        x = self.middle(x + str_feats[3])

        x = self.convt1(x)
        x = self.bnt1(x.to(torch.float32))
        x = self.act(x)

        x = self.convt2(x)
        x = self.bnt2(x.to(torch.float32))
        x = self.act(x)

        x = self.convt3(x)
        x = self.bnt3(x.to(torch.float32))
        x = self.act(x)

        x = self.padt(x)
        x = self.convt4(x)
        x = self.act_last(x)
        x = (x + 1) / 2
        x = x.view(b, t, 3, h, w)
        return x


class ReZeroFFC_video_3D(LaMa_model_video_3D):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, x, rel_pos_emb=None, direct_emb=None, str_feats=None):
        x = self.pad1(x)
        x = self.conv1(x)
        if self.config.use_MPE:
            inp = x.to(torch.float32) + rel_pos_emb.unsqueeze(2) + direct_emb.unsqueeze(2)
        else:
            inp = x.to(torch.float32)
        x = self.bn1(inp)
        x = self.act(x)

        x = self.conv2(x + str_feats[0].unsqueeze(2))
        x = self.bn2(x.to(torch.float32))
        x = self.act(x)

        x = self.conv3(x + str_feats[1].unsqueeze(2))
        x = self.bn3(x.to(torch.float32))
        x = self.act(x)

        x = self.conv4(x + str_feats[2].unsqueeze(2))
        x = self.bn4(x.to(torch.float32))
        x = self.act(x)

        x = self.middle(x + str_feats[3].unsqueeze(2))

        x = self.convt1(x)
        x = self.bnt1(x.to(torch.float32))
        x = self.act(x)

        x = self.convt2(x)
        x = self.bnt2(x.to(torch.float32))
        x = self.act(x)

        x = self.convt3(x)
        x = self.bnt3(x.to(torch.float32))
        x = self.act(x)

        x = self.padt(x)
        x = self.convt4(x)
        x = self.act_last(x)
        x = (x + 1) / 2
        return x


class StructureEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.rezero_for_mpe is None:
            self.rezero_for_mpe = False
        else:
            self.rezero_for_mpe = config.rezero_for_mpe

        self.channel1 = config.generator['ngf']
        self.channel2 = self.channel1 * 2
        self.channel3 = self.channel2 * 2
        self.channel4 = self.channel3 * 2

        self.pad1 = nn.ReflectionPad2d(3)
        self.conv1 = GateConv(in_channels=3, out_channels=self.channel1, kernel_size=7, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(self.channel1)
        self.act = nn.ReLU(True)

        self.conv2 = GateConv(in_channels=self.channel1, out_channels=self.channel2, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(self.channel2)

        self.conv3 = nn.Conv2d(in_channels=self.channel2, out_channels=self.channel3, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(self.channel3)

        self.conv4 = nn.Conv2d(in_channels=self.channel3, out_channels=self.channel4, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(self.channel4)

        blocks = []
        # resnet blocks
        for i in range(3):
            blocks.append(ResnetBlock(input_dim=self.channel4, out_dim=None, dilation=2))

        self.middle = nn.Sequential(*blocks)
        self.alpha1 = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=True)

        self.convt1 = GateConv(self.channel4, self.channel3, kernel_size=4, stride=2, padding=1, transpose=True)
        self.bnt1 = nn.BatchNorm2d(self.channel3)
        self.alpha2 = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=True)

        self.convt2 = GateConv(self.channel3, self.channel2, kernel_size=4, stride=2, padding=1, transpose=True)
        self.bnt2 = nn.BatchNorm2d(self.channel2)
        self.alpha3 = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=True)

        self.convt3 = GateConv(self.channel2, self.channel1, kernel_size=4, stride=2, padding=1, transpose=True)
        self.bnt3 = nn.BatchNorm2d(self.channel1)
        self.alpha4 = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=True)

        if self.rezero_for_mpe:
            self.rel_pos_emb = MaskedSinusoidalPositionalEmbedding(num_embeddings=config.rel_pos_num,
                                                                   embedding_dim=self.channel1)
            self.direct_emb = MultiLabelEmbedding(num_positions=4, embedding_dim=self.channel1)
            self.alpha5 = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=True)
            self.alpha6 = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=True)

    def forward(self, x, rel_pos=None, direct=None):
        # x = torch.cat([batch['edges'], batch['lines'], mask], dim=2) # [B, T, 3, H, W] for video 
        if not self.config.use_SPI:  # never mind the mask
            x = x[:, :, :2, :, :]

        # print(f"2D original shape: {x.shape}") # test
        # print(f"1 pad before: {x.shape}") # test
        x = self.pad1(x)
        # print(f"1 pad after: {x.shape}") # test
        x = self.conv1(x)
        # print(f"1 conv1 after: {x.shape}") # test
        x = self.bn1(x.to(torch.float32))
        # print(f"1 bn1 after: {x.shape}") # test
        x = self.act(x)
        # print(f"1 act after: {x.shape}") # test

        x = self.conv2(x)
        # print(f"2 conv2 after: {x.shape}") # test
        x = self.bn2(x.to(torch.float32))
        # print(f"2 bn2 after: {x.shape}") # test
        x = self.act(x)
        # print(f"2 act2 after: {x.shape}") # test

        x = self.conv3(x)
        # print(f"3 conv3 after: {x.shape}") # test
        x = self.bn3(x.to(torch.float32))
        # print(f"3 bn3 after: {x.shape}") # test
        x = self.act(x)
        # print(f"3 act3 after: {x.shape}") # test

        x = self.conv4(x)
        # print(f"4 conv4 after: {x.shape}") # test
        x = self.bn4(x.to(torch.float32))
        # print(f"4 bn4 after: {x.shape}") # test
        x = self.act(x)
        # print(f"4 act4 after: {x.shape}") # test

        return_feats = []
        x = self.middle(x)
        return_feats.append(x * self.alpha1)

        x = self.convt1(x)
        # print(f"1 convt1 after: {x.shape}") # test
        x = self.bnt1(x.to(torch.float32))
        # print(f"1 bnt1 after: {x.shape}") # test
        x = self.act(x)
        # print(f"1 act1 after: {x.shape}") # test
        return_feats.append(x * self.alpha2)

        x = self.convt2(x)
        # print(f"2 convt2 after: {x.shape}") # test
        x = self.bnt2(x.to(torch.float32))
        # print(f"2 bnt2 after: {x.shape}") # test
        x = self.act(x)
        # print(f"2 act2 after: {x.shape}") # test
        return_feats.append(x * self.alpha3)

        x = self.convt3(x)
        # print(f"3 convt3 after: {x.shape}") # test
        x = self.bnt3(x.to(torch.float32))
        # print(f"3 bnt3 after: {x.shape}") # test
        x = self.act(x)
        # print(f"3 act3 after: {x.shape}") # test
        return_feats.append(x * self.alpha4)

        return_feats = return_feats[::-1]

        if not self.rezero_for_mpe:
            return return_feats
        else:
            b, h, w = rel_pos.shape
            rel_pos = rel_pos.reshape(b, h * w)
            rel_pos_emb = self.rel_pos_emb(rel_pos).reshape(b, h, w, -1).permute(0, 3, 1, 2) * self.alpha5
            direct = direct.reshape(b, h * w, 4).to(torch.float32)
            direct_emb = self.direct_emb(direct).reshape(b, h, w, -1).permute(0, 3, 1, 2) * self.alpha6

            return return_feats, rel_pos_emb, direct_emb


class StructureEncoder_video_2D(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.rezero_for_mpe is None:
            self.rezero_for_mpe = False
        else:
            self.rezero_for_mpe = config.rezero_for_mpe
        self.struc_encoder = StructureEncoder(config)

    def forward(self, x, rel_pos=None, direct=None):
        b, t, c, h, w = x.shape
        return_feats_t = []
        rel_pos_emb_t = []
        direct_emb_t  = []
        for i in range(t):
            return_feats, rel_pos_emb, direct_emb = self.struc_encoder(x[:, i], rel_pos[:, i].squeeze(1), direct[:, i])
            # return list with len 4 refer to four different scales of structure embedding
            
            return_feats_t.append(return_feats) # for the time
            rel_pos_emb_t.append(rel_pos_emb)
            direct_emb_t.append(direct_emb)
            
        return_feats_t = [torch.stack([return_feats_t[j][i] for j in range(t)], dim=1) for i in range(4)]
        
        if not self.rezero_for_mpe:
            return return_feats_t
        else:
            return return_feats_t, torch.stack(rel_pos_emb_t).permute(1,0,2,3,4), torch.stack(direct_emb_t).permute(1,0,2,3,4)

from src.models.fuseformer_origin import AddPosEmb, MultiHeadedAttention

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout = 0.1):
        super().__init__() 
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.act = nn.ReLU(True)
    def forward(self, x):
        x = self.dropout(self.act(self.linear_1(x)))
        x = self.linear_2(x)
        return x
    
class TransformerBlock(nn.Module):
    """
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden=128, num_head=4, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadedAttention(d_model=hidden, head=num_head, p=dropout)
        self.ffn = FeedForward(d_model=hidden, d_ff=512, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input):
        x = self.norm1(input)
        x = input + self.dropout(self.attention(x))
        y = self.norm2(x)
        x = x + self.ffn(y)
        return x
    
class StructureEncoder_video_3D(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.rezero_for_mpe is None:
            self.rezero_for_mpe = False
        else:
            self.rezero_for_mpe = config.rezero_for_mpe

        self.channel1 = config.generator['ngf']
        self.channel2 = self.channel1 * 2
        self.channel3 = self.channel2 * 2
        self.channel4 = self.channel3 * 2

        self.pad1 = nn.ReflectionPad3d((3, 3, 3, 3, 0, 0)) # [b, c, h, w] = [2, 3, 240, 432] => [2, 3, 246, 438]
        # self.conv1 = GateConv3D(in_channels=3, out_channels=64, kernel_size=(1, 7, 7), stride=1, padding=0) # [2, 3, 246, 438] => [2, 64, 240, 432]
        self.conv1 = GateConv3D(in_channels=3, out_channels=self.channel1, kernel_size=(1, 7, 7), stride=1, padding=0) # [2, 3, 246, 438] => [2, 64, 240, 432]
        # self.bn1 = nn.BatchNorm3d(64) # [2, 64, 240, 432]
        self.bn1 = nn.BatchNorm3d(self.channel1) # [2, 64, 240, 432]
        self.act = nn.ReLU(True) # [2, 64, 240, 432]

        # self.conv2 = GateConv3D(in_channels=64, out_channels=128, kernel_size=(1, 4, 4), stride=2, padding=(2, 1, 1)) # [2, 64, 240, 432] => [2, 128, 120, 216]
        self.conv2 = GateConv3D(in_channels=self.channel1, out_channels=self.channel2, kernel_size=(1, 4, 4), stride=2, padding=(2, 1, 1)) # [2, 64, 240, 432] => [2, 128, 120, 216]
        # self.bn2 = nn.BatchNorm3d(128)
        self.bn2 = nn.BatchNorm3d(self.channel2)

        # self.conv3 = nn.Conv3d(in_channels=128, out_channels=256, kernel_size=(1, 4, 4), stride=2, padding=(2, 1, 1))
        self.conv3 = nn.Conv3d(in_channels=self.channel2, out_channels=self.channel3, kernel_size=(1, 4, 4), stride=2, padding=(2, 1, 1))
        # self.bn3 = nn.BatchNorm3d(256)
        self.bn3 = nn.BatchNorm3d(self.channel3)

        # self.conv4 = nn.Conv3d(in_channels=256, out_channels=512, kernel_size=(1, 4, 4), stride=2, padding=(2, 1, 1))
        self.conv4 = nn.Conv3d(in_channels=self.channel3, out_channels=self.channel4, kernel_size=(1, 4, 4), stride=2, padding=(2, 1, 1))
        # self.bn4 = nn.BatchNorm3d(512)
        self.bn4 = nn.BatchNorm3d(self.channel4)

        blocks = []
        # resnet blocks
        # for i in range(3):
        #     blocks.append(ResnetBlock(input_dim=512, out_dim=None, dilation=2))

        stack_num = 3
        n_vecs = config.ref_frame_num*30*54
        hidden = self.channel4 # origin 512
        for _ in range(stack_num):
            blocks.append(TransformerBlock(hidden=self.channel4, num_head=4, dropout=0.1)) # origin hidden 512
        self.transformer = nn.Sequential(*blocks)
        
        self.add_pos_emb = AddPosEmb(n_vecs, hidden)

        # self.middle = nn.Sequential(*blocks)
        self.alpha1 = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=True)

        # self.convt1 = GateConv3D(512, 256, kernel_size=(1, 4, 4), stride=2, padding=(2, 1, 1), transpose=True)
        # self.bnt1 = nn.BatchNorm3d(256)
        self.convt1 = GateConv3D(self.channel4, self.channel3, kernel_size=(1, 4, 4), stride=2, padding=(2, 1, 1), transpose=True)
        self.bnt1 = nn.BatchNorm3d(self.channel3)
        self.alpha2 = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=True)

        # self.convt2 = GateConv3D(256, 128, kernel_size=(1, 4, 4), stride=2, padding=(2, 1, 1), transpose=True)
        # self.bnt2 = nn.BatchNorm3d(128)
        self.convt2 = GateConv3D(self.channel3, self.channel2, kernel_size=(1, 4, 4), stride=2, padding=(2, 1, 1), transpose=True)
        self.bnt2 = nn.BatchNorm3d(self.channel2)
        self.alpha3 = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=True)

        # self.convt3 = GateConv3D(128, 64, kernel_size=(1, 4, 4), stride=2, padding=(2, 1, 1), transpose=True)
        # self.bnt3 = nn.BatchNorm3d(64)
        self.convt3 = GateConv3D(self.channel2, self.channel1, kernel_size=(1, 4, 4), stride=2, padding=(2, 1, 1), transpose=True)
        self.bnt3 = nn.BatchNorm3d(self.channel1)
        self.alpha4 = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=True)

        if self.rezero_for_mpe:
            # self.rel_pos_emb = MaskedSinusoidalPositionalEmbedding(num_embeddings=config.rel_pos_num,
            #                                                        embedding_dim=64)
            # self.direct_emb = MultiLabelEmbedding(num_positions=4, embedding_dim=64)
            self.rel_pos_emb = MaskedSinusoidalPositionalEmbedding(num_embeddings=config.rel_pos_num,
                                                                   embedding_dim=self.channel1)
            self.direct_emb = MultiLabelEmbedding(num_positions=4, embedding_dim=self.channel1)
            self.alpha5 = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=True)
            self.alpha6 = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=True)

    def forward(self, x, rel_pos=None, direct=None):
        # print(f"input x before: {x.shape}") # test
        x = x.permute(0, 2, 1, 3, 4)
        # print(f"input x after: {x.shape}") # test

        x = self.pad1(x) # [b, c, h, w] = [2, 3, 240, 432] => [2, 3, 246, 438]
        # print(f"pad after: {x.shape}") # test
        x = self.conv1(x) # [2, 3, 246, 438] => [2, 64, 240, 432]
        # print(f"conv1 after: {x.shape}")
        x = self.bn1(x.to(torch.float32)) # [2, 64, 240, 432]
        x = self.act(x) # [2, 64, 240, 432]

        x = self.conv2(x) # [2, 64, 240, 432] => [2, 128, 120, 216]
        x = self.bn2(x.to(torch.float32)) # [2, 128, 120, 216]
        x = self.act(x) # [2, 128, 120, 216]

        x = self.conv3(x) # [2, 128, 120, 216] => [2, 256, 60, 108]
        x = self.bn3(x.to(torch.float32)) # [2, 256, 60, 108]
        x = self.act(x) # [2, 256, 60, 108]

        x = self.conv4(x) # [2, 256, 60, 108] => [2, 512, 30, 54]
        x = self.bn4(x.to(torch.float32)) # [2, 512, 30, 54]
        x = self.act(x) # [2, 512, 30, 54]

        return_feats = []
        # print(f"trans feature before: {x.shape}") # test ([2, 512, 5, 30, 54])
        b, c, t, h, w = x.shape
        trans_feat = x.permute(0, 2, 3, 4, 1) # change to [b, t, c, h, w]
        trans_feat = trans_feat.view(b, t*h*w, c)
        trans_feat = self.add_pos_emb(trans_feat)
        # print(f"trans_feat before shape: {trans_feat.shape}") # test
        trans_feat = self.transformer(trans_feat)
        # print(f"trans_feat after shape: {trans_feat.shape}") # test
        trans_feat = trans_feat.view(b, t, h, w, c)
        # print(f"trans_feat after view shape: {trans_feat.shape}") # test
        trans_feat = trans_feat.permute(0, 4, 1, 2, 3) # change to [b, t, c, h, w]
        # print(f"trans feature after: {x.shape}") # test ([2, 512, 5, 30, 54])
        # print(f"ResNet after: {trans_feat.shape}") # test
        x = x + trans_feat
        # print(f"x shape: {x.shape}") # test
        # print(f"return_feats shape: {x.shape}") # test
        # print(f"return_feats permute shape: {x.permute(0, 2, 1, 3, 4).shape}") # test
        return_feats.append(x.permute(0, 2, 1, 3, 4) * self.alpha1)

        # print(f"x before transpose: {x.shape}") # test
        x = self.convt1(x) # [2, 512, 30, 54] => [2, 256, 60, 108] # [2, 256, 5, 60, 108]
        # print(f" x after convt1: {x.shape}") # test
        x = self.bnt1(x.to(torch.float32)) # [2, 256, 60, 108]
        x = self.act(x) # [2, 256, 60, 108]
        return_feats.append(x.permute(0, 2, 1, 3, 4) * self.alpha2)
        # print(f"x after transpose: {x.shape}") # test

        x = self.convt2(x) # [2, 256, 60, 108] => [2, 128, 120, 216] # [2, 128, 5, 120, 216]
        x = self.bnt2(x.to(torch.float32)) # [2, 128, 120, 216]
        x = self.act(x) # [2, 128, 120, 216]
        # print(f" x after convt2: {x.shape}") # test
        return_feats.append(x.permute(0, 2, 1, 3, 4) * self.alpha3)

        x = self.convt3(x) # [2, 128, 120, 216] => [2. 64, 240, 432] # [2, 64, 5, 240, 432]
        x = self.bnt3(x.to(torch.float32)) # [2. 64, 240, 432]
        x = self.act(x) # [2. 64, 240, 432]
        # print(f" x after convt3: {x.shape}") # test
        return_feats.append(x.permute(0, 2, 1, 3, 4) * self.alpha4)

        return_feats = return_feats[::-1]

        if not self.rezero_for_mpe:
            return return_feats
        else:
            # print(f"rel_pos shape: {rel_pos.shape}") # test
            b, t, h, w = rel_pos.shape
            rel_pos = rel_pos.reshape(b * t, h * w)
            # rel_pos_emb = self.rel_pos_emb(rel_pos).reshape(b, t, h, w, -1).permute(0, 3, 1, 2) * self.alpha5
            rel_pos_emb = self.rel_pos_emb(rel_pos)
            # print(f'rel_pos_emb output shape: {rel_pos_emb.reshape(b, t, h, w, -1).permute(0, 1, 4, 2, 3).shape}') # test
            # print(f"direct shape: {direct.shape}") # test
            direct = direct.reshape(b * t, h * w, 4).to(torch.float32)
            # print(f"direct shape after: {direct.shape}") # test
            direct_emb = self.direct_emb(direct).reshape(b, t, h, w, -1).permute(0, 1, 4, 2, 3) * self.alpha6
            # print(f"direct_emb shape: {direct_emb.shape}") # test

            return return_feats, rel_pos_emb, direct_emb


class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc):
        super().__init__()
        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=input_nc, out_channels=64, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=kw, stride=2, padding=padw)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=kw, stride=2, padding=padw)
        self.bn3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=kw, stride=2, padding=padw)
        self.bn4 = nn.BatchNorm2d(512)

        self.conv5 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=kw, stride=1, padding=padw)
        self.bn5 = nn.BatchNorm2d(512)

        self.conv6 = nn.Conv2d(512, 1, kernel_size=kw, stride=1, padding=padw)

    def forward(self, x):
        conv1 = self.conv1(x)

        conv2 = self.conv2(conv1)
        conv2 = self.bn2(conv2.to(torch.float32))
        conv2 = self.act(conv2)

        conv3 = self.conv3(conv2)
        conv3 = self.bn3(conv3.to(torch.float32))
        conv3 = self.act(conv3)

        conv4 = self.conv4(conv3)
        conv4 = self.bn4(conv4.to(torch.float32))
        conv4 = self.act(conv4)

        conv5 = self.conv5(conv4)
        conv5 = self.bn5(conv5.to(torch.float32))
        conv5 = self.act(conv5)

        conv6 = self.conv6(conv5)

        outputs = conv6

        return outputs, [conv1, conv2, conv3, conv4, conv5]

class NLayerDiscriminator_video_2D(nn.Module):
    def __init__(self, config):
        super().__init__()
        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))

        self.input_nc = config.discriminator['input_nc']

        self.channel1 = config.generator['ngf']
        self.channel2 = self.channel1 * 2
        self.channel3 = self.channel2 * 2
        self.channel4 = self.channel3 * 2

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=self.input_nc, out_channels=self.channel1, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=self.channel1, out_channels=self.channel2, kernel_size=kw, stride=2, padding=padw)
        self.bn2 = nn.BatchNorm2d(self.channel2)

        self.conv3 = nn.Conv2d(in_channels=self.channel2, out_channels=self.channel3, kernel_size=kw, stride=2, padding=padw)
        self.bn3 = nn.BatchNorm2d(self.channel3)

        self.conv4 = nn.Conv2d(in_channels=self.channel3, out_channels=self.channel4, kernel_size=kw, stride=2, padding=padw)
        self.bn4 = nn.BatchNorm2d(self.channel4)

        self.conv5 = nn.Conv2d(in_channels=self.channel4, out_channels=self.channel4, kernel_size=kw, stride=1, padding=padw)
        self.bn5 = nn.BatchNorm2d(self.channel4)

        self.conv6 = nn.Conv2d(self.channel4, 1, kernel_size=kw, stride=1, padding=padw)

    def forward(self, x):
        b, t, c, h, w = x.shape
        x = x.view(b*t, c, h, w)
        conv1 = self.conv1(x)

        conv2 = self.conv2(conv1)
        conv2 = self.bn2(conv2.to(torch.float32))
        conv2 = self.act(conv2)

        conv3 = self.conv3(conv2)
        conv3 = self.bn3(conv3.to(torch.float32))
        conv3 = self.act(conv3)

        conv4 = self.conv4(conv3)
        conv4 = self.bn4(conv4.to(torch.float32))
        conv4 = self.act(conv4)

        conv5 = self.conv5(conv4)
        conv5 = self.bn5(conv5.to(torch.float32))
        conv5 = self.act(conv5)

        conv6 = self.conv6(conv5)

        outputs = conv6

        outputs = outputs.view(b, t, conv6.shape[-3], conv6.shape[-2], conv6.shape[-1])
        conv1 = conv1.view(b, t, conv1.shape[-3], conv1.shape[-2], conv1.shape[-1])
        conv2 = conv2.view(b, t, conv2.shape[-3], conv2.shape[-2], conv2.shape[-1])
        conv3 = conv3.view(b, t, conv3.shape[-3], conv3.shape[-2], conv3.shape[-1])
        conv4 = conv4.view(b, t, conv4.shape[-3], conv4.shape[-2], conv4.shape[-1])
        conv5 = conv5.view(b, t, conv5.shape[-3], conv5.shape[-2], conv5.shape[-1])

        return outputs, [conv1, conv2, conv3, conv4, conv5]

class NLayerDiscriminator_video_3D(nn.Module):
    def __init__(self, input_nc, num_frames):
        super().__init__()
        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=input_nc*num_frames, out_channels=64, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=kw, stride=2, padding=padw)
        self.bn2 = nn.BatchNorm3d(128)

        self.conv3 = nn.Conv3d(in_channels=128, out_channels=256, kernel_size=kw, stride=2, padding=padw)
        self.bn3 = nn.BatchNorm3d(256)

        self.conv4 = nn.Conv3d(in_channels=256, out_channels=512, kernel_size=kw, stride=2, padding=padw)
        self.bn4 = nn.BatchNorm3d(512)

        self.conv5 = nn.Conv3d(in_channels=512, out_channels=512, kernel_size=kw, stride=1, padding=padw)
        self.bn5 = nn.BatchNorm3d(512)

        self.conv6 = nn.Conv3d(512, 1, kernel_size=kw, stride=1, padding=padw)

    def forward(self, x):
        conv1 = self.conv1(x)

        conv2 = self.conv2(conv1)
        conv2 = self.bn2(conv2.to(torch.float32))
        conv2 = self.act(conv2)

        conv3 = self.conv3(conv2)
        conv3 = self.bn3(conv3.to(torch.float32))
        conv3 = self.act(conv3)

        conv4 = self.conv4(conv3)
        conv4 = self.bn4(conv4.to(torch.float32))
        conv4 = self.act(conv4)

        conv5 = self.conv5(conv4)
        conv5 = self.bn5(conv5.to(torch.float32))
        conv5 = self.act(conv5)

        conv6 = self.conv6(conv5)

        outputs = conv6

        return outputs, [conv1, conv2, conv3, conv4, conv5]
