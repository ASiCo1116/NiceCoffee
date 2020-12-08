import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import select_conv, select_norm, conv_argument_transform

#--------------------------------------------------------------------------------
# residual_block.py contains the model of residual base CNN block
#--------------------------------------------------------------------------------


__all__ = ['BasicBlock', 'BottleBlock']


class BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel,
            dim = '2d',
            conv_fsize = (1, 1),
            stride = 1,
            short = None):

        super(BasicBlock, self).__init__()

        if dim.lower() not in ['1d', '2d']:
            raise ValueError("Argument: dim must be a str ['1d', '2d']")

        self.conv_fsize = conv_argument_transform(conv_fsize, dim)
        self.stride = conv_argument_transform(stride, dim)

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.dim = dim
        self.short = short

        self.conv1 = select_conv(dim, in_channel, out_channel, self.conv_fsize, self.stride)
        self.bn1 = select_norm(dim, out_channel, select = 'batch')

        self.relu = nn.ReLU(inplace = True)

        self.conv2 = select_conv(dim, out_channel, out_channel, self.conv_fsize, self.stride)
        self.bn2 = select_norm(dim, out_channel, select = 'batch')

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.short is not None:
            residual = self.short(x)

        out += residual
        out = self.relu(out)

        return out


class BottleBlock(nn.Module):
    def __init__(self, in_channel, out_channel,
            dim = '2d',
            conv_fsize = (1, 1),
            stride = 1,
            short = None):

        super(BottleBlock, self).__init__()

        if dim.lower() not in ['1d', '2d']:
            raise ValueError("Argument: dim must be a str ['1d', '2d']")

        self.conv_fsize = conv_argument_transform(conv_fsize, dim)
        self.stride = conv_argument_transform(stride, dim)

        self.latent_conv_fsize = conv_argument_transform(1, dim)
        self.latent_stride = conv_argument_transform(1, dim)

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.dim = dim
        self.short = short

        self.conv1 = select_conv(dim, in_channel, out_channel,
                self.latent_conv_fsize, self.latent_stride)
        self.bn1 = select_norm(dim, out_channel, select = 'batch')

        self.relu = nn.ReLU(inplace = True)

        self.conv2 = select_conv(dim, out_channel, out_channel, self.conv_fsize, self.stride)
        self.bn2 = select_norm(dim, out_channel, select = 'batch')

        self.conv3 = select_conv(dim, out_channel, out_channel,
                self.latent_conv_fsize, self.latent_stride)
        self.bn3 = select_norm(dim, out_channel, select = 'batch')

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.short is not None:
            residual = self.short(x)

        out += residual
        out = self.relu(out)

        return out


