"""
    Pytorch Implementation of: 
        Learning Binary Residual Representations for Domain-specific Video Streaming
"""
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config

config = Config()
C = config.numChannels
L = config.numLayers
kz = config.kernelSize
pad = int(kz / 2)

class EncodeBlock(nn.Module):
    def __init__(self, in_channel, activation=True):
        super(EncodeBlock, self).__init__()
        
        block = [ nn.Conv2d(in_channel, C, kz, stride=2, padding=pad),
                  nn.BatchNorm2d(C) ]
        if activation:
            block += [nn.ReLU()]

        self.block = nn.Sequential(*(block))

    def forward(self, x):
        return self.block(x)

class DecodeBlock(nn.Module):
    def __init__(self, last_layer=False, out_channel=3):
        super(DecodeBlock, self).__init__()

        if not last_layer:
            block = [ nn.Conv2d(C, 4*C, kz, stride=1, padding=pad),
                      nn.PixelShuffle(2),
                      nn.BatchNorm2d(C), 
                      nn.ReLU() ]
        else:
            block = [ nn.Conv2d(C, 4*out_channel, kz, stride=1, padding=pad),
                      nn.PixelShuffle(2)]

        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)

class Binarize_bp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = input.clone()
        out[input < 0] = -1
        out[input >= 0] = 1
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input =  grad_output.clone()
        grad_input[input < -1] = 0
        grad_input[input >  1] = 0
        return grad_input

class Binarizer(nn.Module):
    def __init__(self):
        block = [ nn.Hardtanh() ]
        self.block = block

    def forward(self, x):
        return self.block(x)

class Encoder(nn.Module):
    def __init__(self, in_channel=3):
        super(Encoder, self).__init__()

        block = [('encblk' + str(1), EncodeBlock(in_channel))]
        for i in range(1, L - 1):
            block += [('encblk' + str(i+1), EncodeBlock(C))]
        block += [('encblk' + str(L), EncodeBlock(C, activation=False))]

        self.block = nn.Sequential(OrderedDict(block))
        self.binarizer = Binarize_bp.apply

    def forward(self, x):
        x = self.block(x)
        out = self.binarizer(x)
        return out


class Decoder(nn.Module):
    def __init__(self, out_channel=3):
        super(Decoder, self).__init__()
        
        block = [('decblk' + str(1), DecodeBlock())]
        for i in range(1, L - 1):
            block += [('decblk' + str(i+1), DecodeBlock())]
        block += [('decblk' + str(L), DecodeBlock(last_layer=True, out_channel=out_channel))]

        self.block = nn.Sequential(OrderedDict(block))

    def forward(self, x):
        return self.block(x)
