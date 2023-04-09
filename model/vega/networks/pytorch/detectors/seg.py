from __future__ import absolute_import, division, print_function

from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

def upsample(x):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=2, mode="nearest")

class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out

class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """

    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out

class SegDecoder(nn.Module):
    def __init__(self,
                 num_ch_enc,
                 num_ch_dec=None,
                 num_output_channels=10,
                 use_skips=True,
                 ):
        super(SegDecoder, self).__init__()
        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = num_ch_dec

        # decoder
        self.convs = OrderedDict()
        for i in range(len(num_ch_enc)-1, -1, -1):
            
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == len(num_ch_enc)-1 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        self.convs[("output")] = Conv3x3(self.num_ch_dec[0], self.num_output_channels)

        self.decoder = nn.ModuleList(self.convs.values())

    def forward(self, input_features):
        output_guide = list()
        # decoder
        x = input_features[-1]
        for i in range(0,len(input_features)):
            
            # up 1
            x = self.decoder[2*i](x)
            x = [upsample(x)]
            if self.use_skips and i < len(input_features)-1:
                x += [input_features[ len(input_features)-2 - i]]
            x = torch.cat(x, 1)
            
            # up 2
            x = self.decoder[2*i + 1](x)

            # for semantic guide
            output_guide.append(x)

        output_seg = self.decoder[-1](upsample(x))

        return output_seg, output_guide

