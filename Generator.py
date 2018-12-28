import torch
from torch import nn
from math import log
import hyperparams as hp
from SpectralNorm import SpecNorm
from ConditionalBatchNorm import ConditionalBatchNorm2d
from Attention import AttentionMech


class Generator(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.block_1 = initial_conv_block()
        self.block_2 = up_and_conv_block(512,512)
        self.block_3 = up_and_conv_block(512)
        self.block_4 = up_and_conv_block(256)
        self.block_5 = up_and_conv_block(128, upscale=False)
        self.block_6 = up_and_conv_block(64, upscale=False)
        
        self.attn_1 = AttentionMech(512)
        self.attn_2 = AttentionMech(256)
        
        self.toRGB = toRGB(32)
        
    def forward(self, x, cls):
        x = self.block_1(x, cls)
        x = self.block_2(x, cls)
        x = self.attn_1(x)
        x = self.block_3(x, cls)
        x = self.attn_2(x)
        x = self.block_4(x, cls)
        x = self.block_5(x, cls)
        x = self.block_6(x, cls)
        return self.toRGB(x)

    
class initial_conv_block(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = conv_layer(512, 512, 4, padding=3)
        self.conv_2 = conv_layer(512, 512)

    def forward(self,latent_vector, cls):
        x = self.conv_1(latent_vector, cls)
        x = self.conv_2(x, cls)
        return x
  

class up_and_conv_block(nn.Module):
    def __init__(self, in_c, out_c=None, upscale=True):
        super().__init__()
        self.upscale = upscale
        if not out_c: out_c = int(in_c/2)
            
        self.conv_1 = conv_layer(in_c, out_c)
        self.conv_2 = conv_layer(out_c, out_c)

        if in_c != out_c:
            self.res_conv = SpecNorm(nn.Conv2d(in_c, out_c, 1))

    def forward(self, x, cls):
        if self.upscale:
            x = nn.functional.interpolate(x, scale_factor=2)
        fx = self.conv_1(x, cls)
        fx = self.conv_2(fx, cls)
        if hasattr(self, 'res_conv'):
            return fx + self.res_conv(x)
        else:
            return fx + x
        

class toRGB(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.final_conv = SpecNorm(nn.Conv2d(in_c, 3, 1, padding=0))
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.final_conv(x)
        return self.tanh(x)

    
class conv_layer(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3,
                 padding=1):
        super().__init__()
        self.conv = SpecNorm(nn.Conv2d(in_c, out_c,
                                       kernel_size=kernel_size,
                                       padding=padding))
        self.cond_bn = ConditionalBatchNorm2d(out_c, hp.num_classes)
        self.l_relu = nn.LeakyReLU(negative_slope=hp.leakiness)
        self.dropout = nn.Dropout(p=hp.dropout)

    def forward(self, x, cls):
        x = self.conv(x)
        x = self.cond_bn(x, cls)
        x = self.dropout(x)
        return self.l_relu(x)