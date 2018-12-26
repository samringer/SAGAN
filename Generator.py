import torch
from torch import nn
from math import log
import hyperparams as hp
from SpectralNorm import SpecNorm
from ConditionalBatchNorm import ConditionalBatchNorm2d

class conv_layer(nn.Module):
    def __init__(self, in_channels,
                      out_channels,
                      kernel_size = 3,
                      stride = 1,
                      padding = 1,
                      leakiness = 0.2):
        super().__init__()

        self.conv = SpecNorm(nn.Conv2d(in_channels,
                                out_channels,
                                kernel_size = kernel_size,
                                stride = stride,
                                padding = padding))

        self.cond_bn = ConditionalBatchNorm2d(out_channels,
                                              hp.num_classes)
        self.l_relu = nn.LeakyReLU(negative_slope=leakiness)
        self.dropout = nn.Dropout(p=hp.dropout)


    def forward(self, x, cls):
        x = self.conv(x)
        x = self.cond_bn(x, cls)
        x = self.dropout(x)
        return self.l_relu(x)
    
    
class toRGB(nn.Module):
    def __init__(self,
                 in_c,
                 out_c = 3,
                 kernel_size = 1,
                 stride = 1,
                 padding = 0):
        super().__init__()
        self.final_conv = SpecNorm(nn.Conv2d(in_c,
                                    out_c,
                                    kernel_size = kernel_size,
                                    padding = padding,
                                    stride = stride))
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.final_conv(x)
        return self.tanh(x)

    
class initial_conv_block(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = conv_layer(512,
                                     512,
                                     kernel_size=4,
                                     padding = 3)
        self.conv_2 = conv_layer(512, 512)


    def forward(self,latent_vector, cls):
        x = self.conv_1(latent_vector, cls)
        x = self.conv_2(x, cls)
        return x

    
class up_and_conv_block(nn.Module):
    def __init__(self, initial_number_channels,
                 o_n_c=None, upscale=True):
        super().__init__()
        self.i_n_c = initial_number_channels
        self.o_n_c = o_n_c
        self.upscale = upscale
        if not self.o_n_c:
            self.o_n_c = int(initial_number_channels / 2)
        assert log(self.o_n_c,2) % 1 == 0
        self.conv_1 = conv_layer(self.i_n_c, self.o_n_c)
        self.conv_2 = conv_layer(self.o_n_c, self.o_n_c)

        if self.i_n_c != self.o_n_c:
            self.res_conv = SpecNorm(nn.Conv2d(self.i_n_c,self.o_n_c,kernel_size=1,stride=1))

    def forward(self, x, cls):
        if self.upscale:
            x = nn.functional.interpolate(x, scale_factor = 2)
        fx_1 = self.conv_1(x, cls)
        fx_2 = self.conv_2(fx_1, cls)
        if hasattr(self, 'res_conv'):
            return fx_2 + self.res_conv(x)
        else:
            return fx_2 + x
        
        
class Generator(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.block_1 = initial_conv_block()
        self.block_2 = up_and_conv_block(512,512)
        self.block_3 = up_and_conv_block(512)
        self.block_4 = up_and_conv_block(256)
        self.block_5 = up_and_conv_block(128, upscale=False)
        self.block_6 = up_and_conv_block(64, upscale=False)
        self.toRGB = toRGB(32)
        
    def forward(self, x, cls):
        x = self.block_1(x, cls)
        x = self.block_2(x, cls)
        x = self.block_3(x, cls)
        x = self.block_4(x, cls)
        x = self.block_5(x, cls)
        x = self.block_6(x, cls)
        return self.toRGB(x)
