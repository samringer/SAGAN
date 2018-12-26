import torch
from torch import nn
from DataLoader import CIFAR10DataLoader
import hyperparams as hp
from SpectralNorm import SpecNorm
from math import log


class conv_layer(nn.Module):
    def __init__(self, input_channels, output_channels,
                 leakiness = 0.2, kernel_size = 3,
                 stride = 1,padding = 1):
        super().__init__()
        self.conv = SpecNorm(nn.Conv2d(input_channels,
                              output_channels,
                              kernel_size=kernel_size,
                              stride = stride,
                              padding = padding))
        self.l_relu = nn.LeakyReLU(negative_slope=leakiness)
        self.dropout = nn.Dropout(p=hp.dropout)

    def forward(self,x):
        x = self.conv(x)
        x = self.dropout(x)
        return self.l_relu(x)
    
    
class conv_block(nn.Module):
    def __init__(self, input_channels, output_channels, leakiness = 0.2, pool=True):
        super().__init__()
        self.in_c = input_channels
        self.out_c = output_channels
        assert log(self.in_c,2) % 1 == 0
        self.conv_1 = conv_layer(self.in_c, self.in_c)
        self.conv_2 = conv_layer(self.in_c, self.out_c)
        self.pool = pool
        self.ave_pool = nn.AvgPool2d(kernel_size = 2)

        if self.in_c != self.out_c:
            self.res_conv = SpecNorm(nn.Conv2d(self.in_c,self.out_c,kernel_size=1,stride=1))

    def forward(self, x):
        fx_1 = self.conv_1(x)
        fx_2 = self.conv_2(fx_1)
        if hasattr(self, 'res_conv'):
            pre_pooled_output = fx_2 + self.res_conv(x)
        else:
            pre_pooled_output = fx_2 + x
        x = pre_pooled_output
        if self.pool:
            x = self.ave_pool(x)
        return x

class final_block(nn.Module): #Always takes in 513x4x4
    def __init__(self):
        super().__init__()
        self.conv_1 = conv_layer(512, 512)
        self.conv_2 = conv_layer(512, 512, kernel_size = 4, padding = 0)
        self.linear = SpecNorm(nn.Linear(512, 1, bias=False))

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x.squeeze_()
        x.squeeze_()
        x = self.linear(x)
        return x
    
    
class Discriminator(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.fromRGB = conv_layer(3,64, leakiness = 0.2, kernel_size = 1, stride = 1, padding = 0)
        self.block_1 = conv_block(64, 128, pool = False)
        self.block_2 = conv_block(128, 256, pool = False)
        self.block_3 = conv_block(256, 512)
        self.block_4 = conv_block(512, 512)
        self.block_5 = conv_block(512, 512)
        self.final_block = final_block()
    
    def forward(self, x):
        x = self.fromRGB(x)
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)
        return self.final_block(x)