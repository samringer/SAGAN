import torch
from torch import nn
from DataLoader import CIFAR10DataLoader
import hyperparams as hp
from SpectralNorm import SpecNorm
from Attention import AttentionMech
from math import log


class Discriminator(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.fromRGB = conv_layer(3,64, kernel_size=1, padding=0)
        self.block_1 = conv_block(64, 128, pool=False)
        self.block_2 = conv_block(128, 256, pool=False)
        self.block_3 = conv_block(256, 512)
        self.block_4 = conv_block(512, 512)
        self.block_5 = conv_block(512, 512)
        
        self.attn_1 = AttentionMech(512)
        self.attn_2 = AttentionMech(512)
        
        self.cls_embed = SpecNorm(nn.Embedding(hp.num_classes, 512*4*4))
        self.fc = SpecNorm(nn.Linear(512*4*4, 1))
    
    def forward(self, x, cls):
        x = self.fromRGB(x)
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.attn_1(x)
        x = self.block_4(x)
        x = self.attn_2(x)
        x = self.block_5(x)
        x = x.view(-1, 4*4*512)
        cls_embed = self.cls_embed(cls).view(-1, 4*4*512, 1)
        return self.fc(x) + torch.bmm(x.view(-1, 1, 4*4*512), cls_embed)
    
    
class conv_layer(nn.Module):
    def __init__(self, in_c, out_c,
                 kernel_size=3,
                 padding=1):
        super().__init__()
        self.conv = SpecNorm(nn.Conv2d(in_c, out_c, kernel_size,
                              padding=padding))
        self.l_relu = nn.LeakyReLU(negative_slope=hp.leakiness)
        self.dropout = nn.Dropout(p=hp.dropout)

    def forward(self,x):
        x = self.conv(x)
        x = self.dropout(x)
        return self.l_relu(x)
    
    
class conv_block(nn.Module):
    def __init__(self, in_c, out_c, pool=True):
        super().__init__()
        self.conv_1 = conv_layer(in_c, in_c)
        self.conv_2 = conv_layer(in_c, out_c)
        self.pool = pool
        self.ave_pool = nn.AvgPool2d(kernel_size = 2)

        if in_c != out_c:
            self.res_conv = SpecNorm(nn.Conv2d(in_c,out_c,kernel_size=1,stride=1))

    def forward(self, x):
        fx = self.conv_1(x)
        fx = self.conv_2(fx)
        if hasattr(self, 'res_conv'):
            x = fx + self.res_conv(x)
        else:
            x = fx + x
        if self.pool: x = self.ave_pool(x)
        return x

    
class final_block(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = conv_layer(512, 512)
        self.conv_2 = conv_layer(512, 512, 4, padding=0)
        self.linear = SpecNorm(nn.Linear(512, 1, bias=False))

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x.squeeze_().squeeze_()
        x = self.linear(x)
        return x