import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
import numpy as np
import json
from torchvision import transforms

class CIFAR10Dataset(Dataset):
    
    def __init__(self, data_path):
        
        self.data_path = Path(data_path)
        self.im_dir = self.data_path/'PNG'
        
        with open(self.data_path/'labels.json', 'rb') as labs:
            self.labels = json.load(labs)
        
        self.ims = list(self.labels.keys())
        
        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        index = 5
        im_name = self.ims[index]
        im_path = self.im_dir/im_name
        im = Image.open(im_path)
        im = self.trans(im)
        lab = self.labels[im_name]
        return {'image':im,
                'labels':lab}

def CIFAR10DataLoader(bs):
    train_path = '/home/sam/data/CIFAR10/data/train'
    ds = CIFAR10Dataset(train_path)
    return DataLoader(ds, batch_size=bs,
                      shuffle=True, num_workers=4)