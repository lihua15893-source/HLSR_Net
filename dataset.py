from torch.utils.data import Dataset, DataLoader
import os
from utils.utils import *
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from utils.augmentations import Compose, ColorJitter, RandomHorizontalFlip, RandomCrop, RandomScale
import torchvision.transforms as transform

#数据预处理
scale_range = (0.5,2.0)
crop_size = (1024,1024)
brightness = 0.5
contrast = 0.5
saturation = 0.5
p = 0.5

transform = transform.Compose([
    transform.ToTensor(),  
])

class my_Dataset(Dataset):
    def __init__(self, size, mode = 'train', num_class = 2, enhance=True):
        self.mode = mode    # train or val
        self.enhance = enhance  
        self.scale_range = (0.5, 2.0)
        self.crop_size = (size, size)
        self.brightness = 0.5
        self.contrast = 0.5
        self.saturation = 0.5
        self.p = 0.5
        # if self.mode=='train':
        #     self.img_path = '../../data/Data_BSD/train'

        # elif self.mode=='val':
        #     self.img_path = '../../data/Data_BSD/val'
        # else :
        #     self.img_path = '../../data/Data_BSD/test'

        
        if self.mode=='train':
            self.img_path = '../../data/SLSD/train'
        elif self.mode=='val':
            self.img_path = '../../data/SLSD/val'
        else :
            self.img_path = '../../data/SLSD/test'

        
        self.label_list = os.listdir(os.path.join(self.img_path, 'label'))


        self.size = size
        self.num_class = num_class
 
        self.aug_train = Compose([
            ColorJitter(brightness=self.brightness,  contrast=self.contrast, saturation=self.saturation),
            RandomHorizontalFlip(self.p),
            RandomScale(self.scale_range),
            RandomCrop(self.crop_size, pad_if_needed=True)
        ])
        self.aug_val = Compose([
            RandomCrop(self.crop_size, pad_if_needed=True)
        ])

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, idx):
        label_item = self.label_list[idx]
        label_name = label_item.split()[0]
        label_path = os.path.join(self.img_path, 'label', label_name)
        img_path = os.path.join(self.img_path, 'image', label_name)
        img = read_image(img_path,isLabel=False)
        label = read_image(label_path,isLabel=True)
        img = Resize_Image(img, self.size, mode='image')
        label = Resize_Image(label, self.size, mode='label')

        sample = {'image':img, 'label':label}

        if self.mode=='train':
            if self.enhance :
                sample = self.aug_train(sample)

            img = transform(sample['image'])   
            label = torch.from_numpy(label_norm(sample['label'])).long()   
        if self.mode=='val':
            if self.enhance:
                sample = self.aug_val(sample)
            img = transform(sample['image'])
            label = torch.from_numpy(label_norm(sample['label'])).long() 

        return img, label


