import os
from PIL import Image
import torch.utils.data as data
import numpy as np
import torchvision.transforms as transforms
import utils
import torch
import cv2
import albumentations as A
import random
from PIL import ImageFilter
from facenet_pytorch import fixed_image_standardization

class NormalRAFDB(data.Dataset):
    def __init__(self,config,phase):
        super(NormalRAFDB, self).__init__()
        self.config = config
        self.phase = phase

        self.img_root = config['root']
        self.list_root = config[phase]
        self.sz = config['img_size']

        if not config['eval']:
            self.data_transforms = transforms.Compose([
                transforms.RandomResizedCrop(self.sz, scale=(0.2, 1.)),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            if self.phase == 'train':
                self.data_transforms = transforms.Compose([
                    transforms.RandomResizedCrop(self.sz),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                ])
            else:
                self.data_transforms = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                ])


        self.data_list = self.get_list(self.list_root)

    def get_list(self,path):
        data_list = []
        with open(path, 'r') as imf:
            for i, line in enumerate(imf):
                arr = line.strip().split(' ')
                img_path = arr[0].split('.')[0] + '_aligned.jpg'
                label = arr[1]
                data_list.append((img_path, int(label)-1))
        return data_list

    def __getitem__(self, item):
        img_path, label = self.data_list[item]
        img = Image.open(os.path.join(self.img_root,img_path)).convert('RGB')

        img_tensor = self.data_transforms(img)
        return {'img':img_tensor,'label':label,'path':img_path}

    def __len__(self):
        return len(self.data_list)

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
