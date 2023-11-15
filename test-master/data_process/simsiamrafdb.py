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
from facenet_pytorch import fixed_image_standardization
from PIL import ImageFilter
import torch.nn as nn
import torch.nn.functional as F

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class GradLayer(nn.Module):

    def __init__(self):
        super(GradLayer, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False)
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False)

    def get_gray(self,x):
        '''
        Convert image to its gray one.
        '''
        gray_coeffs = [65.738, 129.057, 25.064]
        convert = x.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
        x_gray = x.mul(convert).sum(dim=1)
        return x_gray.unsqueeze(1)

    def forward(self, x):
        # x_list = []
        # for i in range(x.shape[1]):
        #     x_i = x[:, i]
        #     x_i_v = F.conv2d(x_i.unsqueeze(1), self.weight_v, padding=1)
        #     x_i_h = F.conv2d(x_i.unsqueeze(1), self.weight_h, padding=1)
        #     x_i = torch.sqrt(torch.pow(x_i_v, 2) + torch.pow(x_i_h, 2) + 1e-6)
        #     x_list.append(x_i)

        # x = torch.cat(x_list, dim=1)
        if x.shape[1] == 3:
            x = self.get_gray(x)

        x_v = F.conv2d(x, self.weight_v, padding=1)
        x_h = F.conv2d(x, self.weight_h, padding=1)
        x = torch.sqrt(torch.pow(x_v, 2) + torch.pow(x_h, 2) + 1e-6)

        return x

class Sobel(object):
    def __init__(self,device):
        self.model = GradLayer()
        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()
        self.device = device

    def __call__(self,x):
        x_tensor = self.pil_to_tensor(x)
        c,w,h = x_tensor.size()
        x_tensor = x_tensor.unsqueeze(dim=0)
        x_grad = self.model(x_tensor)
        x_grad = x_grad.squeeze(dim=0).expand(c,w,h)
        x_img = self.tensor_to_pil(x_grad)
        return x_img

class SimSiamRAFDB(data.Dataset):
    def __init__(self,config,phase):
        super(SimSiamRAFDB, self).__init__()
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

        self.normal_data_transform = transforms.Compose([transforms.Resize((self.sz,self.sz)),
                                                   transforms.ToTensor()])

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

        img_tensor_q = self.data_transforms(img)
        img_tensor_k = self.data_transforms(img)
        img_tensor = self.normal_data_transform(img)
        return {'img_q':img_tensor_q,
                'img_k':img_tensor_k,
                'img':img_tensor,
                'label':label,
                'path':img_path}

    def __len__(self):
        return len(self.data_list)

