import os
from PIL import Image
import torch.utils.data as data
import numpy as np
import torchvision.transforms as transforms
import utils
import torch
import torch.nn as nn
from data_process.PoseCrop import PoseCrop
from data_process.DistanceCrop import DistanceCrop
from data_process.Sobel import Sobel

class PoseDB(data.Dataset):
    def __init__(self,config,phase):
        super(PoseDB, self).__init__()
        self.config = config
        self.phase = phase

        self.img_root = config['root']
        self.list_root = config[phase]
        self.sz = config['img_size']
        self.dataset = config['dataset']

        self.eval = config['eval']

        self.crop = PoseCrop()
        #self.crop = DistanceCrop()
        self.data_transforms = transforms.Compose([  # transforms.RandomResizedCrop(size=size),
            transforms.Resize((self.sz,self.sz)),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur(kernel_size=int(0.1 * self.sz))],p=0.8),
            transforms.RandomApply([Sobel()],p=0.6),
            transforms.ToTensor()])

        self.normal_data_transform = transforms.Compose([transforms.Resize((self.sz, self.sz)),
                                                         transforms.ToTensor()])

        if self.dataset == 'BU3D':
            self.data_list = self.get_list_BU3D(self.list_root)
        elif self.dataset == 'RAFDB':
            self.data_list = self.get_list_rafdb(self.list_root)
        else:
            raise Exception('Not Found dataset !')

    def get_list_rafdb(self,path):
        data_list = []
        with open(path, 'r') as imf:
            for i, line in enumerate(imf):
                arr = line.strip().split(' ')
                img_path = arr[0].split('.')[0] + '_aligned.jpg'
                label = arr[1]
                data_list.append((img_path, int(label)-1))
        return data_list

    def get_list_BU3D(self,path):
        data_list = []
        with open(path,'r') as imf:
            for i, line in enumerate(imf):
                arr = line.strip().split(',')
                img_path = arr[0]
                label = arr[2]
                data_list.append((img_path,int(label)))

        return data_list


    def __getitem__(self, item):
        img_path, label = self.data_list[item]
        img = Image.open(os.path.join(self.img_root,img_path)).convert('RGB')

        #images = [self.data_transforms(img) for i in range(2)]
        img_crop1,img_crop2 = self.crop(img)
        images = [self.data_transforms(img) for img in [img_crop1,img_crop2]]
        img_normal = self.normal_data_transform(img)

        return {'images':images,'img':img_normal,'path':img_path,'label':label}

    def __len__(self):
        return len(self.data_list)

class GaussianBlur(object):
    """blur a single image on CPU"""
    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        img = self.tensor_to_pil(img)

        return img