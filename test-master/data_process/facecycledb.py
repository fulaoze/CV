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

# root = r'I:\EST\WECLNet_end_to_end2\data_path\Bu3d_face_95img\Train\F025\Disgust\013.png'
#
# img = Image.open(root).convert('RGB')
# transform = transforms.RandomHorizontalFlip(1.)
# img_flip = transform(img)
# img_flip.show()

class FaceCycleDB(data.Dataset):
    def __init__(self,config,phase):
        super(FaceCycleDB, self).__init__()
        self.config = config
        self.phase = phase

        self.img_root = config['root']
        self.list_root = config[phase]
        self.sz = config['img_size']
        if config['use_aug']:
            self.data_transforms = transforms.Compose([transforms.Resize((self.sz,self.sz)),
                                                       transforms.RandomHorizontalFlip(0.5),
                                                       transforms.ToTensor()])
        else:
            self.data_transforms = transforms.Compose([transforms.Resize((self.sz, self.sz)),
                                                       transforms.ToTensor()])

        self.data_list = self.get_list(self.list_root)

    def get_list(self,path):
        data_list = []
        with open(path,'r') as imf:
            for i, line in enumerate(imf):
                arr = line.strip().split(' ')
                mid_face_path = arr[0]
                exp_face_path = arr[1]
                label = int(arr[2])
                data_list.append((mid_face_path,exp_face_path,label))

        return data_list

    def __getitem__(self, item):
        mid_face_path,exp_face_path,label = self.data_list[item]
        mid_img = Image.open(os.path.join(self.img_root,mid_face_path)).convert('RGB')
        exp_img = Image.open(os.path.join(self.img_root,exp_face_path)).convert('RGB')

        mid_img_tensor = self.data_transforms(mid_img)
        exp_img_tensor = self.data_transforms(exp_img)

        return {'mid_img':mid_img_tensor,
                'exp_img':exp_img_tensor,
                'mid_img_path':mid_face_path,
                'exp_img_path':exp_face_path,
                'label':label}

    def __len__(self):
        return len(self.data_list)