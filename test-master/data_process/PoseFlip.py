import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torchvision.transforms import transforms
import math
from PIL import Image
from data_process.Sobel import Sobel
from data_process.reconsimclrdb import *

class PoseFlip(object):
    def __init__(self):
        self.flip = transforms.RandomHorizontalFlip(p=1.0)
        self.rotation = transforms.RandomRotation(15)

    def __call__(self, x):
        x = self.flip(x)
        x = self.rotation(x)
        return x


if __name__ == '__main__':
    root = r'I:\Dataset\voxceleb_sample\processed_align\processed_align\Aaron_Tveit\8mWxQ6DRO-U\2\02.jpg'

    img = Image.open(root).convert('RGB')

    pf = PoseFlip()

    img_ro = pf(img)
    img_ro.show()