import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torchvision.transforms import transforms
from PIL import Image

class PoseCrop(object):
    def __init__(self,ratio=0.1):
        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

        self.ratio = ratio

    def __call__(self, x):
        x_tensor = self.pil_to_tensor(x)
        c,h,w = x_tensor.size()
        new_h = h // 2

        h_border = int(new_h * self.ratio)

        tensor_crop1 = x_tensor[:,:new_h-h_border,:]
        tensor_crop2 = x_tensor[:,new_h+h_border:,:]

        img_crop1 = self.tensor_to_pil(tensor_crop1)
        img_crop2 = self.tensor_to_pil(tensor_crop2)

        return img_crop1,img_crop2

if __name__ == '__main__':
    root = r'I:\Dataset\RAFDB\Basic\Image\aligned\aligned\train_02749_aligned.jpg'
    pc = PoseCrop()
    img1_crop,img2_crop = pc(Image.open(root).convert('RGB'))
    img1_crop.show()
    img2_crop.show()