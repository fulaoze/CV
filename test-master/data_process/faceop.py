import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torchvision.transforms import transforms
import math
from PIL import Image
import math
import numpy as np
import cv2 as cv
import os

class FaceOP(object):
    def __init__(self,file_root):
        self.sunny_file_list = os.listdir(os.path.join(file_root,'sun'))
        self.shallow_file_list = os.listdir(os.path.join(file_root,'shallow'))

        self.sunny_matrix = []
        for p in self.sunny_file_list:
            full_path = os.path.join(file_root,'sun',p)
            op = np.loadtxt(full_path)
            op = np.repeat(np.expand_dims(op,2),3,2)
            self.sunny_matrix.append(op)
        
        self.shallow_matrix = []
        for p in self.shallow_file_list:
            full_path = os.path.join(file_root,'shallow',p)
            op = np.loadtxt(full_path)
            op = np.repeat(np.expand_dims(op,2),3,2)
            self.shallow_matrix.append(op)

    def cv2pil(self,cv_img):
        return Image.fromarray(cv.cvtColor(cv_img,cv.COLOR_BGR2RGB))

    def pil2cv(self,pil_img):
        return cv.cvtColor(np.asarray(pil_img),cv.COLOR_RGB2BGR)

    def __call__(self,image):
        w,h=image.size
        if w==64:
            image = image.resize((100,100))
        sunny_prob = random.random()
        cv_img = self.pil2cv(image)
        if sunny_prob>0.5:
            cv_img = self.sunny(cv_img)
        else:
            cv_img = self.shallow(cv_img)

        # erase_prob = random.random()
        # if erase_prob > 0.5:
        #     cv_img = self.erase(cv_img)
        
        pil_img = self.cv2pil(cv_img)
        return pil_img

    def sunny(self,image):
        # x, y,_ = image.shape  # 获取图片大小
        # radius = np.random.randint(10, int(min(x, y)), 1)  #
        # pos_x = np.random.randint(0, (min(x, y) - radius), 1)  # 获取人脸光照区域的中心点坐标
        # pos_y = np.random.randint(0, (min(x, y) - radius), 1)  # 获取人脸光照区域的中心坐标
        # pos_x = int(pos_x[0])
        # pos_y = int(pos_y[0])
        # radius = int(radius[0])
        # strength = 100
        # for j in range(pos_y - radius, pos_y + radius):
        #     for i in range(pos_x-radius, pos_x+radius):

        #         distance = math.pow((pos_x - i), 2) + math.pow((pos_y - j), 2)
        #         distance = np.sqrt(distance)
        #         if distance < radius:
        #             result = 1 - distance / radius
        #             result = result*strength
        #             # print(result)
        #             image[i, j, 0] = min((image[i, j, 0] + result),255)
        #             image[i, j, 1] = min((image[i, j, 1] + result),255)
        #             image[i, j, 2] = min((image[i, j, 2] + result),255)
        # image = image.astype(np.uint8)
        op = random.choice(self.sunny_matrix)
        image = np.clip(image + op, 0, 255).astype('uint8')
        return image

    def shallow(self,image):
        # x, y,_ = image.shape  # 获取图片大小
        # radius = np.random.randint(10, int(min(x, y)), 1)  #
        # pos_x = np.random.randint(0, (min(x, y) - radius), 1)  # 获取人脸光照区域的中心点坐标
        # pos_y = np.random.randint(0, (min(x, y) - radius), 1)  # 获取人脸光照区域的中心坐标
        # pos_x = int(pos_x[0])
        # pos_y = int(pos_y[0])
        # radius = int(radius[0])
        # strength = 100
        # for j in range(pos_y - radius, pos_y + radius):
        #     for i in range(pos_x-radius, pos_x+radius):

        #         distance = math.pow((pos_x - i), 2) + math.pow((pos_y - j), 2)
        #         distance = np.sqrt(distance)
        #         if distance < radius:
        #             result = 1 - distance / radius
        #             result = result*strength
        #             # print(result)
        #             image[i, j, 0] = max((image[i, j, 0] - result),0)
        #             image[i, j, 1] = max((image[i, j, 1] - result),0)
        #             image[i, j, 2] = max((image[i, j, 2] - result),0)
        # image = image.astype(np.uint8)
        op = random.choice(self.shallow_matrix)
        image = np.clip(image - op, 0, 255).astype('uint8')
        return image

    def erase(self,image):
        x, y,_ = image.shape  # 获取图片大小
        mmin = 10
        mmax = min([30,x,y])
        mask_size = np.random.randint(mmin, mmax, 1)
        pos_x = np.random.randint(mmin, (min(x, y) - mmax), 1)  # 获取人脸光照区域的中心点坐标
        pos_y = np.random.randint(mmin, (min(x, y) - mmax), 1)  # 获取人脸光照区域的中心坐标
        pos_x = int(pos_x[0])
        pos_y = int(pos_y[0])
        mask_size = int(mask_size[0])
        image[pos_x:pos_x + mask_size, pos_y:pos_y + mask_size] = 0
        return image

#if __name__ == '__main__':
    # root = r'E:\\DL\\self_sup\\pre_process\\test_0001_aligned.jpg'
    # txt_root = r'I:\\Dataset\\transform\\transform\\txt\\shallow\\53_40_40.txt'

    # img = Image.open(root).convert('RGB')
    # op = np.loadtxt(txt_root)
    # op = np.repeat(np.expand_dims(op,2),3,2)

    # #op = FaceOP()
    # img = cv.cvtColor(np.asarray(img),cv.COLOR_RGB2BGR)
    # # shallow
    # #img = np.clip(img - op,0,255).astype('uint8')

    # # sunny
    # img = np.clip(img + op,0,255).astype('uint8')

    # img = Image.fromarray(cv.cvtColor(img,cv.COLOR_BGR2RGB))
    # img = img.resize((64,64))
    # #img1 = op(img)
    # img.save(r'E:\\DL\\self_sup\\pre_process\\test_sunny.jpg')