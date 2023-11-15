import cv2 as cv
import math
import numpy as np

def En(image): # 光线照射
    x, y,_ = image.shape  # 获取图片大小
    radius = np.random.randint(10, int(min(x, y)), 1)  #
    pos_x = np.random.randint(0, (min(x, y) - radius), 1)  # 获取人脸光照区域的中心点坐标
    pos_y = np.random.randint(0, (min(x, y) - radius), 1)  # 获取人脸光照区域的中心坐标
    pos_x = int(pos_x[0])
    pos_y = int(pos_y[0])
    radius = int(radius[0])
    strength = 100
    for j in range(pos_y - radius, pos_y + radius):
        for i in range(pos_x-radius, pos_x+radius):

            distance = math.pow((pos_x - i), 2) + math.pow((pos_y - j), 2)
            distance = np.sqrt(distance)
            if distance < radius:
                result = 1 - distance / radius
                result = result*strength
                # print(result)
                image[i, j, 0] = min((image[i, j, 0] + result),255)
                image[i, j, 1] = min((image[i, j, 1] + result),255)
                image[i, j, 2] = min((image[i, j, 2] + result),255)
    image = image.astype(np.uint8)
    return image

def De(image): # 阴影
    x, y,_ = image.shape  # 获取图片大小
    radius = np.random.randint(10, int(min(x, y)), 1)  #
    pos_x = np.random.randint(0, (min(x, y) - radius), 1)  # 获取人脸光照区域的中心点坐标
    pos_y = np.random.randint(0, (min(x, y) - radius), 1)  # 获取人脸光照区域的中心坐标
    pos_x = int(pos_x[0])
    pos_y = int(pos_y[0])
    radius = int(radius[0])
    strength = 100
    for j in range(pos_y - radius, pos_y + radius):
        for i in range(pos_x-radius, pos_x+radius):

            distance = math.pow((pos_x - i), 2) + math.pow((pos_y - j), 2)
            distance = np.sqrt(distance)
            if distance < radius:
                result = 1 - distance / radius
                result = result*strength
                # print(result)
                image[i, j, 0] = max((image[i, j, 0] - result),0)
                image[i, j, 1] = max((image[i, j, 1] - result),0)
                image[i, j, 2] = max((image[i, j, 2] - result),0)
    image = image.astype(np.uint8)
    return image

def Ma(image): # 遮挡
    x, y,_ = image.shape  # 获取图片大小
    mask_size = np.random.randint(10, 50, 1)
    pos_x = np.random.randint(10, (min(x, y) - 50), 1)  # 获取人脸光照区域的中心点坐标
    pos_y = np.random.randint(10, (min(x, y) - 50), 1)  # 获取人脸光照区域的中心坐标
    pos_x = int(pos_x[0])
    pos_y = int(pos_y[0])
    mask_size = int(mask_size[0])
    image[pos_x:pos_x + mask_size, pos_y:pos_y + mask_size] = 0
    return image

img = cv.imread(r'E:\DL\self_sup\pre_process\test_0016_aligned.jpg')
img1 = En(img)
#img1 = De(img)

cv.imwrite(r'E:\DL\self_sup\pre_process\sunny.jpg',img1)
