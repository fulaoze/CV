import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torchvision.transforms import transforms
import math
from PIL import Image


class DistanceCrop(object):
    def __init__(self,ratio=0.4):
        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()
        self.ratio = ratio
        self.border = ratio / 2
        self.transform_resize = transforms.Resize((224,224))

        self.area_id = [0,1,2,3]

    def __call__(self,x):
        x_tensor = self.pil_to_tensor(x)
        c,h,w = x_tensor.size()
        border_h = int(h * self.border)
        border_w = int(w * self.border)

        assert border_w < w // 2 and border_h < h //2

        pick_area = random.sample(self.area_id,2)

        point_list = []
        for pick_ids in pick_area:
            if pick_ids == 0:
                x = random.randint(border_w,w//2-border_w)
                y = random.randint(border_h,h//2 - border_h)
                point_list.append((x,y))
            elif pick_ids == 1:
                x = random.randint(w//2+border_w,w-border_w)
                y = random.randint(border_h,h//2 - border_h)
                point_list.append((x,y))
            elif pick_ids == 2:
                x = random.randint(border_w,w//2-border_w)
                y = random.randint(h//2+border_h,h-border_h)
                point_list.append((x,y))
            elif pick_ids == 3:
                x = random.randint(w//2+border_w,w-border_w)
                y = random.randint(h//2+border_h,h-border_h)
                point_list.append((x,y))

        #print(point_list)
        img1 = x_tensor[:,point_list[0][1]-border_h:point_list[0][1]+border_h,point_list[0][0]-border_w:point_list[0][0]+border_w]
        img2 = x_tensor[:,point_list[1][1]-border_h:point_list[1][1]+border_h,point_list[1][0]-border_w:point_list[1][0]+border_w]

        img1_crop = self.tensor_to_pil(img1)
        img2_crop = self.tensor_to_pil(img2)

        #img1_crop = self.transform_resize(img1_crop)
        #img2_crop = self.transform_resize(img2_crop)

        return img1_crop,img2_crop


# if __name__ == '__main__':
#     root = r'I:\Dataset\RAFDB\Basic\Image\aligned\aligned\train_02749_aligned.jpg'
#
#     dc = DistanceCrop()
#     img1_crop,img2_crop = dc(Image.open(root).convert('RGB'))
#     img1_crop.show()
#     img2_crop.show()



