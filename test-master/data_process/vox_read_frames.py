import cv2
import numpy as np
import os
import multiprocessing

root = '/media/disk1T/vox/dev/mp4'
out_root = '/media/disk1T/vox/dev/frames'
suffix = '.jpg'


def save_image(root, num, image):
    file_name = os.path.join(root, str(num) + suffix)
    # print(file_name)
    image = cv2.resize(image, (64, 64))
    # print(image.shape)
    cv2.imwrite(file_name, image)
    # print(file_name)


def process(vid_path, dir_name):
    if len(os.listdir(dir_name)) > 12:
        #print(len(os.listdir(dir_name)),dir_name)
        return
    videoCapture = cv2.VideoCapture(vid_path)
    fps = int(videoCapture.get(5))
    frames_num = int(videoCapture.get(7))
    i = 0
    # print(fps)
    ######每25帧取6帧##########
    index_in_25 = 0
    index = 0
    while True:
        success, frame = videoCapture.read()
        if success:
            index_in_25 = i % 25
            # 25里面分别取[ 0  4  8 12 16 20]
            if index_in_25 % 4 == 0 and index_in_25 < 24:
                # print(index_in_25)
                save_image(dir_name, index, frame)
                index += 1
            i = i + 1
            # print('save image vid name: ', file_name, '; frame num: ', i)
        else:
            break
    print('save image in : ', dir_name, '; frame num: ', frames_num, ';fps: ', fps)


def main(root):
    if not os.path.exists(out_root):
        os.mkdir(out_root)
    # path_list = os.listdir(root)
    pool = multiprocessing.Pool(processes=4)
    i = 0
    #### 读取root文件夹下的视频信息 ####
    for parent, dirnames, filenames in os.walk(root):
        #  traversal the files
        for filename in filenames:
            # print("Parent folder:", parent)
            # print("Filename:", filename)
            path = os.path.join(parent, filename)
            preffix = filename.split('.')[0]
            x1 = parent.split('/')[-1]
            x2 = parent.split('/')[-2]
            # dir_name = os.path.join(out_root, x2, x1, preffix)
            dir_name = os.path.join(out_root, x2)
            if not os.path.exists(dir_name):
                os.mkdir(dir_name)
            dir_name = os.path.join(dir_name, x1)
            if not os.path.exists(dir_name):
                os.mkdir(dir_name)
            dir_name = os.path.join(dir_name, preffix)
            if not os.path.exists(dir_name):
                os.mkdir(dir_name)
            pool.apply_async(process, args=(path, dir_name))
            # process(path, dir_name)
        #     i += 1
        #     break
        # if i != 0:
        #     break
    pool.close()
    pool.join()


if __name__ == '__main__':
    main(root)
    print("finish!!!!!!!!!!!!!!!!!!")
