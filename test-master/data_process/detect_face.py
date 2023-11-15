from facenet_pytorch import MTCNN,InceptionResnetV1
from PIL import Image
import face_recognition
import cv2
import math
import numpy as np
import multiprocessing
import os

# img = Image.open(root).convert('RGB')
# img = np.array(img)
# face_landmarks_list = face_recognition.face_landmarks(img,model='large')
# face_landmarks_dict = face_landmarks_list[0]
# print(face_landmarks_dict)
# mtcnn = MTCNN(image_size=64)

def align_face(image_array,landmarks):
    left_eye = landmarks['left_eye']
    right_eye = landmarks['right_eye']
    # calculate the mean point of landmarks of left and right eye
    left_eye_center = np.mean(left_eye, axis=0).astype("int")
    right_eye_center = np.mean(right_eye, axis=0).astype("int")
    # compute the angle between the eye centroids
    dy = right_eye_center[1] - left_eye_center[1]
    dx = right_eye_center[0] - left_eye_center[0]
    # compute angle between the line of 2 centeroids and the horizontal line
    angle = math.atan2(dy, dx) * 180. / math.pi
    # calculate the center of 2 eyes
    eye_center = ((left_eye_center[0] + right_eye_center[0]) / 2,
                  (left_eye_center[1] + right_eye_center[1]) / 2)
    # at the eye_center, rotate the image by the angle
    rotate_matrix = cv2.getRotationMatrix2D(eye_center, angle, scale=1)
    rotated_img = cv2.warpAffine(image_array, rotate_matrix, (image_array.shape[1], image_array.shape[0]))
    return rotated_img, eye_center, angle

# algned_face,eye_center,angle = align_face(img,landmarks=face_landmarks_dict)
# algn_img = Image.fromarray(algned_face)
# img_cropped = mtcnn(algn_img,save_path=out_root)
#img_cropped.save(out_root)

#img_cropped = mtcnn(img,save_path=out_root)

def process(save_dir_name,dir_name,mtcnn):
    if not os.path.exists(save_dir_name):
        os.makedirs(save_dir_name)
    elif len(os.listdir(dir_name)) == len(os.listdir(save_dir_name)):
        return
        

    img_path_list = os.listdir(dir_name)

    for img_path in img_path_list:
        try:
            img = Image.open(os.path.join(dir_name,img_path)).convert('RGB')
            img_array = np.array(img)
            face_landmarks_list = face_recognition.face_landmarks(img_array,model='large')
            face_landmarks_dict = face_landmarks_list[0]

            aligned_face, eye_center,angle = align_face(img_array,face_landmarks_dict)
            aligned_img = Image.fromarray(aligned_face)

            save_img_path = os.path.join(save_dir_name,img_path)
            mtcnn(aligned_img,save_path=save_img_path)
        except Exception:
            continue

def main(root,out_root):
    if not os.path.exists(out_root):
        os.mkdir(out_root)

    pool = multiprocessing.Pool(processes=1)
    mtcnn = MTCNN(image_size=64)

    id_list = os.listdir(root)
    for ids in id_list:
        id_path = os.path.join(root,ids)
        out_id_path = os.path.join(out_root,ids)

        peo_list = os.listdir(id_path)
        for peo in peo_list:
            peo_path = os.path.join(id_path,peo)
            out_peo_path = os.path.join(out_id_path,peo)

            seq_list = os.listdir(peo_path)

            for seq in seq_list:
                seq_path = os.path.join(peo_path,seq)
                out_seq_path = os.path.join(out_peo_path,seq)
                try:
                    nums = len(os.listdir(seq_path))
                except Exception:
                    continue

                if nums == 0:
                    continue

                pool.apply_async(process,args=(out_seq_path, seq_path, mtcnn))
                #process(out_seq_path, seq_path, mtcnn)

    pool.close()
    pool.join()

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    root = r'/media/disk1T/vox/dev/frames'
    #root = r'/data/yuanyuan/wwb/self_sup/dataset/vox'
    out_root = r'/data/yuanyuan/wwb/dataset/vox_cropped'
    suffix = '.jpg'
    main(root,out_root)

