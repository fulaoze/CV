# Sample-cohesive Pose-aware Contrastive Facial Representation Learning

This repository is the Pytorch implementation for  **Sample-cohesive Pose-aware Contrastive Facial Representation Learning**.


## 0. Contents

1. Requirements
2. Data Preparation
3. Pre-trained Models
4. Training
5. Evaluation

## 1. Requirements

To install requirements:
Python Version: 3.7.9

```
pip install -r requirements.txt
```

## 2. Data Preparation

You need to download the related datasets  and put in the folder which namely dataset.

[VoxCeleb1、VoxCeleb2](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/)

[FER-2013](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)

[RAF-DB](http://www.whdeng.cn/raf/model1.html)

[LFW](https://vis-www.cs.umass.edu/lfw/)

[CPLFW](http://www.whdeng.cn/CPLFW/index.html?reload=true)

[DISFA](http://mohammadmahoor.com/disfa/)

[300W-LP](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm)

[AFLW2000](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm)

## 3. Pre-trained Models

You can download our trained models from [Baidu Drive](https://pan.baidu.com/s/1bgJ-t-8CiIVpIl0LLfLr5w?pwd=1234).


## 4. Evaluation

We used the linear evaluation protocol for evaluation.

### 4.1 FER

To evaluate on RAF-DB, run:

```
python main.py --config_file configs/remote_PCL_linear_eval.yaml
```

### 4.2 Pose regression

To trained on 300W-LP and evaluated on AFLW2000, run:

```
python main_pose.py --config_file configs/remote_PCL_linear_eval_pose.yaml
```



