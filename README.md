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

## 3. Pre-trained Models

You can download our trained models from [Baidu Drive](xxx).


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



