import torch
import utils
import argparse
from torch.utils.tensorboard import SummaryWriter
import os
import torchvision
import tqdm
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pytorch_warmup as warmup
from sklearn.metrics import f1_score
import torch.nn.functional as F

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--config_file',required=True,type=str)
parser.add_argument('--local_rank',type=int,default=-1)
parser.add_argument('--use_ddp',action='store_true',default=False)

def eval(config,val_loader,model):
    cos_dis = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    label_list = []
    pred_list = []
    prob = []
    theta = config['theta']

    for i, data in tqdm.tqdm(enumerate(val_loader)):
        label = data['label']

        fea1, fea2 = model.linear_eval_id(data)
        pred = cos_dis(fea1, fea2)
        pred_bool = torch.zeros_like(pred)

        pred_bool[pred >= theta] = 1.
        pred_bool[pred<theta] = -1.
        pred_list.extend(pred_bool.tolist())
        prob.extend(pred.tolist())
        label_list.extend(label.tolist())

    acc_count = 0
    for i in range(len(label_list)):
        print('index: {},\t pred: {},\t prob: {},\t label: {}'.format(i + 1, pred_list[i], prob[i], label_list[i]))
        if pred_list[i] == label_list[i]:
            acc_count += 1

    linear_acc = acc_count / len(label_list)
    return linear_acc

def linear_eval(config,train_loader,val_loader,model,logger):
    best_linear_acc = 0.
    theta = config['theta']
    #linear_classifier = torch.nn.Linear(in_features=config['linear_dim'],out_features=config['out_dim']).cuda()
    linear_classifier = torch.nn.Sequential(torch.nn.BatchNorm1d(num_features=config['linear_dim']),
                                            torch.nn.Linear(in_features=config['linear_dim'],out_features=config['out_dim'])).cuda()
    #linear_classifier.weight.data.normal_(mean=0.0,std=0.01)
    #linear_classifier.bias.data.zero_()
    optimizer = torch.optim.Adam(linear_classifier.parameters(),lr=config['linear_lr'])
    #model = torchvision.models.resnet18(pretrained=False).cuda()
    #optimizer = torch.optim.Adam(model.model.parameters(), lr=config['linear_lr'])
    #optimizer = torch.optim.Adam(model.model.parameters(),lr=config['linear_lr'])
    lr_schduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['eval_epochs'])
    criterizer = torch.nn.CosineEmbeddingLoss(margin=theta).cuda()

    if config['linear_eval']:
        for eval_step in range(config['eval_epochs']):
            train_count = 0.
            train_loss = 0.
            cos_dis = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
            label_list = []
            pred_list = []
            prob = []
            linear_classifier.train()
            model.model.train()
            for i, data in tqdm.tqdm(enumerate(train_loader)):
                label = data['label'].cuda()

                fea1,fea2 = model.linear_eval_id(data)
                #fea1 = model(data['img_normal1'].cuda())
                #fea2 = model(data['img_normal2'].cuda())
                #print(fea1.size())
                #fea1 = F.normalize(fea1, dim=1)
                #fea2 = F.normalize(fea2, dim=1)
                fea1 = linear_classifier(fea1)
                fea2 = linear_classifier(fea2)
                loss = criterizer(fea1,fea2,label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                #print(loss.item())

                train_count += label.size(0)
                pred = cos_dis(fea1,fea2)
                pred_bool = torch.zeros_like(pred)

                pred_bool[pred>=theta] = 1.
                pred_bool[pred<theta] = -1.
                pred_list.extend(pred_bool.tolist())
                prob.extend(pred.tolist())
                label_list.extend(label.tolist())
                train_loss += loss.item()

            acc_count = 0.
            #lr_schduler.step()
            for i in range(len(label_list)):
                if pred_list[i] == label_list[i]:
                    acc_count += 1
            train_linear_acc = acc_count / len(label_list)

            linear_classifier.eval()
            model.model.eval()
            eval_pred_list = []
            eval_label_list = []
            eval_loss = 0.
            for i,data in tqdm.tqdm(enumerate(val_loader)):
                label = data['label'].cuda()

                #fea1,fea2 = model.linear_forward_id(data)

                with torch.no_grad():
                    fea1, fea2 = model.linear_eval_id(data)
                    #fea1 = F.normalize(fea1,dim=1)
                    #fea2 = F.normalize(fea2,dim=1)
                    #fea1 = model(data['img_normal1'].cuda())
                    #fea2 = model(data['img_normal2'].cuda())
                    fea1 = linear_classifier(fea1)
                    fea2 = linear_classifier(fea2)

                    loss = criterizer(fea1,fea2,label)
                pred = cos_dis(fea1,fea2)
                pred_bool = torch.zeros_like(pred)

                pred_bool[pred>=theta] = 1.
                pred_bool[pred < theta] = -1.
                eval_pred_list.extend(pred_bool.tolist())
                eval_label_list.extend(label.tolist())
                eval_loss += loss.item()

            eval_acc_count = 0.
            for i in range(len(eval_label_list)):
                if eval_label_list[i] == eval_pred_list[i]:
                    eval_acc_count += 1.
            eval_acc = eval_acc_count / len(eval_label_list)

            txt = 'eval step: {},\t train acc: {},\t train loss: {},\t eval acc: {},\t eval loss: {}'.format(eval_step,train_linear_acc,train_loss,eval_acc,eval_loss)
            print(txt)
            logger.add_scalar('linear_train_acc_id',train_linear_acc,eval_step)
            logger.add_scalar('linear_train_loss_id',train_loss,eval_step)
            logger.add_scalar('linear_eval_acc_id',eval_acc,eval_step)
            logger.add_scalar('linear_eval_loss_id',eval_loss,eval_step)

            if eval_acc > best_linear_acc:
                best_linear_acc = eval_acc
                model_state = model.model.state_dict()
                linear_state = linear_classifier.state_dict()
                for k in list(linear_state.keys()):
                    model_state[k] = linear_state[k]
                utils.save_checkpoint({
                    'epoch': eval_step + 1,
                    'state_dict': model_state,
                }, config)
    else:
        best_linear_acc = eval(config,val_loader,model)
    return best_linear_acc

def main(config,logger):
    model = utils.create_model(config)

    train_dataset = utils.create_dataset(config,'train')
    test_dataset = utils.create_dataset(config,'test')

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        num_workers=config['num_threads'],
        pin_memory=True,
        drop_last=False,
        shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        num_workers=config['num_threads'],
        pin_memory=True,
        drop_last=False,
        shuffle=False
    )
    linear_acc = linear_eval(config,train_loader,test_loader,model,logger)
    print('test linear acc is : {}'.format(linear_acc))
    exit(0)


if __name__ == '__main__':
    opt = parser.parse_args()

    config = utils.read_config(opt.config_file)
    utils.init(config,opt.local_rank,opt.use_ddp)
    logger = SummaryWriter(log_dir=os.path.join(config['log_path'], config['experiment_name']),
                           comment=config['experiment_name'])

    main(config, logger)

    logger.close()