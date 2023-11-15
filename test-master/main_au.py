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
import time

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--config_file',required=True,type=str)
parser.add_argument('--local_rank',type=int,default=-1)
parser.add_argument('--use_ddp',action='store_true',default=False)

def train(config,train_loader,model,logger,step):
    running_dic = None
    count = 0
    total_num = len(train_loader)
    for i, data in tqdm.tqdm(enumerate(train_loader)):
        data['epoch'] = step
        dic = model.optimize_parameters(data)
        count += 1

        if running_dic == None:
            running_dic = {}
            for k, v in dic.items():
                if k != 'train_print_img':
                    running_dic[k] = v
        else:
            for k, v in dic.items():
                if k != 'train_print_img' and k != 'recon_weight':
                    running_dic[k] += v

        if i % config['print_loss'] == 0:
            txt = 'epoch: {},\t step: {},\t'.format(step, i)
            for k in list(dic.keys()):
                if k != 'train_print_img':
                    txt += ',{}: {},\t'.format(k, dic[k])
            print(txt)

        if config['print_img'] != None and i % config['print_img'] == 0 and 'train_print_img' in dic and dic['train_print_img'] != None:
            print_img = dic['train_print_img']
            grid = torchvision.utils.make_grid(print_img,nrow=1)
            logger.add_image('train_img',grid,global_step=total_num * step + i)

    running_dic['train_loss'] /= count
    if 'train_acc1' in running_dic.keys():
        running_dic['train_acc1'] /= count
    if 'train_acc5' in running_dic.keys():
        running_dic['train_acc5'] /= count

    if 'train_acc1_exp' in running_dic.keys():
        running_dic['train_acc1_exp'] /= count
    if 'train_acc5_exp' in running_dic.keys():
        running_dic['train_acc5_exp'] /= count

    if 'train_acc1_pose' in running_dic.keys():
        running_dic['train_acc1_pose'] /= count
    if 'train_acc5_pose' in running_dic.keys():
        running_dic['train_acc5_pose'] /= count

    if 'train_acc1_flip' in running_dic.keys():
        running_dic['train_acc1_flip'] /= count
    if 'train_acc5_flip' in running_dic.keys():
        running_dic['train_acc5_flip'] /= count

    for k, v in running_dic.items():
        logger.add_scalar(k, v, global_step=step)

def eval(config,val_loader,model,logger,step):
    running_dic = None
    count = 0
    total_num = len(val_loader)

    for i, data in tqdm.tqdm(enumerate(val_loader)):
        dic = model.eval(data)
        count += 1

        if running_dic == None:
            running_dic = {}
            for k, v in dic.items():
                if k != 'eval_print_img':
                    running_dic[k] = v
        else:
            for k, v in dic.items():
                if k != 'eval_print_img':
                    running_dic[k] += v

        if i % config['print_loss'] == 0:
            txt = 'epoch: {},\t step: {},\t'.format(step, i)
            for k in list(dic.keys()):
                if k != 'eval_print_img':
                    txt += ',{}: {},\t'.format(k, dic[k])
            print(txt)

        if config['print_img'] != None and i % config['print_img'] == 0:
            print_img = dic['eval_print_img']
            grid = torchvision.utils.make_grid(print_img,nrow=1)
            logger.add_image('test_img',grid,global_step=total_num * step + i)

    running_dic['eval_loss'] /= count
    for k, v in running_dic.items():
        logger.add_scalar(k, v, global_step=step)

    return running_dic['eval_loss']

def cal_acc(pred,label,threadhold=0.5):
    pred_bool = torch.zeros_like(label)
    pred_bool[pred>threadhold] = 1.
    pred_bool[pred<=threadhold] = -1.

    label_bool = torch.zeros_like(label)
    label_bool[label>threadhold] = 1.

    correct_pred = torch.sum(pred_bool==label_bool,dim=0)

    return correct_pred

def cal_acc_ave(pred_list,label_list,total):
    ans = 0.
    for i in range(0,len(label_list)):
        ans += ((pred_list[i]/label_list[i]))
    ans /= len(label_list)
    return ans

def linear_eval(config,train_loader,val_loader,model,logger):
    count = 0

    #linear_classifier = torch.nn.Linear(in_features=config['linear_dim'],out_features=config['classes_num']).cuda()
    linear_classifier = torch.nn.Sequential(torch.nn.BatchNorm1d(config['linear_dim']),torch.nn.Linear(in_features=config['linear_dim'],out_features=config['classes_num'])).cuda()
    sigmoid = torch.nn.Sigmoid()
    optimizer = torch.optim.Adam(linear_classifier.parameters(),lr=config['linear_lr'])
    #optimizer = torch.optim.SGD(linear_classifier.parameters(),lr=config['linear_lr'])
    lr_schduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=config['eval_epochs'])
    criterizer = torch.nn.BCELoss().cuda()

    #warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period=10)
    #warmup_scheduler.last_step = -1

    best_knn_acc = 0.
    best_linear_acc = 0.
    knn_not_use = True

    for eval_step in range(config['eval_epochs']):

        train_count = torch.zeros(config['classes_num']).cuda()
        train_acc_count = torch.zeros(config['classes_num']).cuda()
        train_loss = 0.
        train_total = 0.

        #warmup_scheduler.dampen()
        data_time = time.time()

        for i,data in tqdm.tqdm(enumerate(train_loader)):
            #print('cost data time: {}'.format(time.time() - data_time))
            #start_time = time.time()
            label = data['label'].cuda()
            #print('convert label time: {}'.format(time.time() - start_time))

            #model_time = time.time()
            fea = model.linear_eval(data)
            #print('model time: {}'.format(time.time()-model_time))

            count += 1
            #print(fea.size())

            #linear_time = time.time()
            pred = linear_classifier(fea)
            pred = sigmoid(pred)
            #print('linear time: {}'.format(time.time() - linear_time))

            #cal_loss = time.time()
            loss = criterizer(pred,label)
            #print('cost loss time: {}'.format(time.time() - cal_loss))

            #optimizer_time = time.time()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #print('optimizer time: {}'.format(time.time() - optimizer_time))

            #cal_acc_time = time.time()
            train_total += label.size(0)
            train_acc_count += cal_acc(pred,label,threadhold=config['eval_threads'])
            train_count += torch.sum(label,dim=0)
            train_loss += loss.item()
            #print('cal acc time: {}'.format(time.time() - cal_acc_time))

            #print('1 iter cost time: {}'.format(time.time() - start_time))
            #data_time = time.time()

        train_acc = train_acc_count / train_count
        train_ave_acc = cal_acc_ave(train_acc_count.cpu().tolist(),train_count.cpu().tolist(),train_total)
        train_loss = train_loss
        lr_schduler.step()

        test_acc_count = torch.zeros(config['classes_num']).cuda()
        test_count = torch.zeros(config['classes_num']).cuda()
        test_loss = 0.
        test_total = 0.
        test_label = []
        test_pred = []
        linear_classifier.eval()

        for i,data in tqdm.tqdm(enumerate(val_loader)):
            label = data['label'].cuda().float()

            fea = model.linear_eval(data)

            with torch.no_grad():
                pred = linear_classifier(fea)
                pred = sigmoid(pred)
                pred_bool = torch.zeros_like(pred)
                pred_bool[pred>0.0005] = 1.

            loss = criterizer(pred,label)
            #test_acc_count += cal_acc(pred,label,threadhold=config['eval_threads'])
            #test_count += torch.sum(label,dim=0)
            test_label.extend(label.cpu().tolist())
            test_pred.extend(pred_bool.cpu().tolist())
            test_loss += loss.item()
            test_total += label.size(0)
        #test_linear_acc = test_acc_count / test_count
        #test_ave_acc = cal_acc_ave(test_acc_count.cpu().tolist(),test_count.cpu().tolist(),test_total)
        #print(len(test_label),len(test_pred))
        #print(len(test_label[0]))
        from sklearn.metrics import recall_score
        from sklearn.metrics import precision_score
        test_recall = recall_score(test_label,test_pred,average=None)
        test_precision = precision_score(test_label,test_pred,average=None)
        test_f1 = np.mean(f1_score(test_label,test_pred,average=None))
        print(test_recall)
        print(test_precision)
        print(f1_score(test_label,test_pred,average=None))
        test_linear_loss = test_loss

        train_linear_acc_list = train_acc.tolist()
        #test_linear_acc_list = test_linear_acc.tolist()

        #if eval_step % config['print_loss'] == 0:
        txt = 'eval step: {},\n'
        for tj in range(0, len(train_linear_acc_list)):
            txt += 'linear train acc au_{}: {}\n'.format(tj, train_linear_acc_list[tj])
        txt += 'linear train loss: {},\t linear train ave acc: {}\n'.format(train_loss, train_ave_acc)
        txt += 'linear eval f1: {},\t'.format(test_f1)
        txt += 'linear eval loss: {}\n'.format(test_linear_loss)
        print(txt)

        for tj in range(0,len(train_linear_acc_list)):
            logger.add_scalar('linear_train_acc_au_{}'.format(tj),train_linear_acc_list[tj],eval_step)
        logger.add_scalar('linear_train_loss',train_loss,eval_step)
        logger.add_scalar('linear_train_ave_acc',train_ave_acc,eval_step)
        logger.add_scalar('linear_eval_loss', test_linear_loss, eval_step)
        logger.add_scalar('linear_eval_f1',test_f1,eval_step)

        #best_linear_acc = max(best_linear_acc,test_linear_acc)
        if best_linear_acc < test_f1:
            best_linear_acc = test_f1
            model_state = model.model.state_dict()
            linear_state = linear_classifier.state_dict()
            for k in list(linear_state.keys()):
                model_state[k] = linear_state[k]
            utils.save_checkpoint({
                'epoch': eval_step + 1,
                'state_dict': model_state,
        }, config)

    return best_linear_acc

def main(config,logger):
    model = utils.create_model(config)
    train_dataset = utils.create_dataset(config,'train')
    val_dataset = utils.create_dataset(config,'val')

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        num_workers=config['num_threads'],
        pin_memory=True,
        drop_last=True,
        shuffle=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        num_workers=config['num_threads'],
        pin_memory=True,
        drop_last=False,
        shuffle=False
    )

    if config['eval']:
        test_dataset = utils.create_dataset(config,'test')
        train_dataset = utils.create_dataset(config,'train')

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            num_workers=config['num_threads'],
            pin_memory=True,
            drop_last=False,
            shuffle=False
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

    lr_schduler = torch.optim.lr_scheduler.CosineAnnealingLR(model.optimizer, T_max=config['epochs'])
    best_metric = None
    for step in range(config['start_epochs'],config['epochs']+1):
        train(config,train_loader,model,logger,step)
        #metric = eval(config,val_loader,model,logger,step)
        lr_schduler.step()

        #flag,cur_best = model.metric_better(metric,best_metric)
        if step % config['save_epoch'] == 0:
            #best_metric = cur_best
            utils.save_checkpoint({
                'epoch': step + 1,
                'state_dict': model.model.state_dict(),
            },config)

if __name__ == '__main__':
    opt = parser.parse_args()

    config = utils.read_config(opt.config_file)
    utils.init(config,opt.local_rank,opt.use_ddp)
    logger = SummaryWriter(log_dir=os.path.join(config['log_path'], config['experiment_name']),
                           comment=config['experiment_name'])

    main(config, logger)

    logger.close()