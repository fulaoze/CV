import numpy as np
import torch
from Models.networks import *
from Models.BaseModel import BaseModel
import utils

class Recon(BaseModel):
    def __init__(self,config):
        BaseModel.__init__(self,config)
        self.model = ReconModel(config['dim'])

        if config['continue_train'] or config['eval']:
            self.model = torch.nn.DataParallel(self.model).cuda()
            self.model.load_state_dict(torch.load(config['load_model'])['state_dict'])
        else:
            self.model = torch.nn.DataParallel(self.model).cuda()

        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=config['lr'],weight_decay=config['wd'])
        self.criterion = torch.nn.L1Loss().cuda()

    def optimize_parameters(self,data):
        self.model.train()
        _,recon_img = self.forward(data)

        ori_img = data['img'].cuda()
        loss = self.criterion(recon_img,ori_img)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        print_img = torch.cat([recon_img[:5],ori_img[:5]],dim=3)
        return {'train_loss':loss.item(),'train_print_img':print_img}


    def forward(self,data):
        img = data['img'].cuda()
        emo_fea,recon_img = self.model(img)
        return emo_fea,recon_img

    def eval(self,data):
        self.model.eval()
        with torch.no_grad():
            fea,recon_img = self.forward(data)

        ori_img = data['img'].cuda()
        loss = self.criterion(recon_img,ori_img)
        print_img = torch.cat([recon_img[:5],ori_img[:5]],dim=3)
        return {'eval_loss':loss.item(),
                'eval_print_img':print_img}

    def linear_eval(self,data):
        self.model.eval()
        with torch.no_grad():
            fea,_ = self.forward(data)
        return fea

    def metric_better(self,cur,best):
        ans = best
        flag = False
        if best == None or cur < best:
            flag = True
            ans = cur
        return flag,ans
