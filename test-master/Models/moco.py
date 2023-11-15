import numpy as np
import torch
from Models.networks import *
from Models.BaseModel import BaseModel
import utils
import torchvision.models as models

class MoCo(BaseModel):
    def __init__(self,config):
        BaseModel.__init__(self,config)
        if not config['eval']:
            self.model = MoCoNetwork(dim=config['dim'],K=config['K'],m=config['m'],T=config['T'],neg_alpha=config['neg_alpha'])
        else:
            self.model = FaceCycleBackbone()
        if config['continue_train']:
            self.model = torch.nn.DataParallel(self.model).cuda()
            self.model.load_state_dict(torch.load(config['load_model'])['state_dict'])
        elif config['eval']:
            self.model.fc = torch.nn.Identity()
            state_dict = torch.load(config['load_model'])['state_dict']
            for k in list(state_dict.keys()):
                if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                    state_dict[k[len("module.encoder_q."):]] = state_dict[k]
                del state_dict[k]

            #self.model = torch.nn.DataParallel(self.model).cuda()
            self.model = self.model.cuda()
            msg = self.model.load_state_dict(state_dict,strict=False)
            assert set(msg.missing_keys) == set()
        else:
            self.model = torch.nn.DataParallel(self.model).cuda()

        self.criterion = nn.CrossEntropyLoss().cuda()
        if not config['eval']:
            self.optimizer = torch.optim.SGD(self.model.parameters(),config['lr'],
                                         momentum=config['momentum'],
                                         weight_decay=config['wd'])

    def optimize_parameters(self,data):
        self.model.train()

        logits,labels = self.forward(data)
        loss = self.criterion(logits,labels)

        acc1,acc5 = utils.accuracy(logits,labels,(1,5))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'train_acc1':acc1,'train_acc5':acc5,'train_loss':loss}


    def forward(self,data):
        q_img = data['exp_images'][0].cuda()
        k_img = data['exp_images'][1].cuda()
        logits,labels = self.model(q_img,k_img)
        return logits,labels

    def linear_forward(self,data):
        img = data['img'].cuda()
        fea = self.model(img)
        return fea

    def eval(self,data):
        self.model.eval()
        with torch.no_grad():
            logits, labels = self.forward(data)

        loss = self.criterion(logits, labels)

        acc1, acc5 = utils.accuracy(logits, labels, (1, 5))

        return {'eval_acc1': acc1, 'eval_acc5': acc5, 'eval_loss': loss}

    def linear_forward_id(self,data):
        img1 = data['img_normal1'].cuda()
        fea1 = self.model(img1)

        img2 = data['img_normal2'].cuda()
        fea2 = self.model(img2)

        return fea1,fea2

    def linear_eval_id(self,data):
        self.model.eval()
        with torch.no_grad():
            fea1,fea2 = self.linear_forward_id(data)
        return fea1,fea2

    def linear_eval(self,data):
        self.model.eval()
        with torch.no_grad():
            fea = self.linear_forward(data)
        return fea

    def metric_better(self,cur,best):
        ans = best
        flag = False
        if best == None or cur < best:
            flag = True
            ans = cur
        return flag,ans

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
