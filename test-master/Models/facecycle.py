import numpy as np
import torch
from Models.networks import *
from Models.BaseModel import BaseModel
import utils

class FaceCycle(BaseModel):
    def __init__(self,config):
        BaseModel.__init__(self,config)
        self.model = FaceCycleGenerator(3,3,8,config['ngf'],linear_eval=config['linear_eval'])
        self.discri = NLayerDiscriminator(6, config['ndf'], n_layers=3).cuda()

        if config['continue_train'] or config['eval']:
            self.model = torch.nn.DataParallel(self.model).cuda()
            self.model.load_state_dict(torch.load(config['load_model'])['state_dict'])
        else:
            self.model = torch.nn.DataParallel(self.model).cuda()

        self.optimizer_g = torch.optim.Adam(self.model.parameters(),lr=config['lr'],betas=(0.5, 0.999))
        self.optimizer_d = torch.optim.Adam(self.discri.parameters(),lr=config['lr'],betas=(0.5, 0.999))
        self.criterionL1 = torch.nn.L1Loss().cuda()
        self.criterionGAN = GANLoss('lsgan').cuda()

    def backward_D(self,recon_img,mid_img,exp_img):
        # fake
        fake_AB = torch.cat((mid_img,recon_img),1)
        pred_fake = self.discri(fake_AB.detach())
        loss_D_fake = self.criterionGAN(pred_fake,False)

        # real
        real_AB = torch.cat((mid_img,exp_img),1)
        pred_real = self.discri(real_AB)
        loss_D_real = self.criterionGAN(pred_real,True)

        loss_D = (loss_D_fake + loss_D_real) * 0.5
        loss_D.backward()

        return loss_D.item()

    def backward_G(self,recon_img,mid_img,exp_img,attn_map):
        fake_AB = torch.cat((mid_img,recon_img),1)
        pred_fake = self.discri(fake_AB)
        loss_G_GAN = self.criterionGAN(pred_fake,True)

        loss_G_L1 = self.criterionL1(recon_img,exp_img) * self.config['lambda_l1']
        loss_G_norm = torch.norm(attn_map,p=1)

        loss_G = loss_G_GAN + loss_G_L1 + loss_G_norm * 0.1
        loss_G.backward()

        return loss_G_L1.item(),loss_G_GAN.item(),loss_G.item()

    def optimize_parameters(self,data):
        mid_img = data['mid_img'].cuda()
        exp_img = data['exp_img'].cuda()

        recon_img,attn_map = self.forward(data)
        # update D
        self.set_requires_grad(self.discri,True)
        self.optimizer_d.zero_grad()
        loss_D = self.backward_D(recon_img,mid_img,exp_img)
        self.optimizer_d.step()

        # update G
        self.set_requires_grad(self.discri,False)
        self.optimizer_g.zero_grad()
        loss_G_L1,loss_G_GAN,loss_G = self.backward_G(recon_img,mid_img,exp_img,attn_map)
        self.optimizer_g.step()

        # expand
        b,c,h,w = attn_map.size()
        attn_map_print = attn_map.expand(b,3,h,w)
        print_img = torch.cat([recon_img[:1],attn_map_print[:1],exp_img[:1],mid_img[:1]],dim=3)
        return {'loss_D':loss_D,'loss_G_L1':loss_G_L1,
                'loss_G_GAN':loss_G_GAN,'loss_G':loss_G,'train_print_img':print_img,'train_loss':loss_D + loss_G + loss_G_L1}


    def forward(self,data):
        exp_img = data['exp_img'].cuda()
        mid_img = data['mid_img'].cuda()
        recon_exp,attn_map = self.model(exp_img,mid_img)
        return recon_exp,attn_map

    def linear_forward(self,data):
        exp_img = data['exp_img'].cuda()
        #exp_img = data['img'].cuda()
        #fea,attn_map = self.model(x=exp_img)
        fea = self.model(x=exp_img)
        return fea

    def eval(self,data):
        self.model.eval()
        with torch.no_grad():
            recon_exp,attn_map = self.forward(data)

        mid_img = data['mid_img'].cuda()
        exp_img = data['exp_img'].cuda()
        loss = self.criterionL1(recon_exp,exp_img)
        b, c, h, w = attn_map.size()
        attn_map_print = attn_map.expand(b, 3, h, w)
        print_img = torch.cat([recon_exp[:1], attn_map_print[:1], exp_img[:1], mid_img[:1]], dim=3)
        #print_img = attn_map_print[:1]
        return {'eval_loss':loss.item(),
                'eval_print_img':print_img}

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
