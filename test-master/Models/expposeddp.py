import numpy as np
import torch
from Models.networks import *
from Models.BaseModel import BaseModel
import utils
import torchvision.models as models
import time
import math

class ExpPoseDDP(BaseModel):
    def __init__(self,config):
        BaseModel.__init__(self,config)

        self.temperature = config['T']
        self.batch_size = config['batch_size']
        self.neg_alpha = config['neg_alpha']
        self.local_rank = config['local_rank']

        if not config['eval']:
            self.model = ExpPoseModel()
        else:
            face_cycle_backbone = FaceCycleBackbone()
            self.model = face_cycle_backbone

        if config['continue_train']:
            device = torch.device('cuda', config['local_rank'])
            self.model.to(device)
            self.model = torch.nn.parallel.DistributedDataParallel(self.model,device_ids=[config['local_rank']])
            self.model.load_state_dict(torch.load(config['load_model'])['state_dict'])
            print('load continue model !')
        elif config['eval']:
            #self.model.last_linear = torch.nn.Identity()
            self.model.fc = torch.nn.Identity()
            state_dict = torch.load(config['load_model'])['state_dict']
            if config['eval_mode'] == 'exp':
                for k in list(state_dict.keys()):
                    if k.startswith('module.exp_encoder') and not k.startswith('module.exp_encoder.fc'):
                        state_dict[k[len("module.exp_encoder."):]] = state_dict[k]
                    del state_dict[k]
            elif config['eval_mode'] == 'pose':
                for k in list(state_dict.keys()):
                    if k.startswith('module.pose_encoder') and not k.startswith('module.pose_encoder.fc'):
                        state_dict[k[len("module.pose_encoder."):]] = state_dict[k]
                    del state_dict[k]

            self.model = self.model.cuda()
            msg = self.model.load_state_dict(state_dict,strict=False)
            assert set(msg.missing_keys) == set()
        else:
            device = torch.device('cuda',config['local_rank'])
            self.model.to(device)
            self.model = torch.nn.parallel.DistributedDataParallel(self.model,device_ids=[config['local_rank']])

        self.criterion = nn.CrossEntropyLoss().cuda()
        self.recon_criterion = nn.L1Loss().cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=config['lr'],weight_decay=config['wd'])

    def get_lambda(self,epoch):
        def sigmoid(x):
            if x >= 0:
                z = math.exp(-x)
                sig = 1 / (1 + z)
                return sig
            else:
                z = math.exp(x)
                sig = z / (1 + z)
                return sig

        lam = (sigmoid(epoch/5.0) - 0.5) * 2.0

        return lam

    def info_nce_loss(self, features):
        rest_features = self.concat_all_gather(features)

        labels = torch.cat([torch.arange(self.batch_size) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.cuda()

        add_rest_features = torch.cat([features,rest_features],dim=0)
        features = F.normalize(features, dim=1)
        add_rest_features = F.normalize(add_rest_features,dim=1)
        #print(features.size(),add_rest_features.size())
        similarity_matrix = torch.matmul(features,add_rest_features.T)
        #print('similarity matrix:', similarity_matrix.size())
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda()
        labels = labels[~mask].view(labels.shape[0], -1) # ~mask 取反
        zero_matrix = torch.zeros(labels.shape[0],rest_features.size(0)).cuda()
        mask = torch.cat([mask,zero_matrix],dim=1).cuda()
        labels = torch.cat([labels,zero_matrix],dim=1)
        #print('mask:',mask.size())
        #print('labels:',labels.size())

        similarity_matrix = similarity_matrix[~mask.bool()].view(similarity_matrix.shape[0], -1) # 去除了自己
        #print('similarity matrix:', similarity_matrix.size())
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        logits = logits / self.temperature
        return logits, labels

    def optimize_parameters(self,data):
        self.model.train()

        img_normal = data['img_normal'].cuda()
        img_flip = data['img_flip'].cuda()
        cur_epoch = data['epoch']

        exp_fea, pose_fea, recon_normal_exp_flip_pose_img, recon_flip_exp_normal_pose_img, recon_normal_exp_normal_pose_img, recon_fea = self.forward(
            data)

        exp_logits, exp_labels = self.neg_inter_info_nce_loss(exp_fea)
        exp_contra_loss = self.criterion(exp_logits, exp_labels)

        pose_logits, pose_labels = self.info_nce_loss(pose_fea)
        pose_contra_loss = self.criterion(pose_logits, pose_labels)

        flip_logits, flip_labels = self.info_nce_loss(recon_fea)
        flip_contra_loss = self.criterion(flip_logits, flip_labels)

        recon_normal_loss = self.recon_criterion(recon_flip_exp_normal_pose_img, img_normal)
        recon_flip_loss = self.recon_criterion(recon_normal_exp_flip_pose_img, img_flip)
        recon_orin_loss = self.recon_criterion(recon_normal_exp_normal_pose_img, img_normal)

        recon_weight = self.get_lambda(cur_epoch)

        loss = exp_contra_loss + pose_contra_loss + flip_contra_loss + recon_weight * (
                    recon_normal_loss + recon_flip_loss + recon_orin_loss)

        exp_acc1, exp_acc5 = utils.accuracy(exp_logits, exp_labels, (1, 5))
        pose_acc1, pose_acc5 = utils.accuracy(pose_logits, pose_labels, (1, 5))
        flip_acc1, flip_acc5 = utils.accuracy(flip_logits, flip_labels, (1, 5))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        print_img = torch.cat(
            [img_normal[:2], img_flip[:2], recon_flip_exp_normal_pose_img[:2], recon_normal_exp_flip_pose_img[:2],
             recon_normal_exp_normal_pose_img[:2]], dim=3)

        return {'train_acc1_exp': exp_acc1, 'train_acc5_exp': exp_acc5, 'train_loss': loss,
                'train_acc1_pose': pose_acc1, 'train_acc5_pose': pose_acc5,
                'train_acc1_flip': flip_acc1, 'train_acc5_flip': flip_acc5,
                'train_flip_contra_loss': flip_contra_loss,
                'train_exp_contra_loss': exp_contra_loss,
                'train_pose_contra_loss': pose_contra_loss,
                'train_recon_normal_loss': recon_normal_loss,
                'train_recon_flip_loss': recon_flip_loss,
                'train_recon_orin_loss': recon_orin_loss,
                'train_print_img': print_img, 'recon_weight': recon_weight}

    def neg_inter_info_nce_loss(self,features):
        b, dim = features.size()

        #time_start = time.time()
        rest_features = self.concat_all_gather(features)
        rest_features = rest_features.expand((b,b,dim))
        print('rest_features:',rest_features.size())

        labels = torch.cat([torch.arange(b // 2) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

        labels_flag = (1 - labels).bool()
        features_expand = features.expand((b, b, dim))  # 512 * 512 * dim
        fea_neg_li = list(features_expand[labels_flag].chunk(b, dim=0))
        fea_neg_tensor = torch.stack(fea_neg_li, dim=0)  # 512 * 510 * dim
        fea_neg_tensor = torch.cat([fea_neg_tensor,rest_features],dim=1) # 512 * 1022 * dim
        print('fea_neg_tensor:',fea_neg_tensor.size())

        #time_alpha = time.time()
        neg_mask = np.random.beta(self.neg_alpha, self.neg_alpha,size=(fea_neg_tensor.shape[0],fea_neg_tensor.shape[1]))
        time_alpha_finish = time.time()
        #print('cost alpha time: {}'.format(time_alpha_finish - time_alpha))
        if isinstance(neg_mask, np.ndarray):
            neg_mask = torch.from_numpy(neg_mask).float().cuda()
            neg_mask = neg_mask.unsqueeze(dim=2)
        indices = torch.randperm(fea_neg_tensor.shape[1])
        fea_neg_tensor = fea_neg_tensor * neg_mask + (1 - neg_mask) * fea_neg_tensor[:, indices]

        features = F.normalize(features, dim=1)
        q, k = features.chunk(2, dim=0)
        fea_neg_tensor = F.normalize(fea_neg_tensor, dim=2)

        pos = torch.cat(
            [torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1), torch.einsum('nc,nc->n', [k, q]).unsqueeze(-1)], dim=0)

        fea_neg_tensor = fea_neg_tensor.transpose(2, 1)
        neg = torch.bmm(features.view(b, 1, -1), fea_neg_tensor).view(b, -1)
        print('neg: ',neg.size())

        logits = torch.cat([pos, neg], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        logits = logits / self.temperature
        #print('cost time: {}'.format(time.time() - time_start))

        return logits, labels

    def forward(self,data):
        pose_images = data['pose_images']
        exp_images = data['exp_images']
        img_normal = data['img_normal'].cuda()
        img_flip = data['img_flip'].cuda()
        pose_images = torch.cat(pose_images,dim=0).cuda()
        exp_images = torch.cat(exp_images,dim=0).cuda()
        exp_fea, pose_fea, recon_normal_exp_flip_pose_img, recon_flip_exp_normal_pose_img, recon_normal_exp_normal_pose_img, recon_fea = self.model(
            exp_images, pose_images,
            img_normal, img_flip)
        return exp_fea, pose_fea, recon_normal_exp_flip_pose_img, recon_flip_exp_normal_pose_img, recon_normal_exp_normal_pose_img, recon_fea


    def linear_forward(self,data):
        img = data['img_normal'].cuda()
        fea = self.model(img)
        return fea

    def metric_better(self,cur,best):
        ans = best
        flag = False
        if best == None or cur < best:
            flag = True
            ans = cur
        return flag,ans

    def eval(self,data):
        pass

    def linear_eval(self,data):
        self.model.eval()
        with torch.no_grad():
            fea = self.linear_forward(data)
        return fea

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

    #@torch.no_grad()
    def concat_all_gather(self,tensor):
        with torch.no_grad():
            tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather(tensors_gather,tensor,async_op=True)

        tensors_gather_rest_list = tensors_gather[0:self.local_rank] + tensors_gather[self.local_rank+1:]
        output = torch.cat(tensors_gather_rest_list,dim=0)
        return output

# if __name__ == '__main__':
#     a = torch.tensor([[1,2,3],[3,4,5],[2,3,4],[1,2,3]],dtype=torch.float).cuda(0)
#     output = neg_inter(a)

# if __name__ == '__main__':
#     import os
#
#     os.environ['CUDA_VISIBLE_DEVICES'] = '3'
#
#     model = ExpPoseModel()
#     model = torch.nn.DataParallel(model).cuda()
#     model.load_state_dict(torch.load('/data/yuanyuan/wwb/self_sup/checkpoints/ExpPose/3801.pth')['state_dict'])
#     model = model.eval()
#
#     from torchvision.transforms import transforms
#
#     trans = transforms.Compose([
#         transforms.Resize((64,64)),
#         transforms.ToTensor()
#     ])
#
#     flip = transforms.RandomHorizontalFlip(1.0)
#     to_img = transforms.ToPILImage()
#
#     from PIL import Image
#
#     img = Image.open('/data/yuanyuan/wwb/dataset/RAFDB/img/aligned/test_0286_aligned.jpg').convert('RGB')
#     img_flip = flip(img)
#     tensor_flip = trans(img_flip).unsqueeze(dim=0).cuda()
#     tensor_normal = trans(img).unsqueeze(dim=0).cuda()
#
#     _, _, recon_normal_img, recon_flip_img, _ = model(tensor_normal,tensor_normal,tensor_normal,tensor_flip)
#
#     recon_normal_img = to_img(recon_normal_img.squeeze())
#     recon_flip_img = to_img(recon_flip_img.squeeze())
#
#     recon_normal_img.save('./recon_normal.jpg')
#     recon_flip_img.save('./recon_flip_img.jpg')



