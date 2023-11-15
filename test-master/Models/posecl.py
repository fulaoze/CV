import numpy as np
import torch
from Models.networks import *
from Models.BaseModel import BaseModel
import utils
import torchvision.models as models
import time

class PoseCL(BaseModel):
    def __init__(self,config):
        BaseModel.__init__(self,config)

        self.temperature = config['T']
        self.batch_size = config['batch_size']
        self.neg_alpha = config['neg_alpha']

        if not config['eval']:
            self.model = FaceCycleBackbone()
        else:
            face_cycle_backbone = FaceCycleBackbone()

            self.model = face_cycle_backbone
            dim_mlp = self.model.fc.weight.shape[1]
            dim_mlp_out = self.model.fc.weight.shape[0]
            self.model.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                         nn.ReLU(),
                                         self.model.fc,
                                         nn.BatchNorm1d(dim_mlp_out))

        if config['continue_train']:
            self.model = torch.nn.DataParallel(self.model).cuda()
            self.model.load_state_dict(torch.load(config['load_model'])['state_dict'])
            print('load continue model !')
        elif config['eval']:
            #self.model.last_linear = torch.nn.Identity()
            self.model.fc = torch.nn.Identity()
            state_dict = torch.load(config['load_model'])['state_dict']
            for k in list(state_dict.keys()):
                if k.startswith('module') and not k.startswith('module.fc'):
                    state_dict[k[len("module."):]] = state_dict[k]
                del state_dict[k]

            self.model = self.model.cuda()
            msg = self.model.load_state_dict(state_dict,strict=False)
            assert set(msg.missing_keys) == set()
        else:
            self.model = torch.nn.DataParallel(self.model).cuda()

        self.criterion = nn.CrossEntropyLoss().cuda()
        self.recon_criterion = nn.L1Loss().cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=config['lr'],weight_decay=config['wd'])

    def info_nce_loss(self, features,use_mix=False):

        labels = torch.cat([torch.arange(self.batch_size) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.cuda()

        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features,features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda()
        labels = labels[~mask].view(labels.shape[0], -1) # ~mask 取反
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1) # 去除了自己
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

        fea = self.forward(data)
        logits,labels = self.info_nce_loss(fea)
        contra_loss = self.criterion(logits,labels)

        loss = contra_loss

        acc1,acc5 = utils.accuracy(logits,labels,(1,5))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'train_acc1':acc1,'train_acc5':acc5,'train_loss':loss}

    def neg_inter_info_nce_loss(self,features):

        #time_start = time.time()

        b, dim = features.size()

        labels = torch.cat([torch.arange(b // 2) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

        labels_flag = (1 - labels).bool()
        features_expand = features.expand((b, b, dim))  # 512 * 512 * dim
        fea_neg_li = list(features_expand[labels_flag].chunk(b, dim=0))
        fea_neg_tensor = torch.stack(fea_neg_li, dim=0)  # 512 * 510 * dim

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

        logits = torch.cat([pos, neg], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        logits = logits / self.temperature
        #print('cost time: {}'.format(time.time() - time_start))

        return logits, labels

    def forward(self,data):
        imgs = data['images']
        imgs = torch.cat(imgs,dim=0).cuda()
        fea = self.model(imgs)
        return fea

    def linear_forward(self,data):
        img = data['img'].cuda()
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


# if __name__ == '__main__':
#     a = torch.tensor([[1,2,3],[3,4,5],[2,3,4],[1,2,3]],dtype=torch.float).cuda(0)
#     output = neg_inter(a)

