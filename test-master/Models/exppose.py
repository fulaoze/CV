import numpy as np
import torch
from Models.networks import *
from Models.BaseModel import BaseModel
import utils
import torchvision.models as models
import time
import math
from torch.cuda.amp import GradScaler, autocast
# from weighting.PCGrad import *
# from weighting.PCGrad_ori import *


'''
exppose_gradnorm,train->exppose
'''


class DiffLoss(nn.Module):

    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, input1, input2):
        # Zero mean
        input1_mean = torch.mean(input1, dim=0, keepdims=True)
        input2_mean = torch.mean(input2, dim=0, keepdims=True)
        input1 = input1 - input1_mean
        input2 = input2 - input2_mean

        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)

        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)

        diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))
        # diff_loss = torch.mean((input1_l2 * input2_l2).sum(dim=1).pow(2))

        return diff_loss


class ExpPose(BaseModel):
    def __init__(self, config):
        BaseModel.__init__(self, config)

        self.temperature = config['T']
        self.batch_size = config['batch_size']
        self.neg_alpha = config['neg_alpha']
        self.pose_alpha = config['pose_alpha']
        self.lr = config['lr']
        self.weight_decay = config['wd']
        self.loss_train = []

        self.exp_grad = 0.
        self.pose_grad = 0.
        self.cos = torch.nn.CosineSimilarity(dim=1)

        if not config['eval'] and not config['t_sne']:
            self.model = ExpPoseModel()
        else:
            # face_cycle_backbone = FaceCycleBackbone()
            # face_cycle_backbone = SimCLRNetwork()
            face_cycle_backbone = FaceCycleBackbone()
            self.model = face_cycle_backbone.cuda()

        if config['continue_train']:
            self.model = torch.nn.DataParallel(self.model).cuda()
            self.model.load_state_dict(torch.load(config['load_model'])['state_dict'])
            print('load continue model !')
        elif config['eval'] or (config['t_sne'] != None and config['t_sne']):
            # self.model.last_linear = torch.nn.Identity()
            # self.model.fc = torch.nn.Identity()
            # # self.model.fc = nn.Sequential(nn.Linear(2048, 2048),
            # #                              nn.ReLU(),
            # #                              nn.Linear(2048,512),
            # #                              nn.BatchNorm1d(512))
            if config['eval_mode'] == 'exp':  # exp
                state_dict = torch.load(config['load_model'])['state_dict']
                for k in list(state_dict.keys()):
                    if k.startswith('module.encoder'):
                        state_dict[k[len("module.encoder."):]] = state_dict[k]
                    elif k.startswith('module.exp_fc') or k.startswith('module.pose_fc'):
                        state_dict[k[len("module."):]] = state_dict[k]
                    #     # if k.startswith('module.exp_fc'):
                    #     #     state_dict[k[len("module.exp_"):]] = state_dict[k]
                    #     del state_dict[k]
                    # if k.startswith('module.exp_encoder'):
                    #     state_dict[k[len("module.exp_encoder."):]] = state_dict[k]
                    del state_dict[k]
            ######## simclr
            # state_dict = torch.load(config['load_model'])['state_dict']
            # if config['eval_mode'] == 'exp':
            #     self.model = torch.nn.DataParallel(self.model).cuda()
            elif config['eval_mode'] == 'pose':  # pose
                state_dict = torch.load(config['load_model'])['state_dict']
                for k in list(state_dict.keys()):
                    if k.startswith('module.encoder'):
                        state_dict[k[len("module.encoder."):]] = state_dict[k]
                    elif k.startswith('module.exp_fc') or k.startswith('module.pose_fc'):
                        state_dict[k[len("module."):]] = state_dict[k]
                    del state_dict[k]
                print('pose loaded!')
            elif config['eval_mode'] == 'face_cycle':
                state_dict = torch.load(config['load_model'])['codegeneration']
                for k in list(state_dict.keys()):
                    if k.startswith('expresscode.'):
                        del state_dict[k]
            elif config['eval_mode'] == 'TCAE':
                self.model.fc = nn.Sequential(nn.Linear(32768, 2048),
                                              nn.ReLU(),
                                              nn.Linear(2048, 2048),
                                              nn.BatchNorm1d(2048))
                state_dict = torch.load(config['load_model'])['state_dict']
                for k in list(state_dict.keys()):
                    if k.startswith('encoder.exp.'):
                        state_dict['fc.' + k[len("encoder.exp.")]] = state_dict[k]
                    if k.startswith('encoder.'):
                        state_dict[k[len("encoder."):]] = state_dict[k]
                    del state_dict[k]

            # self.model = self.model.cuda()
            # self.model = torch.nn.DataParallel(self.model).cuda()
            msg = self.model.load_state_dict(state_dict, strict=False)
            assert set(msg.missing_keys) == set()
            print('load model !')
        else:
            # self.model = torch.nn.DataParallel(self.model,device_ids=[0]).cuda()
            self.model = self.model.cuda()

        self.criterion = nn.CrossEntropyLoss().cuda()
        self.recon_criterion = nn.L1Loss().cuda()
        self.diff_loss = DiffLoss().cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['lr'], weight_decay=config['wd'])
        self.scaler = GradScaler()

        self.select_weight = "UW"  # 选择权重方法
        ##################################################
        # self.pcgrad = PCGrad()
        # self.optimizer1=PCGrad(self.optimizer)
        # # ########################################
        # # #GradNorm
        # self.Weightloss1 = torch.tensor(torch.FloatTensor([1]), requires_grad=True)
        # self.Weightloss2 = torch.tensor(torch.FloatTensor([1]), requires_grad=True)
        # self.params = [self.Weightloss1, self.Weightloss2]
        # self.optimizer2 = torch.optim.Adam(self.params, lr=self.lr, weight_decay=self.weight_decay)
        # self.alpha = 0.16
        #     UW
        #     self.loss_scale = nn.Parameter(torch.tensor([-0.5] * 2, device='cuda', requires_grad=True))
        self.loss_scale = torch.tensor([-0.5] * 2, device='cuda', requires_grad=True)

    def get_lambda(self, epoch):
        def sigmoid(x):
            if x >= 0:
                z = math.exp(-x)
                sig = 1 / (1 + z)
                return sig
            else:
                z = math.exp(x)
                sig = z / (1 + z)
                return sig

        def exp_decay(x):
            z = math.exp(-x)
            z = max(1e-10, z)
            return z

        if epoch < 200:
            lam = (sigmoid(epoch / 5.0) - 0.5) * 2.0
        else:
            lam = exp_decay((epoch - 200) / 5.0)

        return lam

    def info_nce_loss(self, features):

        labels = torch.cat([torch.arange(self.batch_size) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.cuda()

        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda()
        labels = labels[~mask].view(labels.shape[0], -1)  # ~mask 取反
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)  # 去除了自己
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        logits = logits / self.temperature
        return logits, labels

    def optimize_parameters(self, data):
        self.model.train()

        img_normal = data['img_normal'].cuda()
        img_flip = data['img_flip'].cuda()
        # img_normal = data['exp_images'][0].cuda()
        # img_flip = data['exp_images'][1].cuda()
        cur_epoch = data['epoch']

        with autocast():
            # exp_fea, pose_fea, recon_normal_exp_flip_pose_img, recon_flip_exp_normal_pose_img,recon_normal_exp_normal_pose_img,normal_exp_fea_fc,normal_pose_fea_fc,flip_exp_fea_fc,flip_pose_fea_fc = self.forward(data)
            exp_fea_fc, pose_fea_fc, exp_fea, pose_fea, recon_normal_exp_flip_pose_img, recon_flip_exp_normal_pose_img, recon_normal_exp_normal_pose_img, recon_flip_exp_flip_psoe_img = self.forward(
                data)

            exp_logits, exp_labels = self.neg_inter_info_nce_loss(exp_fea_fc)
            exp_contra_loss = self.criterion(exp_logits, exp_labels)

            pose_logits, pose_labels = self.neg_inter_info_nce_loss(pose_fea_fc)
            pose_contra_loss = self.criterion(pose_logits, pose_labels)

            recon_normal_loss = self.recon_criterion(recon_flip_exp_normal_pose_img, img_normal)  # || s-s'||
            # recon_normal_loss = 0.
            recon_flip_loss = self.recon_criterion(recon_normal_exp_flip_pose_img, img_flip)  # ||f-f'||
            # recon_flip_loss = 0.
            recon_orin_loss = self.recon_criterion(recon_normal_exp_normal_pose_img, img_normal)  # ||s-s''||
            # recon_orin_loss = 0.
            recon_flip_ori_loss = self.recon_criterion(recon_flip_exp_flip_psoe_img, img_flip)

            # recon_weight = self.get_lambda(cur_epoch)
            recon_weight = 1.

            # 约束Fpi和Ffi或者Fpj和Ffj应相互垂直，以尽量减少姿势感知特征对人脸感知特征的影响
            diff_loss = self.diff_loss(exp_fea_fc, pose_fea_fc)
            # diff_loss = 0.
            # diff_loss = self.diff_loss(normal_exp_fea_fc,normal_pose_fea_fc) + self.diff_loss(flip_exp_fea_fc,flip_pose_fea_fc)
            # diff_loss = 0.

            # 对比损失exp_contra_loss 和 pose_contra_loss
            loss = exp_contra_loss + pose_contra_loss * self.pose_alpha + diff_loss + recon_weight * (
                    recon_normal_loss + recon_flip_loss + recon_orin_loss + recon_flip_ori_loss)
            # loss = exp_contra_loss + diff_loss + (recon_normal_loss + recon_flip_loss + recon_orin_loss + recon_flip_ori_loss)
            ################################################
            loss_other = diff_loss + recon_weight * (
                    recon_normal_loss + recon_flip_loss + recon_orin_loss + recon_flip_ori_loss)
            losses = []
            loss1 = exp_contra_loss
            loss2 = pose_contra_loss
            losses.append(loss1)
            losses.append(loss2)
            if self.select_weight == "PCGrad":  # PCGrad

                self.optimizer1.zero_grad()
                self.optimizer1.pc_backward(losses, loss_other)
                self.optimizer1.step()
            elif self.select_weight == "GradNorm11":

                self.params[0] = self.params[0].cuda()
                self.params[1] = self.params[1].cuda()
                # loss1=loss1.cpu()
                # loss2=loss2.cpu()
                L1 = self.params[0] * loss1
                L2 = self.params[1] * loss2
                loss = torch.div(torch.add(L1, L2), 2) + loss_other
                # loss = torch.add(L1, L2)
                if cur_epoch == 0:
                    L_0_1 = L1.data
                    L_0_2 = L2.data

                self.optimizer.zero_grad()
                loss.backward(retain_graph=True)

                # Getting gradients of the first layers of each tower and calculate their l2-norm
                param = list(self.model.get_last_share_param().parameters())
                G1R = torch.autograd.grad(L1, param[0], retain_graph=True, create_graph=True)
                G1 = torch.norm(G1R[0], 2)
                # G1 = torch.norm(torch.mul(G1R[0],self.params[0]))
                G2R = torch.autograd.grad(L2, param[0], retain_graph=True, create_graph=True)
                G2 = torch.norm(G2R[0], 2)
                # G2 = torch.norm(torch.mul(G2R[0], self.params[1]))
                G_avg = torch.div(torch.add(G1, G2), 2.0)
                # Calculating relative losses
                Lhat1 = torch.div(L1, L_0_1)
                Lhat2 = torch.div(L2, L_0_2)
                Lhat_avg = torch.div(torch.add(Lhat1, Lhat2), 2.0)
                # Calculating relative inverse training rates for tasks
                inv_rate1 = torch.div(Lhat1, Lhat_avg)
                inv_rate2 = torch.div(Lhat2, Lhat_avg)
                # Calculating the constant target for Eq. 2 in the GradNorm paper
                C1 = G_avg * (inv_rate1) ** self.alpha
                C2 = G_avg * (inv_rate2) ** self.alpha
                C1 = C1.detach()
                C2 = C2.detach()

                self.optimizer2.zero_grad()
                # Calculating the gradient loss according to Eq. 2 in the GradNorm paper
                Lgrad = torch.add(self.recon_criterion(G1, C1), self.recon_criterion(G2, C2))
                Lgrad.backward()

                # Updating loss weights
                self.optimizer2.step()
                # Updating the model weights
                self.optimizer.step()
                # Renormalizing the losses weights
                coef = 2 / torch.add(self.Weightloss1, self.Weightloss2)
                self.params = [coef * self.Weightloss1, coef * self.Weightloss2]

            elif self.select_weight == "GradNorm":
                task_loss = torch.stack(losses).cuda()
                # compute the weighted loss w_i(t) * L_i(t)
                weighted_task_loss = torch.mul(self.model.weights, task_loss)
                # initialize the initial loss L(0) if t=0
                if cur_epoch == 0:
                    # set L(0)
                    if torch.cuda.is_available():
                        initial_task_loss = task_loss.data.cpu()
                    else:
                        initial_task_loss = task_loss.data
                    initial_task_loss = initial_task_loss.numpy()
                # get the total loss
                loss = torch.sum(weighted_task_loss) + loss_other
                # clear the gradients
                self.optimizer.zero_grad()
                # do the backward pass to compute the gradients for the whole set of weights
                # This is equivalent to compute each \nabla_W L_i(t)
                loss.backward(retain_graph=True)
                # set the gradients of w_i(t) to zero because these gradients have to be updated using the GradNorm loss
                # print('Before turning to 0: {}'.format(model.weights.grad))
                self.model.weights.grad.data = self.model.weights.grad.data * 0.0
                # print('Turning to 0: {}'.format(model.weights.grad))
                # get layer of shared weights
                W = self.model.get_last_share_param()

                # get the gradient norms for each of the tasks
                # G^{(i)}_w(t)
                norms = []
                # print(task_loss[0])
                for i in range(len(task_loss)):
                    # get the gradient of this task loss with respect to the shared parameters
                    gygw = torch.autograd.grad(losses[i], W.parameters(), retain_graph=True)
                    # compute the norm
                    norms.append(torch.norm(torch.mul(self.model.weights[i], gygw[0])))
                norms = torch.stack(norms)
                # print('G_w(t): {}'.format(norms))

                # compute the inverse training rate r_i(t)
                # \curl{L}_i
                if torch.cuda.is_available():
                    loss_ratio = task_loss.data.cpu().numpy() / initial_task_loss
                else:
                    loss_ratio = task_loss.data.numpy() / initial_task_loss
                # r_i(t)
                inverse_train_rate = loss_ratio / np.mean(loss_ratio)
                # print('r_i(t): {}'.format(inverse_train_rate))

                # compute the mean norm \tilde{G}_w(t)
                if torch.cuda.is_available():
                    mean_norm = np.mean(norms.data.cpu().numpy())
                else:
                    mean_norm = np.mean(norms.data.numpy())
                # print('tilde G_w(t): {}'.format(mean_norm))

                # compute the GradNorm loss
                # this term has to remain constant
                constant_term = torch.tensor(mean_norm * (inverse_train_rate ** self.alpha), requires_grad=False)
                if torch.cuda.is_available():
                    constant_term = constant_term.cuda()
                # print('Constant term: {}'.format(constant_term))
                # this is the GradNorm loss itself
                # grad_norm_loss = torch.tensor(torch.sum(torch.abs(norms - constant_term)))
                grad_norm_loss = torch.sum(torch.abs(norms - constant_term))
                # print('GradNorm loss {}'.format(grad_norm_loss))

                # compute the gradient for the weights
                self.model.weights.grad = torch.autograd.grad(grad_norm_loss, self.model.weights)[0]
                # grad_norm_loss.backward()
                # do a step with the optimizer
                self.optimizer.step()
                # renormalize
                normalize_coeff = 2 / torch.sum(self.model.weights.data, dim=0)
                self.model.weights.data = self.model.weights.data * normalize_coeff

            elif self.select_weight == "UW":
                self.optimizer.zero_grad()
                loss = loss1 / (2 * self.loss_scale[0].exp()) + loss2 / (2 * self.loss_scale[1].exp()) + \
                       self.loss_scale[0] / 2 + self.loss_scale[1] / 2 + loss_other
                # losses=torch.stack(losses)
                # loss = (losses / (2* self.loss_scale.exp())+self.loss_scale/2).sum()
                # print(self.loss_scale.grad)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            elif self.select_weight == "DWA":
                # self.loss_train.append([loss1, loss2])
                # b = torch.tensor(self.loss_train)
                # self.optimizer.zero_grad()
                #
                # if cur_epoch > 1:
                #     w_i = torch.Tensor(
                #         b[:, cur_epoch - 1] / b[:, cur_epoch - 2])
                #     batch_weight = 2 * F.softmax(w_i / 2, dim=-1)
                # else:
                #     batch_weight = torch.ones_like(torch.tensor(losses))
                # loss = torch.mul(torch.tensor(losses), batch_weight).sum() + loss_other
                # self.scaler.scale(loss).backward()
                # self.scaler.step(self.optimizer)
                # self.scaler.update()
                self.optimizer.zero_grad()
                self.loss_train.append(torch.tensor([loss1, loss2],requires_grad=True))


                if cur_epoch > 1:
                    w1=self.loss_train[cur_epoch-1][0]/self.loss_train[cur_epoch-2][0]
                    w2=self.loss_train[cur_epoch-1][1]/self.loss_train[cur_epoch-2][1]
                    batch_weight1 = 2 * ((w1/2).exp()/((w1/2).exp()+(w2/2).exp()))
                    batch_weight2 = 2 * ((w2/2).exp()/((w1/2).exp()+(w2/2).exp()))
                else:
                    batch_weight1=1
                    batch_weight2=1
                loss = loss1*batch_weight1+loss2*batch_weight2 + loss_other
                # losses=torch.stack(losses)
                # loss = (losses / (2* self.loss_scale.exp())+self.loss_scale/2).sum()
                # print(self.loss_scale.grad)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
        ################################################

        exp_acc1, exp_acc5 = utils.accuracy(exp_logits, exp_labels, (1, 5))
        pose_acc1, pose_acc5 = utils.accuracy(pose_logits, pose_labels, (1, 5))

        # self.optimizer.zero_grad()
        # self.scaler.scale(loss).backward()
        # self.scaler.step(self.optimizer)
        # self.scaler.update()
        # print(self.exp_grad)

        exp_grad = 0.
        pose_grad = 0.
        exp_grad = self.model.exp_fc.linear2.weight.grad.detach()
        pose_grad = self.model.pose_fc.linear2.weight.grad.detach()
        sim = self.cos(exp_grad.view(1, -1), pose_grad.view(1, -1))
        # for params in self.model.module.exp_fc.parameters():
        #     exp_grad += torch.sum(params.grad).detach().cpu().item()
        #
        # for params in self.model.module.pose_fc.parameters():
        #     pose_grad += torch.sum(params.grad).detach().cpu().item()

        print_img = torch.cat(
            [img_normal[:2], img_flip[:2], recon_flip_exp_normal_pose_img[:2], recon_normal_exp_flip_pose_img[:2],
             recon_normal_exp_normal_pose_img[:2]], dim=3)

        return {'train_acc1_exp': exp_acc1, 'train_acc5_exp': exp_acc5, 'train_loss': loss,
                'train_acc1_pose': pose_acc1, 'train_acc5_pose': pose_acc5,
                'train_diff_loss': diff_loss,
                'train_exp_contra_loss': exp_contra_loss,
                'train_pose_contra_loss': pose_contra_loss,
                'train_recon_normal_loss': recon_normal_loss,
                'train_recon_flip_loss': recon_flip_loss,
                'train_recon_orin_loss': recon_orin_loss,
                'train_recon_flip_orin_loss': recon_flip_ori_loss,
                'train_print_img': print_img,
                'exp_grad': exp_grad.sum().item(),
                'pose_grad': pose_grad.sum().item(),
                'grad_sim': sim.item()
                # 'recon_weight':recon_weight
                }

    # 对比学习部分（再看看）
    def neg_inter_info_nce_loss(self, features, pose_flag=False, flip_pose=None):

        # time_start = time.time()

        b, dim = features.size()

        labels = torch.cat([torch.arange(b // 2) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

        labels_flag = (1 - labels).bool()
        features_expand = features.expand((b, b, dim))  # 512 * 512 * dim
        fea_neg_li = list(features_expand[labels_flag].chunk(b, dim=0))
        fea_neg_tensor = torch.stack(fea_neg_li, dim=0)  # 512 * 510 * dim
        if pose_flag:
            flip_pose = flip_pose.repeat(2, 1)
            fea_neg_tensor = torch.cat([fea_neg_tensor, flip_pose.view(b, 1, -1)], dim=1)

        # time_alpha = time.time()
        neg_mask = np.random.beta(self.neg_alpha, self.neg_alpha,
                                  size=(fea_neg_tensor.shape[0], fea_neg_tensor.shape[1]))
        time_alpha_finish = time.time()
        # print('cost alpha time: {}'.format(time_alpha_finish - time_alpha))
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
        # print('cost time: {}'.format(time.time() - time_start))

        return logits, labels

    def forward(self, data):
        exp_images = data['exp_images']
        img_normal = (data['img_normal'] + torch.randn_like(data['img_normal']) * 4e-1).cuda()
        img_flip = (data['img_flip'] + torch.randn_like(data['img_normal']) * 4e-1).cuda()
        exp_images = torch.cat(exp_images, dim=0).cuda()
        return self.model(exp_images, img_normal, img_flip)

    def linear_forward(self, data):
        img = data['img_normal'].cuda()
        fea, _ = self.model(img)
        return fea

    def linear_forward_id(self, data):
        img1 = data['img_normal1'].cuda()
        fea1 = self.model(img1)

        img2 = data['img_normal2'].cuda()
        fea2 = self.model(img2)

        return fea1, fea2

    def linear_eval_id(self, data):
        self.model.eval()
        with torch.no_grad():
            fea1, fea2 = self.linear_forward_id(data)
        return fea1, fea2

    def metric_better(self, cur, best):
        ans = best
        flag = False
        if best == None or cur < best:
            flag = True
            ans = cur
        return flag, ans

    def eval(self, data):
        pass

    def linear_eval(self, data):
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

# if __name__ == '__main__':
#     import os
#
#     os.environ['CUDA_VISIBLE_DEVICES'] = '2'
#
#     model = ExpPoseModel()
#     model = torch.nn.DataParallel(model).cuda()
#     model.load_state_dict(torch.load('/data/yuanyuan/wwb/self_sup/checkpoints/ExpPose_single_encoder/1101.pth')['state_dict'])
#     # model.load_state_dict(
#     #     torch.load('/data/yuanyuan/wwb/self_sup/checkpoints/ExpPose_single_encoder_vox_del_same_id_pose_alpha_0.001/451.pth')['state_dict'])
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
#     to_img = transforms.Compose([transforms.ToPILImage(),transforms.Resize((100,100))])
#     #gray = transforms.Grayscale(num_output_channels=3)
#
#     from PIL import Image
#
#     img = Image.open('/data/yuanyuan/wwb/dataset/RAFDB/img/aligned/test_0093_aligned.jpg').convert('RGB')
#     #img = Image.open('/data/yuanyuan/wwb/SEST/datasets/BU-3DFE/BU-3DFE/M043/Happy/001/0.png').convert('RGB')
#     # img2 = Image.open('/data/yuanyuan/wwb/dataset/RAFDB/img/aligned/test_0282_aligned.jpg').convert('RGB')
#     # img3 = Image.open('/data/yuanyuan/wwb/dataset/RAFDB/img/aligned/test_0288_aligned.jpg').convert('RGB')
#
#     img_flip = flip(img)
#     tensor_flip = trans(img_flip).unsqueeze(dim=0).cuda()
#     tensor_normal = trans(img).unsqueeze(dim=0).cuda()
#
#     # img_flip2 = flip(img2)
#     # tensor_flip2 = trans(img_flip2).unsqueeze(dim=0).cuda()
#     # tensor_normal2 = trans(img2).unsqueeze(dim=0).cuda()
#     #
#     # img_flip3 = flip(img3)
#     # tensor_flip3 = trans(img_flip3).unsqueeze(dim=0).cuda()
#     # tensor_normal3 = trans(img3).unsqueeze(dim=0).cuda()
#
#     # tensor_flip = torch.cat([tensor_flip,tensor_flip2,tensor_flip3],dim=0)
#     # tensor_normal = torch.cat([tensor_normal,tensor_normal2,tensor_normal3],dim=0)
#
#     recon_normal_exp_flip_pose_img,recon_flip_exp_normal_pose_img,recon_normal_exp_normal_pose_img,recon_normal_pose_normal_pose_img,recon_flip_pose_flip_pose_img = model(tensor_normal,tensor_normal,tensor_flip)
#
#     recon_normal_exp_flip_pose_img = to_img(recon_normal_exp_flip_pose_img[0])
#     recon_flip_exp_normal_pose_img = to_img(recon_flip_exp_normal_pose_img[0])
#     recon_normal_exp_normal_exp_img = to_img(recon_normal_exp_normal_pose_img[0])
#     recon_normal_pose_normal_pose_img = to_img(recon_normal_pose_normal_pose_img[0])
#     recon_flip_pose_flip_pose_img = to_img(recon_flip_pose_flip_pose_img[0])
#
#     img_name = 'test_0093_aligned'
#     if not os.path.exists(os.path.join('/data/yuanyuan/wwb/self_sup/save_img/',img_name)):
#         os.makedirs(os.path.join('/data/yuanyuan/wwb/self_sup/save_img/',img_name))
#     recon_normal_exp_flip_pose_img.save('/data/yuanyuan/wwb/self_sup/save_img/'+img_name + '/' + 'recon_normal_exp_flip_pose.jpg')
#     recon_flip_exp_normal_pose_img.save('/data/yuanyuan/wwb/self_sup/save_img/'+img_name + '/' + 'recon_flip_exp_normal_pose.jpg')
#     recon_normal_exp_normal_exp_img.save('/data/yuanyuan/wwb/self_sup/save_img/'+img_name + '/' + 'recon_normal_exp_normal_exp.jpg')
#     recon_normal_pose_normal_pose_img.save('/data/yuanyuan/wwb/self_sup/save_img/'+img_name + '/' + 'recon_normal_pose_normal_pose.jpg')
#     recon_flip_pose_flip_pose_img.save('/data/yuanyuan/wwb/self_sup/save_img/'+img_name + '/' + 'recon_flip_pose_flip_pose.jpg')

# recon_normal_img.save('./recon_decay_recon_normal_exp_flip_pose.jpg')
# recon_flip_img.save('./recon_decay_recon_flip_exp_normal_pose.jpg')
