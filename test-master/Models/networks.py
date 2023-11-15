import torch
import torch.nn as nn
import torchvision
import numpy as np
import torch.nn.functional as F
from functools import partial
import math
from facenet_pytorch import InceptionResnetV1
import torchvision.models as models
import functools
import torchvision.models as models
import random
from Models.resnet import *

class Generator(torch.nn.Module):
    def __init__(self,d_model):
        super(Generator, self).__init__()
        self.d_model = d_model

        up = nn.Upsample(scale_factor=2, mode='bilinear')

        self.fc = nn.Linear(d_model,512)
        self.d_model = 512

        dconv1 = nn.Conv2d(self.d_model, self.d_model//2, 3, 1, 1) # 2*2 512
        dconv2 = nn.Conv2d(self.d_model//2, self.d_model//2, 3, 1, 1) # 4*4 256
        dconv3 = nn.Conv2d(self.d_model//2, self.d_model//2, 3, 1, 1) # 16*16 256
        dconv4 = nn.Conv2d(self.d_model//2, self.d_model//2, 3, 1, 1) # 32 * 32 * 256
        dconv5 = nn.Conv2d(self.d_model//2, self.d_model//4, 3, 1, 1) #  64 * 64 *128
        #dconv6 = nn.Conv2d(self.d_model//4, self.d_model//8, 3, 1, 1) # 128 * 128 *32
        dconv7 = nn.Conv2d(self.d_model//4, 3, 3, 1, 1)

        # batch_norm2_1 = nn.BatchNorm2d(self.d_model//8)
        batch_norm4_1 = nn.BatchNorm2d(self.d_model//4)
        batch_norm8_4 = nn.BatchNorm2d(self.d_model//2)
        batch_norm8_5 = nn.BatchNorm2d(self.d_model//2)
        batch_norm8_6 = nn.BatchNorm2d(self.d_model//2)
        batch_norm8_7 = nn.BatchNorm2d(self.d_model//2)

        relu = nn.ReLU()
        tanh = nn.Tanh()

        self.model = torch.nn.Sequential(relu, up, dconv1, batch_norm8_4, \
                             relu, up, dconv2, batch_norm8_5, relu,
                             up, dconv3, batch_norm8_6, relu, up, dconv4,
                             batch_norm8_7, relu, up, dconv5, batch_norm4_1,
                             relu, up, dconv7, tanh)

    def forward(self,x):
        x = self.fc(x)
        x = x.unsqueeze(dim=2).unsqueeze(dim=3)
        out = self.model(x)
        return out

class ReconModel(torch.nn.Module):
    def __init__(self,d_model=512):
        super(ReconModel, self).__init__()
        self.exp = InceptionResnetV1()
        self.decoder = Generator(d_model=d_model)

    def forward(self,x):
        emo_fea = self.exp(x)
        recon_img = self.decoder(emo_fea)
        return emo_fea,recon_img


class SimSiamNetwork(torch.nn.Module):
    def __init__(self, dim=2048, pred_dim=512):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(SimSiamNetwork, self).__init__()

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        #self.encoder = models.resnet50(num_classes=dim, zero_init_residual=True)
        self.encoder = FaceCycleBackbone()
        self.predictor = nn.Sequential(nn.Linear(2048, 2048),
                                         nn.ReLU(),
                                         nn.Linear(2048,2048),
                                         nn.BatchNorm1d(2048))


        # build a 3-layer projector
        # prev_dim = self.encoder.fc.weight.shape[1]
        # self.encoder.fc = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
        #                                 nn.BatchNorm1d(prev_dim),
        #                                 nn.ReLU(inplace=True), # first layer
        #                                 nn.Linear(prev_dim, prev_dim, bias=False),
        #                                 nn.BatchNorm1d(prev_dim),
        #                                 nn.ReLU(inplace=True), # second layer
        #                                 self.encoder.fc,
        #                                 nn.BatchNorm1d(dim, affine=False)) # output layer
        # self.encoder.fc[6].bias.requires_grad = False # hack: not use bias as it is followed by BN
        #
        # # build a 2-layer predictor
        # self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
        #                                 nn.BatchNorm1d(pred_dim),
        #                                 nn.ReLU(inplace=True), # hidden layer
        #                                 nn.Linear(pred_dim, dim)) # output layer

    def forward(self, x1, x2):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """

        # compute features for one view
        z1 = self.encoder(x1) # NxC
        z2 = self.encoder(x2) # NxC


        p1 = self.predictor(z1) # NxC
        p2 = self.predictor(z2) # NxC

        return p1, p2, z1.detach(), z2.detach()

class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False,linear_eval=False,is_attn_gen=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        self.linear_eval = linear_eval
        self.fea_container = []
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True,linear_eval=linear_eval,fea_container=self.fea_container)  # add the innermost layer
        self.extractor = unet_block
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout,linear_eval=linear_eval)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer,linear_eval=linear_eval)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer,linear_eval=linear_eval)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer,linear_eval=linear_eval)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer,linear_eval=linear_eval,is_attn_gen=is_attn_gen)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        if self.linear_eval:
            attn_map = self.model(input)
            return self.fea_container[0].squeeze(),attn_map
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False,linear_eval=False,fea_container=None,is_attn_gen=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        self.innermost = innermost
        self.linear_eval = linear_eval
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)
        self.fea_container = fea_container
        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            if is_attn_gen:
                up = [uprelu, upconv, nn.Sigmoid()]
            else:
                up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
            self.model = nn.Sequential(*model)
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            #model = down + up
            self.down = nn.Sequential(*down)
            self.up = nn.Sequential(*up)
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

            self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        elif self.innermost:
            if self.linear_eval:
                fea = self.down(x)
                self.fea_container.clear()
                self.fea_container.append(fea)
                return torch.cat([x, self.up(fea)], 1)
                #return self.up(fea)
            else:
                return torch.cat([x,self.up(self.down(x))],1)
                #return self.up(self.down(x))
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)
            #return self.model(x)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class FaceCycleGenerator(nn.Module):
    def __init__(self,input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False,linear_eval=False):
        super(FaceCycleGenerator, self).__init__()
        #self.attn_gene = UnetGenerator(input_nc,1,num_downs=num_downs,ngf=ngf,norm_layer=norm_layer,use_dropout=use_dropout,linear_eval=linear_eval,is_attn_gen=True)
        #self.recon_gene = UnetGenerator(input_nc + 1, output_nc,num_downs=num_downs,ngf=ngf,norm_layer=norm_layer,use_dropout=use_dropout,linear_eval=linear_eval)
        self.attn_gene = ResnetGenerator(input_nc,1,ngf=ngf,norm_layer=norm_layer,use_dropout=use_dropout,n_blocks=9)
        self.recon_gene = ResnetGenerator(input_nc+1,3,ngf=ngf,norm_layer=norm_layer,use_dropout=use_dropout,n_blocks=9)
        self.linear_eval = linear_eval

    def forward(self,x,mid_face=None):
        if self.linear_eval:
            attn_map = self.attn_gene(x)
            b = attn_map.size(0)
            return attn_map.sigmoid().reshape(b,-1)

        with torch.no_grad():
            attn_map = self.attn_gene(x)
        #attn_map = attn_map.sigmoid()

        mid_face_with_attn = torch.cat([attn_map,mid_face],dim=1)
        recon_exp = self.recon_gene(mid_face_with_attn)
        return recon_exp,attn_map


class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss

class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class MoCoNetwork(nn.Module):
    def __init__(self,dim=128,K=65536,m=0.999,T=0.07,neg_alpha=1.6):
        super(MoCoNetwork, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        self.neg_alpha = neg_alpha

        #self.encoder_q = models.resnet50(num_classes=dim)
        #self.encoder_k = models.resnet50(num_classes=dim)
        self.encoder_q = FaceCycleBackbone()
        self.encoder_k = FaceCycleBackbone()

        #dim_mlp = self.encoder_q.fc.weight.shape[1]
        self.encoder_q.fc = nn.Sequential(nn.Linear(2048, 2048),
                                         nn.ReLU(),
                                         nn.Linear(2048,512),
                                         nn.BatchNorm1d(512))
        self.encoder_k.fc = nn.Sequential(nn.Linear(2048, 2048),
                                         nn.ReLU(),
                                         nn.Linear(2048,512),
                                         nn.BatchNorm1d(512))

        for param_q,param_k in zip(self.encoder_q.parameters(),self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        self.register_buffer("queue",torch.randn(dim,K))
        self.queue = nn.functional.normalize(self.queue,dim=0)

        self.register_buffer("queue_ptr",torch.zeros(1,dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):

        for param_q,param_k in zip(self.encoder_q.parameters(),self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self,keys):

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0

        self.queue[:,ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K

        self.queue_ptr[0] = ptr

    def forward(self,im_q,im_k=None,linear_eval=False):
        if linear_eval:
            fea = self.encoder_q(im_q)
            return fea

        q = self.encoder_q(im_q)
        q = nn.functional.normalize(q,dim=1)

        with torch.no_grad():
            self._momentum_update_key_encoder()

            k = self.encoder_k(im_k)
            k = nn.functional.normalize(k,dim=1)

        l_pos = torch.einsum('nc,nc->n',[q,k]).unsqueeze(-1)

        ######################nag
        queue = self.queue.clone().detach().T
        neg_mask = np.random.beta(self.neg_alpha,self.neg_alpha,size=(queue.shape[0]))
        if isinstance(neg_mask,np.ndarray):
            neg_mask = torch.from_numpy(neg_mask).float().cuda()
            neg_mask = neg_mask.unsqueeze(dim=1)
        indices = torch.randperm(queue.shape[0]).cuda()
        queue = neg_mask * queue + (1 - neg_mask) * queue[indices]
        queue = queue.T

        l_neg = torch.einsum('nc,ck->nk',[q,queue])

        logits = torch.cat([l_pos,l_neg],dim=1)

        logits /= self.T

        labels = torch.zeros(logits.shape[0],dtype=torch.long).cuda()

        self._dequeue_and_enqueue(k)

        return logits, labels

class ReconSimCLRNetwork(torch.nn.Module):
    def __init__(self,config):
        super(ReconSimCLRNetwork, self).__init__()
        self.extractor = FaceCycleBackbone()
        #self.extractor = models.resnet50(num_classes=512)
        #dim_mlp = self.extractor.fc.weight.shape[1]
        #dim_mlp_out = self.extractor.fc.weight.shape[0]

        self.fc = nn.Sequential(nn.Linear(2048, 2048),
                                         nn.ReLU(),
                                         nn.Linear(2048,512),
                                         nn.BatchNorm1d(512))

        self.decoder = Generator(d_model=512)

    def forward(self,x_normal,x=None):

        fea = self.fc(self.extractor(x))

        b = x_normal.size(0)
        assert b % 2 == 0
        x_normal_half_1 = x_normal[:b//2]
        x_normal_half_2 = x_normal[b//2:]

        beta_alpha = np.random.beta(1.6,1.6,size=(b//2,1))
        if isinstance(beta_alpha,np.ndarray):
            beta_alpha = torch.from_numpy(beta_alpha).float().cuda()

        x_normal_interp = x_normal_half_1 * beta_alpha.unsqueeze(dim=2).unsqueeze(dim=3) + (1-beta_alpha).unsqueeze(dim=2).unsqueeze(dim=3) * x_normal_half_2

        fea_normal = self.extractor(x_normal)

        fea_normal_half_1 = fea_normal[:b//2]
        fea_normal_half_2 = fea_normal[b//2:]

        fea_normal_interp = fea_normal_half_1 * beta_alpha + (1 - beta_alpha) * fea_normal_half_2

        recon_img_intep = self.decoder(fea_normal_interp)
        recon_img = self.decoder(fea_normal)

        return fea,recon_img,x_normal_interp,recon_img_intep

class ReconMoCoNetwork(nn.Module):
    def __init__(self,K=65536,m=0.999,T=0.07):
        super(ReconMoCoNetwork, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        self.encoder_q = InceptionResnetV1()
        self.encoder_k = InceptionResnetV1()

        dim_mlp = self.encoder_q.last_linear.weight.shape[1]
        dim_mlp_out = self.encoder_q.last_linear.weight.shape[0]

        self.encoder_q.last_linear = torch.nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp),
            nn.ReLU(),
            self.encoder_q.last_linear
        )

        self.encoder_k.last_linear = torch.nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp),
            nn.ReLU(),
            self.encoder_k.last_linear
        )

        for param_q,param_k in zip(self.encoder_q.parameters(),self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        self.register_buffer("queue",torch.randn(dim_mlp_out,K))
        self.queue = nn.functional.normalize(self.queue,dim=0)

        self.register_buffer("queue_ptr",torch.zeros(1,dtype=torch.long))

        self.decoder = Generator(d_model=dim_mlp_out)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):

        for param_q,param_k in zip(self.encoder_q.parameters(),self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self,keys):

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0

        self.queue[:,ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K

        self.queue_ptr[0] = ptr

    def forward(self,im_normal,im_q,im_k=None):

        q = self.encoder_q(im_q)
        q = nn.functional.normalize(q,dim=1)

        with torch.no_grad():
            self._momentum_update_key_encoder()

            k = self.encoder_k(im_k)
            k = nn.functional.normalize(k,dim=1)

        l_pos = torch.einsum('nc,nc->n',[q,k]).unsqueeze(-1)

        l_neg = torch.einsum('nc,ck->nk',[q,self.queue.clone().detach()])

        logits = torch.cat([l_pos,l_neg],dim=1)

        logits /= self.T

        labels = torch.zeros(logits.shape[0],dtype=torch.long).cuda()

        self._dequeue_and_enqueue(k)

        fea_normal = self.encoder_q(im_normal)
        recon_img = self.decoder(fea_normal)

        return logits, labels, recon_img


class selfattention(nn.Module):
    def __init__(self, inplanes):
        super(selfattention, self).__init__()

        self.interchannel = inplanes
        self.inplane = inplanes
        self.g = nn.Conv2d(inplanes, inplanes, kernel_size=1, stride=1, padding=0)
        self.theta = nn.Conv2d(inplanes, self.interchannel, kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(inplanes, self.interchannel, kernel_size=1, stride=1, padding=0)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x):
        b, c, h, w = x.size()
        g_y = self.g(x).view(b, c, -1)  # BXcXN
        theta_x = self.theta(x).view(b, self.interchannel, -1)
        theta_x = F.softmax(theta_x, dim=-1)  # softmax on N
        theta_x = theta_x.permute(0, 2, 1).contiguous()  # BXNXC'

        phi_x = self.phi(x).view(b, self.interchannel, -1)  # BXC'XN

        similarity = torch.bmm(phi_x, theta_x)  # BXc'Xc'

        g_y = F.softmax(g_y, dim=1)
        attention = torch.bmm(similarity, g_y)  # BXCXN
        attention = attention.view(b, c, h, w).contiguous()
        y = self.act(x + attention)
        return y

class BasicBlockNormal(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlockNormal, self).__init__()

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv2d(inplanes,planes,3,stride,1)
        self.relu = nn.LeakyReLU(negative_slope=0.1,inplace=True)
        self.conv2 = nn.Conv2d(planes,planes,3,1,1)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)
        #out = self.relu(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = (out + identity)
        return self.relu(out)

class FaceCycleBackbone(torch.nn.Module):
    def __init__(self):
        super(FaceCycleBackbone, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, 7, 2, 3, bias=True),
                                   nn.LeakyReLU(negative_slope=0.1),
                                   nn.Conv2d(64, 64, 3, 1, 1, bias=True),
                                   nn.LeakyReLU(negative_slope=0.1))

        self.layer1 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1, bias=True),
                                    nn.LeakyReLU(negative_slope=0.1),
                                    selfattention(64),
                                    nn.Conv2d(64, 64, 3, 1, 1, bias=True),
                                    nn.LeakyReLU(negative_slope=0.1))  # 64

        self.layer2_1 = nn.Sequential(nn.Conv2d(64, 128, 3, 2, 1, bias=True),
                                      nn.LeakyReLU(negative_slope=0.1),
                                      selfattention(128),
                                      nn.Conv2d(128, 128, 3, 1, 1, bias=True),
                                      nn.LeakyReLU(negative_slope=0.1), )  # 64

        self.resblock1 = BasicBlockNormal(128, 128)
        self.resblock2 = BasicBlockNormal(128, 128)

        self.layer2_2 = nn.Sequential(nn.Conv2d(128, 128, 3, 2, 1, bias=True),
                                      nn.LeakyReLU(negative_slope=0.1),
                                      nn.Conv2d(128, 128, 3, 1, 1, bias=True),
                                      nn.LeakyReLU(negative_slope=0.1), )  # 64

        self.layer3_1 = nn.Sequential(nn.Conv2d(128, 256, 3, 2, 1, bias=True),
                                      nn.LeakyReLU(negative_slope=0.1),
                                      nn.Conv2d(256, 256, 3, 1, 1, bias=True),
                                      nn.LeakyReLU(negative_slope=0.1), )  # 64

        self.layer3_2_exp = nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1, bias=True),  # stride 2 for 128x128
                                      nn.LeakyReLU(negative_slope=0.1),
                                      nn.Conv2d(256, 128, 3, 1, 1, bias=True),
                                      nn.LeakyReLU(negative_slope=0.1),
                                      )  # 64

        self.layer3_2_pose = nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1, bias=True),  # stride 2 for 128x128
                                          nn.LeakyReLU(negative_slope=0.1),
                                          nn.Conv2d(256, 128, 3, 1, 1, bias=True),
                                          nn.LeakyReLU(negative_slope=0.1))  # 64

        #self.fc = nn.Identity()
        # self.exp_fc = nn.Sequential(nn.Linear(2048, 2048),
        #                             nn.ReLU(),
        #                             nn.Linear(2048, 512),
        #                             nn.BatchNorm1d(512))
        #
        # self.pose_fc = nn.Sequential(nn.Linear(2048, 2048),
        #                              nn.ReLU(),
        #                              nn.Linear(2048, 512),
        #                              nn.BatchNorm1d(512))

        #self.decoder = Generator(d_model=512)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        #encoder
        '''

        :param x: [batch,3,64,64]
        :return:
        '''
        out_1 = self.conv1(x) # [batch,64,32,32]
        out_1 = self.layer1(out_1) # [batch,64,32,32]
        out_2 = self.layer2_1(out_1) # [batch,128,16,16]
        out_2 = self.resblock1(out_2) # [batch,128,16,16]
        out_2 = self.resblock2(out_2) # [batch,128,16,16]
        out_2 = self.layer2_2(out_2) # [batch,128,8,8]
        out_3 = self.layer3_1(out_2) # [batch,256,4,4]

        out_3_exp = self.layer3_2_exp(out_3) # [batch,128,4,4]
        out_3_exp = out_3_exp.view(x.size()[0],-1) # [batch,2048]

        out_3_pose = self.layer3_2_pose(out_3)
        out_3_pose = out_3_pose.view(x.size()[0],-1)
        #print(out_3.size())
        # expcode = self.fc(out_3) # [batch,256]
        #out_3 = self.fc(out_3)
        #exp_fea = self.exp_fc(out_3)
        #pose_fea = self.pose_fc(out_3)
        #fea = torch.cat([exp_fea,pose_fea],dim=1)
        #fea = exp_fea + pose_fea
        #return fea
        #return out_3
        #return exp_fea
        #return pose_fea
        #return exp_fea,pose_fea
        #return expcode
        return out_3_exp,out_3_pose

class Projection(nn.Module):
    def __init__(self,in_dim=2048,out_dim=512):
        super(Projection,self).__init__()
        self.linear1 = nn.Linear(in_dim,in_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(in_dim,out_dim)
        self.bn = nn.BatchNorm1d(out_dim)

    def forward(self,x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.bn(out)
        return out

class ExpPoseModel(nn.Module):
    def __init__(self):
        super(ExpPoseModel, self).__init__()

        self.encoder = FaceCycleBackbone()
        #self.encoder = resnet18()
        #self.encoder = resnet34()
        #self.encoder = resnet50()

        self.exp_fc = Projection(in_dim=2048,out_dim=512)

        self.pose_fc = Projection(in_dim=2048,out_dim=512)

        self.decoder = Generator(d_model=2048)

    def forward(self,exp_img,normal_img,flip_img):
        #exp_fea = self.exp_encoder_fc(self.exp_encoder(exp_img))
        #pose_fea = self.pose_encoder_fc(self.pose_encoder(pose_img))
        exp_fea,pose_fea = self.encoder(exp_img)
        exp_fea_fc = self.exp_fc(exp_fea)
        pose_fea_fc = self.pose_fc(pose_fea)

        b = normal_img.size(0)

        normal_exp_fea,normal_pose_fea = self.encoder(normal_img)
        flip_exp_fea,flip_pose_fea = self.encoder(flip_img)

        pose_fea_fc = torch.cat([pose_fea_fc[0:b],self.pose_fc(normal_pose_fea)],dim=0)

        # normal_exp_fea_fc = self.exp_fc(normal_exp_fea)
        # flip_exp_fea_fc = self.exp_fc(flip_exp_fea)
        #
        # normal_pose_fea_fc = self.pose_fc(normal_pose_fea)
        # flip_pose_fea_fc = self.pose_fc(flip_pose_fea)

        # ########### test fea
        # recon_normal_fea = F.normalize(normal_pose_fea_fc+flip_pose_fea_fc,dim=1)
        # recon_flip_fea = F.normalize(flip_pose_fea_fc+normal_pose_fea_fc, dim=1)
        #
        # recon_normal_img = self.decoder(recon_normal_fea)
        # recon_flip_img = self.decoder(recon_flip_fea)
        #
        # return None,None,recon_normal_img,recon_flip_img,None

        recon_normal_exp_flip_pose_fea = F.normalize(normal_exp_fea+flip_pose_fea, dim=1)
        recon_flip_exp_normal_pose_fea = F.normalize(flip_exp_fea+normal_pose_fea, dim=1)
        recon_normal_exp_normal_pose_fea = F.normalize(normal_exp_fea+normal_pose_fea, dim=1)
        recon_flip_exp_flip_pose_fea = F.normalize(flip_exp_fea+flip_pose_fea,dim=1)

        recon_normal_exp_flip_pose_img = self.decoder(recon_normal_exp_flip_pose_fea)
        recon_flip_exp_normal_pose_img = self.decoder(recon_flip_exp_normal_pose_fea)
        recon_normal_exp_normal_pose_img = self.decoder(recon_normal_exp_normal_pose_fea)
        recon_flip_exp_flip_psoe_img = self.decoder(recon_flip_exp_flip_pose_fea)

        return exp_fea_fc, pose_fea_fc, exp_fea,pose_fea, recon_normal_exp_flip_pose_img,recon_flip_exp_normal_pose_img,recon_normal_exp_normal_pose_img,recon_flip_exp_flip_psoe_img
        ########### test
        # recon_normal_exp_flip_pose_fea = F.normalize(normal_exp_fea_fc + flip_pose_fea_fc, dim=1)
        # recon_flip_exp_normal_pose_fea = F.normalize(flip_exp_fea_fc+normal_pose_fea_fc, dim=1)
        # recon_normal_exp_normal_exp_fea = F.normalize(normal_exp_fea_fc + normal_exp_fea_fc, dim=1)
        # recon_normal_pose_normal_pose_fea = F.normalize(normal_pose_fea_fc + normal_pose_fea_fc,dim=1)
        # recon_flip_pose_flip_pose_fea = F.normalize(flip_pose_fea_fc + flip_pose_fea_fc,dim=1)
        #
        # recon_normal_exp_flip_pose_img = self.decoder(recon_normal_exp_flip_pose_fea)
        # recon_flip_exp_normal_pose_img = self.decoder(recon_flip_exp_normal_pose_fea)
        # recon_normal_exp_normal_exp_img = self.decoder(recon_normal_exp_normal_exp_fea)
        # recon_normal_pose_normal_pose_img = self.decoder(recon_normal_pose_normal_pose_fea)
        # recon_flip_pose_flip_pose_img = self.decoder(recon_flip_pose_flip_pose_fea)
        # #
        # return recon_normal_exp_flip_pose_img,recon_flip_exp_normal_pose_img,recon_normal_exp_normal_exp_img,recon_normal_pose_normal_pose_img,recon_flip_pose_flip_pose_img


        # return exp_fea,pose_fea,recon_normal_exp_flip_pose_img,recon_flip_exp_normal_pose_img,\
        #        recon_normal_exp_normal_pose_img,normal_exp_fea_fc,normal_pose_fea_fc,flip_exp_fea_fc,flip_pose_fea_fc

class SimCLRNetwork(torch.nn.Module):
    def __init__(self):
        super(SimCLRNetwork, self).__init__()

        self.encoder = FaceCycleBackbone()
        self.fc = nn.Sequential(nn.Linear(2048, 2048),
                                         nn.ReLU(),
                                         nn.Linear(2048,512),
                                         nn.BatchNorm1d(512))

    def forward(self,x):
        fea = self.fc(self.encoder(x))
        return fea

#
# if __name__ == '__main__':
#     #encoder = ReconSimCLRNetwork(config=None).cuda(0)
#     #encoder = models.resnet50().cuda(0)
#     encoder = InceptionResnetV1()
#     #encoder.fc = torch.nn.Identity()
#     print(encoder)
#     x = torch.randn(2,3,128,128).cuda(0)
#     output = encoder(x)
#     print(output.size())