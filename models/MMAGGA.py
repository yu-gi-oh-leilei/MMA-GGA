from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import functional as F
import math
import random
from .resnet import ResNet, BasicBlock, Bottleneck, ResNetNonLocal
import numpy as np
from .non_local import Non_local
import torchvision

__all__ = ['MMAGGA']

# x=torch.random(32,8,3,256,128)
# model=MMAGGA()
# model(x)

class MMAGGA(nn.Module):
    def __init__(self, num_classes, loss={'xent'}):
        super(MMAGGA, self).__init__()
        print('=================', num_classes, '=================')

        self.link_pred = False
        self.hidden_dim = 512
        self.p1 = 4. #4.
        self.p2 = 8.
        self.p3 = 2.

        self.loss = loss
        self.base = torchvision.models.resnet50(pretrained=True)
        self.base.layer4[0].downsample[0].stride = (1, 1)
        self.base.layer4[0].conv2.stride = (1, 1)

        layers = [3, 4, 6, 3]
        non_layers = [0, 2, 3, 0]
        self.NL_1 = nn.ModuleList(
            [Non_local(256) for i in range(non_layers[0])])
        self.NL_1_idx = sorted([layers[0] - (i + 1) for i in range(non_layers[0])])
        self.NL_2 = nn.ModuleList(
            [Non_local(512) for i in range(non_layers[1])])
        self.NL_2_idx = sorted([layers[1] - (i + 1) for i in range(non_layers[1])])
        self.NL_3 = nn.ModuleList(
            [Non_local(1024) for i in range(non_layers[2])])
        self.NL_3_idx = sorted([layers[2] - (i + 1) for i in range(non_layers[2])])
        self.NL_4 = nn.ModuleList(
            [Non_local(2048) for i in range(non_layers[3])])
        self.NL_4_idx = sorted([layers[3] - (i + 1) for i in range(non_layers[3])])

        self.downsample0=nn.Linear(2048, 1024)
        self.downsample1=nn.Linear(2048, 1024)
        self.downsample2=nn.Linear(2048, 1024)
        self.downsample3=nn.Linear(2048, 1024)
        #=======global================
        self.bnneck1 = nn.BatchNorm1d(4*2*self.hidden_dim)
        self.bnneck1.bias.requires_grad_(False)  # no shift
        # classifier
        self.fc_list1 = nn.ModuleList()
        for i in range(4):
            classifier = nn.Linear(2*self.hidden_dim, num_classes,bias=False)
            nn.init.normal_(classifier.weight, std=0.001)
            self.fc_list1.append(classifier)
        
        
        #GGA
        self.attconv0=nn.Conv1d(8,  8,  1024)
        '''
        self.attconv1=nn.Conv1d(32, 32, 1024)
        self.attconv2=nn.Conv1d(64, 64, 1024)
        self.attconv3=nn.Conv1d(16, 16, 1024)
        '''
        self.gl11=nn.Linear(1024, 1024)
        self.gl01=nn.Linear(1024, 1024)
        self.gl22=nn.Linear(1024, 1024)
        self.gl02=nn.Linear(1024, 1024)
        self.gl33=nn.Linear(1024, 1024)
        self.gl03=nn.Linear(1024, 1024)

        # MMA

        self.mgcov1 = torch.nn.Parameter(torch.FloatTensor([[0.5,0.5,0.01,0.01,0.01,0.01,0.01,0.01],[0.01,0.01,0.5,0.5,0.01,0.01,0.01,0.01],[0.01,0.01,0.01,0.01,0.5,0.5,0.01,0.01],[0.01,0.01,0.01,0.01,0.01,0.01,0.5,0.5]]), requires_grad=True)
        self.mgcov2 = torch.nn.Parameter(torch.FloatTensor([[0.25,0.25,0.25,0.25,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001],[0.001,0.001,0.001,0.001,0.25,0.25,0.25,0.25,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001],[0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.25,0.25,0.25,0.25,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001],[0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.25,0.25,0.25,0.25,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001],[0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.25,0.25,0.25,0.25,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001],[0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.25,0.25,0.25,0.25,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001],[0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.25,0.25,0.25,0.25,0.001,0.001,0.001,0.001], [0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.25,0.25,0.25,0.25]]), requires_grad=True)
        self.mgcov3 = torch.nn.Parameter(torch.FloatTensor([[1,0.1],[0.1,1]]), requires_grad=True)

        '''
        self.mgcov1=nn.Conv1d(8, 4, 1)
        self.mgcov2=nn.Conv1d(32,8, 1)
        self.mgcov3=nn.Conv1d(2, 2, 1)
        '''

    def frozen_without_mma(self):
        for n, p in self.named_parameters():
            if 'mgcov' in n:
                p.requires_grad=True
            else:
                p.requires_grad=False
        self.bnneck1.bias.requires_grad_(False)

    def unfrozen_mma(self):
        for n, p in self.named_parameters():
            if 'mgcov' in n:
                p.requires_grad=True
        self.bnneck1.bias.requires_grad_(False)

    def frozen_mma(self):
        for n, p in self.named_parameters():
            if 'mgcov' in n:
                p.requires_grad=False
        self.bnneck1.bias.requires_grad_(False)

    def unfrozen_whole(self):
        for n, p in self.named_parameters():
            p.requires_grad=True
        self.bnneck1.bias.requires_grad_(False)

    def get_optimizer(self, args):
        param_dicts = [{
            "params": [p for n, p in self.named_parameters() if p.requires_grad],
            "lr": args.lr,
            "weight_decay": args.weight_decay}]
        optimizer = torch.optim.Adam(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
        return optimizer


    def forward(self, x):
        # print(x.shape) # [64, 6, 3, 256, 128])
        b, t, c, h, w = x.shape
        x=x.view(b*t, c, h, w)
        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x = self.base.relu(x)
        x = self.base.maxpool(x)

        NL1_counter = 0
        if len(self.NL_1_idx) == 0: self.NL_1_idx = [-1]
        for i in range(len(self.base.layer1)):
            x = self.base.layer1[i](x)
            if i == self.NL_1_idx[NL1_counter]:
                _, C, H, W = x.shape
                x = self.NL_1[NL1_counter](x)
                NL1_counter += 1
        # Layer 2
        NL2_counter = 0
        if len(self.NL_2_idx) == 0: self.NL_2_idx = [-1]
        for i in range(len(self.base.layer2)):
            x = self.base.layer2[i](x)
            if i == self.NL_2_idx[NL2_counter]:
                _, C, H, W = x.shape
                x = self.NL_2[NL2_counter](x)
                NL2_counter += 1
        # Layer 3
        NL3_counter = 0
        if len(self.NL_3_idx) == 0: self.NL_3_idx = [-1]
        for i in range(len(self.base.layer3)):
            x = self.base.layer3[i](x)
            if i == self.NL_3_idx[NL3_counter]:
                _, C, H, W = x.shape
                x = self.NL_3[NL3_counter](x)
                NL3_counter += 1
        # Layer 4
        NL4_counter = 0
        if len(self.NL_4_idx) == 0: self.NL_4_idx = [-1]
        for i in range(len(self.base.layer4)):
            x = self.base.layer4[i](x)
            if i == self.NL_4_idx[NL4_counter]:
                _, C, H, W = x.shape
                x = self.NL_4[NL4_counter](x)
                NL4_counter += 1

        x = x.view(b*t, 2048, 16, 8)
        x=x.clamp(min=3e-6).pow(3)

        #========global=============
        x0 = F.avg_pool2d(x, x.size()[2:]).clamp(min=3e-6).pow(1./3)
        x0 = x0.view(b, t, -1)
        xglobe= x0.view(b*t, 1, -1)

        xframelist=[]
        xframelist.append(xglobe)
        

        x=x.view(b*t,2048,16,8)
        
        x1 = F.avg_pool2d(x,(4,4)).clamp(min=3e-6).pow(1./3) 
        x1 = x1.view(b*t,2048,8)
        x1 = x1.permute(0,2,1)
        x1 = torch.matmul(self.mgcov1,x1)
        x1 = F.leaky_relu(x1)
        xframelist.append(x1)

        x2 = F.avg_pool2d(x,(2,2)).clamp(min=3e-6).pow(1./3)
        x2 = x2.view(b*t,2048,32)
        x2 = x2.permute(0,2,1)
        x2 = torch.matmul(self.mgcov2,x2)
        x2 = F.leaky_relu(x2)
        xframelist.append(x2)

        x3 = F.avg_pool2d(x,(8,8)).clamp(min=3e-6).pow(1./3)
        x3 = x3.view(b*t,2048,2)
        x3 = x3.permute(0,2,1)
        x3 = torch.matmul(self.mgcov3,x3)
        x3 = F.leaky_relu(x3)
        xframelist.append(x3)
        
        xframe=torch.cat(xframelist,dim=1)

        x0=xframe[:,0,:]
        x1=xframe[:,1:5,:].contiguous()
        x2=xframe[:,5:13,:].contiguous()
        x3=xframe[:,13:,:].contiguous()
        
        x0 = x0.view(b,t,-1)
        x1 = x1.view(b, t*int(self.p1), -1)
        x2 = x2.view(b, t*int(self.p2), -1)
        x3 = x3.view(b, t*int(self.p3), -1)
        
        x0 = x0.view(b*t,-1)
        x0 = self.downsample0(x0)
        x0 = F.relu(x0)
        x0 = x0.view(b,t,-1)

        attf0 = self.attconv0(x0)
        attf0 = attf0.view(b,8)
        attf0 = F.softmax(attf0,dim=1)
        attf0 = attf0.view(b,1,8)


        f0 = torch.matmul(attf0,x0)
        f0 = f0.view(b, 2*self.hidden_dim)


        #======== part p1 ============
        x1 = x1.view(b*t*int(self.p1),-1)
        x1 = self.downsample1(x1)
        x1 = F.relu(x1)
        x1 = x1.view(b, t*int(self.p1),-1)
        #======== gga p1 ============
        a1 = self.gl11(x1)
        a1 = F.relu(a1)
        a1 = F.normalize(a1,dim=2,p=2)
        b1 = self.gl01(f0)
        b1 = F.relu(b1)
        b1 = b1.view(b,2*self.hidden_dim)
        b1 = F.normalize(b1,dim=1,p=2)
        b1 = b1.view(b,2*self.hidden_dim,1)
        att1 = torch.matmul(a1,b1)
        att1 = att1.view(b,t*int(self.p1))
        att1 = F.softmax(att1,dim=1)
        att1 = att1.view(b,1,t*int(self.p1))
        f1 = torch.matmul(att1,x1)
        f1 = f1.view(b, 2*self.hidden_dim)
        '''
        #====== pooling p1 =============
        attf1 = self.attconv1(x1)
        attf1 = attf1.view(b,32)
        attf1 = F.softmax(attf1,dim=1)
        attf1 = attf1.view(b,1,32)

        f1 = torch.matmul(attf1,x1)
        f1 = f1.view(b, 2*self.hidden_dim)
        '''

        #========part p2============
        x2 = x2.view(b*t*int(self.p2),-1)
        x2 = self.downsample2(x2)
        x2 = F.relu(x2)
        x2 = x2.view(b, t*int(self.p2),-1)
        #======== gga p2 ============
        a2 = self.gl22(x2)
        a2 = F.relu(a2)
        a2 = F.normalize(a2,dim=2,p=2)
        b2 = self.gl02(f0)
        b2 = F.relu(b2)
        b2 = b2.view(b,2*self.hidden_dim)
        b2 = F.normalize(b2,dim=1,p=2)
        b2 = b2.view(b,2*self.hidden_dim,1)

        att2 = torch.matmul(a2,b2)
        att2 = att2.view(b,t*int(self.p2))
        att2 = F.softmax(att2,dim=1)
        att2 = att2.view(b,1,t*int(self.p2))
        f2 = torch.matmul(att2,x2)
        f2 = f2.view(b, 2*self.hidden_dim)
        '''
        #======pooling p2=============
        attf2 = self.attconv2(x2)
        attf2 = attf2.view(b,64)
        attf2 = F.softmax(attf2,dim=1)
        attf2 = attf2.view(b,1,64)
        f2 = torch.matmul(attf2,x2)
        f2 = f2.view(b, 2*self.hidden_dim)
        '''

        #========part p3============
        x3 = x3.view(b*t*int(self.p3),-1)
        x3 = self.downsample3(x3)
        x3 = F.relu(x3)
        x3 = x3.view(b, t*int(self.p3),-1)
        #======== gga p3 ============
        a3 = self.gl33(x3)
        a3 = F.relu(a3)
        a3 = F.normalize(a3,dim=2,p=2)
        b3 = self.gl03(f0)
        b3 = F.relu(b3)
        b3 = b3.view(b,2*self.hidden_dim)
        b3 = F.normalize(b3,dim=1,p=2)
        b3 = b3.view(b,2*self.hidden_dim,1)
        att3 = torch.matmul(a3,b3)
        att3 = att3.view(b,t*int(self.p3))
        att3 = F.softmax(att3,dim=1)
        att3 = att3.view(b,1,t*int(self.p3))
        f3 = torch.matmul(att3,x3)
        f3 = f3.view(b, 2*self.hidden_dim)
        '''
        #======pooling p3=============
        attf3 = self.attconv3(x3)
        attf3 = attf3.view(b,16)
        attf3 = F.softmax(attf3,dim=1)
        attf3 = attf3.view(b,1,16)
        f3 = torch.matmul(attf3,x3)
        f3 = f3.view(b, 2*self.hidden_dim)
        '''
        #=====bnneck==================
        f = torch.cat((f0, f3, f1, f2), 1)
        f_bn = self.bnneck1(f)
        
        local_feat = [f_bn[:,0:2*self.hidden_dim], f_bn[:,2*self.hidden_dim:4*self.hidden_dim], f_bn[:, 4*self.hidden_dim:6*self.hidden_dim], f_bn[:, 6*self.hidden_dim:]]
        logits_list = []
        
        for i in range(4):
            logits_list.append(self.fc_list1[i](local_feat[i]))
        
        if not self.training:
            return f_bn        
        return logits_list, f