#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataloader import *
import random
import pickle
import math
import time
from utils import *
import os


# TIME_TYPE = {'no-time':0, 'point-in-time':1, 'only-begin':2, 'only-end':3, 'full-interval':4}

def Identity(x):
    return x

class SetIntersection(nn.Module):
    def __init__(self, mode_dims, expand_dims, agg_func=torch.min):
        super(SetIntersection, self).__init__()
        self.agg_func = agg_func
        self.pre_mats = nn.Parameter(torch.FloatTensor(expand_dims, mode_dims))
        nn.init.xavier_uniform_(self.pre_mats)
        self.register_parameter("premat", self.pre_mats)
        self.post_mats = nn.Parameter(torch.FloatTensor(mode_dims, expand_dims))
        nn.init.xavier_uniform_(self.post_mats)
        self.register_parameter("postmat", self.post_mats)
        self.pre_mats_im = nn.Parameter(torch.FloatTensor(expand_dims, mode_dims))
        nn.init.xavier_uniform_(self.pre_mats_im)
        self.register_parameter("premat_im", self.pre_mats_im)
        self.post_mats_im = nn.Parameter(torch.FloatTensor(mode_dims, expand_dims))
        nn.init.xavier_uniform_(self.post_mats_im)
        self.register_parameter("postmat_im", self.post_mats_im)

    def forward(self, embeds1, embeds2, embeds3 = [], name='real'):
        if name == 'real':
            temp1 = F.relu(embeds1.mm(self.pre_mats))
            temp2 = F.relu(embeds2.mm(self.pre_mats))
            if len(embeds3) > 0:
                temp3 = F.relu(embeds3.mm(self.pre_mats))
                combined = torch.stack([temp1, temp2, temp3])
            else:
                combined = torch.stack([temp1, temp2])
            combined = self.agg_func(combined, dim=0)
            if type(combined) == tuple:
                combined = combined[0]
            combined = combined.mm(self.post_mats)

        elif name == 'img':
            temp1 = F.relu(embeds1.mm(self.pre_mats_im))
            temp2 = F.relu(embeds2.mm(self.pre_mats_im))
            if len(embeds3) > 0:
                temp3 = F.relu(embeds3.mm(self.pre_mats_im))
                combined = torch.stack([temp1, temp2, temp3])
            else:
                combined = torch.stack([temp1, temp2])
            combined = self.agg_func(combined, dim=0)
            if type(combined) == tuple:
                combined = combined[0]
            combined = combined.mm(self.post_mats_im)
        return combined

class CenterSet(nn.Module):
    def __init__(self, mode_dims, expand_dims, center_use_offset, agg_func=torch.min, bn='no', nat=1, name='Real_center'):
        super(CenterSet, self).__init__()
        assert nat == 1, 'vanilla method only support 1 nat now'
        self.center_use_offset = center_use_offset
        self.agg_func = agg_func
        self.bn = bn
        self.nat = nat
        if center_use_offset:
            self.pre_mats = nn.Parameter(torch.FloatTensor(expand_dims*2, mode_dims))
        else:
            self.pre_mats = nn.Parameter(torch.FloatTensor(expand_dims, mode_dims))

        nn.init.xavier_uniform_(self.pre_mats)
        self.register_parameter("premat_%s"%name, self.pre_mats)
        if bn != 'no':
            self.bn1 = nn.BatchNorm1d(mode_dims)
            self.bn2 = nn.BatchNorm1d(mode_dims)
            self.bn3 = nn.BatchNorm1d(mode_dims)

        self.post_mats = nn.Parameter(torch.FloatTensor(mode_dims, expand_dims))
        nn.init.xavier_uniform_(self.post_mats)
        self.register_parameter("postmat_%s"%name, self.post_mats)

    def forward(self, embeds1, embeds1_o, embeds2, embeds2_o, embeds3 = [], embeds3_o=[]):
        if self.center_use_offset:
            temp1 = torch.cat([embeds1, embeds1_o], dim=1)
            temp2 = torch.cat([embeds2, embeds2_o], dim=1)
            if len(embeds3) > 0:
                temp3 = torch.cat([embeds3, embeds3_o], dim=1)
        else:
            temp1 = embeds1
            temp2 = embeds2
            if len(embeds3) > 0:
                temp3 = embeds3

        if self.bn == 'no':
            temp1 = F.relu(temp1.mm(self.pre_mats))
            temp2 = F.relu(temp2.mm(self.pre_mats))
        elif self.bn == 'before':
            temp1 = F.relu(self.bn1(temp1.mm(self.pre_mats)))
            temp2 = F.relu(self.bn2(temp2.mm(self.pre_mats)))
        elif self.bn == 'after':
            temp1 = self.bn1(F.relu(temp1.mm(self.pre_mats)))
            temp2 = self.bn2(F.relu(temp2.mm(self.pre_mats)))
        if len(embeds3) > 0:
            if self.bn == 'no':
                temp3 = F.relu(temp3.mm(self.pre_mats))
            elif self.bn == 'before':
                temp3 = F.relu(self.bn3(temp3.mm(self.pre_mats)))
            elif self.bn == 'after':
                temp3 = self.bn3(F.relu(temp3.mm(self.pre_mats)))
            combined = torch.stack([temp1, temp2, temp3])
        else:
            combined = torch.stack([temp1, temp2])
        combined = self.agg_func(combined, dim=0)
        if type(combined) == tuple:
            combined = combined[0]
        combined = combined.mm(self.post_mats)
        return combined

class MeanSet(nn.Module):
    def __init__(self):
        super(MeanSet, self).__init__()

    def forward(self, embeds1, embeds1_o, embeds2, embeds2_o, embeds3 = [], embeds3_o=[]):
        if len(embeds3) > 0:
            return torch.mean(torch.stack([embeds1, embeds2, embeds3], dim=0), dim=0)
        else:
            return torch.mean(torch.stack([embeds1, embeds2], dim=0), dim=0)

class MinSet(nn.Module):
    def __init__(self):
        super(MinSet, self).__init__()

    def forward(self, embeds1, embeds1_o, embeds2, embeds2_o, embeds3 = [], embeds3_o=[]):
        if len(embeds3) > 0:
            return torch.min(torch.stack([embeds1, embeds2, embeds3], dim=0), dim=0)[0]
        else:
            return torch.min(torch.stack([embeds1, embeds2], dim=0), dim=0)[0]

class OffsetSet(nn.Module):
    def __init__(self, mode_dims, expand_dims, offset_use_center, agg_func=torch.min, name='Real_offset'):
        super(OffsetSet, self).__init__()
        self.offset_use_center = offset_use_center
        self.agg_func = agg_func
        self.act_func = F.relu
        if offset_use_center:
            self.pre_mats = nn.Parameter(torch.FloatTensor(expand_dims*2, mode_dims))
            nn.init.xavier_uniform_(self.pre_mats)
            self.register_parameter("premat_%s"%name, self.pre_mats)
        else:
            self.pre_mats = nn.Parameter(torch.FloatTensor(expand_dims, mode_dims))
            nn.init.xavier_uniform_(self.pre_mats)
            self.register_parameter("premat_%s"%name, self.pre_mats)

        self.post_mats = nn.Parameter(torch.FloatTensor(mode_dims, expand_dims))
        nn.init.xavier_uniform_(self.post_mats)
        self.register_parameter("postmat_%s"%name, self.post_mats)

    def forward(self, embeds1, embeds1_o, embeds2, embeds2_o, embeds3 = [], embeds3_o=[]):
        length = len(embeds3_o)
        assert type(embeds3_o) == list
        if self.offset_use_center:
            temp1 = torch.cat([embeds1, embeds1_o], dim=1)
            temp2 = torch.cat([embeds2, embeds2_o], dim=1)
            if length > 0:
                embeds3 = torch.cat(embeds3, dim=0)
                embeds3_o = torch.cat(embeds3_o, dim=0)
                temp3 = torch.cat([embeds3, embeds3_o], dim=1)
        else:
            temp1 = embeds1_o
            temp2 = embeds2_o
            if length > 0:
                embeds3_o = torch.cat(embeds3_o, dim=0)
                temp3 = embeds3_o
        temp1 = self.act_func(temp1.mm(self.pre_mats)) # linear transformation
        temp2 = self.act_func(temp2.mm(self.pre_mats)) # linear transformation 
        if length > 0:
            temp3 = self.act_func(temp3.mm(self.pre_mats))
            temp3 = torch.chunk(temp3, length, dim=0)
            temps = [temp1, temp2] + [i for i in temp3]
            combined = torch.stack(temps)
        else:
            combined = torch.stack([temp1, temp2])
        combined = self.agg_func(combined, dim=0) # element-wise agg_func: mean, min, max, etc.; they choose mean;
        if type(combined) == tuple:
            combined = combined[0]
        combined = combined.mm(self.post_mats) # post-preocess; pass through a linear transformation
        return combined

# the difference between offsetSet and InductiveOffsetSet is that for inductiveOffsetSet, it has weights, which use offset_min as weights
class InductiveOffsetSet(nn.Module):
    def __init__(self, mode_dims, expand_dims, offset_use_center, off_reg, agg_func=torch.min, name='Real_offset'):
        super(InductiveOffsetSet, self).__init__()
        self.offset_use_center = offset_use_center
        self.agg_func = agg_func
        self.off_reg = off_reg
        self.OffsetSet_Module = OffsetSet(mode_dims, expand_dims, offset_use_center, self.agg_func)
        

    def forward(self, embeds1, embeds1_o, embeds2, embeds2_o, embeds3 = [], embeds3_o=[]):
        assert isinstance(embeds3_o, list) == True
        # assert len(embeds3_o) == 1
        if len(embeds3_o) > 0:
            offset_min = torch.min(torch.stack([embeds1_o, embeds2_o] + embeds3_o), dim=0)[0]
        else:
            offset_min = torch.min(torch.stack([embeds1_o, embeds2_o]), dim=0)[0] 
            ## if only stack, a new dimension will be added. if 'cat', then merge on the given dimension
            ## then the output of the min function has two dimensions: values & indices; element-wise min 
        offset = offset_min * torch.sigmoid(self.OffsetSet_Module(embeds1, embeds1_o, embeds2, embeds2_o, embeds3, embeds3_o))
        return offset

class SubOffsetSet(nn.Module):
    def __init__(self, mode_dims, expand_dims, off_reg, agg_func=torch.min, name='Sub_offset'):
        super(SubOffsetSet, self).__init__()
        self.agg_func = agg_func
        self.off_reg = off_reg
        self.pre_mats = nn.Parameter(torch.FloatTensor(expand_dims, mode_dims))
        nn.init.xavier_uniform_(self.pre_mats)
        self.register_parameter("premat_%s"%name, self.pre_mats)

        self.post_mats = nn.Parameter(torch.FloatTensor(mode_dims, expand_dims))
        nn.init.xavier_uniform_(self.post_mats)
        self.register_parameter("postmat_%s"%name, self.post_mats)

    def forward(self, embeds1, embeds1_o, embeds2, embeds_o, embeds3 = [], embeds3_o = []):
        temp1 = embeds1_o
        temp2 = embeds2_o 

        if len(embeds3_o) > 0:
            temp3 = embeds3_o 

        temp1 = F.relu(temp1.mm(self.pre_mats))
        temp2 = F.relu(temp2.mm(self.pre_mats))

        ## get the lower left corner & the top right corner for each 
        box1_min = embeds1 - 0.5 * temp1
        box1_max = embeds1 + 0.5 * temp1

        box2_min = embeds2 - 0.5 * temp2
        box2_max = embeds2 + 0.5 * temp2

        if len(embeds3_o) > 0:
            temp3 = F.relu(temp3.mm(self.pre_mats))
            box3_min = embeds3 - 0.5 * temp3
            box3_max = embeds3 + 0.5 * temp3

            combined_min = torch.stack([box1_min, box2_min, box3_min])
            combined_max = torch.stack([box1_max, box2_max, box3_max])
        else:
            combined_min = torch.stack([box1_min, box2_min])
            combined_max = torch.stack([box1_max, box2_max])

        ## element-wise max on combined_min; element-wise min on combined_max

class AttentionSet(nn.Module):
    def __init__(self, mode_dims, expand_dims, center_use_offset, att_reg=0., att_tem=1., att_type="whole", bn='no', nat=1, name="Real"):
        super(AttentionSet, self).__init__()
        self.center_use_offset = center_use_offset
        self.att_reg = att_reg
        self.att_type = att_type
        self.att_tem = att_tem
        self.Attention_module = Attention(mode_dims, expand_dims, center_use_offset, att_type=att_type, bn=bn, nat=nat)

    def forward(self, embeds1, embeds1_o, embeds2, embeds2_o, embeds3=[], embeds3_o=[]):
        # print('the attention module:', self.Attention_module.atten_mats1[:10])
        # print('the attention module 2:', self.Attention_module.atten_mats2[:10])
        assert type(embeds3) == list
        length = len(embeds3)
        temp1 = (self.Attention_module(embeds1, embeds1_o) + self.att_reg)/(self.att_tem+1e-4) # impose linear transformations on them;
        temp2 = (self.Attention_module(embeds2, embeds2_o) + self.att_reg)/(self.att_tem+1e-4) 
        if len(embeds3) > 0:
            embeds = [embeds1, embeds2] + embeds3
            embeds3 = torch.cat(embeds3, dim=0)
            embeds3_o = torch.cat(embeds3_o, dim=0)
            temp3 = (self.Attention_module(embeds3, embeds3_o) + self.att_reg)/(self.att_tem+1e-4)
            # split them into parts
            temp3 = torch.chunk(temp3, length, dim=0)
            temps = [temp1, temp2] + [i for i in temp3]
            center = torch.zeros_like(embeds1)
            
            if self.att_type == 'whole':
                combined = F.softmax(torch.cat(temps, dim=1), dim=1)
                for i in range(length+2):
                    center += embeds[i]*(combined[:,i].view(embeds1.size(0), 1))
                            # embeds2*(combined[:,1].view(embeds2.size(0), 1)) + \
                            # embeds3*(combined[:,2].view(embeds3.size(0), 1))
            elif self.att_type == 'ele':
                combined = F.softmax(torch.stack(temps), dim=0)
                for i in range(length + 2):
                    center += embeds[i]*combined[i] 
        else:
            if self.att_type == 'whole':
                combined = F.softmax(torch.cat([temp1, temp2], dim=1), dim=1)
                center = embeds1*(combined[:,0].view(embeds1.size(0), 1)) + \
                        embeds2*(combined[:,1].view(embeds2.size(0), 1))
            elif self.att_type == 'ele':
                combined = F.softmax(torch.stack([temp1, temp2]), dim=0) #softmax on each dimension of tensors and then used as weights
                center = embeds1*combined[0] + embeds2*combined[1]

        return center

class Attention(nn.Module):
    def __init__(self, mode_dims, expand_dims, center_use_offset, att_type, bn, nat, name="Real", act_func=F.relu):
        super(Attention, self).__init__()
        self.center_use_offset = center_use_offset
        self.bn = bn
        self.nat = nat
        self.act_func = F.relu

        if center_use_offset:
            self.atten_mats1 = nn.Parameter(torch.FloatTensor(expand_dims*2, mode_dims))
        else:
            self.atten_mats1 = nn.Parameter(torch.FloatTensor(expand_dims, mode_dims))
        nn.init.xavier_uniform_(self.atten_mats1)
        self.register_parameter("atten_mats1_%s"%name, self.atten_mats1)
        if self.nat >= 2:
            self.atten_mats1_1 = nn.Parameter(torch.FloatTensor(mode_dims, mode_dims))
            nn.init.xavier_uniform_(self.atten_mats1_1)
            self.register_parameter("atten_mats1_1_%s"%name, self.atten_mats1_1)
        if self.nat >= 3:
            self.atten_mats1_2 = nn.Parameter(torch.FloatTensor(mode_dims, mode_dims))
            nn.init.xavier_uniform_(self.atten_mats1_2)
            self.register_parameter("atten_mats1_2_%s"%name, self.atten_mats1_2)
        if bn != 'no':
            self.bn1 = nn.BatchNorm1d(mode_dims)
            self.bn1_1 = nn.BatchNorm1d(mode_dims)
            self.bn1_2 = nn.BatchNorm1d(mode_dims)
        if att_type == 'whole':
            self.atten_mats2 = nn.Parameter(torch.FloatTensor(mode_dims, 1))
        elif att_type == 'ele':
            self.atten_mats2 = nn.Parameter(torch.FloatTensor(mode_dims, mode_dims))
        nn.init.xavier_uniform_(self.atten_mats2)
        self.register_parameter("atten_mats2_%s"%name, self.atten_mats2)

    def forward(self, center_embed, offset_embed=None):
        if self.center_use_offset:
            temp1 = torch.cat([center_embed, offset_embed], dim=1)
        else:
            temp1 = center_embed
        if self.nat >= 1:
            if self.bn == 'no':
                temp2 = self.act_func(temp1.mm(self.atten_mats1))
            elif self.bn == 'before':
                temp2 = self.act_func(self.bn1(temp1.mm(self.atten_mats1)))
            elif self.bn == 'after':
                temp2 = self.bn1(self.act_func(temp1.mm(self.atten_mats1)))

        if self.nat >= 2:
            if self.bn == 'no':
                temp2 = self.act_func(temp2.mm(self.atten_mats1_1))
            elif self.bn == 'before':
                temp2 = self.act_func(self.bn1_1(temp2.mm(self.atten_mats1_1)))
            elif self.bn == 'after':
                temp2 = self.bn1_1(self.act_func(temp2.mm(self.atten_mats1_1)))

        if self.nat >= 3:
            if self.bn == 'no':
                temp2 = self.act_func(temp2.mm(self.atten_mats1_2))
            elif self.bn == 'before':
                temp2 = self.act_func(self.bn1_2(temp2.mm(self.atten_mats1_2)))
            elif self.bn == 'after':
                temp2 = self.bn1_2(self.act_func(temp2.mm(self.atten_mats1_2)))
        temp3 = temp2.mm(self.atten_mats2)
        return temp3

class Query2box(nn.Module):
    def __init__(self, model_name, nentity, nrelation, ntimestamp, hidden_dim, gamma, 
                 writer=None, geo=None, 
                 cen=None, offset_deepsets=None,
                 center_deepsets=None, offset_use_center=None, center_use_offset=None,
                 att_reg = 0., off_reg = 0., att_tem = 1., euo = False, 
                 gamma2=0, bn='no', nat=1, activation='relu', act_time='sigmoid', use_fixed_time_fun = False,  time_reg=None, use_separate_relation_embedding=False,use_relation_time=False):
        super(Query2box, self).__init__()
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.ntimestamp = ntimestamp
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        self.writer=writer
        self.geo = geo
        self.cen = cen
        self.offset_deepsets = offset_deepsets
        self.center_deepsets = center_deepsets
        self.offset_use_center = offset_use_center
        self.center_use_offset = center_use_offset
        self.att_reg = att_reg
        self.off_reg = off_reg
        self.att_tem = att_tem
        self.euo = euo
        self.his_step = 0
        self.bn = bn
        self.nat = nat
        self.time_reg = time_reg
        self.use_separate_relation_embedding = use_separate_relation_embedding
        self.use_relation_time = use_relation_time
        # self.double_point_in_time = double_point_in_time

        if activation == 'none':
            self.func = Identity
        elif activation == 'relu':
            self.func = F.relu
        elif activation == 'softplus':
            self.func = F.softplus
        # elif activation == 'leakyrelu':
        #     self.func = torch.nn.LeakyReLU

        if act_time == 'sigmoid':
            self.time_act_func = torch.sigmoid
        elif act_time == 'relu':
            self.time_act_func = F.relu
        elif act_time == 'none':
            self.time_act_func = None
        elif act_time == 'sin':
            self.time_act_func = torch.sin
        else:
            raise NotImplementedError
        
        self.gamma = nn.Parameter(
            torch.Tensor([gamma]), 
            requires_grad=False
        )

        if gamma2 == 0:
            gamma2 = gamma

        self.gamma2 = nn.Parameter(
            torch.Tensor([gamma2]), 
            requires_grad=False
        )

        self.entity_dim = hidden_dim
        # self.time_embedding = self.get_time_representation()  
        if self.model_name == 'BoxRotatE': 
            self.relation_dim = hidden_dim // 2
            self.time_dim = hidden_dim
            # self.re_time_mats = nn.Parameter(torch.FloatTensor(hidden_dim, hidden_dim//2))
            # nn.init.xavier_uniform_(self.re_time_mats)
            # self.register_parameter("re_transformation", self.re_time_mats)
            # self.im_time_mats = nn.Parameter(torch.FloatTensor(hidden_dim,hidden_dim//2))
            # nn.init.xavier_uniform_(self.im_time_mats)
            # self.register_parameter("im_transformation", self.im_time_mats)
            # self.time_dim = hidden_dim // 2
        elif self.model_name == 'BoxTransETimeAsRotation':
            self.time_dim = hidden_dim // 2
            self.relation_dim = hidden_dim
        else:
            self.relation_dim = hidden_dim
            self.time_dim = hidden_dim

        self.ent_embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / self.entity_dim]), 
            requires_grad=False
        )

        self.rel_embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / self.relation_dim]), 
            requires_grad=False
        )

        self.time_embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / self.time_dim]), 
            requires_grad=False
        )
        
        self.use_fixed_time_fun = use_fixed_time_fun

        if self.use_fixed_time_fun:
            # define time embeddings; which follows predefined functions
            self.time_frequency = nn.Parameter(torch.zeros(1, self.time_dim))
            # nn.init.uniform_(
            #     tensor=self.time_frequency, 
            #     a=-self.embedding_range.item(), 
            #     b=self.embedding_range.item()
            # )
            nn.init.xavier_uniform_(self.time_frequency)
            self.time_shift = nn.Parameter(torch.zeros(1, self.time_dim))
            nn.init.xavier_uniform_(self.time_shift)
            # nn.init.uniform_(
            #     tensor=self.time_shift, 
            #     a=-self.embedding_range.item(), 
            #     b=self.embedding_range.item()
            # )
            # add one linear layer to improve flexibility 
            self.time_mats = nn.Parameter(torch.FloatTensor(self.time_dim, self.time_dim))
            nn.init.xavier_uniform_(self.time_mats)
            self.register_parameter("time_transformation", self.time_mats)
        else: 
            ## pure learning with regularization 
            self.time_embedding = nn.Parameter(torch.zeros(ntimestamp, self.time_dim))
            nn.init.uniform_(
                tensor=self.time_embedding, 
                a=-self.time_embedding_range.item(),
                b=self.time_embedding_range.item()
            )

        self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))
        nn.init.uniform_(
            tensor=self.entity_embedding, 
            a=-self.ent_embedding_range.item(),
            b=self.ent_embedding_range.item()
        )

        
        self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding, 
            a=-self.rel_embedding_range.item(), 
            b=self.rel_embedding_range.item()
        )

        if self.use_relation_time:
            self.relation_time_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
            nn.init.uniform_(
                tensor=self.relation_time_embedding, 
                a=-self.rel_embedding_range.item(), 
                b=self.rel_embedding_range.item()
            )

        # if self.use_separate_relation_embedding:
            # self.relation_embedding_time = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
            # nn.init.uniform_(
            #     tensor=self.relation_embedding_time, 
            #     a=-self.rel_embedding_range.item(), 
            #     b=self.rel_embedding_range.item()
            # )

        if self.geo == 'vec':
            if self.center_deepsets == 'vanilla':
                self.deepsets = CenterSet(self.relation_dim, self.relation_dim, False, agg_func = torch.mean, bn=bn, nat=nat)
            elif self.center_deepsets == 'attention':
                self.deepsets = AttentionSet(self.relation_dim, self.relation_dim, False, 
                                                    att_reg = self.att_reg, att_tem = self.att_tem, bn=bn, nat=nat)
            elif self.center_deepsets == 'eleattention':
                self.deepsets = AttentionSet(self.relation_dim, self.relation_dim, False, 
                                                    att_reg = self.att_reg, att_type='ele', att_tem=self.att_tem, bn=bn, nat=nat)
            elif self.center_deepsets == 'mean':
                self.deepsets = MeanSet()
            else:
                assert False

        if self.geo == 'circle':
            ## the radius of each circle/global in high-dimension
            self.offsets = nn.Parameter(torch.zeros(nrelation+ntimestamp, 1))
            nn.init.uniform_(
                tensor=self.offsets, 
            )

            if self.offset_deepsets == 'vanilla':
                self.offset_sets = OffsetSet(self.relation_dim, self.relation_dim, self.offset_use_center, agg_func = torch.mean)
            elif self.offset_deepsets == 'inductive':
                self.offset_sets = InductiveOffsetSet(self.relation_dim, self.relation_dim, self.offset_use_center, self.off_reg, agg_func=torch.mean) # use element-wise mean to get the center
            # elif self.offset_deepsets == 'eleattention':
            #     self.offset_sets = AttentionSet(self.relation_dim, self.relation_dim, False, 
            #                                         att_reg = self.att_reg, att_type='ele', att_tem=self.att_tem, bn=bn, nat=nat)
            elif self.offset_deepsets == 'min':
                self.offset_sets = MinSet()
            else:
                assert False

        if self.geo == 'box':
            # generate offset_embedding for each relation, including time as predicates 
            self.relation_offset_embedding = nn.Parameter(torch.zeros(nrelation, self.entity_dim))
            nn.init.uniform_(
                tensor=self.relation_offset_embedding, 
                a=0., 
                b=self.ent_embedding_range.item()
            )

            self.time_offset_embedding = nn.Parameter(torch.zeros(ntimestamp, self.entity_dim))
            nn.init.uniform_(
                tensor=self.time_offset_embedding, 
                a=0., 
                b=self.ent_embedding_range.item()
            )

            if self.euo:
                self.entity_offset_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))
                nn.init.uniform_(
                    tensor=self.entity_offset_embedding, 
                    a=0., 
                    b=self.ent_embedding_range.item()
                )

            if self.model_name in ['BoxTransE', 'BoxDistMult']:
                if self.center_deepsets == 'vanilla':
                    self.center_sets = CenterSet(self.entity_dim, self.entity_dim, self.center_use_offset, agg_func = torch.mean, bn=bn, nat=nat)
                elif self.center_deepsets == 'attention':
                    self.center_sets = AttentionSet(self.entity_dim, self.entity_dim, self.center_use_offset, 
                                                        att_reg = self.att_reg, att_tem = self.att_tem, bn=bn, nat=nat)
                elif self.center_deepsets == 'eleattention': # use element-wise attention to find the new center--> a vector
                    self.center_sets = AttentionSet(self.entity_dim, self.entity_dim, self.center_use_offset, 
                                                        att_reg = self.att_reg, att_type='ele', att_tem=self.att_tem, bn=bn, nat=nat)
                elif self.center_deepsets == 'mean':
                    self.center_sets = MeanSet()
                else:
                    assert False

                if self.offset_deepsets == 'vanilla':
                    self.offset_sets = OffsetSet(self.entity_dim, self.entity_dim, self.offset_use_center, agg_func = torch.mean)
                elif self.offset_deepsets == 'inductive':
                    self.offset_sets = InductiveOffsetSet(self.entity_dim, self.entity_dim, self.offset_use_center, self.off_reg, agg_func = torch.mean) # use element-wise mean to get the center
                elif self.offset_deepsets == 'min':
                    self.offset_sets = MinSet()
                else:
                    assert False
            elif self.model_name in ['BoxRotatE', 'BoxTransETimeAsRotation']:
                ## only have eleattention 
                if self.center_deepsets == 'eleattention': # use element-wise attention to find the new center--> a vector
                    self.re_center_sets = AttentionSet(self.entity_dim//2, self.entity_dim//2, self.center_use_offset, 
                                                        att_reg = self.att_reg, att_type='ele', att_tem=self.att_tem, bn=bn, nat=nat)
                    self.im_center_sets = AttentionSet(self.entity_dim//2, self.entity_dim//2, self.center_use_offset, 
                                                        att_reg = self.att_reg, att_type='ele', att_tem=self.att_tem, bn=bn, nat=nat)
                if self.offset_deepsets == 'inductive':
                    self.re_offset_sets = InductiveOffsetSet(self.entity_dim//2, self.entity_dim//2, self.offset_use_center, self.off_reg, agg_func=torch.mean) # use element-wise mean to get the center
                    self.im_offset_sets = InductiveOffsetSet(self.entity_dim//2, self.entity_dim//2, self.offset_use_center, self.off_reg, agg_func=torch.mean) # use element-wise mean to get the center
            else:
                raise NotImplementedError

        if model_name not in ['TransE', 'BoxTransE', 'CircleTransE', 'BoxRotatE', 'BoxDistMult', 'BoxHyTE', 'BoxTransETimeAsRotation']:
            raise ValueError('model %s not supported' % model_name)
    
    def position_embedding(length, hidden_size, min_timescale=900, max_timescale=1.0e4):
        position = torch.arange(length*1.0)
        num_timescales = (hidden_size+1) // 2
        log_timescale_increment = (math.log(float(max_timescale) / float(min_timescale)) / 
            (num_timescales - 1))
        inv_timescales = min_timescale * torch.exp(
            torch.arange(num_timescales*1.0) * -log_timescale_increment)
        scaled_time = position.view(-1, 1) * inv_timescales.view(1,-1)
        signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)],dim=1)
        return signal[:,:hidden_size]

    def get_time_representation(self, time_indices):
        # here, we use time2vec implementation; the first dim with no activation function and others with activation functions
        # the activation functions can be periodical, e.g., sin/cos or non-periodical, e.g., sigmoid, relu;
        # result in an embedding table  with shape [ntimestamp, relation_dim]
        if self.use_fixed_time_fun:
            time_indices = time_indices.view(-1,1)
            signal = time_indices*self.time_frequency + self.time_shift
            if self.time_act_func == None:
                return signal 
            else:
                mask = torch.ones(self.time_dim).cuda()
                mask[::2] = 0 
                assert mask[0]==0, mask[1]==1
                return mask*self.time_act_func(signal) + (1-mask)*self.time_act_func(3.14/2 - signal)
        else:
            return torch.index_select(self.time_embedding, dim=0, index=time_indices)

    def forward(self, sample, rel_len, qtype, mode='single', use_relation_time=False):
        relation_time = None
        if qtype == '2-inter' or qtype == '3-inter' or qtype == '2-3-inter':
            if mode == 'single': ## used in positive samples
                batch_size, negative_sample_size = sample.size(0), 1

                head = torch.index_select(self.entity_embedding, dim=0, index=sample[:, 0]).unsqueeze(1)
                # head_2 = torch.index_select(self.entity_embedding, dim=0, index=sample[:, 2]).unsqueeze(1)
                # head = torch.cat([head_1, head_1], dim=0)
                # head = head
                if self.euo and self.geo == 'box':
                    head_offset = torch.index_select(self.entity_offset_embedding, dim=0, index=sample[:, 0]).unsqueeze(1)
                    # head_offset_2 = torch.index_select(self.entity_offset_embedding, dim=0, index=sample[:, 2]).unsqueeze(1)
                    # head_offset = torch.cat([head_offset_1, head_offset_2], dim=0)
                # if rel_len == 3:
                #     head = torch.cat([head, head_1], dim=0)
                #     head_3 = torch.index_select(self.entity_embedding, dim=0, index=sample[:, 4]).unsqueeze(1)
                #     head = torch.cat([head, head_3], dim=0)
                #     if self.euo and self.geo == 'box':
                #         head_offset_3 = torch.index_select(self.entity_offset_embedding, dim=0, index=sample[:, 4]).unsqueeze(1)
                #         head_offset = torch.cat([head_offset, head_offset_3], dim=0)

                ## Under testing seting, tails are all the possibel entities;
                tail = torch.index_select(self.entity_embedding, dim=0, index=sample[:,-1]).unsqueeze(1)
                # if rel_len == 2:
                #     tail = torch.cat([tail, tail], dim=0)
                # elif rel_len == 3:
                #     tail = torch.cat([tail, tail, tail], dim=0) ##concatenate three elements along 0 axis

                relation_1 = torch.index_select(self.relation_embedding, dim=0, index=sample[:,1]).unsqueeze(1)
                
                # relation_2 = torch.index_select(self.relation_embedding, dim=0, index=sample[:,3]).unsqueeze(1).unsqueeze(1)
                
                relation_2 = self.get_time_representation(sample[:,2]).unsqueeze(1)
                if use_relation_time:
                    relation_time = torch.index_select(self.relation_time_embedding, dim=0, index=sample[:,1]).unsqueeze(1)
                    # relation_2 = relation_2 * relation_time
                # relation = torch.cat([relation_1, relation_2], dim=0)
                relation = [relation_1, relation_2]

                if rel_len == 3:
                    # relation_3 = torch.index_select(self.relation_embedding, dim=0, index=sample[:,5]).unsqueeze(1).unsqueeze(1)
                    relation_3 = self.get_time_representation(sample[:,3]).unsqueeze(1)
                    # if use_relation_time:
                    #     relation_3 = relation_3 * relation_time
                    # relation = torch.cat([relation, relation_3], dim=0)
                    relation = relation + [relation_3]
                    
                if self.geo == 'box':
                    offset_1 = torch.index_select(self.relation_offset_embedding, dim=0, index=sample[:,1]).unsqueeze(1)
                    offset_2 = torch.index_select(self.time_offset_embedding, dim=0, index=sample[:,2]).unsqueeze(1)
                    # offset = torch.cat([offset_1, offset_2], dim=0)
                    offset = [offset_1, offset_2]
                    if rel_len == 3:
                        offset_3 = torch.index_select(self.time_offset_embedding, dim=0, index=sample[:,3]).unsqueeze(1)
                        # offset = torch.cat([offset, offset_3], dim=0)
                        offset = offset + [offset_3]
                # if self.geo == 'circle':
                #     offset_1 = torch.index_select(self.offsets, dim=0, index=sample[:,1]).unsqueeze(1).unsqueeze(1)
                #     offset_2 = torch.index_select(self.offsets, dim=0, index=sample[:,3]).unsqueeze(1).unsqueeze(1)
                #     offset = torch.cat([offset_1, offset_2], dim=0)
                #     if rel_len == 3:
                #         offset_3 = torch.index_select(self.offsets, dim=0, index=sample[:,5]).unsqueeze(1).unsqueeze(1)
                #         offset = torch.cat([offset, offset_3], dim=0)
            elif mode == 'tail-batch':
                head_part, tail_part = sample
                batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
                
                head = torch.index_select(self.entity_embedding, dim=0, index=head_part[:, 0]).unsqueeze(1)
                #head_2 = torch.index_select(self.entity_embedding, dim=0, index=head_part[:, 2]).unsqueeze(1)
                # head = torch.cat([head_1, head_1], dim=0)
                if self.euo and self.geo == 'box':
                    head_offset = torch.index_select(self.entity_offset_embedding, dim=0, index=head_part[:, 0]).unsqueeze(1)
                    # head_offset_2 = torch.index_select(self.entity_offset_embedding, dim=0, index=head_part[:, 2]).unsqueeze(1)
                    # head_offset = torch.cat([head_offset_1, head_offset_1], dim=0)
                # if rel_len == 3:
                #     # head_3 = torch.index_select(self.endity_embedding, dim=0, index=head_part[:, 4]).unsqueeze(1)
                #     head = torch.cat([head, head_1], dim=0)
                #     if self.euo and self.geo == 'box':
                #         # head_offset_3 = torch.index_select(self.entity_offset_embedding, dim=0, index=head_part[:, 4]).unsqueeze(1)
                #         head_offset = torch.cat([head_offset, head_offset_1], dim=0)

                tail = torch.index_select(self.entity_embedding, dim=0, index=tail_part.view(-1)).view(batch_size, negative_sample_size, -1)
                # if rel_len == 2:
                #     tail = torch.cat([tail, tail], dim=0)
                # elif rel_len == 3:
                #     tail = torch.cat([tail, tail, tail], dim=0)

                relation_1 = torch.index_select(self.relation_embedding, dim=0, index=head_part[:, 1]).unsqueeze(1)

                relation_2 = self.get_time_representation(head_part[:, 2]).unsqueeze(1)
                if use_relation_time:
                    relation_time = torch.index_select(self.relation_time_embedding, dim=0, index=head_part[:,1]).unsqueeze(1)
                    # relation_2 = relation_2 * relation_time
                # relation_2 = torch.index_select(self.relation_embedding, dim=0, index=head_part[:, 3]).unsqueeze(1).unsqueeze(1)
                # relation = torch.cat([relation_1, relation_2], dim=0)
                relation = [relation_1, relation_2]
                if rel_len == 3:
                    # relation_3 = torch.index_select(self.relation_embedding, dim=0, index=head_part[:, 5]).unsqueeze(1).unsqueeze(1)
                    relation_3 = self.get_time_representation(head_part[:, 3]).unsqueeze(1)
                    # if use_relation_time:
                    #     relation_3 = relation_3 * relation_time
                    # relation =torch.cat([relation_1, relation_2, relation_3], dim=0)
                    relation = relation + [relation_3]

                if self.geo == 'box':
                    offset_1 = torch.index_select(self.relation_offset_embedding, dim=0, index=head_part[:, 1]).unsqueeze(1)
                    offset_2 = torch.index_select(self.time_offset_embedding, dim=0, index=head_part[:, 2]).unsqueeze(1)
                    # offset = torch.cat([offset_1, offset_2], dim=0)
                    offset = [offset_1, offset_2]
                    if rel_len == 3:
                        offset_3 = torch.index_select(self.time_offset_embedding, dim=0, index=head_part[:, 3]).unsqueeze(1)
                        offset = [offset_1, offset_2, offset_3]
                        # offset = torch.cat([offset, offset_3], dim=0)
                # elif self.geo == 'circle':
                #     offset_1 = torch.index_select(self.offsets, dim=0, index=head_part[:,1]).unsqueeze(1).unsqueeze(1)
                #     offset_2 = torch.index_select(self.offsets, dim=0, index=head_part[:,3]).unsqueeze(1).unsqueeze(1)
                #     offset = torch.cat([offset_1, offset_2], dim=0)
                #     if rel_len == 3:
                #         offset_3 = torch.index_select(self.offsets, dim=0, index=head_part[:,5]).unsqueeze(1).unsqueeze(1)
                #         offset = torch.cat([offset, offset_3], dim=0)
            elif mode == 'time-batch':
                head_part, tail_part = sample  # head_part <s, r, t, o> tails: negative timestamps
                batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

                head = torch.index_select(self.entity_embedding, dim=0, index=head_part[:, 0]).unsqueeze(1)
                tail = torch.index_select(self.entity_embedding, dim=0, index=head_part[:, -1]).unsqueeze(1) #[batch_size, 1, ndim]

                if self.euo and self.geo == 'box':
                    head_offset = torch.index_select(self.entity_offset_embedding, dim=0, index=head_part[:, 0]).unsqueeze(1)

                negative_ts = self.get_time_representation(tail_part.view(-1)).view(batch_size, negative_sample_size, -1)

                ## get embeddings for relations
                relation_1 = torch.index_select(self.relation_embedding, dim=0, index=head_part[:, 1]).unsqueeze(1)

                # relation_2 = self.get_time_representation(head_part[:, 2]).unsqueeze(1) ## groundtruth timestamp

                relation = [relation_1, negative_ts]

                if self.geo == 'box':
                    offset_1 = torch.index_select(self.relation_offset_embedding, dim=0, index=head_part[:, 1]).unsqueeze(1)
                    # offset_2 = torch.index_select(self.time_offset_embedding, dim=0, index=head_part[:, 2]).unsqueeze(1)
                    # offset = torch.cat([offset_1, offset_2], dim=0)
                    
                    offset_ts = torch.index_select(self.time_offset_embedding, dim=0, index=tail_part.view(-1)).view(batch_size, negative_sample_size, -1)
                    offset = [offset_1, offset_ts]
                # print("head", head)
                # print("relation", relation)
                # print("offset", offset)

        elif qtype == '1-chain' or qtype == '2-chain' or qtype == '3-chain':
            # if self.use_separate_relation_embedding:
            #relation_time = None
            if mode == 'single':
                batch_size, negative_sample_size = sample.size(0), 1
                head = torch.index_select(self.entity_embedding, dim=0, index=sample[:,0]).unsqueeze(1) # [batch_size, 1, n_dims]
                
                relation = torch.index_select(self.relation_embedding, dim=0, index=sample[:,1]).unsqueeze(1) # [batch_size, 1, 1, n_dims]
                if self.geo == 'box':
                    offset = torch.index_select(self.relation_offset_embedding, dim=0, index=sample[:,1]).unsqueeze(1)
                    # time_offset = torch.index_select(self.time_offset_embedding, dim=0, index=torch.zeros_like(sample[:,-1])).unsqueeze(1)
                    if self.euo:
                        head_offset = torch.index_select(self.entity_offset_embedding, dim=0, index=sample[:,0]).unsqueeze(1)
                elif self.geo == 'circle':
                    offset = torch.index_select(self.offsets, dim=0, index=sample[:, 1]).unsqueeze(1).unsqueeze(1) # the radius of the real relation
                
                assert relation.size(1) == rel_len
                if self.geo == 'box':
                    assert offset.size(1) == rel_len  
                tail = torch.index_select(self.entity_embedding, dim=0, index=sample[:,-1]).unsqueeze(1)
                # time = torch.index_select(self.time_embedding, dim=0, index=torch.zeros_like(sample[:,-1])).unsqueeze(1)
                relation = [relation]
                offset = [offset]

            elif mode == 'tail-batch':
                # print('qtype', qtype)
                head_part, tail_part = sample
                batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
                
                head = torch.index_select(self.entity_embedding, dim=0, index=head_part[:, 0]).unsqueeze(1)
                
                relation = torch.index_select(self.relation_embedding, dim=0, index=head_part[:, 1]).unsqueeze(1) #[batch_size, 1, 1, n_dims]
                # time = torch.index_select(self.time_embedding, dim=0, index=torch.zeros_like(head_part[:,1])).unsqueeze(1)
                if self.geo == 'box':
                    offset = torch.index_select(self.relation_offset_embedding, dim=0, index=head_part[:, 1]).unsqueeze(1) #[batch_size, 1, 1, n_dims]
                    # time_offset = torch.index_select(self.time_offset_embedding, dim=0, index=torch.zeros_like(head_part[:,1])).unsqueeze(1)
                    if self.euo:
                        head_offset = torch.index_select(self.entity_offset_embedding, dim=0, index=head_part[:, 0]).unsqueeze(1)
                elif self.geo == 'circle':
                    offset = torch.index_select(self.offsets, dim=0, index=head_part[:, 1]).unsqueeze(1) # the radius of the real relation
                    assert offset.size(3) == 1

                assert relation.size(1) == rel_len
                if self.geo == 'box':
                    assert offset.size(1) == rel_len    
                tail = torch.index_select(self.entity_embedding, dim=0, index=tail_part.view(-1)).view(batch_size, negative_sample_size, -1)           
                relation = [relation]
                offset = [offset]
            # relation = [relation]
            # offset = [offset]
            # elif mode == 'head-batch':
            #     positive_part, negative_part = sample
            #     batch_size, negative_sample_size = negative_part.size(0), negative_part.size(1)

            #     head = torch.index_select(self.entity_embedding, dim=0, index=negative_part.view(-1)).view(batch_size, negative_sample_size, -1)  

            #     relation = torch.index_select(self.relation_embedding, dim=0, index=positive_part[:, 1]).unsqueeze(1).unsqueeze(1) #[batch_size, 1, 1, n_dims]

            #     if self.geo == 'box':
            #         offset = torch.index_select(self.offset_embedding, dim=0, index=positive_part[:, 1]).unsqueeze(1).unsqueeze(1) #[batch_size, 1, 1, n_dims]
            #         if self.euo:
            #             head_offset = torch.index_select(self.offset_embedding, dim=0, index=negative_part.view(-1)).view(batch_size, negative_sample_size, -1)  
            #     elif self.geo == 'circle':
            #         offset = torch.index_select(self.offsets, dim=0, index=positive_part[:, 1]).unsqueeze(1).unsqueeze(1) # the radius of the real relation
            #         assert offset.size(3) == 1

            #     assert relation.size(1) == rel_len
            #     if self.geo == 'box':
            #         assert offset.size(1) == rel_len    
            #     tail = torch.index_select(self.entity_embedding, dim=0, index=positive_part[:, 2]).unsqueeze(1)

        # elif qtype == '1-chain-t': # the difference lies in that for relation, we get from different places. For ordinary 1-c, we get from relation embedding; otherwise, we get from time embedding
        #     if mode == 'single':
        #         batch_size, negative_sample_size = sample.size(0), 1
        #         head = torch.index_select(self.entity_embedding, dim=0, index=sample[:,0]).unsqueeze(1) # [batch_size, 1, n_dims]
        #         relation = self.get_time_representation(sample[:,1]-self.nrelation).unsqueeze(1).unsqueeze(1) # [batch_size, 1, 1, n_dims]
                
        #         if self.geo == 'box':
        #             offset = torch.index_select(self.offset_embedding, dim=0, index=sample[:,1]).unsqueeze(1).unsqueeze(1)
        #             if self.euo:
        #                 head_offset = torch.index_select(self.entity_offset_embedding, dim=0, index=sample[:,0]).unsqueeze(1)
        #         elif self.geo == 'circle':
        #             offset = torch.index_select(self.offsets, dim=0, index=sample[:, 1]).unsqueeze(1).unsqueeze(1) # the radius of the real relation
                
        #         assert relation.size(1) == rel_len
        #         if self.geo == 'box':
        #             assert offset.size(1) == rel_len  
        #         tail = torch.index_select(self.entity_embedding, dim=0, index=sample[:,-1]).unsqueeze(1)

        #     elif mode == 'tail-batch':
        #         # print('qtype', qtype)
        #         head_part, tail_part = sample
        #         batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
                
        #         head = torch.index_select(self.entity_embedding, dim=0, index=head_part[:, 0]).unsqueeze(1)
                
        #         relation = self.get_time_representation(head_part[:,1]).unsqueeze(1).unsqueeze(1) # [batch_size, 1, 1, n_dims] #[batch_size, 1, 1, n_dims]

        #         if self.geo == 'box':
        #             offset = torch.index_select(self.offset_embedding, dim=0, index=head_part[:, 1]).unsqueeze(1).unsqueeze(1) #[batch_size, 1, 1, n_dims]
        #             if self.euo:
        #                 head_offset = torch.index_select(self.entity_offset_embedding, dim=0, index=head_part[:, 0]).unsqueeze(1)
        #         elif self.geo == 'circle':
        #             offset = torch.index_select(self.offsets, dim=0, index=head_part[:, 1]).unsqueeze(1).unsqueeze(1) # the radius of the real relation
        #             assert offset.size(3) == 1

        #         assert relation.size(1) == rel_len
        #         if self.geo == 'box':
        #             assert offset.size(1) == rel_len    
        #         tail = torch.index_select(self.entity_embedding, dim=0, index=tail_part.view(-1)).view(batch_size, negative_sample_size, -1)      
        else:
            raise ValueError('mode %s not supported' % mode)
            
        model_func = {
            'BoxTransE': self.BoxTransE,
            'TransE': self.TransE,
            'CircleTransE':self.CircleTransE,
            'BoxRotatE':self.BoxRotatE,
            'BoxDistMult':self.BoxDistMult,
            'BoxHyTE':self.BoxHyTE,
            'BoxTransETimeAsRotation':self.BoxTransETimeAsRotation,
        }
        if self.geo == 'vec':
            offset = None
            head_offset = None
        if self.geo == 'box':
            if not self.euo:
                head_offset = None
        if self.geo == 'circle':
            if not self.euo:
                head_offset = None
        time_score_reg = None
        if self.model_name in model_func: ## calculate TransE and BoxTransE
            if qtype == '2-inter' or '3-inter' == qtype or qtype == '2-3-inter':
                score, score_cen, offset_norm, score_cen_plus, time_score_reg = model_func[self.model_name](head, relation, tail, relation_time, mode, offset, head_offset, 1, qtype)
            else:
                score, score_cen, offset_norm, score_cen_plus, _ = model_func[self.model_name](head, relation, tail, relation_time, mode, offset, head_offset, rel_len, qtype)
        else:
            raise ValueError('model %s not supported' % self.model_name)
        
        return score, score_cen, offset_norm, score_cen_plus, None, time_score_reg

    def CircleTransE(self, head, relation, tail, mode, offset, head_offset, rel_len, qtype):
        if qtype in ['2-inter', '3-inter']:
            ## first merge the boxes and then add to head;
            rel_len = int(qtype.split('-')[0])
            query_center = torch.chunk(head, rel_len, dim=0)[0]
            relation = relation.squeeze(1)
            relation = torch.chunk(relation, rel_len, dim=0)
            offset_ = torch.abs(offset).squeeze(1)
            offset_ = torch.chunk(offset_, rel_len, dim=0)
            
            query_center = query_center + relation[0]
            tails = torch.chunk(tail, rel_len, dim=0)
        
            if rel_len == 2:
                time_norm = torch.norm(relation[1], dim=-1).view(-1, 1)
                new_query_center = (query_center + relation[1]/time_norm.unsqueeze(1)*offset_[0]*0.5)
                new_offset =  torch.min(offset_[0]*0.5, offset_[1]) # use mean/min/max to aggregate information  
                score_center = torch.norm(new_query_center - tail[0], dim=-1) # outside distance
                score_center_plus = torch.abs(new_offset).squeeze(1) # inside distance

                ## modify the distance to two parts: inside distance and outside distance
                mask_outside_distance = (score_center == torch.max(score_center, score_center_plus))
                score_center_plus = torch.min(score_center, score_center_plus) # inside distance
                score_offset = (score_center - score_center_plus) * mask_outside_distance # outside distance

                score_center = self.gamma2.item() - score_center
                score_center_plus = self.gamma.item() - torch.norm(score_offset, p=1, dim=-1) - self.cen * torch.norm(score_center_plus, p=1, dim=-1)  
                return None, score_center, None, score_center_plus.view(-1, 1), None

            elif rel_len == 3:
                time_norm_1 = torch.norm(relation[1], dim=-1).view(-1, 1)
                time_norm_2 = torch.norm(relation[2], dim=-1).view(-1, 1)
                
                relation2 = query_center + relation[1]/time_norm_1.unsqueeze(1)*offset_[0]*0.5
                relation3 = query_center + relation[2]/time_norm_2.unsqueeze(1)*offset_[0]*0.5

                ## the objective is to find an entity which is close to any of the three circles
                score_center = torch.norm(query_center - tail[0], dim=-1) # outside distance
                score_center_plus_1 = torch.abs(offset_[0]).squeeze(1) # .inside distance

                ## modify the distance to two parts: inside distance and outside distance
                mask_outside_distance_1 = (score_center == torch.max(score_center, score_center_plus_1))
                score_center_plus_1 = torch.min(score_center, score_center_plus_1) # inside distance
                score_offset_1 = (score_center - score_center_plus_1) * mask_outside_distance_1 # outside distance

                score_center_plus_1 = - torch.norm(score_offset_1, dim=-1) - self.cen * torch.norm(score_center_plus_1, dim=-1) 

                ## the objective is to find an entity which is close to any of the three circles
                score_center_2 = torch.norm(relation2 - tail[0],  dim=-1) # outside distance
                score_center_plus_2 = torch.abs(offset_[1]).squeeze(1) # inside distance

                ## modify the distance to two parts: inside distance and outside distance
                mask_outside_distance_2 = (score_center_2 == torch.max(score_center_2, score_center_plus_2))
                score_center_plus_2 = torch.min(score_center_2, score_center_plus_2) # inside distance
                score_offset_2 = (score_center_2 - score_center_plus_2) * mask_outside_distance_2 # outside distance

                score_center_plus_2 = - torch.norm(score_offset_2, dim=-1) - self.cen * torch.norm(score_center_plus_2, dim=-1) 
                ## the objective is to find an entity which is close to any of the three circles
                score_center_3 = torch.norm(relation3 - tail[0], dim=-1) # outside distance
                score_center_plus_3 = torch.abs(offset_[2]).squeeze(1) # inside distance

                ## modify the distance to two parts: inside distance and outside distance
                mask_outside_distance_3 = (score_center_3 == torch.max(score_center_3, score_center_plus_3))
                score_center_plus_3 = torch.min(score_center_3, score_center_plus_3) # inside distance
                score_offset_3 = (score_center_3 - score_center_plus_3) * mask_outside_distance_3 # outside distance

                score_center_plus_3 = self.gamma.item() - torch.norm(score_offset_3, p=1, dim=-1) - self.cen * torch.norm(score_center_plus_3, p=1, dim=-1) 
                score_center = self.gamma2.item() - (score_center + score_center_2 + score_center_3)/3
                return None, score_center, None, (score_center_plus_3 + score_center_plus_2 + score_center_plus_1).view(-1, 1), None
            
        else:# here, only one possibility -- 1-chain case
            query_center = head
            # print('the shape of query_center', query_center.shape)
            # print('the shape of relation', relation.shape)
           
            query_center = query_center + relation[:,0,:,:]
            # print('the shape of query_center - tail', (query_center - tail).shape)

            score_center = torch.norm(query_center - tail, dim=-1) # outside distance
            # print('the shape of score_offset', score_offset.shape)

            score_center_plus = torch.abs(offset).squeeze(1).squeeze(1) # inside distance
            # print('the shape of score_center_plus', score_center_plus.shape)

            ## modify the distance to two parts: inside distance and outside distance
            mask_outside_distance = (score_center == torch.max(score_center, score_center_plus))
            # print('the shape of mask_outside_distance', mask_outside_distance.shape)
            score_center_plus = torch.min(score_center, score_center_plus) # inside distance
            # print('score_center_plus', score_center_plus[:10])
            score_offset = (score_center - score_center_plus) * mask_outside_distance # outside distance
            # print('score_offset', score_offset[:10])

            score_center_plus = self.gamma.item() - score_offset - self.cen * score_center_plus
            # print('score_center_plus', score_center_plus)
            score_center = self.gamma2.item() - score_center
        return None, score_center, None, score_center_plus, None
    
    def BoxRotatE(self, head, relation, tail, relation_time, mode, offset, head_offset, rel_len, qtype):
        pi = 3.14159265358979323846

        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        phase_relation_r = relation[0]/(self.rel_embedding_range.item()/pi) # real predicate

        re_relation_r = torch.cos(phase_relation_r)
        im_relation_r = torch.sin(phase_relation_r)

        # re_relation_r, im_relation_r = torch.chunk(relation[0], 2, dim=2)

        re_offset_r, im_offset_r = torch.chunk(offset[0], 2, dim=2)

        re_query_center = re_head * re_relation_r - im_head * im_relation_r
        im_query_center = re_head * im_relation_r + im_head * re_relation_r

        re_query_min = re_query_center 
        im_query_min = im_query_center 
        re_query_max = re_query_center 
        im_query_max = im_query_center

        re_query_min = re_query_min - 0.5 * self.func(re_offset_r)
        im_query_min = im_query_min - 0.5 * self.func(im_offset_r)
        re_query_max = re_query_max + 0.5 * self.func(re_offset_r)
        im_query_max = im_query_max + 0.5 * self.func(im_offset_r)
        
        re_offsets = re_query_max - re_query_min # all are positive numbers
        im_offsets = im_query_max - im_query_min # all are positive numbers

        ## implememt three ways: 1c, 2i and 3i
        if qtype in ['2-inter', '3-inter', '2-3-inter']:
            rel_len = int(qtype.split('-')[0])
            re_relation_t, im_relation_t = torch.chunk(relation[1], 2, dim=-1)
            re_offset_t, im_offset_t = torch.chunk(offset[1], 2, dim=-1)

            # phase_relation_t = relation[1]/(self.time_embedding_range.item()/pi)

            # re_relation_t = torch.cos(phase_relation_t)
            # im_relation_t = torch.sin(phase_relation_t)

            # re_query_center_t = re_head * re_relation_t - im_head * im_relation_t
            # im_query_center_t = re_head * im_relation_t + im_head * re_relation_t

            re_query_center_t = re_query_center + re_relation_t 
            im_query_center_t = im_query_center + im_relation_t
            # im_query_center_t = re_head * im_relation_t + im_head * re_relation_t
                
            re_query_min_t = re_query_center_t 
            im_query_min_t = im_query_center_t 
            re_query_max_t = re_query_center_t 
            im_query_max_t = im_query_center_t

            re_query_min_t = re_query_min_t - 0.5 * self.func(re_offset_t)
            im_query_min_t = im_query_min_t - 0.5 * self.func(im_offset_t)
            re_query_max_t = re_query_max_t + 0.5 * self.func(re_offset_t)
            im_query_max_t = im_query_max_t + 0.5 * self.func(im_offset_t)
            
            re_offsets_t = re_query_max_t - re_query_min_t # all are positive numbers
            im_offsets_t = im_query_max_t - im_query_min_t # all are positive numbers

            ### combine relationa and time
            if relation_time == None:
                # re_query_center_relation_t = torch.cat([re_relation_r, re_relation_t], dim=-1)
                # im_query_center_relation_t = torch.cat([im_relation_r, im_relation_t], dim=-1)
                # re_query_center_relation_t = re_query_center_relation_t.squeeze(1).mm(self.re_time_mats).unsqueeze(1)
                # im_query_center_relation_t = im_query_center_relation_t.squeeze(1).mm(self.im_time_mats).unsqueeze(1)
                re_query_center_relation_t = re_relation_r * re_relation_t - im_relation_r * im_relation_t
                im_query_center_relation_t = re_relation_r * im_relation_t + im_relation_r * re_relation_t
            else:
                re_relation_time, im_relation_time = torch.chunk(relation_time, 2, dim=-1)
                re_query_center_relation_t = re_relation_time * re_relation_t - im_relation_time * im_relation_t
                im_query_center_relation_t = re_relation_time * im_relation_t + im_relation_time * re_relation_t
            re_query_min_relation_t = re_query_center_relation_t 
            im_query_min_relation_t = im_query_center_relation_t 
            re_query_max_relation_t = re_query_center_relation_t 
            im_query_max_relation_t = im_query_center_relation_t

            re_query_min_relation_t = re_query_min_relation_t - 0.5 * self.func(re_offset_t)
            im_query_min_relation_t = im_query_min_relation_t - 0.5 * self.func(im_offset_t)
            re_query_max_relation_t = re_query_max_relation_t + 0.5 * self.func(re_offset_t)
            im_query_max_relation_t = im_query_max_relation_t + 0.5 * self.func(im_offset_t)
            
            re_offsets_relation_t = re_query_max_relation_t - re_query_min_relation_t # all are positive numbers
            im_offsets_relation_t = im_query_max_relation_t - im_query_min_relation_t # all are positive numbers

            if rel_len == 2:     
                re_new_query_center = self.re_center_sets(re_query_center.squeeze(1), re_offsets.squeeze(1), 
                                                re_query_center_t.squeeze(1), re_offsets_t.squeeze(1), # use attention to get the new center
                                                [re_query_center_relation_t.squeeze(1)], [re_offsets_relation_t.squeeze(1)]).squeeze(1) # use attention to get the new center
                im_new_query_center = self.im_center_sets(im_query_center.squeeze(1), im_offsets.squeeze(1), 
                                                im_query_center_t.squeeze(1), im_offsets_t.squeeze(1), # use attention to get the new center
                                                [im_query_center_relation_t.squeeze(1)], [im_offsets_relation_t.squeeze(1)]).squeeze(1) # use attention to get the new center
                re_new_offset = self.re_offset_sets(re_query_center.squeeze(1), re_offsets.squeeze(1), 
                                                re_query_center_t.squeeze(1), re_offsets_t.squeeze(1)) # use mean/min/max to aggregate information  
                im_new_offset = self.im_offset_sets(im_query_center.squeeze(1), im_offsets.squeeze(1), 
                                                im_query_center_t.squeeze(1), im_offsets_t.squeeze(1)) # use mean/min/max to aggregate information 

            elif rel_len == 3:
                raise NotImplementedError
                # phase_relation_2 = relation[1]/(self.rel_embedding_range.item()/pi)

                # re_relation_2 = torch.cos(phase_relation_2)
                # im_relation_2 = torch.sin(phase_relation_2)

                # re_relation_2 = re_head * re_relation_2 - im_head * im_relation_2
                # im_relation_2 = re_head * im_relation_2 + im_head * re_relation_2

                # ## chuck the offset into imagenary and the real parts
                # re_offset_2, im_offset_2 = torch.chunk(self.func(offset_[1]), 2, dim=2)

                # phase_relation_3 = relation[2]/(self.re_embedding_range.item()/pi)

                # re_relation_3 = torch.cos(phase_relation_3)
                # im_relation_3 = torch.sin(phase_relation_3)

                # re_relation_3 = re_head * re_relation_3 - im_head * im_relation_3
                # im_relation_3 = re_head * im_relation_3 + im_head * re_relation_3

                # ## chuck the offset into imagenary and the real parts
                # re_offset_3, im_offset_3 = torch.chunk(self.func(offset_[2]), 2, dim=2)

                # re_new_query_center = self.re_center_sets(re_query_center.squeeze(1), re_offsets.squeeze(1), 
                #                                 re_relation_2.squeeze(1), re_offset_2.squeeze(1),
                #                                 re_relation_3.squeeze(1), re_offset_3.squeeze(1)) # use attention to get the new center
                # im_new_query_center = self.im_center_sets(im_query_center.squeeze(1), im_offsets.squeeze(1), 
                #                                 im_relation_2.squeeze(1), im_offset_2.squeeze(1),
                #                                 im_relation_3.squeeze(1), im_offset_3.squeeze(1),) # use attention to get the new center
                # re_new_offset = self.re_offset_sets(re_query_center.squeeze(1), re_offsets.squeeze(1), 
                #                                 re_relation_2.squeeze(1), re_offset_2.squeeze(1),
                #                                 re_relation_3.squeeze(1), re_offset_3.squeeze(1)) # use mean/min/max to aggregate information                        
                # im_new_offset = self.im_offset_sets(im_query_center.squeeze(1), im_offsets.squeeze(1), 
                #                                 im_relation_2.squeeze(1), im_offset_2.squeeze(1),
                #                                 im_relation_3.squeeze(1), im_offset_3.squeeze(1)) # use mean/min/max to aggregate information                        

            re_new_query_min = (re_new_query_center - 0.5*self.func(re_new_offset)).unsqueeze(1)
            re_new_query_max = (re_new_query_center + 0.5*self.func(re_new_offset)).unsqueeze(1)
            im_new_query_min = (im_new_query_center - 0.5*self.func(im_new_offset)).unsqueeze(1)
            im_new_query_max = (im_new_query_center + 0.5*self.func(im_new_offset)).unsqueeze(1)

            # re_score_offset = F.relu(re_new_query_min - re_tail) + F.relu(re_tail - re_new_query_max) # dist_outside
            # im_score_offset = F.relu(im_new_query_min - im_tail) + F.relu(im_tail - im_new_query_max) # dist_outside
            # score_offset = torch.stack([re_score_offset, im_score_offset], dim=0)
            new_query_min = torch.cat([re_new_query_min, im_new_query_min], dim=-1)
            new_query_max = torch.cat([re_new_query_max, im_new_query_max], dim=-1)

            score_offset = F.relu(new_query_min - tail) + F.relu(tail - new_query_max) # dist_outside
            score_offset =  torch.stack(torch.chunk(score_offset, 2, dim=-1), dim=0)
            ## 
            score_offset = score_offset.norm(dim = 0)

            re_score_center = re_new_query_center.unsqueeze(1) - re_tail
            im_score_center = im_new_query_center.unsqueeze(1) - im_tail
            score_center = torch.stack([re_score_center, im_score_center], dim=0)
            score_center = score_center.norm(dim = 0)

            # re_score_center_plus = torch.min(re_new_query_max, torch.max(re_new_query_min, re_tail)) - re_new_query_center.unsqueeze(1) # dist_inside
            # im_score_center_plus = torch.min(im_new_query_max, torch.max(im_new_query_min, im_tail)) - im_new_query_center.unsqueeze(1) # dist_inside
            # score_center_plus = torch.stack([re_score_center_plus, im_score_center_plus], dim=0) # dist_inside
            # score_center_plus =  score_center_plus.norm(dim = 0) # dist_inside
            new_query_center = torch.cat([re_new_query_center, im_new_query_center], dim=-1)
            score_center_plus = torch.min(new_query_max, torch.max(new_query_min, tail)) - new_query_center.unsqueeze(1) # dist_inside
            score_center_plus =  torch.stack(torch.chunk(score_center_plus, 2, dim=-1), dim=0)
            score_center_plus =  score_center_plus.norm(dim = 0) # dist_inside

        else:
            # new_query_center = torch.stack([re_query_center, im_query_center], dim = 0)
            # new_query_min = torch.stack([re_query_min, im_query_min], dim = 0)
            # new_query_max = torch.stack([re_query_max, im_query_max], dim = 0)
            
            # re_score_offset = F.relu(re_query_min - re_tail) + F.relu(re_tail - re_query_max) # dist_outside
            # im_score_offset = F.relu(im_query_min - im_tail) + F.relu(im_tail - im_query_max) # dist_outside
            # score_offset = torch.stack([re_score_offset, im_score_offset], dim=0)

            new_query_min = torch.cat([re_query_min, im_query_min], dim=-1)
            new_query_max = torch.cat([re_query_max, im_query_max], dim=-1)

            score_offset = F.relu(new_query_min - tail) + F.relu(tail - new_query_max) # dist_outside
            score_offset =  torch.stack(torch.chunk(score_offset, 2, dim=-1), dim=0)
            score_offset = score_offset.norm(dim = 0)

            re_score_center = re_query_center.unsqueeze(1) - re_tail
            im_score_center = im_query_center.unsqueeze(1) - im_tail
            score_center = torch.stack([re_score_center, im_score_center], dim=0)
            score_center = score_center.norm(dim = 0)
            
            # re_score_center_plus = torch.min(re_query_max, torch.max(re_query_min, re_tail)) - re_query_center.unsqueeze(1) 
            # im_score_center_plus = torch.min(im_query_max, torch.max(im_query_min, im_tail)) - im_query_center.unsqueeze(1) 
            # score_center_plus = torch.stack([re_score_center_plus, im_score_center_plus], dim=0) # dist_inside
            # score_center_plus = score_center_plus(dim=0) # dist_inside

            new_query_center = torch.cat([re_query_center, im_query_center], dim=-1)
            score_center_plus = torch.min(new_query_max, torch.max(new_query_min, tail)) - new_query_center.unsqueeze(1) # dist_inside
            score_center_plus =  torch.stack(torch.chunk(score_center_plus, 2, dim=-1), dim=0)
            score_center_plus =  score_center_plus.norm(dim = 0) # dist_inside


        offset_norm = score_offset.norm(p=1, dim = -1)
        score = self.gamma.item() - offset_norm

        score_center_norm = score_center.norm(p=1, dim = -1)
        score_center = self.gamma2.item() - score_center_norm
        
        score_center_plus_norm = score_center_plus.norm(p=1, dim = -1)      
        score_center_plus = self.gamma.item() - offset_norm - self.cen * score_center_plus_norm 

        # offset_norm = score_offset.sum(dim = -1)
        # score = self.gamma.item() - offset_norm

        # score_center_norm = score_center.sum(dim = -1)
        # score_center = self.gamma2.item() - score_center_norm
        
        # score_center_plus_norm = score_center_plus.sum(dim = -1)      
        # score_center_plus = self.gamma.item() - offset_norm - self.cen * score_center_plus_norm

        return score, score_center, None, score_center_plus, None

    # def BoxDistMult(self, head, relation, tail, relation_time, mode, offset, head_offset, rel_len, qtype):
    #     query_center = head * relation[0] # s + o

    #     if self.euo: ## inititalize the size of the box at the first stage using the subject
    #         query_min = query_center - 0.5 * self.func(head_offset)
    #         query_max = query_center + 0.5 * self.func(head_offset)
    #     else:
    #         query_min = query_center
    #         query_max = query_center

    #     # update box size ## enlarge the size of the box using relation information
    #     query_min = query_min - 0.5 * self.func(offset[0])
    #     query_max = query_max + 0.5 * self.func(offset[0])

    #     if '1-chain' == qtype: # assume this is relative easy; for statements with missing information
    #         score_offset = F.relu(query_min - tail) + F.relu(tail - query_max)
    #         score_center = query_center - tail
    #         score_center_plus = torch.min(query_max, torch.max(query_min, tail)) - query_center

    #         score = self.gamma.item() - torch.norm(score_offset, p=1, dim=-1)  
    #         score_center = self.gamma2.item() - torch.norm(score_center, p=1, dim=-1)  
    #         score_center_plus = self.gamma.item() - torch.norm(score_offset, p=1, dim=-1) - self.cen * torch.norm(score_center_plus, p=1, dim=-1)
    #     elif qtype in ['2-inter', '3-inter', '2-3-inter']:
    #         rel_len = int(qtype.split('-')[0])
    #         offsets = query_max - query_min
    #         if rel_len == 2:
    #             query_center_time = head * relation[1] # s + t
    #             if self.euo: ## inititalize the size of the box at the first stage using the subject
    #                 query_min_time = query_center_time - 0.5 * self.func(head_offset)
    #                 query_max_time = query_center_time + 0.5 * self.func(head_offset)
    #             else:
    #                 query_min_time = query_center_time
    #                 query_max_time = query_center_time
    #             query_min_time = query_min_time - 0.5 * self.func(offset[1])
    #             query_max_time = query_max_time + 0.5 * self.func(offset[1])
    #             offsets_time = query_max_time - query_min_time
    #             if self.use_relation_time:# if this is true, then use another relation to enhance the expressivity
    #                 query_center_relation_time = relation_time * relation[1] # r + t 
    #             else:
    #                 query_center_relation_time = relation[0] * relation[1] # r + t 
    #             if self.euo: ## inititalize the size of the box at the first stage using the subject
    #                 query_min_relation_time = query_center_relation_time - 0.5 * self.func(head_offset)
    #                 query_max_relation_time = query_center_relation_time + 0.5 * self.func(head_offset)
    #             else:
    #                 query_min_relation_time = query_center_relation_time
    #                 query_max_relation_time = query_center_relation_time
    #             query_min_relation_time = query_min_relation_time - 0.5 * self.func(offset[1])
    #             query_max_relation_time = query_max_relation_time + 0.5 * self.func(offset[1])
    #             offsets_relation_time = query_max_relation_time - query_min_relation_time
    #             ## generate new center and offset
    #             new_query_center = self.center_sets(query_center.squeeze(1), offsets.squeeze(1), 
    #                                             query_center_time.squeeze(1),offsets_time.squeeze(1), 
    #                                             [query_center_relation_time.squeeze(1)], [offsets_relation_time.squeeze(1)]).squeeze(1)
    #             new_offset = self.offset_sets(query_center.squeeze(1), offsets.squeeze(1),
    #                                             query_center_time.squeeze(1), offsets_time.squeeze(1))
    #                                             # query_center_relation_time.squeeze(1), offsets_relation_time.squeeze(1))
    #             new_query_min = (new_query_center - 0.5*self.func(new_offset)).unsqueeze(1)
    #             new_query_max = (new_query_center + 0.5*self.func(new_offset)).unsqueeze(1)
    #             score_offset = F.relu(new_query_min - tail) + F.relu(tail - new_query_max) # dist_outside
    #             score_center = new_query_center.unsqueeze(1) - tail
    #             score_center_plus = torch.min(new_query_max, torch.max(new_query_min, tail)) - new_query_center.unsqueeze(1) # dist_inside
                
    #             score = self.gamma.item() - torch.norm(score_offset, p=1, dim=-1)  
    #             score_center = self.gamma2.item() - torch.norm(score_center, p=1, dim=-1)  
    #             score_center_plus = self.gamma.item() - torch.norm(score_offset, p=1, dim=-1) - self.cen * torch.norm(score_center_plus, p=1, dim=-1)
    #         elif rel_len == 3:
    #             print('Unknow qtype')
    #             raise NotImplementedError

    #     else:
    #         print('Unknow qtype')
    #         raise NotImplementedError

    #     return score, score_center, None, score_center_plus, None

    def BoxTransE(self, head, relation, tail, relation_time, mode, offset, head_offset, rel_len, qtype):
        query_center = head + relation[0] # s + r => o

        # if self.euo: ## inititalize the size of the box at the first stage using the subject
        #     query_min = query_center - 0.5 * self.func(head_offset)
        #     query_max = query_center + 0.5 * self.func(head_offset)
        # else:
        query_min = query_center
        query_max = query_center

        # update box size ## enlarge the size of the box using relation information
        query_min = query_min - 0.5 * self.func(offset[0])
        query_max = query_max + 0.5 * self.func(offset[0])

        if '1-chain' == qtype: # assume this is relative easy; for statements with missing information
            score_offset = F.relu(query_min - tail) + F.relu(tail - query_max)
            score_center = query_center - tail
            score_center_plus = torch.min(query_max, torch.max(query_min, tail)) - query_center

            score = self.gamma.item() - torch.norm(score_offset, p=1, dim=-1)  
            score_center = self.gamma2.item() - torch.norm(score_center, p=1, dim=-1)  
            score_center_plus = self.gamma.item() - torch.norm(score_offset, p=1, dim=-1) - self.cen * torch.norm(score_center_plus, p=1, dim=-1)
        #el
        elif qtype in ['2-inter', '2-3-inter', '3-inter']:
            # rel_len = int(qtype.split('-')[0])
            offsets = query_max - query_min

            if mode == "time-batch": ## used in negative sampling process
                query_center_time = head + relation[1] # relation[2] stores a batch of negative samples --> [batch_size, num_neg, ndim]
                if self.euo: ## inititalize the size of the box at the first stage using the subject
                    query_min_time = query_center_time - 0.5 * self.func(head_offset)
                    query_max_time = query_center_time + 0.5 * self.func(head_offset)
                else:
                    query_min_time = query_center_time
                    query_max_time = query_center_time

                query_min_time = query_min_time - 0.5 * self.func(offset[1])
                query_max_time = query_max_time + 0.5 * self.func(offset[1])
                offsets_time = query_max_time - query_min_time #[batch_size, num_neg, ndim]
                
                if self.use_relation_time:# if this is true, then use another relation to enhance the expressivity
                    query_center_relation_time = relation_time + relation[1] # r + t 
                else:
                    query_center_relation_time = relation[0] + relation[1] # r + t 
                # if self.euo: ## inititalize the size of the box at the first stage using the subject
                #     query_min_relation_time = query_center_relation_time - 0.5 * self.func(head_offset)
                #     query_max_relation_time = query_center_relation_time + 0.5 * self.func(head_offset)
                # else:
                #     query_min_relation_time = query_center_relation_time
                #     query_max_relation_time = query_center_relation_time
                # query_min_relation_time = query_min_relation_time - 0.5 * self.func(offset[2])
                # query_max_relation_time = query_max_relation_time + 0.5 * self.func(offset[2])
                # offsets_relation_time = query_max_relation_time - query_min_relation_time
                ## reshape and expand tensors: expand head and relation --> [batch_size, num_negs, ndim]
                
                batch_size, num_negative_samples = relation[1].size(0), relation[1].size(1)
                query_center = query_center.expand(batch_size, num_negative_samples, -1).clone().view(batch_size*num_negative_samples, -1)
                query_center_time = query_center_time.view(batch_size*num_negative_samples, -1)
                query_center_relation_time = query_center_relation_time.view(batch_size*num_negative_samples, -1)

                offsets = offsets.expand(batch_size, num_negative_samples, -1).clone().view(batch_size*num_negative_samples, -1)
                offsets_time = offsets_time.view(batch_size*num_negative_samples, -1)
                # offsets_relation_time = offsets_relation_time.view(batch_size*num_negative_samples, -1)

                assert qtype == '2-inter'
                ## generate new center and offset
                new_query_center = self.center_sets(query_center, offsets, 
                                                query_center_time,offsets_time,
                                                [query_center_relation_time], [offsets_time]).squeeze(1)
                new_offset = self.offset_sets(query_center, offsets,
                                                query_center_time, offsets_time)

                new_query_center = new_query_center.view(batch_size, num_negative_samples, -1)
                new_offset = new_offset.view(batch_size, num_negative_samples, -1)

                # print("shape of new_query_center", new_query_center.shape)
                # print("shape of new_offset", new_offset.shape)
                # print("shape of query_center_relation_time", query_center_relation_time.shape)

                # print("shape of query_center", offsets.shape)
                # print("shape of query_center_time", offsets_time.shape)
                # print("shape of query_center_relation_time", offsets_relation_time.shape)

                new_query_min = (new_query_center - 0.5*self.func(new_offset)) 
                new_query_max = (new_query_center + 0.5*self.func(new_offset))
                ## the shape of tail is [batch_size, 1, ndim]
                score_offset = F.relu(new_query_min - tail) + F.relu(tail - new_query_max) # dist_outside
                score_center = new_query_center - tail
                score_center_plus = torch.min(new_query_max, torch.max(new_query_min, tail)) - new_query_center# dist_inside
                
                score = self.gamma.item() - torch.norm(score_offset, p=1, dim=-1)  
                score_center = self.gamma2.item() - torch.norm(score_center, p=1, dim=-1)  
                score_center_plus = self.gamma.item() - torch.norm(score_offset, p=1, dim=-1) - self.cen * torch.norm(score_center_plus, p=1, dim=-1)

                return score, score_center, None, score_center_plus, None
            else:
                query_center_time = head + relation[1] # s + t => o
                # if self.euo: ## inititalize the size of the box at the first stage using the subject
                #     query_min_time = query_center_time - 0.5 * self.func(head_offset)
                #     query_max_time = query_center_time + 0.5 * self.func(head_offset)
                # else:
                query_min_time = query_center_time
                query_max_time = query_center_time

                query_min_time = query_min_time - 0.5 * self.func(offset[1])
                query_max_time = query_max_time + 0.5 * self.func(offset[1])
                offsets_time = query_max_time - query_min_time
                if self.use_relation_time:# if this is true, then use another relation to enhance the expressivity
                    query_center_relation_time = relation_time + relation[1] # r + t 
                else:
                    query_center_relation_time = relation[0] + relation[1] # r + t 
                # if self.euo: ## inititalize the size of the box at the first stage using the subject
                #     query_min_relation_time = query_center_relation_time - 0.5 * self.func(head_offset)
                #     query_max_relation_time = query_center_relation_time + 0.5 * self.func(head_offset)
                # else:
                query_min_relation_time = query_center_relation_time
                query_max_relation_time = query_center_relation_time

                query_min_relation_time = query_min_relation_time - 0.5 * self.func(offset[1])
                query_max_relation_time = query_max_relation_time + 0.5 * self.func(offset[1])
                offsets_relation_time = query_max_relation_time - query_min_relation_time

            if qtype == '2-inter':
                ## generate new center and offset
                new_query_center = self.center_sets(query_center.squeeze(1), offsets.squeeze(1), 
                                                query_center_time.squeeze(1),offsets_time.squeeze(1),
                                                [query_center_relation_time.squeeze(1)], [offsets_relation_time.squeeze(1)]).squeeze(1)
                                                # [query_center_relation_time.squeeze(1)], [self.func(head_offset).squeeze(1)]).squeeze(1)
                new_offset = self.offset_sets(query_center.squeeze(1), offsets.squeeze(1),
                                                query_center_time.squeeze(1), offsets_time.squeeze(1))
                                               # [query_center_relation_time.squeeze(1)],  [offsets_relation_time.squeeze(1)])
                                                # [query_center_relation_time.squeeze(1)], [self.func(head_offset).squeeze(1)])
            elif qtype == '2-3-inter':
                ## generate new center and offset
                sub_query_centers = [query_center_time.squeeze(1), query_center_relation_time.squeeze(1), query_center_relation_time.squeeze(1)]
                sub_offsets = [offsets_time.squeeze(1), offsets_relation_time.squeeze(1), offsets_relation_time.squeeze(1)]
                new_query_center = self.center_sets(query_center.squeeze(1), offsets.squeeze(1), 
                                                query_center_time.squeeze(1), offsets_time.squeeze(1), 
                                                sub_query_centers, sub_offsets).squeeze(1)
                new_offset = self.offset_sets(query_center.squeeze(1), offsets.squeeze(1),
                                                query_center_time.squeeze(1), offsets_time.squeeze(1),
                                                sub_query_centers, sub_offsets)
            elif qtype == '3-inter': 
                # print('get here')
                # generate new center and offset
                # new_query_center_1 = self.center_sets(query_center.squeeze(1), offsets.squeeze(1), 
                #                                 query_center_time.squeeze(1),offsets_time.squeeze(1),
                #                                 [query_center_relation_time.squeeze(1)], [offsets_relation_time.squeeze(1)]).squeeze(1)
                #                                 # [query_center_relation_time.squeeze(1)], [self.func(head_offset).squeeze(1)]).squeeze(1)
                # new_offset_1 = self.offset_sets(query_center.squeeze(1), offsets.squeeze(1),
                #                                 query_center_time.squeeze(1), offsets_time.squeeze(1),
                #                                 [query_center_relation_time.squeeze(1)], [])

                # ## deal with the other time information
                # query_center_time_2 = head + relation[1] # s + t => o
                # # if self.euo: ## inititalize the size of the box at the first stage using the subject
                # #     query_min_time = query_center_time - 0.5 * self.func(head_offset)
                # #     query_max_time = query_center_time + 0.5 * self.func(head_offset)
                # # else:
                # query_min_time_2 = query_center_time_2
                # query_max_time_2 = query_center_time_2

                # query_min_time_2 = query_min_time_2 - 0.5 * self.func(offset[2])
                # query_max_time_2 = query_max_time_2 + 0.5 * self.func(offset[2])
                # offsets_time_2 = query_max_time_2 - query_min_time_2
                # if self.use_relation_time:# if this is true, then use another relation to enhance the expressivity
                #     query_center_relation_time_2 = relation_time + relation[1] # r + t 
                # else:
                #     query_center_relation_time_2 = relation[0] + relation[1] # r + t 
                # # if self.euo: ## inititalize the size of the box at the first stage using the subject
                # #     query_min_relation_time = query_center_relation_time - 0.5 * self.func(head_offset)
                # #     query_max_relation_time = query_center_relation_time + 0.5 * self.func(head_offset)
                # # else:
                # query_min_relation_time_2 = query_center_relation_time_2
                # query_max_relation_time_2 = query_center_relation_time_2

                # query_min_relation_time_2 = query_min_relation_time_2 - 0.5 * self.func(offset[2])
                # query_max_relation_time_2 = query_max_relation_time_2 + 0.5 * self.func(offset[2])
                # offsets_relation_time_2 = query_max_relation_time_2 - query_min_relation_time_2

                # new_query_center_２ = self.center_sets(query_center.squeeze(1), offsets.squeeze(1), 
                #                                 query_center_time_2.squeeze(1),offsets_time_2.squeeze(1),
                #                                 [query_center_relation_time_2.squeeze(1)], [offsets_relation_time_2.squeeze(1)]).squeeze(1)
                #                                 # [query_center_relation_time.squeeze(1)], [self.func(head_offset).squeeze(1)]).squeeze(1)
                # new_offset_２ = self.offset_sets(query_center.squeeze(1), offsets.squeeze(1),
                #                                 query_center_time_2.squeeze(1), offsets_time_2.squeeze(1),
                #                                 [query_center_relation_time_2.squeeze(1)], [])

                # new_query_min_1 = (new_query_center_1 - 0.5*self.func(new_offset_1)).unsqueeze(1)
                # new_query_max_1 = (new_query_center_1 + 0.5*self.func(new_offset_1)).unsqueeze(1)
                # score_offset_1 = F.relu(new_query_min_1 - tail) + F.relu(tail - new_query_max_1) # dist_outside
                # score_center_1 = new_query_center_1.unsqueeze(1) - tail
                # score_center_plus_1 = torch.min(new_query_max_1, torch.max(new_query_min_1, tail)) - new_query_center_1.unsqueeze(1) # dist_inside
                
                # score_1 = self.gamma.item() - torch.norm(score_offset_1, p=1, dim=-1)  
                # score_center_1 = self.gamma2.item() - torch.norm(score_center_1, p=1, dim=-1)  
                # score_center_plus_1 = self.gamma.item() - torch.norm(score_offset_1, p=1, dim=-1) - self.cen * torch.norm(score_center_plus_1, p=1, dim=-1)

                # new_query_min_2 = (new_query_center_2 - 0.5*self.func(new_offset_2)).unsqueeze(1)
                # new_query_max_2 = (new_query_center_2 + 0.5*self.func(new_offset_2)).unsqueeze(1)
                # score_offset_2 = F.relu(new_query_min_2 - tail) + F.relu(tail - new_query_max_2) # dist_outside
                # score_center_2 = new_query_center_2.unsqueeze(1) - tail
                # score_center_plus_2 = torch.min(new_query_max_2, torch.max(new_query_min_2, tail)) - new_query_center_2.unsqueeze(1) # dist_inside
                
                # score_2 = self.gamma.item() - torch.norm(score_offset_2, p=1, dim=-1)  
                # score_center_2 = self.gamma2.item() - torch.norm(score_center_2, p=1, dim=-1)  
                # score_center_plus_2 = self.gamma.item() - torch.norm(score_offset_2, p=1, dim=-1) - self.cen * torch.norm(score_center_plus_2, p=1, dim=-1)

                # return (score_1+score_2)/2, (score_center_1+score_center_2)/2, None, (score_center_plus_1+score_center_plus_2)/2,  torch.norm(score_center_plus_1-score_center_plus_2, p=1, dim=-1)

                query_center_time_2 = head + relation[2] # s + end_t
                if self.euo: ## inititalize the size of the box at the first stage using the subject
                    query_min_time_2 = query_center_time_2 - 0.5 * self.func(head_offset)
                    query_max_time_2 = query_center_time_2 + 0.5 * self.func(head_offset)
                else:
                    query_min_time_2 = query_center_time_2
                    query_max_time_2 = query_center_time_2
                query_min_time_2 = query_min_time_2 - 0.5 * self.func(offset[2])
                query_max_time_2 = query_max_time_2 + 0.5 * self.func(offset[2])
                offsets_time_2 = query_max_time_2 - query_min_time_2
                # if self.use_relation_time:# if this is true, then use another relation to enhance the expressivity
                #     query_center_relation_time_2 = relation_time + relation[2] # r + t_end 
                # else:
                #     query_center_relation_time_2 = relation[0] + relation[2] # r + t_end 
                # if self.euo: ## inititalize the size of the box at the first stage using the subject
                #     query_min_relation_time_2 = query_center_relation_time_2 - 0.5 * self.func(head_offset)
                #     query_max_relation_time_2 = query_center_relation_time_2 + 0.5 * self.func(head_offset)
                # else:
                #     query_min_relation_time_2 = query_center_relation_time_2
                #     query_max_relation_time_2 = query_center_relation_time_2
                # query_min_relation_time_2 = query_min_relation_time_2 - 0.5 * self.func(offset[2])
                # query_max_relation_time_2 = query_max_relation_time_2 + 0.5 * self.func(offset[2])
                # offsets_relation_time_2 = query_max_relation_time_2 - query_min_relation_time_2
                
                sub_query_centers = [query_center_time_2.squeeze(1)]
                # sub_query_centers = [query_center_relation_time.squeeze(1), query_center_time_2.squeeze(1), query_center_relation_time_2.squeeze(1)]
                # sub_offsets = [offsets_relation_time.squeeze(1), offsets_time_2.squeeze(1), offsets_relation_time_2.squeeze(1)]
                sub_offsets = [offsets_time_2.squeeze(1)]
                new_query_center = self.center_sets(query_center.squeeze(1), offsets.squeeze(1), 
                                                query_center_time.squeeze(1), offsets_time.squeeze(1), 
                                                sub_query_centers, sub_offsets).squeeze(1)
                new_offset = self.offset_sets(query_center.squeeze(1), offsets.squeeze(1),
                                                query_center_time.squeeze(1), offsets_time.squeeze(1),
                                                sub_query_centers, sub_offsets)
            else:
                raise NotImplementedError

            ## the shape of tail is [batch_size, num_negative_samples, ndim]
            # if  self.enumerate_time and qtype == '3-inter' and neg_batch_size == 1: ## indicate that must be enumerate_time mode and 3-inter when *testing*!
            #     tail = 
            new_query_min = (new_query_center - 0.5*self.func(new_offset)).unsqueeze(1)
            new_query_max = (new_query_center + 0.5*self.func(new_offset)).unsqueeze(1)
            score_offset = F.relu(new_query_min - tail) + F.relu(tail - new_query_max) # dist_outside
            score_center = new_query_center.unsqueeze(1) - tail
            score_center_plus = torch.min(new_query_max, torch.max(new_query_min, tail)) - new_query_center.unsqueeze(1) # dist_inside
            
            score = self.gamma.item() - torch.norm(score_offset, p=1, dim=-1)  
            score_center = self.gamma2.item() - torch.norm(score_center, p=1, dim=-1)  
            score_center_plus = self.gamma.item() - torch.norm(score_offset, p=1, dim=-1) - self.cen * torch.norm(score_center_plus, p=1, dim=-1)
            
        else:
            print('Unknow qtype')
            raise NotImplementedError

        return score, score_center, None, score_center_plus, None

    def BoxDistMult(self, head, relation, tail, relation_time, mode, offset, head_offset, rel_len, qtype):
        query_center = head * relation[0] # s + r => o

        # if self.euo: ## inititalize the size of the box at the first stage using the subject
        #     query_min = query_center - 0.5 * self.func(head_offset)
        #     query_max = query_center + 0.5 * self.func(head_offset)
        # else:
        query_min = query_center
        query_max = query_center

        # update box size ## enlarge the size of the box using relation information
        query_min = query_min - 0.5 * self.func(offset[0])
        query_max = query_max + 0.5 * self.func(offset[0])

        if '1-chain' == qtype: # assume this is relative easy; for statements with missing information
            score_offset = F.relu(query_min - tail) + F.relu(tail - query_max)
            score_center = query_center - tail
            score_center_plus = torch.min(query_max, torch.max(query_min, tail)) - query_center

            score = self.gamma.item() - torch.norm(score_offset, p=1, dim=-1)  
            score_center = self.gamma2.item() - torch.norm(score_center, p=1, dim=-1)  
            score_center_plus = self.gamma.item() - torch.norm(score_offset, p=1, dim=-1) - self.cen * torch.norm(score_center_plus, p=1, dim=-1)
        #el
        elif qtype in ['2-inter', '2-3-inter', '3-inter']:
            # rel_len = int(qtype.split('-')[0])
            offsets = query_max - query_min

            if mode == "time-batch": ## used in negative sampling process
                query_center_time = head * relation[1] # relation[2] stores a batch of negative samples --> [batch_size, num_neg, ndim]
                if self.euo: ## inititalize the size of the box at the first stage using the subject
                    query_min_time = query_center_time - 0.5 * self.func(head_offset)
                    query_max_time = query_center_time + 0.5 * self.func(head_offset)
                else:
                    query_min_time = query_center_time
                    query_max_time = query_center_time

                query_min_time = query_min_time - 0.5 * self.func(offset[1])
                query_max_time = query_max_time + 0.5 * self.func(offset[1])
                offsets_time = query_max_time - query_min_time #[batch_size, num_neg, ndim]
                
                if self.use_relation_time:# if this is true, then use another relation to enhance the expressivity
                    query_center_relation_time = relation_time * relation[1] # r + t 
                else:
                    query_center_relation_time = relation[0] * relation[1] # r + t 
                # if self.euo: ## inititalize the size of the box at the first stage using the subject
                #     query_min_relation_time = query_center_relation_time - 0.5 * self.func(head_offset)
                #     query_max_relation_time = query_center_relation_time + 0.5 * self.func(head_offset)
                # else:
                #     query_min_relation_time = query_center_relation_time
                #     query_max_relation_time = query_center_relation_time
                # query_min_relation_time = query_min_relation_time - 0.5 * self.func(offset[2])
                # query_max_relation_time = query_max_relation_time + 0.5 * self.func(offset[2])
                # offsets_relation_time = query_max_relation_time - query_min_relation_time
                ## reshape and expand tensors: expand head and relation --> [batch_size, num_negs, ndim]
                
                batch_size, num_negative_samples = relation[1].size(0), relation[1].size(1)
                query_center = query_center.expand(batch_size, num_negative_samples, -1).clone().view(batch_size*num_negative_samples, -1)
                query_center_time = query_center_time.view(batch_size*num_negative_samples, -1)
                query_center_relation_time = query_center_relation_time.view(batch_size*num_negative_samples, -1)

                offsets = offsets.expand(batch_size, num_negative_samples, -1).clone().view(batch_size*num_negative_samples, -1)
                offsets_time = offsets_time.view(batch_size*num_negative_samples, -1)
                # offsets_relation_time = offsets_relation_time.view(batch_size*num_negative_samples, -1)

                assert qtype == '2-inter'
                ## generate new center and offset
                new_query_center = self.center_sets(query_center, offsets, 
                                                query_center_time,offsets_time, 
                                                [query_center_relation_time], [offsets_time]).squeeze(1)
                new_offset = self.offset_sets(query_center, offsets,
                                                query_center_time, offsets_time)

                new_query_center = new_query_center.view(batch_size, num_negative_samples, -1)
                new_offset = new_offset.view(batch_size, num_negative_samples, -1)

                # print("shape of new_query_center", new_query_center.shape)
                # print("shape of new_offset", new_offset.shape)
                # print("shape of query_center_relation_time", query_center_relation_time.shape)

                # print("shape of query_center", offsets.shape)
                # print("shape of query_center_time", offsets_time.shape)
                # print("shape of query_center_relation_time", offsets_relation_time.shape)

                new_query_min = (new_query_center - 0.5*self.func(new_offset)) 
                new_query_max = (new_query_center + 0.5*self.func(new_offset))
                ## the shape of tail is [batch_size, 1, ndim]
                score_offset = F.relu(new_query_min - tail) + F.relu(tail - new_query_max) # dist_outside
                score_center = new_query_center - tail
                score_center_plus = torch.min(new_query_max, torch.max(new_query_min, tail)) - new_query_center# dist_inside
                
                score = self.gamma.item() - torch.norm(score_offset, p=1, dim=-1)  
                score_center = self.gamma2.item() - torch.norm(score_center, p=1, dim=-1)  
                score_center_plus = self.gamma.item() - torch.norm(score_offset, p=1, dim=-1) - self.cen * torch.norm(score_center_plus, p=1, dim=-1)

                return score, score_center, None, score_center_plus, None
            else:
                query_center_time = head * relation[1] # s + t => o
                # if self.euo: ## inititalize the size of the box at the first stage using the subject
                #     query_min_time = query_center_time - 0.5 * self.func(head_offset)
                #     query_max_time = query_center_time + 0.5 * self.func(head_offset)
                # else:
                query_min_time = query_center_time
                query_max_time = query_center_time

                query_min_time = query_min_time - 0.5 * self.func(offset[1])
                query_max_time = query_max_time + 0.5 * self.func(offset[1])
                offsets_time = query_max_time - query_min_time
                if self.use_relation_time:# if this is true, then use another relation to enhance the expressivity
                    query_center_relation_time = relation_time * relation[1] # r + t 
                else:
                    query_center_relation_time = relation[0] * relation[1] # r + t 
                # if self.euo: ## inititalize the size of the box at the first stage using the subject
                #     query_min_relation_time = query_center_relation_time - 0.5 * self.func(head_offset)
                #     query_max_relation_time = query_center_relation_time + 0.5 * self.func(head_offset)
                # else:
                query_min_relation_time = query_center_relation_time
                query_max_relation_time = query_center_relation_time

                query_min_relation_time = query_min_relation_time - 0.5 * self.func(offset[1])
                query_max_relation_time = query_max_relation_time + 0.5 * self.func(offset[1])
                offsets_relation_time = query_max_relation_time - query_min_relation_time

            if qtype == '2-inter':
                ## generate new center and offset
                new_query_center = self.center_sets(query_center.squeeze(1), offsets.squeeze(1), 
                                                query_center_time.squeeze(1),offsets_time.squeeze(1),
                                                [query_center_relation_time.squeeze(1)], [offsets_relation_time.squeeze(1)]).squeeze(1)
                                                # [query_center_relation_time.squeeze(1)], [self.func(head_offset).squeeze(1)]).squeeze(1)
                new_offset = self.offset_sets(query_center.squeeze(1), offsets.squeeze(1),
                                                query_center_time.squeeze(1), offsets_time.squeeze(1),
                                               [query_center_relation_time.squeeze(1)],  [offsets_relation_time.squeeze(1)])
                                                # [query_center_relation_time.squeeze(1)], [self.func(head_offset).squeeze(1)])
            elif qtype == '2-3-inter':
                ## generate new center and offset
                sub_query_centers = [query_center_time.squeeze(1), query_center_relation_time.squeeze(1), query_center_relation_time.squeeze(1)]
                sub_offsets = [offsets_time.squeeze(1), offsets_relation_time.squeeze(1), offsets_relation_time.squeeze(1)]
                new_query_center = self.center_sets(query_center.squeeze(1), offsets.squeeze(1), 
                                                query_center_time.squeeze(1), offsets_time.squeeze(1), 
                                                sub_query_centers, sub_offsets).squeeze(1)
                new_offset = self.offset_sets(query_center.squeeze(1), offsets.squeeze(1),
                                                query_center_time.squeeze(1), offsets_time.squeeze(1),
                                                sub_query_centers, sub_offsets)
            elif qtype == '3-inter': 
                # print('get here')
                # generate new center and offset
                new_query_center_1 = self.center_sets(query_center.squeeze(1), offsets.squeeze(1), 
                                                query_center_time.squeeze(1),offsets_time.squeeze(1),
                                                [query_center_relation_time.squeeze(1)], [offsets_relation_time.squeeze(1)]).squeeze(1)
                                                # [query_center_relation_time.squeeze(1)], [self.func(head_offset).squeeze(1)]).squeeze(1)
                new_offset_1 = self.offset_sets(query_center.squeeze(1), offsets.squeeze(1),
                                                query_center_time.squeeze(1), offsets_time.squeeze(1),
                                                [query_center_relation_time.squeeze(1)], [])

                ## deal with the other time information
                query_center_time_2 = head * relation[1] # s + t => o
                # if self.euo: ## inititalize the size of the box at the first stage using the subject
                #     query_min_time = query_center_time - 0.5 * self.func(head_offset)
                #     query_max_time = query_center_time + 0.5 * self.func(head_offset)
                # else:
                query_min_time_2 = query_center_time_2
                query_max_time_2 = query_center_time_2

                query_min_time_2 = query_min_time_2 - 0.5 * self.func(offset[2])
                query_max_time_2 = query_max_time_2 + 0.5 * self.func(offset[2])
                offsets_time_2 = query_max_time_2 - query_min_time_2
                if self.use_relation_time:# if this is true, then use another relation to enhance the expressivity
                    query_center_relation_time_2 = relation_time + relation[1] # r + t 
                else:
                    query_center_relation_time_2 = relation[0] + relation[1] # r + t 
                # if self.euo: ## inititalize the size of the box at the first stage using the subject
                #     query_min_relation_time = query_center_relation_time - 0.5 * self.func(head_offset)
                #     query_max_relation_time = query_center_relation_time + 0.5 * self.func(head_offset)
                # else:
                query_min_relation_time_2 = query_center_relation_time_2
                query_max_relation_time_2 = query_center_relation_time_2

                query_min_relation_time_2 = query_min_relation_time_2 - 0.5 * self.func(offset[2])
                query_max_relation_time_2 = query_max_relation_time_2 + 0.5 * self.func(offset[2])
                offsets_relation_time_2 = query_max_relation_time_2 - query_min_relation_time_2

                new_query_center_２ = self.center_sets(query_center.squeeze(1), offsets.squeeze(1), 
                                                query_center_time_2.squeeze(1),offsets_time_2.squeeze(1),
                                                [query_center_relation_time_2.squeeze(1)], [offsets_relation_time_2.squeeze(1)]).squeeze(1)
                                                # [query_center_relation_time.squeeze(1)], [self.func(head_offset).squeeze(1)]).squeeze(1)
                new_offset_２ = self.offset_sets(query_center.squeeze(1), offsets.squeeze(1),
                                                query_center_time_2.squeeze(1), offsets_time_2.squeeze(1),
                                                [query_center_relation_time_2.squeeze(1)], [])

                new_query_min_1 = (new_query_center_1 - 0.5*self.func(new_offset_1)).unsqueeze(1)
                new_query_max_1 = (new_query_center_1 + 0.5*self.func(new_offset_1)).unsqueeze(1)
                score_offset_1 = F.relu(new_query_min_1 - tail) + F.relu(tail - new_query_max_1) # dist_outside
                score_center_1 = new_query_center_1.unsqueeze(1) - tail
                score_center_plus_1 = torch.min(new_query_max_1, torch.max(new_query_min_1, tail)) - new_query_center_1.unsqueeze(1) # dist_inside
                
                score_1 = self.gamma.item() - torch.norm(score_offset_1, p=1, dim=-1)  
                score_center_1 = self.gamma2.item() - torch.norm(score_center_1, p=1, dim=-1)  
                score_center_plus_1 = self.gamma.item() - torch.norm(score_offset_1, p=1, dim=-1) - self.cen * torch.norm(score_center_plus_1, p=1, dim=-1)

                new_query_min_2 = (new_query_center_2 - 0.5*self.func(new_offset_2)).unsqueeze(1)
                new_query_max_2 = (new_query_center_2 + 0.5*self.func(new_offset_2)).unsqueeze(1)
                score_offset_2 = F.relu(new_query_min_2 - tail) + F.relu(tail - new_query_max_2) # dist_outside
                score_center_2 = new_query_center_2.unsqueeze(1) - tail
                score_center_plus_2 = torch.min(new_query_max_2, torch.max(new_query_min_2, tail)) - new_query_center_2.unsqueeze(1) # dist_inside
                
                score_2 = self.gamma.item() - torch.norm(score_offset_2, p=1, dim=-1)  
                score_center_2 = self.gamma2.item() - torch.norm(score_center_2, p=1, dim=-1)  
                score_center_plus_2 = self.gamma.item() - torch.norm(score_offset_2, p=1, dim=-1) - self.cen * torch.norm(score_center_plus_2, p=1, dim=-1)

                return (score_1+score_2)/2, (score_center_1+score_center_2)/2, None, (score_center_plus_1+score_center_plus_2)/2,  torch.norm(score_center_plus_1-score_center_plus_2, p=1, dim=-1)

                # query_center_time_2 = head + relation[2] # s + end_t
                # if self.euo: ## inititalize the size of the box at the first stage using the subject
                #     query_min_time_2 = query_center_time_2 - 0.5 * self.func(head_offset)
                #     query_max_time_2 = query_center_time_2 + 0.5 * self.func(head_offset)
                # else:
                #     query_min_time_2 = query_center_time_2
                #     query_max_time_2 = query_center_time_2
                # query_min_time_2 = query_min_time_2 - 0.5 * self.func(offset[2])
                # query_max_time_2 = query_max_time_2 + 0.5 * self.func(offset[2])
                # offsets_time_2 = query_max_time_2 - query_min_time_2
                # if self.use_relation_time:# if this is true, then use another relation to enhance the expressivity
                #     query_center_relation_time_2 = relation_time + relation[2] # r + t_end 
                # else:
                #     query_center_relation_time_2 = relation[0] + relation[2] # r + t_end 
                # if self.euo: ## inititalize the size of the box at the first stage using the subject
                #     query_min_relation_time_2 = query_center_relation_time_2 - 0.5 * self.func(head_offset)
                #     query_max_relation_time_2 = query_center_relation_time_2 + 0.5 * self.func(head_offset)
                # else:
                #     query_min_relation_time_2 = query_center_relation_time_2
                #     query_max_relation_time_2 = query_center_relation_time_2
                # query_min_relation_time_2 = query_min_relation_time_2 - 0.5 * self.func(offset[2])
                # query_max_relation_time_2 = query_max_relation_time_2 + 0.5 * self.func(offset[2])
                # offsets_relation_time_2 = query_max_relation_time_2 - query_min_relation_time_2
                
                # sub_query_centers = [query_center_relation_time.squeeze(1), query_center_time_2.squeeze(1), query_center_relation_time_2.squeeze(1)]
                # # sub_offsets = [offsets_relation_time.squeeze(1), offsets_time_2.squeeze(1), offsets_relation_time_2.squeeze(1)]
                # sub_offsets = [offsets_time_2.squeeze(1)]
                # new_query_center = self.center_sets(query_center.squeeze(1), offsets.squeeze(1), 
                #                                 query_center_time.squeeze(1), offsets_time.squeeze(1), 
                #                                 sub_query_centers, sub_offsets).squeeze(1)
                # new_offset = self.offset_sets(query_center.squeeze(1), offsets.squeeze(1),
                #                                 query_center_time.squeeze(1), offsets_time.squeeze(1),
                #                                 sub_query_centers, sub_offsets)
            else:
                raise NotImplementedError

            ## the shape of tail is [batch_size, num_negative_samples, ndim]
            # if  self.enumerate_time and qtype == '3-inter' and neg_batch_size == 1: ## indicate that must be enumerate_time mode and 3-inter when *testing*!
            #     tail = 
            new_query_min = (new_query_center - 0.5*self.func(new_offset)).unsqueeze(1)
            new_query_max = (new_query_center + 0.5*self.func(new_offset)).unsqueeze(1)
            score_offset = F.relu(new_query_min - tail) + F.relu(tail - new_query_max) # dist_outside
            score_center = new_query_center.unsqueeze(1) - tail
            score_center_plus = torch.min(new_query_max, torch.max(new_query_min, tail)) - new_query_center.unsqueeze(1) # dist_inside
            
            score = self.gamma.item() - torch.norm(score_offset, p=1, dim=-1)  
            score_center = self.gamma2.item() - torch.norm(score_center, p=1, dim=-1)  
            score_center_plus = self.gamma.item() - torch.norm(score_offset, p=1, dim=-1) - self.cen * torch.norm(score_center_plus, p=1, dim=-1)
            

        else:
            print('Unknow qtype')
            raise NotImplementedError

        return score, score_center, None, score_center_plus, None

    def BoxTransETimeAsRotation(self, head, relation, tail, relation_time, mode, offset, head_offset, rel_len, qtype):
        query_center = head + relation[0] # s + r => o

        # if self.euo: ## inititalize the size of the box at the first stage using the subject
        #     query_min = query_center - 0.5 * self.func(head_offset)
        #     query_max = query_center + 0.5 * self.func(head_offset)
        # else:
        query_min = query_center
        query_max = query_center

        # update box size ## enlarge the size of the box using relation information
        query_min = query_min - 0.5 * self.func(offset[0])
        query_max = query_max + 0.5 * self.func(offset[0])

        if '1-chain' == qtype: # assume this is relative easy; for statements with missing information
            score_offset = F.relu(query_min - tail) + F.relu(tail - query_max)
            score_center = query_center - tail
            score_center_plus = torch.min(query_max, torch.max(query_min, tail)) - query_center

            score = self.gamma.item() - torch.norm(score_offset, p=1, dim=-1)  
            score_center = self.gamma2.item() - torch.norm(score_center, p=1, dim=-1)  
            score_center_plus = self.gamma.item() - torch.norm(score_offset, p=1, dim=-1) - self.cen * torch.norm(score_center_plus, p=1, dim=-1)
            return score, score_center, None, score_center_plus, None
        #el
        elif qtype in ['2-inter', '2-3-inter', '3-inter']:
            # rel_len = int(qtype.split('-')[0])
            # offsets = query_max - query_min

            if mode == "time-batch": ## used in negative sampling process
                #raise NotImplementedError

                rel_len = int(qtype.split('-')[0])
                offsets = query_max - query_min
                re_offsets, im_offsets = torch.chunk(offsets, 2, dim=-1)


                pi = 3.14159265358979323846

                re_query_center, im_query_center = torch.chunk(query_center, 2, dim=2)
                # re_head, im_head = torch.chunk(head, 2, dim=2)

                if rel_len == 2:
                    phase_time = relation[1]/(self.time_embedding_range.item()/pi)

                    re_relation_t = torch.cos(phase_time)
                    im_relation_t = torch.sin(phase_time)

                    re_query_center_relation_t = re_query_center * re_relation_t - im_query_center * im_relation_t
                    im_query_center_relation_t = re_query_center * im_relation_t + im_query_center * re_relation_t

                    # query_center_time = torch.stack([a_relation, b_relation], dim = 0)
                    # a_score = a_score - a_tail
                    # b_score = b_score - b_tail

                    re_offset_t, im_offset_t = torch.chunk(offset[1], 2, dim=-1)
                    re_query_min_relation_t = re_query_center_relation_t - 0.5 * self.func(re_offset_t)
                    im_query_min_relation_t = im_query_center_relation_t - 0.5 * self.func(im_offset_t)
                    re_query_max_relation_t = re_query_center_relation_t + 0.5 * self.func(re_offset_t)
                    im_query_max_relation_t = im_query_center_relation_t + 0.5 * self.func(im_offset_t)
                    
                    re_offsets_relation_t = re_query_max_relation_t - re_query_min_relation_t # all are positive numbers
                    im_offsets_relation_t = im_query_max_relation_t - im_query_min_relation_t # all are positive numbers

                    batch_size, num_negative_samples = relation[1].size(0), relation[1].size(1)
                    re_query_center = re_query_center.expand(batch_size, num_negative_samples, -1).clone().view(batch_size*num_negative_samples, -1)
                    im_query_center = im_query_center.expand(batch_size, num_negative_samples, -1).clone().view(batch_size*num_negative_samples, -1)
                    re_query_center_relation_t = re_query_center_relation_t.view(batch_size*num_negative_samples, -1)
                    im_query_center_relation_t = im_query_center_relation_t.view(batch_size*num_negative_samples, -1)

                    re_offsets = re_offsets.expand(batch_size, num_negative_samples, -1).clone().view(batch_size*num_negative_samples, -1)
                    im_offsets = im_offsets.expand(batch_size, num_negative_samples, -1).clone().view(batch_size*num_negative_samples, -1)
                    re_offsets_relation_t = re_offsets_relation_t.view(batch_size*num_negative_samples, -1)
                    im_offsets_relation_t = im_offsets_relation_t.view(batch_size*num_negative_samples, -1)
                    # offsets_relation_time = offsets_relation_time.view(batch_size*num_negative_samples, -1)

                    assert qtype == '2-inter'
                    ## generate new center and offset
                    re_new_query_center = self.re_center_sets(re_query_center.squeeze(1), re_offsets.squeeze(1), 
                                                    re_query_center_relation_t.squeeze(1),re_offsets_relation_t.squeeze(1)).squeeze(1) 
                    im_new_query_center = self.im_center_sets(im_query_center.squeeze(1), im_offsets.squeeze(1), 
                                                    im_query_center_relation_t.squeeze(1),im_offsets_relation_t.squeeze(1)).squeeze(1)
                    re_new_offset = self.re_offset_sets(re_query_center.squeeze(1), re_offsets.squeeze(1),
                                                    re_query_center_relation_t.squeeze(1), re_offsets_relation_t.squeeze(1))
                    im_new_offset = self.im_offset_sets(im_query_center.squeeze(1), im_offsets.squeeze(1),
                                                    im_query_center_relation_t.squeeze(1), im_offsets_relation_t.squeeze(1))

                    re_new_query_center = re_new_query_center.view(batch_size, num_negative_samples, -1)
                    im_new_query_center = im_new_query_center.view(batch_size, num_negative_samples, -1)
                    re_new_offset = re_new_offset.view(batch_size, num_negative_samples, -1)
                    im_new_offset = im_new_offset.view(batch_size, num_negative_samples, -1)

                    new_query_center = torch.cat([re_new_query_center, im_new_query_center], dim=-1)
                    new_offset  = torch.cat([re_new_offset, im_new_offset], dim=-1)

                    new_query_min = (new_query_center - 0.5*self.func(new_offset))
                    new_query_max = (new_query_center + 0.5*self.func(new_offset))

                    # re_new_query_min = (re_new_query_center - 0.5*self.func(re_new_offset)).unsqueeze(1)
                    # re_new_query_max = (re_new_query_center + 0.5*self.func(re_new_offset)).unsqueeze(1)
                    # im_new_query_min = (im_new_query_center - 0.5*self.func(im_new_offset)).unsqueeze(1)
                    # im_new_query_max = (im_new_query_center + 0.5*self.func(im_new_offset)).unsqueeze(1)

                    # re_score_offset = F.relu(re_new_query_min - re_tail) + F.relu(re_tail - re_new_query_max) # dist_outside
                    # im_score_offset = F.relu(im_new_query_min - im_tail) + F.relu(im_tail - im_new_query_max) # dist_outside
                    # score_offset = torch.stack([re_score_offset, im_score_offset], dim=0)
                    # new_query_min = torch.cat([re_new_query_min, im_new_query_min], dim=-1)
                    # new_query_max = torch.cat([re_new_query_max, im_new_query_max], dim=-1)

                    score_offset = F.relu(new_query_min - tail) + F.relu(tail - new_query_max) # dist_outside
                    score_offset =  torch.stack(torch.chunk(score_offset, 2, dim=-1), dim=0)
                    ## 
                    score_offset = score_offset.norm(dim = 0)

                    # re_tail, im_tail = torch.chunk(tail, 2, dim=-1)
                    # re_score_center = re_new_query_center.unsqueeze(1) - re_tail
                    # im_score_center = im_new_query_center.unsqueeze(1) - im_tail
                    # score_center = torch.stack([re_score_center, im_score_center], dim=0)
                    score_center = new_query_center.unsqueeze(1) - tail
                    score_center =  torch.stack(torch.chunk(score_center, 2, dim=-1), dim=0)
                    score_center = score_center.norm(dim = 0)

                    # re_score_center_plus = torch.min(re_new_query_max, torch.max(re_new_query_min, re_tail)) - re_new_query_center.unsqueeze(1) # dist_inside
                    # im_score_center_plus = torch.min(im_new_query_max, torch.max(im_new_query_min, im_tail)) - im_new_query_center.unsqueeze(1) # dist_inside
                    # score_center_plus = torch.stack([re_score_center_plus, im_score_center_plus], dim=0) # dist_inside
                    # score_center_plus =  score_center_plus.norm(dim = 0) # dist_inside
                    # new_query_center = torch.cat([re_new_query_center, im_new_query_center], dim=-1)
                    score_center_plus = torch.min(new_query_max, torch.max(new_query_min, tail)) - new_query_center # dist_inside
                    score_center_plus =  torch.stack(torch.chunk(score_center_plus, 2, dim=-1), dim=0)
                    score_center_plus =  score_center_plus.norm(dim = 0) # dist_inside

                    score = self.gamma.item() - score_offset.norm(p=1, dim = -1)
                    score_center = self.gamma2.item() - score_center.norm(p=1, dim = -1)  
                    score_center_plus = self.gamma.item() - score_offset.norm(p=1, dim = -1) - self.cen *  score_center_plus.norm(p=1, dim = -1)  

                    return score, score_center, None, score_center_plus, None
                # query_center_time = head + relation[1] # relation[2] stores a batch of negative samples --> [batch_size, num_neg, ndim]
                # if self.euo: ## inititalize the size of the box at the first stage using the subject
                #     query_min_time = query_center_time - 0.5 * self.func(head_offset)
                #     query_max_time = query_center_time + 0.5 * self.func(head_offset)
                # else:
                #     query_min_time = query_center_time
                #     query_max_time = query_center_time

                # query_min_time = query_min_time - 0.5 * self.func(offset[1])
                # query_max_time = query_max_time + 0.5 * self.func(offset[1])
                # offsets_time = query_max_time - query_min_time #[batch_size, num_neg, ndim]
                
                # if self.use_relation_time:# if this is true, then use another relation to enhance the expressivity
                #     query_center_relation_time = relation_time + relation[1] # r + t 
                # else:
                #     query_center_relation_time = relation[0] + relation[1] # r + t 
                # # if self.euo: ## inititalize the size of the box at the first stage using the subject
                # #     query_min_relation_time = query_center_relation_time - 0.5 * self.func(head_offset)
                # #     query_max_relation_time = query_center_relation_time + 0.5 * self.func(head_offset)
                # # else:
                # #     query_min_relation_time = query_center_relation_time
                # #     query_max_relation_time = query_center_relation_time
                # # query_min_relation_time = query_min_relation_time - 0.5 * self.func(offset[2])
                # # query_max_relation_time = query_max_relation_time + 0.5 * self.func(offset[2])
                # # offsets_relation_time = query_max_relation_time - query_min_relation_time
                # ## reshape and expand tensors: expand head and relation --> [batch_size, num_negs, ndim]
                
                # batch_size, num_negative_samples = relation[1].size(0), relation[1].size(1)
                # query_center = query_center.expand(batch_size, num_negative_samples, -1).clone().view(batch_size*num_negative_samples, -1)
                # query_center_time = query_center_time.view(batch_size*num_negative_samples, -1)
                # query_center_relation_time = query_center_relation_time.view(batch_size*num_negative_samples, -1)

                # offsets = offsets.expand(batch_size, num_negative_samples, -1).clone().view(batch_size*num_negative_samples, -1)
                # offsets_time = offsets_time.view(batch_size*num_negative_samples, -1)
                # # offsets_relation_time = offsets_relation_time.view(batch_size*num_negative_samples, -1)

                # assert qtype == '2-inter'
                # ## generate new center and offset
                # new_query_center = self.center_sets(query_center, offsets, 
                #                                 query_center_time,offsets_time, 
                #                                 [query_center_relation_time], [offsets_time]).squeeze(1)
                # new_offset = self.offset_sets(query_center, offsets,
                #                                 query_center_time, offsets_time)

                # new_query_center = new_query_center.view(batch_size, num_negative_samples, -1)
                # new_offset = new_offset.view(batch_size, num_negative_samples, -1)

                # # print("shape of new_query_center", new_query_center.shape)
                # # print("shape of new_offset", new_offset.shape)
                # # print("shape of query_center_relation_time", query_center_relation_time.shape)

                # # print("shape of query_center", offsets.shape)
                # # print("shape of query_center_time", offsets_time.shape)
                # # print("shape of query_center_relation_time", offsets_relation_time.shape)

                # new_query_min = (new_query_center - 0.5*self.func(new_offset)) 
                # new_query_max = (new_query_center + 0.5*self.func(new_offset))
                # ## the shape of tail is [batch_size, 1, ndim]
                # score_offset = F.relu(new_query_min - tail) + F.relu(tail - new_query_max) # dist_outside
                # score_center = new_query_center - tail
                # score_center_plus = torch.min(new_query_max, torch.max(new_query_min, tail)) - new_query_center# dist_inside
                
                # score = self.gamma.item() - torch.norm(score_offset, p=1, dim=-1)  
                # score_center = self.gamma2.item() - torch.norm(score_center, p=1, dim=-1)  
                # score_center_plus = self.gamma.item() - torch.norm(score_offset, p=1, dim=-1) - self.cen * torch.norm(score_center_plus, p=1, dim=-1)

                # return score, score_center, None, score_center_plus, None
            else:
                # query_center_time = head + relation[1] # s + t => o
                # if self.euo: ## inititalize the size of the box at the first stage using the subject
                #     query_min_time = query_center_time - 0.5 * self.func(head_offset)
                #     query_max_time = query_center_time + 0.5 * self.func(head_offset)
                # else:

                rel_len = int(qtype.split('-')[0])
                offsets = query_max - query_min
                re_offsets, im_offsets = torch.chunk(offsets, 2, dim=-1)


                pi = 3.14159265358979323846

                re_query_center, im_query_center = torch.chunk(query_center, 2, dim=2)
                re_head, im_head = torch.chunk(head, 2, dim=2)

                if rel_len == 2:
                    phase_time = relation[1]/(self.time_embedding_range.item()/pi)

                    re_relation_t = torch.cos(phase_time)
                    im_relation_t = torch.sin(phase_time)

                    re_query_center_relation_t = re_head * re_relation_t - im_head * im_relation_t
                    im_query_center_relation_t = re_head * im_relation_t + im_head * re_relation_t

                    # query_center_time = torch.stack([a_relation, b_relation], dim = 0)
                    # a_score = a_score - a_tail
                    # b_score = b_score - b_tail

                    re_offset_t, im_offset_t = torch.chunk(offset[1], 2, dim=-1)
                    re_query_min_relation_t = re_query_center_relation_t - 0.5 * self.func(re_offset_t)
                    im_query_min_relation_t = im_query_center_relation_t - 0.5 * self.func(im_offset_t)
                    re_query_max_relation_t = re_query_center_relation_t + 0.5 * self.func(re_offset_t)
                    im_query_max_relation_t = im_query_center_relation_t + 0.5 * self.func(im_offset_t)
                    
                    re_offsets_relation_t = re_query_max_relation_t - re_query_min_relation_t # all are positive numbers
                    im_offsets_relation_t = im_query_max_relation_t - im_query_min_relation_t # all are positive numbers

                    # query_center_time = query_center + relation[1] # s + t
                    # if self.euo: ## inititalize the size of the box at the first stage using the subject
                    #     query_min_time = query_center_time - 0.5 * self.func(head_offset)
                    #     query_max_time = query_center_time + 0.5 * self.func(head_offset)
                    # else:
                    #     query_min_time = query_center_time
                    #     query_max_time = query_center_time
                    # query_min_time = query_min_time - 0.5 * self.func(offset[1])
                    # query_max_time = query_max_time + 0.5 * self.func(offset[1])
                    # offsets_time = query_max_time - query_min_time
                    # if self.use_relation_time:# if this is true, then use another relation to enhance the expressivity
                    #     query_center_relation_time = relation_time + relation[1] # r + t 
                    # else:
                    #     query_center_relation_time = relation[0] + relation[1] # r + t 
                    # if self.euo: ## inititalize the size of the box at the first stage using the subject
                    #     query_min_relation_time = query_center_relation_time - 0.5 * self.func(head_offset)
                    #     query_max_relation_time = query_center_relation_time + 0.5 * self.func(head_offset)
                    # else:
                    #     query_min_relation_time = query_center_relation_time
                    #     query_max_relation_time = query_center_relation_time
                    # query_min_relation_time = query_min_relation_time - 0.5 * self.func(offset[1])
                    # query_max_relation_time = query_max_relation_time + 0.5 * self.func(offset[1])
                    # offsets_relation_time = query_max_relation_time - query_min_relation_time
                    ## generate new center and offset

                    re_new_query_center = self.re_center_sets(re_query_center.squeeze(1), re_offsets.squeeze(1), 
                                                    re_query_center_relation_t.squeeze(1),re_offsets_relation_t.squeeze(1)).squeeze(1) 
                    im_new_query_center = self.im_center_sets(im_query_center.squeeze(1), im_offsets.squeeze(1), 
                                                    im_query_center_relation_t.squeeze(1),im_offsets_relation_t.squeeze(1)).squeeze(1)
                    re_new_offset = self.re_offset_sets(re_query_center.squeeze(1), re_offsets.squeeze(1),
                                                    re_query_center_relation_t.squeeze(1), re_offsets_relation_t.squeeze(1))
                    im_new_offset = self.im_offset_sets(im_query_center.squeeze(1), im_offsets.squeeze(1),
                                                    im_query_center_relation_t.squeeze(1), im_offsets_relation_t.squeeze(1))

                    new_query_center = torch.cat([re_new_query_center, im_new_query_center], dim=-1)
                    new_offset  = torch.cat([re_new_offset, im_new_offset], dim=-1)

                    new_query_min = (new_query_center - 0.5*self.func(new_offset)).unsqueeze(1)
                    new_query_max = (new_query_center + 0.5*self.func(new_offset)).unsqueeze(1) 

                    # re_new_query_min = (re_new_query_center - 0.5*self.func(re_new_offset)).unsqueeze(1)
                    # re_new_query_max = (re_new_query_center + 0.5*self.func(re_new_offset)).unsqueeze(1)
                    # im_new_query_min = (im_new_query_center - 0.5*self.func(im_new_offset)).unsqueeze(1)
                    # im_new_query_max = (im_new_query_center + 0.5*self.func(im_new_offset)).unsqueeze(1)

                    # re_score_offset = F.relu(re_new_query_min - re_tail) + F.relu(re_tail - re_new_query_max) # dist_outside
                    # im_score_offset = F.relu(im_new_query_min - im_tail) + F.relu(im_tail - im_new_query_max) # dist_outside
                    # score_offset = torch.stack([re_score_offset, im_score_offset], dim=0)
                    # new_query_min = torch.cat([re_new_query_min, im_new_query_min], dim=-1)
                    # new_query_max = torch.cat([re_new_query_max, im_new_query_max], dim=-1)

                    score_offset = F.relu(new_query_min - tail) + F.relu(tail - new_query_max) # dist_outside
                    score_offset =  torch.stack(torch.chunk(score_offset, 2, dim=-1), dim=0)
                    ## 
                    score_offset = score_offset.norm(dim = 0)

                    # re_tail, im_tail = torch.chunk(tail, 2, dim=-1)
                    # re_score_center = re_new_query_center.unsqueeze(1) - re_tail
                    # im_score_center = im_new_query_center.unsqueeze(1) - im_tail
                    # score_center = torch.stack([re_score_center, im_score_center], dim=0)
                    score_center = new_query_center.unsqueeze(1) - tail
                    score_center =  torch.stack(torch.chunk(score_center, 2, dim=-1), dim=0)
                    score_center = score_center.norm(dim = 0)

                    # re_score_center_plus = torch.min(re_new_query_max, torch.max(re_new_query_min, re_tail)) - re_new_query_center.unsqueeze(1) # dist_inside
                    # im_score_center_plus = torch.min(im_new_query_max, torch.max(im_new_query_min, im_tail)) - im_new_query_center.unsqueeze(1) # dist_inside
                    # score_center_plus = torch.stack([re_score_center_plus, im_score_center_plus], dim=0) # dist_inside
                    # score_center_plus =  score_center_plus.norm(dim = 0) # dist_inside
                    # new_query_center = torch.cat([re_new_query_center, im_new_query_center], dim=-1)
                    score_center_plus = torch.min(new_query_max, torch.max(new_query_min, tail)) - new_query_center.unsqueeze(1) # dist_inside
                    score_center_plus =  torch.stack(torch.chunk(score_center_plus, 2, dim=-1), dim=0)
                    score_center_plus =  score_center_plus.norm(dim = 0) # dist_inside

                    score = self.gamma.item() - score_offset.norm(p=1, dim = -1)
                    score_center = self.gamma2.item() - score_center.norm(p=1, dim = -1)  
                    score_center_plus = self.gamma.item() - score_offset.norm(p=1, dim = -1) - self.cen *  score_center_plus.norm(p=1, dim = -1)  

                    # new_query_min = (new_query_center - 0.5*self.func(new_offset)).unsqueeze(1)
                    # new_query_max = (new_query_center + 0.5*self.func(new_offset)).unsqueeze(1)
                    # score_offset = F.relu(new_query_min - tail) + F.relu(tail - new_query_max) # dist_outside
                    # score_center = new_query_center.unsqueeze(1) - tail
                    # score_center_plus = torch.min(new_query_max, torch.max(new_query_min, tail)) - new_query_center.unsqueeze(1) # dist_inside
                    
                    # score = self.gamma.item() - torch.norm(score_offset, p=1, dim=-1)  
                    # score_center = self.gamma2.item() - torch.norm(score_center, p=1, dim=-1)  
                    # score_center_plus = self.gamma.item() - torch.norm(score_offset, p=1, dim=-1) - self.cen * torch.norm(score_center_plus, p=1, dim=-1)
                # elif rel_len == 3:
                #     print('Unknow qtype')
                #     raise NotImplementedError

            # else:
            #     print('Unknow qtype')
            #     raise NotImplementedError

            return score, score_center, None, score_center_plus, None
        #         query_min_time = query_center_time
        #         query_max_time = query_center_time

        #         query_min_time = query_min_time - 0.5 * self.func(offset[1])
        #         query_max_time = query_max_time + 0.5 * self.func(offset[1])
        #         offsets_time = query_max_time - query_min_time
        #         if self.use_relation_time:# if this is true, then use another relation to enhance the expressivity
        #             query_center_relation_time = relation_time + relation[1] # r + t 
        #         else:
        #             query_center_relation_time = relation[0] + relation[1] # r + t 
        #         # if self.euo: ## inititalize the size of the box at the first stage using the subject
        #         #     query_min_relation_time = query_center_relation_time - 0.5 * self.func(head_offset)
        #         #     query_max_relation_time = query_center_relation_time + 0.5 * self.func(head_offset)
        #         # else:
        #         query_min_relation_time = query_center_relation_time
        #         query_max_relation_time = query_center_relation_time

        #         query_min_relation_time = query_min_relation_time - 0.5 * self.func(offset[1])
        #         query_max_relation_time = query_max_relation_time + 0.5 * self.func(offset[1])
        #         offsets_relation_time = query_max_relation_time - query_min_relation_time

        #     if qtype == '2-inter':
        #         ## generate new center and offset
        #         new_query_center = self.center_sets(query_center.squeeze(1), offsets.squeeze(1), 
        #                                         query_center_time.squeeze(1),offsets_time.squeeze(1),
        #                                         [query_center_relation_time.squeeze(1)], [offsets_relation_time.squeeze(1)]).squeeze(1)
        #                                         # [query_center_relation_time.squeeze(1)], [self.func(head_offset).squeeze(1)]).squeeze(1)
        #         new_offset = self.offset_sets(query_center.squeeze(1), offsets.squeeze(1),
        #                                         query_center_time.squeeze(1), offsets_time.squeeze(1),
        #                                        [query_center_relation_time.squeeze(1)],  [offsets_relation_time.squeeze(1)])
        #                                         # [query_center_relation_time.squeeze(1)], [self.func(head_offset).squeeze(1)])
        #     elif qtype == '2-3-inter':
        #         ## generate new center and offset
        #         sub_query_centers = [query_center_time.squeeze(1), query_center_relation_time.squeeze(1), query_center_relation_time.squeeze(1)]
        #         sub_offsets = [offsets_time.squeeze(1), offsets_relation_time.squeeze(1), offsets_relation_time.squeeze(1)]
        #         new_query_center = self.center_sets(query_center.squeeze(1), offsets.squeeze(1), 
        #                                         query_center_time.squeeze(1), offsets_time.squeeze(1), 
        #                                         sub_query_centers, sub_offsets).squeeze(1)
        #         new_offset = self.offset_sets(query_center.squeeze(1), offsets.squeeze(1),
        #                                         query_center_time.squeeze(1), offsets_time.squeeze(1),
        #                                         sub_query_centers, sub_offsets)
        #     elif qtype == '3-inter': 
        #         # print('get here')
        #         # generate new center and offset
        #         new_query_center_1 = self.center_sets(query_center.squeeze(1), offsets.squeeze(1), 
        #                                         query_center_time.squeeze(1),offsets_time.squeeze(1),
        #                                         [query_center_relation_time.squeeze(1)], [offsets_relation_time.squeeze(1)]).squeeze(1)
        #                                         # [query_center_relation_time.squeeze(1)], [self.func(head_offset).squeeze(1)]).squeeze(1)
        #         new_offset_1 = self.offset_sets(query_center.squeeze(1), offsets.squeeze(1),
        #                                         query_center_time.squeeze(1), offsets_time.squeeze(1),
        #                                         [query_center_relation_time.squeeze(1)], [])

        #         ## deal with the other time information
        #         query_center_time_2 = head + relation[1] # s + t => o
        #         # if self.euo: ## inititalize the size of the box at the first stage using the subject
        #         #     query_min_time = query_center_time - 0.5 * self.func(head_offset)
        #         #     query_max_time = query_center_time + 0.5 * self.func(head_offset)
        #         # else:
        #         query_min_time_2 = query_center_time_2
        #         query_max_time_2 = query_center_time_2

        #         query_min_time_2 = query_min_time_2 - 0.5 * self.func(offset[2])
        #         query_max_time_2 = query_max_time_2 + 0.5 * self.func(offset[2])
        #         offsets_time_2 = query_max_time_2 - query_min_time_2
        #         if self.use_relation_time:# if this is true, then use another relation to enhance the expressivity
        #             query_center_relation_time_2 = relation_time + relation[1] # r + t 
        #         else:
        #             query_center_relation_time_2 = relation[0] + relation[1] # r + t 
        #         # if self.euo: ## inititalize the size of the box at the first stage using the subject
        #         #     query_min_relation_time = query_center_relation_time - 0.5 * self.func(head_offset)
        #         #     query_max_relation_time = query_center_relation_time + 0.5 * self.func(head_offset)
        #         # else:
        #         query_min_relation_time_2 = query_center_relation_time_2
        #         query_max_relation_time_2 = query_center_relation_time_2

        #         query_min_relation_time_2 = query_min_relation_time_2 - 0.5 * self.func(offset[2])
        #         query_max_relation_time_2 = query_max_relation_time_2 + 0.5 * self.func(offset[2])
        #         offsets_relation_time_2 = query_max_relation_time_2 - query_min_relation_time_2

        #         new_query_center_２ = self.center_sets(query_center.squeeze(1), offsets.squeeze(1), 
        #                                         query_center_time_2.squeeze(1),offsets_time_2.squeeze(1),
        #                                         [query_center_relation_time_2.squeeze(1)], [offsets_relation_time_2.squeeze(1)]).squeeze(1)
        #                                         # [query_center_relation_time.squeeze(1)], [self.func(head_offset).squeeze(1)]).squeeze(1)
        #         new_offset_２ = self.offset_sets(query_center.squeeze(1), offsets.squeeze(1),
        #                                         query_center_time_2.squeeze(1), offsets_time_2.squeeze(1),
        #                                         [query_center_relation_time_2.squeeze(1)], [])

        #         new_query_min_1 = (new_query_center_1 - 0.5*self.func(new_offset_1)).unsqueeze(1)
        #         new_query_max_1 = (new_query_center_1 + 0.5*self.func(new_offset_1)).unsqueeze(1)
        #         score_offset_1 = F.relu(new_query_min_1 - tail) + F.relu(tail - new_query_max_1) # dist_outside
        #         score_center_1 = new_query_center_1.unsqueeze(1) - tail
        #         score_center_plus_1 = torch.min(new_query_max_1, torch.max(new_query_min_1, tail)) - new_query_center_1.unsqueeze(1) # dist_inside
                
        #         score_1 = self.gamma.item() - torch.norm(score_offset_1, p=1, dim=-1)  
        #         score_center_1 = self.gamma2.item() - torch.norm(score_center_1, p=1, dim=-1)  
        #         score_center_plus_1 = self.gamma.item() - torch.norm(score_offset_1, p=1, dim=-1) - self.cen * torch.norm(score_center_plus_1, p=1, dim=-1)

        #         new_query_min_2 = (new_query_center_2 - 0.5*self.func(new_offset_2)).unsqueeze(1)
        #         new_query_max_2 = (new_query_center_2 + 0.5*self.func(new_offset_2)).unsqueeze(1)
        #         score_offset_2 = F.relu(new_query_min_2 - tail) + F.relu(tail - new_query_max_2) # dist_outside
        #         score_center_2 = new_query_center_2.unsqueeze(1) - tail
        #         score_center_plus_2 = torch.min(new_query_max_2, torch.max(new_query_min_2, tail)) - new_query_center_2.unsqueeze(1) # dist_inside
                
        #         score_2 = self.gamma.item() - torch.norm(score_offset_2, p=1, dim=-1)  
        #         score_center_2 = self.gamma2.item() - torch.norm(score_center_2, p=1, dim=-1)  
        #         score_center_plus_2 = self.gamma.item() - torch.norm(score_offset_2, p=1, dim=-1) - self.cen * torch.norm(score_center_plus_2, p=1, dim=-1)

        #         return (score_1+score_2)/2, (score_center_1+score_center_2)/2, None, (score_center_plus_1+score_center_plus_2)/2,  torch.norm(score_center_plus_1-score_center_plus_2, p=1, dim=-1)

        #         # query_center_time_2 = head + relation[2] # s + end_t
        #         # if self.euo: ## inititalize the size of the box at the first stage using the subject
        #         #     query_min_time_2 = query_center_time_2 - 0.5 * self.func(head_offset)
        #         #     query_max_time_2 = query_center_time_2 + 0.5 * self.func(head_offset)
        #         # else:
        #         #     query_min_time_2 = query_center_time_2
        #         #     query_max_time_2 = query_center_time_2
        #         # query_min_time_2 = query_min_time_2 - 0.5 * self.func(offset[2])
        #         # query_max_time_2 = query_max_time_2 + 0.5 * self.func(offset[2])
        #         # offsets_time_2 = query_max_time_2 - query_min_time_2
        #         # if self.use_relation_time:# if this is true, then use another relation to enhance the expressivity
        #         #     query_center_relation_time_2 = relation_time + relation[2] # r + t_end 
        #         # else:
        #         #     query_center_relation_time_2 = relation[0] + relation[2] # r + t_end 
        #         # if self.euo: ## inititalize the size of the box at the first stage using the subject
        #         #     query_min_relation_time_2 = query_center_relation_time_2 - 0.5 * self.func(head_offset)
        #         #     query_max_relation_time_2 = query_center_relation_time_2 + 0.5 * self.func(head_offset)
        #         # else:
        #         #     query_min_relation_time_2 = query_center_relation_time_2
        #         #     query_max_relation_time_2 = query_center_relation_time_2
        #         # query_min_relation_time_2 = query_min_relation_time_2 - 0.5 * self.func(offset[2])
        #         # query_max_relation_time_2 = query_max_relation_time_2 + 0.5 * self.func(offset[2])
        #         # offsets_relation_time_2 = query_max_relation_time_2 - query_min_relation_time_2
                
        #         # sub_query_centers = [query_center_relation_time.squeeze(1), query_center_time_2.squeeze(1), query_center_relation_time_2.squeeze(1)]
        #         # # sub_offsets = [offsets_relation_time.squeeze(1), offsets_time_2.squeeze(1), offsets_relation_time_2.squeeze(1)]
        #         # sub_offsets = [offsets_time_2.squeeze(1)]
        #         # new_query_center = self.center_sets(query_center.squeeze(1), offsets.squeeze(1), 
        #         #                                 query_center_time.squeeze(1), offsets_time.squeeze(1), 
        #         #                                 sub_query_centers, sub_offsets).squeeze(1)
        #         # new_offset = self.offset_sets(query_center.squeeze(1), offsets.squeeze(1),
        #         #                                 query_center_time.squeeze(1), offsets_time.squeeze(1),
        #         #                                 sub_query_centers, sub_offsets)
        #     else:
        #         raise NotImplementedError

        #     ## the shape of tail is [batch_size, num_negative_samples, ndim]
        #     # if  self.enumerate_time and qtype == '3-inter' and neg_batch_size == 1: ## indicate that must be enumerate_time mode and 3-inter when *testing*!
        #     #     tail = 
        #     new_query_min = (new_query_center - 0.5*self.func(new_offset)).unsqueeze(1)
        #     new_query_max = (new_query_center + 0.5*self.func(new_offset)).unsqueeze(1)
        #     score_offset = F.relu(new_query_min - tail) + F.relu(tail - new_query_max) # dist_outside
        #     score_center = new_query_center.unsqueeze(1) - tail
        #     score_center_plus = torch.min(new_query_max, torch.max(new_query_min, tail)) - new_query_center.unsqueeze(1) # dist_inside
            
        #     score = self.gamma.item() - torch.norm(score_offset, p=1, dim=-1)  
        #     score_center = self.gamma2.item() - torch.norm(score_center, p=1, dim=-1)  
        #     score_center_plus = self.gamma.item() - torch.norm(score_offset, p=1, dim=-1) - self.cen * torch.norm(score_center_plus, p=1, dim=-1)
            
        # else:
        #     print('Unknow qtype')
        #     raise NotImplementedError

        # return score, score_center, None, score_center_plus, None


    # def BoxTransETimeAsRotation(self, head, relation, tail, relation_time, mode, offset, head_offset, rel_len, qtype):
    #     query_center = head + relation[0] # s + r

    #     if self.euo: ## inititalize the size of the box at the first stage using the subject
    #         query_min = query_center - 0.5 * self.func(head_offset)
    #         query_max = query_center + 0.5 * self.func(head_offset)
    #     else:
    #         query_min = query_center
    #         query_max = query_center

    #     # update box size ## enlarge the size of the box using relation information
    #     query_min = query_min - 0.5 * self.func(offset[0])
    #     query_max = query_max + 0.5 * self.func(offset[0])

    #     if '1-chain' == qtype: # assume this is relative easy; for statements with missing information
    #         score_offset = F.relu(query_min - tail) + F.relu(tail - query_max)
    #         score_offset =  torch.stack(torch.chunk(score_offset, 2, dim=-1), dim=0)
    #             ## 
    #         score_offset = score_offset.norm(dim = 0)

    #         score_center = query_center - tail
    #         score_center =  torch.stack(torch.chunk(score_center, 2, dim=-1), dim=0)
    #             ## 
    #         score_center = score_center.norm(dim = 0)

    #         score_center_plus = torch.min(query_max, torch.max(query_min, tail)) - query_center
    #         score_center_plus =  torch.stack(torch.chunk(score_center_plus, 2, dim=-1), dim=0)
    #             ## 
    #         score_center_plus = score_center_plus.norm(dim = 0)

    #         score = self.gamma.item() - torch.norm(score_offset, p=1, dim=-1)  
    #         score_center = self.gamma2.item() - torch.norm(score_center, p=1, dim=-1)  
    #         score_center_plus = self.gamma.item() - torch.norm(score_offset, p=1, dim=-1) - self.cen * torch.norm(score_center_plus, p=1, dim=-1)
    #     elif qtype in ['2-inter', '3-inter', '2-3-inter']:
    #         rel_len = int(qtype.split('-')[0])
    #         offsets = query_max - query_min
    #         re_offsets, im_offsets = torch.chunk(offsets, 2, dim=-1)


    #         pi = 3.14159265358979323846

    #         re_query_center, im_query_center = torch.chunk(query_center, 2, dim=2)
    #         # a_tail, b_tail = torch.chunk(tail, 2, dim=2)

    #         if rel_len == 2:
    #             phase_time = relation[1]/(self.time_embedding_range.item()/pi)

    #             re_relation_t = torch.cos(phase_time)
    #             im_relation_t = torch.sin(phase_time)

    #             re_query_center_relation_t = re_query_center * re_relation_t - im_query_center * im_relation_t
    #             im_query_center_relation_t = re_query_center * im_relation_t + im_query_center * re_relation_t

    #             # query_center_time = torch.stack([a_relation, b_relation], dim = 0)
    #             # a_score = a_score - a_tail
    #             # b_score = b_score - b_tail

    #             re_offset_t, im_offset_t = torch.chunk(offset[1], 2, dim=-1)
    #             re_query_min_relation_t = re_query_center_relation_t - 0.5 * self.func(re_offset_t)
    #             im_query_min_relation_t = im_query_center_relation_t - 0.5 * self.func(im_offset_t)
    #             re_query_max_relation_t = re_query_center_relation_t + 0.5 * self.func(re_offset_t)
    #             im_query_max_relation_t = im_query_center_relation_t + 0.5 * self.func(im_offset_t)
                
    #             re_offsets_relation_t = re_query_max_relation_t - re_query_min_relation_t # all are positive numbers
    #             im_offsets_relation_t = im_query_max_relation_t - im_query_min_relation_t # all are positive numbers

    #             # query_center_time = query_center + relation[1] # s + t
    #             # if self.euo: ## inititalize the size of the box at the first stage using the subject
    #             #     query_min_time = query_center_time - 0.5 * self.func(head_offset)
    #             #     query_max_time = query_center_time + 0.5 * self.func(head_offset)
    #             # else:
    #             #     query_min_time = query_center_time
    #             #     query_max_time = query_center_time
    #             # query_min_time = query_min_time - 0.5 * self.func(offset[1])
    #             # query_max_time = query_max_time + 0.5 * self.func(offset[1])
    #             # offsets_time = query_max_time - query_min_time
    #             # if self.use_relation_time:# if this is true, then use another relation to enhance the expressivity
    #             #     query_center_relation_time = relation_time + relation[1] # r + t 
    #             # else:
    #             #     query_center_relation_time = relation[0] + relation[1] # r + t 
    #             # if self.euo: ## inititalize the size of the box at the first stage using the subject
    #             #     query_min_relation_time = query_center_relation_time - 0.5 * self.func(head_offset)
    #             #     query_max_relation_time = query_center_relation_time + 0.5 * self.func(head_offset)
    #             # else:
    #             #     query_min_relation_time = query_center_relation_time
    #             #     query_max_relation_time = query_center_relation_time
    #             # query_min_relation_time = query_min_relation_time - 0.5 * self.func(offset[1])
    #             # query_max_relation_time = query_max_relation_time + 0.5 * self.func(offset[1])
    #             # offsets_relation_time = query_max_relation_time - query_min_relation_time
    #             ## generate new center and offset

    #             re_new_query_center = self.re_center_sets(re_query_center.squeeze(1), re_offsets.squeeze(1), 
    #                                             re_query_center_relation_t.squeeze(1),re_offsets_relation_t.squeeze(1)).squeeze(1) 
    #             im_new_query_center = self.im_center_sets(im_query_center.squeeze(1), im_offsets.squeeze(1), 
    #                                             im_query_center_relation_t.squeeze(1),im_offsets_relation_t.squeeze(1)).squeeze(1)
    #             re_new_offset = self.re_offset_sets(re_query_center.squeeze(1), re_offsets.squeeze(1),
    #                                             re_query_center_relation_t.squeeze(1), re_offsets_relation_t.squeeze(1))
    #             im_new_offset = self.im_offset_sets(im_query_center.squeeze(1), im_offsets.squeeze(1),
    #                                             im_query_center_relation_t.squeeze(1), im_offsets_relation_t.squeeze(1))

    #             re_new_query_min = (re_new_query_center - 0.5*self.func(re_new_offset)).unsqueeze(1)
    #             re_new_query_max = (re_new_query_center + 0.5*self.func(re_new_offset)).unsqueeze(1)
    #             im_new_query_min = (im_new_query_center - 0.5*self.func(im_new_offset)).unsqueeze(1)
    #             im_new_query_max = (im_new_query_center + 0.5*self.func(im_new_offset)).unsqueeze(1)

    #             # re_score_offset = F.relu(re_new_query_min - re_tail) + F.relu(re_tail - re_new_query_max) # dist_outside
    #             # im_score_offset = F.relu(im_new_query_min - im_tail) + F.relu(im_tail - im_new_query_max) # dist_outside
    #             # score_offset = torch.stack([re_score_offset, im_score_offset], dim=0)
    #             new_query_min = torch.cat([re_new_query_min, im_new_query_min], dim=-1)
    #             new_query_max = torch.cat([re_new_query_max, im_new_query_max], dim=-1)

    #             score_offset = F.relu(new_query_min - tail) + F.relu(tail - new_query_max) # dist_outside
    #             score_offset =  torch.stack(torch.chunk(score_offset, 2, dim=-1), dim=0)
    #             ## 
    #             score_offset = score_offset.norm(dim = 0)

    #             re_tail, im_tail = torch.chunk(tail, 2, dim=-1)
    #             re_score_center = re_new_query_center.unsqueeze(1) - re_tail
    #             im_score_center = im_new_query_center.unsqueeze(1) - im_tail
    #             score_center = torch.stack([re_score_center, im_score_center], dim=0)
    #             score_center = score_center.norm(dim = 0)

    #             # re_score_center_plus = torch.min(re_new_query_max, torch.max(re_new_query_min, re_tail)) - re_new_query_center.unsqueeze(1) # dist_inside
    #             # im_score_center_plus = torch.min(im_new_query_max, torch.max(im_new_query_min, im_tail)) - im_new_query_center.unsqueeze(1) # dist_inside
    #             # score_center_plus = torch.stack([re_score_center_plus, im_score_center_plus], dim=0) # dist_inside
    #             # score_center_plus =  score_center_plus.norm(dim = 0) # dist_inside
    #             new_query_center = torch.cat([re_new_query_center, im_new_query_center], dim=-1)
    #             score_center_plus = torch.min(new_query_max, torch.max(new_query_min, tail)) - new_query_center.unsqueeze(1) # dist_inside
    #             score_center_plus =  torch.stack(torch.chunk(score_center_plus, 2, dim=-1), dim=0)
    #             score_center_plus =  score_center_plus.norm(dim = 0) # dist_inside

    #             score = self.gamma.item() - score_offset.norm(p=1, dim = -1)
    #             score_center = self.gamma2.item() - score_center.norm(p=1, dim = -1)  
    #             score_center_plus = self.gamma.item() - score_offset.norm(p=1, dim = -1) - self.cen *  score_center_plus.norm(p=1, dim = -1)  

    #             # new_query_min = (new_query_center - 0.5*self.func(new_offset)).unsqueeze(1)
    #             # new_query_max = (new_query_center + 0.5*self.func(new_offset)).unsqueeze(1)
    #             # score_offset = F.relu(new_query_min - tail) + F.relu(tail - new_query_max) # dist_outside
    #             # score_center = new_query_center.unsqueeze(1) - tail
    #             # score_center_plus = torch.min(new_query_max, torch.max(new_query_min, tail)) - new_query_center.unsqueeze(1) # dist_inside
                
    #             # score = self.gamma.item() - torch.norm(score_offset, p=1, dim=-1)  
    #             # score_center = self.gamma2.item() - torch.norm(score_center, p=1, dim=-1)  
    #             # score_center_plus = self.gamma.item() - torch.norm(score_offset, p=1, dim=-1) - self.cen * torch.norm(score_center_plus, p=1, dim=-1)
    #         elif rel_len == 3:
    #             print('Unknow qtype')
    #             raise NotImplementedError

    #     else:
    #         print('Unknow qtype')
    #         raise NotImplementedError

    #     return score, score_center, None, score_center_plus, None

    def BoxHyTE(self, head, relation, tail, mode, offset, head_offset, rel_len, qtype):
        if '1-chain' == qtype:
            query_min = head + relation[:,0,:,:] - 0.5 * self.func(offset[:,0,:,:])
            query_max = head + relation[:,0,:,:] + 0.5 * self.func(offset[:,0,:,:])
            score_offset = F.relu(query_min - tail) + F.relu(tail - query_max)
            score_center = head - tail
            score_center_plus = torch.min(query_max, torch.max(query_min, tail)) - head
            score_center_plus = self.gamma.item() - torch.norm(score_offset, p=1, dim=-1) - self.cen * torch.norm(score_center_plus, p=1, dim=-1)
        elif 'inter' in qtype:
            rel_len = int(qtype.split('-')[0])
            assert rel_len > 1
            head = torch.chunk(head, rel_len, dim=0)[0]
            relations = torch.chunk(relation.squeeze(1), rel_len, dim=0) # 2*[batch_size, ndim]
            offset = torch.chunk(offset.squeeze(1), rel_len, dim=0)[0]
            query_center = head + relations[0]
            
            if self.euo:
                query_min = query_center - 0.5 * self.func(head_offset)
                query_max = query_center + 0.5 * self.func(head_offset)
            else:
                query_min = query_center
                query_max = query_center
                
            tail = torch.chunk(tail, rel_len, dim=0)[0]
            query_min = query_min - 0.5 * self.func(offset)
            query_max = query_max + 0.5 * self.func(offset)

            score_offset = F.relu(query_min - tail) + F.relu(tail - query_max)
            score_center = head - tail
            score_center_plus_box = torch.min(query_max, torch.max(query_min, tail)) - head
            score_center_plus = self.gamma.item() - torch.norm(score_offset, p=1, dim=-1) - self.cen * torch.norm(score_center_plus_box, p=1, dim=-1)

            # queries_min = torch.chunk(query_min, rel_len, dim=0)
            # queries_max = torch.chunk(query_max, rel_len, dim=0)
             # [512, 128, 400]
            
            offsets = query_max - query_min
            # offsets = torch.chunk(offsets, rel_len, dim=0) 
            # if 'inter' in qtype:
                # relation  = torch.chunk(query_center, rel_len, dim=0)[0]

            batch_size = relations[0].size(0)
            ndim = relations[0].size(2)
            # try:
            #     assert batch_size == 1024
            # except:
            #     print(batch_size)
            # assert ndim == 400
            if rel_len == 2:
                time1 = relations[1]                
                head_t1 = head - torch.bmm(time1.view(batch_size, 1, ndim), head.view(batch_size, ndim, 1))*time1
                relation_t1 = relations[0] - torch.bmm(time1.view(batch_size, 1, ndim), relations[0].view(batch_size, ndim, 1))*time1
                tail_t1 = tail - torch.bmm(time1.view(batch_size, 1, ndim), tail.view(batch_size, ndim, -1)).view(batch_size, -1, 1)*time1 ##[b_s, 1, num_neg] *[b_s, 1, ndim]
                score_center_plus_t1 = head_t1  + relation_t1 - tail_t1
                score_center_plus += self.gamma.item() - torch.norm(score_center_plus_t1, p=1, dim=-1) - (1-torch.norm(time1, p=2, dim=-1))**2
                # new_query_center = self.center_sets(queries_center[0].squeeze(1), offsets[0].squeeze(1), 
                #                                 query_center_2, offsets[1].squeeze(1)) # use attention to get the new center
                # new_offset = self.offset_sets(queries_center[0].squeeze(1), offsets[0].squeeze(1),
                #                                 query_center_2, offsets[1].squeeze(1)) # use mean/min/max to aggregate information                        
            elif rel_len == 3:
                time2 = relations[2]
                head_t2 = head - torch.bmm(time2.view(batch_size, 1, ndim), head.view(batch_size, ndim, 1))*time2
                relation_t2 = relations[0] - torch.bmm(time2.view(batch_size, 1, ndim), relations[0].view(batch_size, ndim, 1))*time2
                tail_t2 = tail - torch.bmm(time2.view(batch_size, 1, ndim), tail.view(batch_size, ndim, -1)).view(batch_size, -1, 1)*time2
                score_center_plus_t2 = head_t2  + relation_t2 - tail_t2
                score_center_plus += self.gamma.item() - torch.norm(score_center_plus_t2, p=1, dim=-1) - (1-torch.norm(time2, p=2, dim=-1))**2

        return None, None, None, score_center_plus, None

    
    def TransE(self, head, relation, tail, mode, offset, head_offset, rel_len, qtype):

        if qtype == 'chain-inter':
            relations = torch.chunk(relation, 3, dim=0)
            heads = torch.chunk(head, 2, dim=0)
            score_1 = (heads[0] + relations[0][:,0,:,:] + relations[1][:,0,:,:]).squeeze(1)
            score_2 = (heads[1] + relations[2][:,0,:,:]).squeeze(1)
            conj_score = self.deepsets(score_1, None, score_2, None).unsqueeze(1)
            score = conj_score - tail
        elif qtype == 'inter-chain':
            relations = torch.chunk(relation, 3, dim=0)
            heads = torch.chunk(head, 2, dim=0)
            score_1 = (heads[0] + relations[0][:,0,:,:]).squeeze(1)
            score_2 = (heads[1] + relations[1][:,0,:,:]).squeeze(1)
            conj_score = self.deepsets(score_1, None, score_2, None).unsqueeze(1)
            score = conj_score + relations[2][:,0,:,:] - tail
        elif qtype == 'union-chain':
            relations = torch.chunk(relation, 3, dim=0)
            heads = torch.chunk(head, 2, dim=0)
            score_1 = heads[0] + relations[0][:,0,:,:] + relations[2][:,0,:,:]
            score_2 = heads[1] + relations[1][:,0,:,:] + relations[2][:,0,:,:]
            conj_score = torch.stack([score_1, score_2], dim=0)
            score = conj_score - tail
        else:
            score = head
            for rel in range(rel_len):
                score = score + relation[:,rel,:,:]

            if 'inter' not in qtype and 'union' not in qtype:
                score = score - tail
            else:
                rel_len = int(qtype.split('-')[0])
                assert rel_len > 1
                score = score.squeeze(1)
                scores = torch.chunk(score, rel_len, dim=0)
                tails = torch.chunk(tail, rel_len, dim=0)
                if 'inter' in qtype:
                    if rel_len == 2:
                        conj_score = self.deepsets(scores[0], None, scores[1], None)
                    elif rel_len == 3:
                        conj_score = self.deepsets(scores[0], None, scores[1], None, scores[2], None)
                    conj_score = conj_score.unsqueeze(1)
                    score = conj_score - tails[0]
                elif 'union' in qtype:
                    conj_score = torch.stack(scores, dim=0)
                    score = conj_score - tails[0]    
                else:
                    assert False, 'qtype not exist: %s'%qtype                    
        
        score = self.gamma.item() - torch.norm(score, p=1, dim=-1)
        if 'union' in qtype:
            score = torch.max(score, dim=0)[0]
        if qtype == '2-union':
            score = score.unsqueeze(0)
        return score, None, None, 0., []

    @staticmethod
    def train_step(model, optimizer, train_iterator, args, step, use_time=False):
        model.train()
        optimizer.zero_grad()
        # start = time.time()
        positive_sample, negative_sample, time_negative_sample, subsampling_weight, mode = next(train_iterator) ## get from the function of collate_fn
        # end = time.time()
        # print('time used in iterating %s -- %f' % (train_iterator.qtype, end - start))
        if args.uni_weight: ## do not consider the frequency information
            subsampling_weight = torch.ones_like(subsampling_weight)

        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()

        qtype = train_iterator.qtype
        rel_len = int(qtype.split('-')[0]) # specify the length of relations 
        
        ## Loss on negative samples
        negative_score, negative_score_cen, negative_offset, negative_score_cen_plus, _, _ = model((positive_sample, negative_sample), rel_len, qtype, mode="tail-batch", use_relation_time=args.use_relation_time)
        # if mode == 'time-batch':
            # print(negative_score_cen_plus)
            # print(positive_sample)
            # print(negative_sample)
            # print(qtype)

        if model.geo == 'box' or model.geo == 'circle': #sigmoid --> +/-infinity to positive
            negative_score = F.logsigmoid(-negative_score_cen_plus).mean(dim = 1)
            # print('negative_score', negative_score)
        else:
            negative_score = F.logsigmoid(-negative_score).mean(dim = 1)

        positive_score, positive_score_cen, positive_offset, positive_score_cen_plus, _, time_score_reg = model(positive_sample, rel_len, qtype, use_relation_time=args.use_relation_time)
        if model.geo == 'box' or  model.geo == 'circle':
            positive_score = F.logsigmoid(positive_score_cen_plus).squeeze(dim = 1)
            # print('positive_score', positive_score)
        else:
            positive_score = F.logsigmoid(positive_score).squeeze(dim = 1)

        # assert time_negative_sample !=None
        if 'time-batch' in args.negative_sample_types and time_negative_sample != None: 
            if  args.cuda: ## exclude 1c
                time_negative_sample = time_negative_sample.cuda()
            negative_score_time, negative_score_cen_time, negative_offset_time, negative_score_cen_plus_time, _, _ = model((positive_sample, time_negative_sample), rel_len, qtype, mode="time-batch", use_relation_time=args.use_relation_time)
            # negative_score_cen_plus = torch.cat ([negative_score_cen_plus, negative_score_cen_plus_time*args.time_score_weight], dim=1)
            
            negative_score += args.time_score_weight * F.logsigmoid(-negative_score_cen_plus_time).mean(dim = 1)
            # positive_score = (1+args.time_score_weight)*positive_score


        # print('positive_score', positive_fscore)
        if args.uni_weight:
            positive_sample_loss = - positive_score.mean()
            negative_sample_loss = - negative_score.mean()
        else:
            positive_sample_loss = - (subsampling_weight * positive_score).sum()
            negative_sample_loss = - (subsampling_weight * negative_score).sum()
            positive_sample_loss /= subsampling_weight.sum()
            negative_sample_loss /= subsampling_weight.sum()
            # negative_sample_loss = - negative_score.mean()

        loss = (positive_sample_loss + negative_sample_loss)/2
        # print('loss', subsampling_weight)
        
        if args.regularization != 0.0:
            regularization = args.regularization * 1/3.0*(
                model.entity_embedding.norm(p = 3)**3 + 
                model.relation_embedding.norm(p = 3).norm(p = 3)**3
            )
            ## if no atemporal statement, use 
            if use_time:
                regularization += args.regularization * 1/3.0 * model.time_embedding.norm(p = 3)**3
            if args.use_relation_time:
                regularization += args.regularization * 1/3.0 * model.relation_time_embedding.norm(p = 3)**3
            loss = loss + regularization
            regularization_log = {'regularization': regularization.item()}
        else:
            regularization_log = {}

        ## deal with time smoothing 
        if  time_score_reg != None:
                l_time = time_score_reg.mean(dim = 0)*args.time_smooth_weight
                loss += l_time
                time_regularization_log = {'time_regularization': l_time.item()}
        else:
            if model.time_reg != None:
            # time_embedding = torch.cat([torch.cos(model.time_embedding), torch.sin(model.time_embedding)], dim=-1)
            # assert time_embedding.size(0)==2002
            # assert time_embedding.size(1)==400
                l_time = model.time_reg.forward(model.time_embedding[1:]-model.time_embedding[:-1])
                loss = loss + l_time
                time_regularization_log = {'time_regularization': l_time.item()}
            else:
                time_regularization_log = {}
           
        loss.backward()
        optimizer.step()
        log = {
            **regularization_log,
            **time_regularization_log,
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item()
        }
        return log, positive_sample.size(0)
    
    @staticmethod
    def test_step(model, test_triples, test_ans, test_ans_hard, args, file_tag='', ans_t=None):
        qtype = test_triples[0][-1] # test_ans and test_ans_hard; the second is the exact result. ## three types - 1-chain, 2-inter, 3-inter (2i --> only-end, only-begin and point-in-time
        time_type = test_triples[0][2] # five types - point-in-time, only-begin, only-end, full-interval, no time

        # if qtype == 'chain-inter' or qtype == 'inter-chain' or qtype == 'union-chain':
        #     rel_len = 2
        # else:
        rel_len = int(test_triples[0][-1].split('-')[0])
        
        model.eval()
        
        if 'inter' in qtype: # here, deal with three situations:  2i, 3i-2i, 3i
            if args.predict_o:
                mode = 'tail-batch'
            if args.predict_t:
                mode = 'time-batch'
                assert file_tag != ''
                assert ans_t != None
            test_dataloader_tail = DataLoader(
                TestInterDataset(
                    test_triples, 
                    test_ans, 
                    test_ans_hard,
                    args.nentity, 
                    args.nrelation, 
                    args.ntimestamp,
                    mode,
                    use_one_sample = args.use_one_sample if qtype == '3-inter' else False, 
                    use_two_sample = args.use_two_sample,
                    double_point_in_time = args.double_point_in_time if  time_type == TIME_TYPE['point-in-time'] else False, ## only when point_in_time is considered, the parammeter will be triggered.
                    enumerate_time = args.enumerate_time,
                    predict_o = args.predict_o, 
                    predict_t = args.predict_t, 
                    predict_r = args.predict_r,
                ), 
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num), 
                collate_fn=TestDataset.collate_fn
            )
            ## deal with time interval; by default, rel_len for 3-inter is 3
            if args.enumerate_time:
                qtype = '2-inter'
                rel_len = 2
            elif args.use_one_sample and qtype =='3-inter': # if use_one_sample is true, then for 3-inter, just sample one from the interval.
                qtype = '2-inter' 
                rel_len = 2

            if (time_type == TIME_TYPE['point-in-time']) and args.double_point_in_time: # if use_one_sample is true, then for 3-inter, just sample one from the interval.
                qtype = '2-3-inter' 

        else:
            if args.predict_o:
                test_dataloader_tail = DataLoader(
                    TestDataset(
                        test_triples, 
                        test_ans, 
                        test_ans_hard,
                        args.nentity, 
                        args.nrelation, 
                        'tail-batch'
                    ), 
                    batch_size=args.test_batch_size,
                    num_workers=max(1, args.cpu_num), 
                    collate_fn=TestDataset.collate_fn
                )

        test_dataset_list = [test_dataloader_tail]
        # test_dataset_list = [test_dataloader_head, test_dataloader_tail]
        step = 0
        total_steps = sum([len(dataset) for dataset in test_dataset_list])
        logs = []

        save_rank_result = []

        with torch.no_grad():
            for test_dataset in test_dataset_list:
                for positive_sample, negative_sample, mode, queries in test_dataset: # positive_samples contain [head, relation, time, gold_tail]
                    org_positive = positive_sample

                    num = positive_sample.size(-1)
                    positive_sample = positive_sample.view(-1, num)
                    size = positive_sample.size(0)

                    gold_ans = [positive_sample[0, -1].item()]

                    if args.cuda:
                        positive_sample = positive_sample.cuda()
                        negative_sample = negative_sample.cuda()

                    if 'inter' in qtype:
                        # if args.cuda:
                        #     positive_sample = positive_sample.cuda()
                        #     negative_sample = negative_sample.cuda()
                        if mode == 'tail-batch':
                            score_cen_plus_list = []

                            if model.geo == 'box' or model.geo == 'circle':
                                mini_batch_size = 2
                                for i in range(0, int(size), mini_batch_size):
                                    start = i
                                    end = min(i + mini_batch_size, size)
                                    sub_positive_sample  = positive_sample[start:end]
                                    _, score_cen, _, score_cen_plus, _, _ = model((sub_positive_sample, negative_sample), rel_len, qtype, mode=mode, use_relation_time=args.use_relation_time)
                                    score_cen_plus_list.append(score_cen_plus)
                                score_cen_plus = torch.cat(score_cen_plus_list)
                                ## when enumerating times, the shape of score_cen_plus is num_timestamps in an interval * num_entitites
                            else:
                                score, score_cen, _, score_cen_plus, _, _ = model((positive_sample, negative_sample), rel_len, qtype, mode=mode, use_relation_time=args.use_relation_time)
                        elif mode == 'time-batch':
                             _, score_cen, _, score_cen_plus, _, _ = model((positive_sample, negative_sample), rel_len, qtype, mode=mode, use_relation_time=args.use_relation_time) ## assume batch == 1, then the output would be [1, ntimestamps] 

                    else:
                        score, score_cen, _, score_cen_plus, _, _ = model((positive_sample, negative_sample), rel_len, qtype, mode=mode, use_relation_time=args.use_relation_time)

                    if step % args.test_log_steps == 0:
                        logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                    step += 1

                    tmp_metrics = {
                        #'lossm':torch.mean(score_cen_plus).item(),
                        'MRRm_new': 0.0,
                        'MRm_new': 0.0,
                        'HITS_1m_new':  0.0,
                        'HITS_3m_new':  0.0,
                        'HITS_10m_new':  0.0,
                        'mean_HITS_1m_new':  0.0,
                        'mean_HITS_3m_new':  0.0,
                        'mean_HITS_10m_new':  0.0,
                        'max_HITS_1m_new':  0.0,
                        'max_HITS_3m_new':  0.0,
                        'max_HITS_10m_new':  0.0,
                    }
                    if mode == 'time-batch': ## only for test dataset
                        ## when it is point-in-time
                        if '2i' in file_tag:
                            all_idx = set(range(args.ntimestamp))
                            ans = ans_t[(org_positive[0,0].item(), org_positive[0,1].item(), org_positive[0,-1].item())] ## all possible timestamps
                            ans = ans - {org_positive[0,2].item()}
                            false_ans = list(all_idx - ans) 
                            vals = np.zeros((1, args.ntimestamp))
                            vals[0, np.array(false_ans)] = 1
                            b = torch.Tensor(vals) if not args.cuda else torch.Tensor(vals).cuda()
                            gold_ans_tensor = org_positive[0,2]

                            score2 = score_cen_plus
                            score2 -= (torch.min(score_cen_plus) - 1)
                            filter_score2 = b*score_cen_plus
                            
                            argsort2 = torch.argsort(filter_score2, dim=1, descending=True)
                            argsort2 = argsort2.view(1, args.ntimestamp)

                            ## save to disk
                            query_result = [(org_positive[0,0].item(), org_positive[0,1].item(), org_positive[0,-1].item(), org_positive[0,2].item(), org_positive[0,2].item()), ans_t[(org_positive[0,0].item(), org_positive[0,1].item(), org_positive[0,-1].item())], argsort2.detach().cpu().numpy().tolist(), score2]
                            save_rank_result.append(query_result)

                            argsort2 = argsort2 - gold_ans_tensor
                            ranking2 = (argsort2 == 0).nonzero(as_tuple=False) ## nonzero is used to change bool into int type
                            ranking2 = ranking2[:, 1]
                            ranking2 = ranking2 + 1

                            mrrm_newd = (1./ranking2.to(torch.float)).item()
                            mrm_newd = ranking2.to(torch.float).item()

                            ## hard hit@1, hit@3, hit@10
                            hits1m_newd = torch.mean((ranking2 <= 1).to(torch.float)).item()
                            hits3m_newd = torch.mean((ranking2 <= 3).to(torch.float)).item()
                            hits10m_newd = torch.mean((ranking2 <= 10).to(torch.float)).item()

                            ## soft hit@1, hit@3, hit@10
                            if file_tag in ['2i-begin', '2i-end'] and args.flag_use_weighted_partial_interval:
                                ## mask the previous ones and use square to penalize that ranking
                                mask = torch.zeros(args.ntimestamp,dtype=torch.float32).view(1,-1)
                                less = (argsort2<0).nonzero(as_tuple=False) ## find the index that are greater than zero; years before golden year
                                # print(less)
                                # print(mask[:10])
                                mask[0, less] = 1 ## preceding years
                                if args.cuda:
                                    mask = mask.cuda()

                                argsort2 = (torch.abs(argsort2)+1)
                                if '2i-begin':
                                    argsort3 = (argsort2*mask)**2 + (argsort2*(1-mask))
                                elif file_tag == '2i-end':
                                    argsort3 = argsort2*(1-mask)**2 + (argsort2*mask)
                            else:
                                argsort3 = (torch.abs(argsort2)+1)

                            mean_hits1m_newd = torch.mean(1./argsort3[0, :1]).to(torch.float).item()
                            mean_hits3m_newd = torch.mean(1./argsort3[0, :3]).to(torch.float).item()
                            mean_hits10m_newd = torch.mean(1./argsort3[0, :10]).to(torch.float).item()

                            max_hits1m_newd = torch.max(1./argsort3[0, :1]).to(torch.float).item()
                            max_hits3m_newd = torch.max(1./argsort3[0, :3]).to(torch.float).item()
                            max_hits10m_newd = torch.max(1./argsort3[0, :10]).to(torch.float).item()

                            ## save to logs
                            tmp_metrics = {
                                #'lossm':torch.mean(score_cen_plus).item(),
                                'MRRm_new': mrrm_newd,
                                'MRm_new': mrm_newd,
                                'HITS_1m_new': hits1m_newd,
                                'HITS_3m_new': hits3m_newd,
                                'HITS_10m_new': hits10m_newd,
                                'mean_HITS_1m_new': mean_hits1m_newd,
                                'mean_HITS_3m_new': mean_hits3m_newd,
                                'mean_HITS_10m_new': mean_hits10m_newd,
                                'max_HITS_1m_new': max_hits1m_newd,
                                'max_HITS_3m_new': max_hits3m_newd,
                                'max_HITS_10m_new': max_hits10m_newd,
                            }

                        elif '3i' in file_tag:
                            score2 = score_cen_plus
                            score2 -= (torch.min(score_cen_plus) - 1)
                            argsort2 = torch.argsort(score2, dim=1, descending=True)
                            ## save to disk
                            query_result = [(org_positive[0,0].item(), org_positive[0,1].item(), org_positive[0,-1].item(), org_positive[0,2].item(), org_positive[0,3].item()), ans_t[(org_positive[0, 0].item(), org_positive[0, 1].item(), org_positive[0, -1].item())], argsort2.detach().cpu().numpy().tolist(), score2]
                            save_rank_result.append(query_result)

                        logs.append(tmp_metrics)

                        
                    elif mode == 'tail-batch':
                        if model.geo == 'box' or model.geo == 'circle':
                            score = score_cen
                            score2 = score_cen_plus

                        ## filtered mrr
                        if isinstance(queries, tuple): ## if queries is a type of list, then it must be in enumerate-time mode.
                            queries = [queries]
                            scores2 = [score2]
                        else:
                             scores2 = score2
                        
                        # print(gold_ans)

                        ## enumerate all the times and store in a list 
                        ## this is the gold tail for this specific query
                        result_enumerate = {'MRm_new':[], 'MRRm_new':[], 'HITS_1m_new':[], 'HITS_3m_new':[], 'HITS_10m_new':[]}
                        for i  in range(len(queries)):
                            query = queries[i]
                            score2 = scores2[i]
                            ans = test_ans[query]
                            # print('ans', ans)
                            # hard_ans = test_ans_hard[query]
                            hard_ans = gold_ans
                            # print('hard_ans', hard_ans)
                            all_idx = set(range(args.nentity))
                            false_ans = all_idx - ans
                            ans_list = list(ans)
                            hard_ans_list = list(hard_ans)
                            false_ans_list = list(false_ans)
                            ans_idxs = np.array(hard_ans_list)
                            vals = np.zeros((len(ans_idxs), args.nentity))
                            vals[np.arange(len(ans_idxs)), ans_idxs] = 1
                            axis2 = np.tile(false_ans_list, len(ans_idxs))
                            axis1 = np.repeat(range(len(ans_idxs)), len(false_ans))
                            vals[axis1, axis2] = 1 # valid entities = those that are not in test_ans + current_gold_ans
                            b = torch.Tensor(vals) if not args.cuda else torch.Tensor(vals).cuda()
                            ans_tensor = torch.LongTensor(hard_ans_list) if not args.cuda else torch.LongTensor(hard_ans_list).cuda()
                            if model.geo == 'box':
                                score2 -= (torch.min(score2) - 1)
                                filter_score2 = b*score2
                                argsort2 = torch.argsort(filter_score2, dim=1, descending=True)

                                

                                argsort3 = torch.transpose(torch.transpose(argsort2, 0, 1) - ans_tensor, 0, 1)
                                ranking2 = (argsort3 == 0).nonzero(as_tuple=False) ## nonzero is used to change bool into int type
                                ranking2 = ranking2[:, 1]
                                ranking2 = ranking2 + 1

                                hits1m_newd = torch.mean((ranking2 <= 1).to(torch.float)).item()
                                hits3m_newd = torch.mean((ranking2 <= 3).to(torch.float)).item()
                                hits10m_newd = torch.mean((ranking2 <= 10).to(torch.float)).item()
                                mrm_newd = torch.mean(ranking2.to(torch.float)).item()
                                mrrm_newd = torch.mean(1./ranking2.to(torch.float)).item()

                                ## save the results into disk
                                if file_tag != '':
                                    query_result = []
                                    for x,y in zip(hard_ans_list, argsort2.detach().cpu().numpy().tolist()):
                                        query_result.append([query, x, y[:10], mrrm_newd])
                                    assert len(query_result) == 1
                                    save_rank_result.append(query_result)

                                result_enumerate['MRm_new'].append(mrm_newd)
                                result_enumerate['MRRm_new'].append(mrrm_newd)
                                result_enumerate['HITS_1m_new'].append(hits1m_newd)
                                result_enumerate['HITS_3m_new'].append(hits3m_newd)
                                result_enumerate['HITS_10m_new'].append(hits10m_newd)
                            else:
                                raise NotImplementedError
                            # elif model.geo == 'vec':
                            #     score -= (torch.min(score) - 1)
                            #     filter_score = b*score
                            #     argsort = torch.argsort(filter_score, dim=1, descending=True)
                            #     argsort = torch.transpose(torch.transpose(argsort, 0, 1) - ans_tensor, 0, 1) 
                            #     ranking = (argsort == 0).nonzero() ## find the position of the current hard_ans
                            #     ranking = ranking[:, 1]
                            #     ranking = ranking + 1

                            #     ans_vec = np.zeros(args.nentity)
                            #     ans_vec[ans_list] = 1
                            #     hits1 = torch.sum((ranking <= 1).to(torch.float)).item()
                            #     hits3 = torch.sum((ranking <= 3).to(torch.float)).item()
                            #     hits10 = torch.sum((ranking <= 10).to(torch.float)).item()
                            #     mr = float(torch.sum(ranking).item())
                            #     mrr = torch.sum(1./ranking.to(torch.float)).item()
                            #     hits1m = torch.mean((ranking <= 1).to(torch.float)).item()
                            #     hits3m = torch.mean((ranking <= 3).to(torch.float)).item()
                            #     hits10m = torch.mean((ranking <= 10).to(torch.float)).item()
                            #     mrm = torch.mean(ranking.to(torch.float)).item()
                            #     mrrm = torch.mean(1./ranking.to(torch.float)).item()
                                
                            
                            #     hits1m_newd = hits1m
                            #     hits3m_newd = hits3m
                            #     hits10m_newd = hits10m
                            #     mrm_newd = mrm
                            #     mrrm_newd = mrrm
                        # num_ans = len(hard_ans_list)
                        ## average ranking results for each sample
                        mrm_newd = np.mean(result_enumerate['MRm_new'])
                        mrrm_newd = np.mean(result_enumerate['MRRm_new'])
                        hits1m_newd = np.mean(result_enumerate['HITS_1m_new'])
                        hits3m_newd = np.mean(result_enumerate['HITS_3m_new'])
                        hits10m_newd = np.mean(result_enumerate['HITS_10m_new'])

                        logs.append({
                            #'lossm':torch.mean(score_cen_plus).item(),
                            'MRRm_new': mrrm_newd,
                            'MRm_new': mrm_newd,
                            'HITS_1m_new': hits1m_newd,
                            'HITS_3m_new': hits3m_newd,
                            'HITS_10m_new': hits10m_newd,
                            # 'time_type': time_type
                            # 'num_answer': num_ans
                        })



        if file_tag != '':
            ## save rank disk for analysis
            with open(os.path.join(args.save_path, 'rank_result', file_tag+'_rank%s.pkl' % ('_time' if mode == 'time-batch' else '')), 'wb') as wf:
                pickle.dump(save_rank_result, wf)

        # if mode == 'tail-batch':
        metrics = {}
        # num_answer = sum([log['num_answer'] for log in logs])
        for metric in logs[0].keys():
            # if metric == 'num_answer':
            #     continue
            # if 'm' in metric:
            metrics[metric] = sum([log[metric] for log in logs])/len(logs)
            # else:
            #     metrics[metric] = sum([log[metric] for log in logs])/num_answer
        return metrics
        # else:
        #     return None