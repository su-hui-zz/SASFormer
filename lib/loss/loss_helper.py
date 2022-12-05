##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Copyright (c) 2022 megvii-model. All Rights Reserved.
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Donny You, RainbowSecret
## Microsoft Research
## yuyua@microsoft.com
## Copyright (c) 2019
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import pdb
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from lib.utils.tools.logger import Logger as Log
from lib.loss.rmi_loss import RMILoss

from kernels.lib_tree_filter.modules.tree_filter import MinimumSpanningTree
from kernels.lib_tree_filter.modules.tree_filter import TreeFilter2D

import pdb

## new
class TreeEnergyLoss(nn.Module):
    def __init__(self, configer=None):
        super(TreeEnergyLoss, self).__init__()
        self.configer = configer
        if self.configer is None:
            print("self.configer is None")

        self.weight = self.configer.get('tree_loss', 'params')['weight']
        self.mst_layers = MinimumSpanningTree(TreeFilter2D.norm2_distance)
        self.tree_filter_layers = TreeFilter2D(groups=1, sigma=self.configer.get('tree_loss', 'sigma'))

    def forward(self, preds, low_feats, high_feats, unlabeled_ROIs):  # preds-[bz,21,32,32], low_feats-[bz,3,512,512], high_feats-[bz,768,32,32]
        # scale low_feats via high_feats
        with torch.no_grad():
            batch, _, h, w = preds.size()
            low_feats = F.interpolate(low_feats, size=(h, w), mode='bilinear', align_corners=False)
            unlabeled_ROIs = F.interpolate(unlabeled_ROIs.unsqueeze(1).float(), size=(h, w), mode='nearest')
            N = unlabeled_ROIs.sum()

        prob = torch.softmax(preds, dim=1) # prob-[bz,21,32,32],

        # low-level MST
        tree = self.mst_layers(low_feats)
        AS = self.tree_filter_layers(feature_in=prob, embed_in=low_feats, tree=tree)  # [b, n, h, w]

        # high-level MST
        if high_feats is not None:
            tree = self.mst_layers(high_feats)
            AS = self.tree_filter_layers(feature_in=AS, embed_in=high_feats, tree=tree, low_tree=False)  # [b, n, h, w]

        tree_loss = (unlabeled_ROIs * torch.abs(prob - AS)).sum()
        if N > 0:
            tree_loss /= N

        return self.weight * tree_loss


class AffinityEnergyLoss(nn.Module):
    def __init__(self, configer=None):
        super(AffinityEnergyLoss, self).__init__()
        self.configer = configer
        if self.configer is None:
            print("self.configer is None")

        self.weight = self.configer.get('affinity_loss', 'params')['weight']

        # 0 - only encode_affinity_loss; 1 - only decoder_affinity_loss; 2 - all_affinity_loss; 
        self.loss_index = self.configer.get('affinity_loss', 'params')['loss_index']  

        # 0-no gt target infos; 1-add gt target infos
        self.gt_add = self.configer.get('affinity_loss', 'params')['gt_add']  
        
        #self.mst_layers = MinimumSpanningTree(TreeFilter2D.norm2_distance)
        #self.tree_filter_layers = TreeFilter2D(groups=1, sigma=self.configer.get('tree_loss', 'sigma'))

    def forward(self, preds, low_feats, high_feats, unlabeled_ROIs, targets, attns, decode_attns):  # preds-[bz,21,32,32], low_feats-[bz,3,512,512], high_feats-[bz,768,32,32]
        # encode attns  [bz, head_num=12, 1+32*32, 1+32*32]
        bz, head_n, total_n, _  = attns[0].size()
        token_n         = total_n - 1
        encode_attn_avg = torch.zeros(bz, token_n, token_n, dtype=attns[0].dtype, device=attns[0].device)
        new_encode_attns    = []  # [bz, 1024, 1024]
        for attn in attns:
            attn = attn[:,:,1:,1:]
            attn = attn / attn.sum(dim=-1,keepdim=True)
            attn = attn.mean(dim=1)
            new_encode_attns.append(attn)
            encode_attn_avg += attn
        encode_attn_avg = encode_attn_avg / len(attns)
        
        # decode attns  [bz, head_num=12, 32*32+class_num, 32*32+class_num] (decode 中 class_embed 放在 patch embed 后面)
        decode_attn_avg = torch.zeros(bz, token_n, token_n, dtype=attns[0].dtype, device=attns[0].device)
        new_decode_attns    = []  # [bz, 1024, 1024]
        for attn in decode_attns:
            attn = attn[:,:,:(token_n),:(token_n)] 
            attn = attn / attn.sum(dim=-1,keepdim=True) 
            attn = attn.mean(dim=1)
            new_decode_attns.append(attn)
            decode_attn_avg += attn
        decode_attn_avg = decode_attn_avg / len(decode_attns)  

        # soft affinity probability 
        _, class_num, h, w = preds.size()
        prob = torch.softmax(preds, dim=1)                           # prob-[bz,21,32,32]
        prob            = prob.view(bz,class_num,-1).permute(0,2,1)  # [bz, 1024, 21]
        prob_map        = prob.unsqueeze(1).expand(bz,token_n, token_n, class_num)
        
        if self.loss_index == 0:
            attn_map = encode_attn_avg.unsqueeze(-1).expand(bz, token_n, token_n, class_num)
        elif self.loss_index == 1:
            attn_map = decode_attn_avg.unsqueeze(-1).expand(bz, token_n, token_n, class_num)
        else:
            total_attn_avg = (encode_attn_avg * len(attns) + decode_attn_avg * len(decode_attns)) / (len(attns) + len(decode_attns))
            attn_map =  total_attn_avg.unsqueeze(-1).expand(bz, token_n, token_n, class_num)

        affinity_prob   = attn_map * prob_map   # [bz, 1024, 1024, 21]
        affinity_prob   = affinity_prob.sum(dim=2)
        affinity_prob   = affinity_prob / affinity_prob.sum(dim=-1,keepdim=True)  # [bz, 1024, 21]

        # affinity loss
        with torch.no_grad():
            unlabeled_ROIs = F.interpolate(unlabeled_ROIs.unsqueeze(1).float(), size=(h, w), mode='nearest')  # [bz, 1, 32, 32]
            unlabeled_ROIs = unlabeled_ROIs.view(bz, -1).unsqueeze(-1)
            N = unlabeled_ROIs.sum()

        if self.gt_add == 0:
            affinity_loss = (unlabeled_ROIs * torch.abs(prob - affinity_prob)).sum() 
            if N > 0:
                affinity_loss = affinity_loss / N
        else:  
            # add gt target infos ( 对有 gt target infos的pixel, affinity*probs = gt target info )
            with torch.no_grad():
                targets = F.interpolate(targets.unsqueeze(1).float(), size=(h, w), mode='nearest').to(prob.device)  # [bz, 1, 32, 32], 待查看resize后的像素值是否正常
                targets = targets.view(bz, -1).long().to(prob.device)   # [bz, 1024]
                
                labeled_ROIs = targets>=0   # [bz, 1024]
                Nl           = labeled_ROIs.sum()

                targets[targets<0] = 0
                targets_onehot = torch.zeros(bz, class_num, token_n, device=prob.device)
                targets_onehot = targets_onehot.scatter(1, targets.unsqueeze(dim=1), 1).permute(0,2,1)  # [bz, 1024, 21]

            labeled_ROIs = labeled_ROIs.unsqueeze(dim=-1).long()
            affinity_prob = affinity_prob * (1-labeled_ROIs) + targets_onehot * labeled_ROIs
            total_diff    = torch.abs(prob - affinity_prob)
            affinity_loss = (unlabeled_ROIs * total_diff).sum() + (labeled_ROIs * total_diff).sum()

            if N+Nl>0:
                affinity_loss = affinity_loss / (N+Nl)

        return self.weight * affinity_loss


class SegformerAffinityEnergyLoss(nn.Module):
    def __init__(self, configer=None):
        super(SegformerAffinityEnergyLoss, self).__init__()
        self.configer = configer
        if self.configer is None:
            print("self.configer is None")

        self.weight = self.configer.get('affinity_loss', 'params')['weight']
        self.class_num = self.configer.get('data', 'num_classes')
        # 0,1,2,3
        self.loss_index = self.configer.get('affinity_loss', 'params')['loss_index']  

        #self.mst_layers = MinimumSpanningTree(TreeFilter2D.norm2_distance)
        #self.tree_filter_layers = TreeFilter2D(groups=1, sigma=self.configer.get('tree_loss', 'sigma'))

    # [bz,21,128,128], [bz,21,16,16], [bz,21,32,32], [bz,21,64,64], _attns- a list of 4
    def forward(self, outputs, low_feats, unlabeled_ROIs, targets): 
        seg, seg_16, seg_32, seg_64, attns = outputs
        if self.loss_index == -1:
            return torch.zeros(1, dtype=seg.dtype, device=seg.device)
        
        # attn_avg1  [bz, 128*128, 16*16]
        bz, _, token_b1_n1, token_b1_n2  = attns[0][0].size()
        attn_avg1 = torch.zeros(bz, token_b1_n1, token_b1_n2, dtype=seg.dtype, device=seg.device)
        for attn in attns[0]:
            attn = attn.mean(dim=1)
            attn = attn / attn.sum(dim=-1,keepdim=True)
            attn_avg1 += attn
        attn_avg1 = attn_avg1 / len(attns[0])
        
        # attn_avg2 [bz, 64*64, 16*16]
        bz, _, token_b2_n1, token_b2_n2 = attns[1][0].size()
        attn_avg2 = torch.zeros(bz, token_b2_n1, token_b2_n2, dtype=seg.dtype, device=seg.device)
        for attn in attns[1]:
            attn = attn.mean(dim=1)
            attn = attn / attn.sum(dim=-1,keepdim=True)
            attn_avg2 += attn
        attn_avg2 = attn_avg2 / len(attns[1])

        # attn_avg3 [bz, 32*32, 16*16]
        bz, _, token_b3_n1, token_b3_n2 = attns[2][0].size()
        attn_avg3 = torch.zeros(bz, token_b3_n1, token_b3_n2, dtype=seg.dtype, device=seg.device)
        for attn in attns[2]:
            attn = attn.mean(dim=1)
            attn = attn / attn.sum(dim=-1,keepdim=True)
            attn_avg3 += attn
        attn_avg3 = attn_avg3 / len(attns[2]) 

        # attn_avg4 [bz, 32*32, 32*32]
        bz, _, token_b4_n1, token_b4_n2 = attns[3][0].size()
        attn_avg4 = torch.zeros(bz, token_b4_n1, token_b4_n2, dtype=seg.dtype, device=seg.device)
        for attn in attns[3]:
            attn = attn.mean(dim=1)
            attn = attn / attn.sum(dim=-1,keepdim=True)
            attn_avg4 += attn
        attn_avg4 = attn_avg4 / len(attns[3])     

        # soft affinity probability 
        _, _, h128,w128    = seg.size()
        prob128            = torch.softmax(seg, dim=1)            # prob-[bz,21,128,128]
        prob128            = prob128.view(bz,self.class_num,-1).permute(0,2,1)  # [bz, 128*128, 21]
        prob128_softmax    = torch.softmax(prob128, dim=-1)

        _, _, h16,w16      = seg_16.size()
        prob16             = torch.softmax(seg_16, dim=1)           
        prob16             = prob16.view(bz,self.class_num,-1).permute(0,2,1)  
        prob16_softmax     = torch.softmax(prob16, dim=-1)

        _, _, h32,w32      = seg_32.size()
        prob32             = torch.softmax(seg_32, dim=1)            
        prob32             = prob32.view(bz,self.class_num,-1).permute(0,2,1)  
        prob32_softmax     = torch.softmax(prob32, dim=-1)

        _, _, h64,w64      = seg_64.size()
        prob64             = torch.softmax(seg_64, dim=1)            
        prob64             = prob64.view(bz,self.class_num,-1).permute(0,2,1)  
        prob64_softmax     = torch.softmax(prob64, dim=-1)

        # loss
        # affinity_loss1     = torch.abs(torch.matmul(attn_avg1, prob16) - prob128)  # [bz, 128*128, 21]
        # affinity_loss2     = torch.abs(torch.matmul(attn_avg2, prob16) - prob64)
        # affinity_loss3     = torch.abs(torch.matmul(attn_avg3, prob16) - prob32)
        # affinity_loss4     = torch.abs(torch.matmul(attn_avg4, prob32) - prob32)

        affinity_loss1     = torch.abs(torch.softmax(torch.matmul(attn_avg1, prob16),dim=-1) - prob128_softmax)  # [bz, 128*128, 21]
        affinity_loss2     = torch.abs(torch.softmax(torch.matmul(attn_avg2, prob16),dim=-1) - prob64_softmax)
        affinity_loss3     = torch.abs(torch.softmax(torch.matmul(attn_avg3, prob16),dim=-1) - prob32_softmax)
        affinity_loss4     = torch.abs(torch.softmax(torch.matmul(attn_avg4, prob32),dim=-1) - prob32_softmax)

        # affinity_loss1     = F.kl_div(F.log_softmax(torch.matmul(attn_avg1, prob16),dim=-1) , prob128_softmax)  # [bz, 128*128, 21]
        # affinity_loss2     = F.kl_div(F.log_softmax(torch.matmul(attn_avg2, prob16),dim=-1) , prob64_softmax)
        # affinity_loss3     = F.kl_div(F.log_softmax(torch.matmul(attn_avg3, prob16),dim=-1) , prob32_softmax)
        # affinity_loss4     = F.kl_div(F.log_softmax(torch.matmul(attn_avg4, prob32),dim=-1) , prob32_softmax)

        # affinity loss number
        with torch.no_grad():
            unlabeled_ROIs128 = F.interpolate(unlabeled_ROIs.unsqueeze(1).float(), size=(h128, w128), mode='nearest')  # [bz, 1, 128, 128]
            unlabeled_ROIs128 = unlabeled_ROIs128.view(bz, -1).unsqueeze(-1)
            N128 = unlabeled_ROIs128.sum()

            # unlabeled_ROIs16 = F.interpolate(unlabeled_ROIs.unsqueeze(1).float(), size=(h16, w16), mode='nearest')  # [bz, 1, 16, 16]
            # unlabeled_ROIs16 = unlabeled_ROIs16.view(bz, -1).unsqueeze(-1)
            # N16 = unlabeled_ROIs16.sum()

            unlabeled_ROIs32 = F.interpolate(unlabeled_ROIs.unsqueeze(1).float(), size=(h32, w32), mode='nearest')  # [bz, 1, 16, 16]
            unlabeled_ROIs32 = unlabeled_ROIs32.view(bz, -1).unsqueeze(-1)
            N32 = unlabeled_ROIs32.sum()

            unlabeled_ROIs64 = F.interpolate(unlabeled_ROIs.unsqueeze(1).float(), size=(h64, w64), mode='nearest')  # [bz, 1, 16, 16]
            unlabeled_ROIs64 = unlabeled_ROIs64.view(bz, -1).unsqueeze(-1)
            N64 = unlabeled_ROIs64.sum()

        if N128>0:
            affinity_loss1 = (unlabeled_ROIs128 * affinity_loss1).sum() / N128
        if N64>0:
            affinity_loss2 = (unlabeled_ROIs64 * affinity_loss2).sum() / N64
        if N32>0:
            affinity_loss3 = (unlabeled_ROIs32 * affinity_loss3).sum() / N32
        if N32>0:
            affinity_loss4 = (unlabeled_ROIs32 * affinity_loss4).sum() / N32
        
        if self.loss_index == 0:
            affinity_loss = affinity_loss1
        elif self.loss_index == 1:
            affinity_loss = affinity_loss1 + affinity_loss2
        elif self.loss_index == 2:
            affinity_loss = affinity_loss1 + affinity_loss2 + affinity_loss3
        elif self.loss_index == 3:
            affinity_loss = affinity_loss1 + affinity_loss2 + affinity_loss3 + affinity_loss4
        else:
            affinity_loss = torch.zeros(1, dtype=seg.dtype, device=seg.device)
        return self.weight * affinity_loss

class WeightedFSOhemCELoss(nn.Module):
    def __init__(self, configer):
        super().__init__()
        self.configer = configer
        self.thresh = self.configer.get('loss', 'params')['ohem_thresh']
        self.reduction = 'elementwise_mean'
        if self.configer.exists('loss', 'params') and 'ce_reduction' in self.configer.get('loss', 'params'):
            self.reduction = self.configer.get('loss', 'params')['ce_reduction']

    def forward(self, predict, target, min_kept=1, weight=None, ignore_index=-1, **kwargs):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
        """
        prob_out = F.softmax(predict, dim=1)
        tmp_target = target.clone()
        tmp_target[tmp_target == ignore_index] = 0
        prob = prob_out.gather(1, tmp_target.unsqueeze(1))
        mask = target.contiguous().view(-1,) != ignore_index
        sort_prob, sort_indices = prob.contiguous().view(-1,)[mask].contiguous().sort()
        min_threshold = sort_prob[min(min_kept, sort_prob.numel() - 1)]
        threshold = max(min_threshold, self.thresh)
        loss_matrix = F.cross_entropy(predict, target, weight=weight, ignore_index=ignore_index, reduction='none').contiguous().view(-1,)
        sort_loss_matrix = loss_matrix[mask][sort_indices]
        select_loss_matrix = sort_loss_matrix[sort_prob < threshold]
        if self.reduction == 'sum':
            return select_loss_matrix.sum()
        elif self.reduction == 'elementwise_mean':
            return select_loss_matrix.mean()
        else:
            raise NotImplementedError('Reduction Error!')

# Cross-entropy Loss
class FSCELoss(nn.Module):
    def __init__(self, configer=None):
        super(FSCELoss, self).__init__()
        self.configer = configer
        weight = None
        if self.configer.exists('loss', 'params') and 'ce_weight' in self.configer.get('loss', 'params'):
            weight = self.configer.get('loss', 'params')['ce_weight']
            weight = torch.FloatTensor(weight).cuda()
            # weight = torch.HalfTensor(weight).cuda()


        elif self.configer.exists('loss', 'params') and 'cls_hist' in self.configer.get('loss', 'params'):
            hist = self.configer.get('loss', 'params')['cls_hist']
            use_logp = self.configer.get('loss', 'params')['use_logp']

            hist = torch.tensor(hist).float()
            hist = F.normalize(hist, p=1, dim=0)

            if use_logp:
                # https://arxiv.org/pdf/1809.09077v1.pdf formula (3)
                log_safety_const = 1.10
                weight = 1.0 / torch.log(hist + log_safety_const)
            else:
                weight = 1.0 / hist.clamp(min=1e-6)

            weight = weight.cuda()

        print("fs_ce loss weight:", weight)

        reduction = 'elementwise_mean'
        if self.configer.exists('loss', 'params') and 'ce_reduction' in self.configer.get('loss', 'params'):
            reduction = self.configer.get('loss', 'params')['ce_reduction']

        ignore_index = -1
        if self.configer.exists('loss', 'params') and 'ce_ignore_index' in self.configer.get('loss', 'params'):
            ignore_index = self.configer.get('loss', 'params')['ce_ignore_index']

        self.ce_loss = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)

    def forward(self, inputs, *targets, weights=None, **kwargs):
        loss = 0.0
        if isinstance(inputs, tuple) or isinstance(inputs, list):
            if weights is None:
                weights = [1.0] * len(inputs)

            for i in range(len(inputs)):
                if len(targets) > 1:
                    target = self._scale_target(targets[i], (inputs[i].size(2), inputs[i].size(3)))
                    loss += weights[i] * self.ce_loss(inputs[i], target)
                else:
                    target = self._scale_target(targets[0], (inputs[i].size(2), inputs[i].size(3)))
                    loss += weights[i] * self.ce_loss(inputs[i], target)

        else:
            target = self._scale_target(targets[0], (inputs.size(2), inputs.size(3)))
            loss = self.ce_loss(inputs, target)

        return loss

    @staticmethod
    def _scale_target(targets_, scaled_size):
        if targets_.dim() == 3:
            targets = targets_.clone().unsqueeze(1).float()
        else:
            targets = targets_.clone().float()
        targets = F.interpolate(targets, size=scaled_size, mode='nearest')

        # print("target value:", torch.unique(targets))
        targets[targets==-2] = -1
        return targets.squeeze(1).long()

class FSOhemCELoss(nn.Module):
    def __init__(self, configer):
        super(FSOhemCELoss, self).__init__()
        self.configer = configer
        self.thresh = self.configer.get('loss', 'params')['ohem_thresh']
        self.min_kept = max(1, self.configer.get('loss', 'params')['ohem_minkeep'])
        weight = None
        if self.configer.exists('loss', 'params') and 'ce_weight' in self.configer.get('loss', 'params'):
            weight = self.configer.get('loss', 'params')['ce_weight']
            weight = torch.FloatTensor(weight).cuda()

        self.reduction = 'elementwise_mean'
        if self.configer.exists('loss', 'params') and 'ce_reduction' in self.configer.get('loss', 'params'):
            self.reduction = self.configer.get('loss', 'params')['ce_reduction']

        ignore_index = -1
        if self.configer.exists('loss', 'params') and 'ce_ignore_index' in self.configer.get('loss', 'params'):
            ignore_index = self.configer.get('loss', 'params')['ce_ignore_index']

        self.ignore_label = ignore_index
        self.ce_loss = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction='none')

    def forward(self, predict, target, **kwargs):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        prob_out = F.softmax(predict, dim=1)
        tmp_target = target.clone()
        tmp_target[tmp_target == self.ignore_label] = 0
        prob = prob_out.gather(1, tmp_target.unsqueeze(1))
        mask = target.contiguous().view(-1,) != self.ignore_label
        mask[0] = 1  # Avoid `mask` being empty
        sort_prob, sort_indices = prob.contiguous().view(-1,)[mask].contiguous().sort()
        min_threshold = sort_prob[min(self.min_kept, sort_prob.numel() - 1)]
        threshold = max(min_threshold, self.thresh)
        loss_matirx = self.ce_loss(predict, target).contiguous().view(-1,)
        sort_loss_matirx = loss_matirx[mask][sort_indices]
        select_loss_matrix = sort_loss_matirx[sort_prob < threshold]
        if self.reduction == 'sum':
            return select_loss_matrix.sum()
        elif self.reduction == 'elementwise_mean':
            return select_loss_matrix.mean()
        else:
            raise NotImplementedError('Reduction Error!')

class FSAuxOhemCELoss(nn.Module):
    def __init__(self, configer=None):
        super(FSAuxOhemCELoss, self).__init__()
        self.configer = configer
        self.ce_loss = FSCELoss(self.configer)
        if self.configer.get('loss', 'loss_type') == 'fs_auxohemce_loss':
            self.ohem_ce_loss = FSOhemCELoss(self.configer)
        else:
            assert self.configer.get('loss', 'loss_type') == 'fs_auxslowohemce_loss'
            self.ohem_ce_loss = FSSlowOhemCELoss(self.configer)

    def forward(self, inputs, targets, **kwargs):
        aux_out, seg_out = inputs
        seg_loss = self.ohem_ce_loss(seg_out, targets)
        aux_loss = self.ce_loss(aux_out, targets)
        loss = self.configer.get('network', 'loss_weights')['seg_loss'] * seg_loss
        loss = loss + self.configer.get('network', 'loss_weights')['aux_loss'] * aux_loss
        return loss

class FSAuxCELoss(nn.Module):
    def __init__(self, configer=None):
        super(FSAuxCELoss, self).__init__()
        self.configer = configer
        self.ce_loss = FSCELoss(self.configer)

    def forward(self, inputs, targets, **kwargs):
        aux_out, seg_out = inputs
        seg_loss = self.ce_loss(seg_out, targets)
        aux_loss = self.ce_loss(aux_out, targets)
        loss = self.configer.get('network', 'loss_weights')['seg_loss'] * seg_loss
        loss = loss + self.configer.get('network', 'loss_weights')['aux_loss'] * aux_loss
        return loss

class FSAuxRMILoss(nn.Module):
    def __init__(self, configer=None):
        super(FSAuxRMILoss, self).__init__()
        self.configer = configer
        self.ce_loss = FSCELoss(self.configer)
        self.rmi_loss = RMILoss(self.configer)

    def forward(self, inputs, targets, **kwargs):
        aux_out, seg_out = inputs
        aux_loss = self.ce_loss(aux_out, targets)
        seg_loss = self.rmi_loss(seg_out, targets)
        loss = self.configer.get('network', 'loss_weights')['seg_loss'] * seg_loss
        loss = loss + self.configer.get('network', 'loss_weights')['aux_loss'] * aux_loss
        return loss

class SegFixLoss(nn.Module):
    """
    We predict a binary mask to categorize the boundary pixels as class 1 and otherwise as class 0
    Based on the pixels predicted as 1 within the binary mask, we further predict the direction for these
    pixels.
    """

    def __init__(self, configer=None):
        super().__init__()
        self.configer = configer
        self.ce_loss = FSCELoss(self.configer)

    def calc_weights(self, label_map, num_classes):

        weights = []
        for i in range(num_classes):
            weights.append((label_map == i).sum().data)
        weights = torch.FloatTensor(weights)
        weights_sum = weights.sum()
        return (1 - weights / weights_sum).cuda()       

    def forward(self, inputs, targets, **kwargs):

        from lib.utils.helpers.offset_helper import DTOffsetHelper

        pred_mask, pred_direction = inputs

        seg_label_map, distance_map, angle_map = targets[0], targets[1], targets[2]
        gt_mask = DTOffsetHelper.distance_to_mask_label(distance_map, seg_label_map, return_tensor=True)

        gt_size = gt_mask.shape[1:]
        mask_weights = self.calc_weights(gt_mask, 2)

        pred_direction = F.interpolate(pred_direction, size=gt_size, mode="bilinear", align_corners=True)
        pred_mask = F.interpolate(pred_mask, size=gt_size, mode="bilinear", align_corners=True)
        mask_loss = F.cross_entropy(pred_mask, gt_mask, weight=mask_weights, ignore_index=-1)

        mask_threshold = float(os.environ.get('mask_threshold', 0.5))
        binary_pred_mask = torch.softmax(pred_mask, dim=1)[:, 1, :, :] > mask_threshold

        gt_direction = DTOffsetHelper.angle_to_direction_label(
            angle_map,
            seg_label_map=seg_label_map,
            extra_ignore_mask=(binary_pred_mask == 0),
            return_tensor=True
        )

        direction_loss_mask = gt_direction != -1
        direction_weights = self.calc_weights(gt_direction[direction_loss_mask], pred_direction.size(1))
        direction_loss = F.cross_entropy(pred_direction, gt_direction, weight=direction_weights, ignore_index=-1)

        if self.training \
           and self.configer.get('iters') % self.configer.get('solver', 'display_iter') == 0 \
           and torch.cuda.current_device() == 0:
            Log.info('mask loss: {} direction loss: {}.'.format(mask_loss, direction_loss))

        mask_weight = float(os.environ.get('mask_weight', 1))
        direction_weight = float(os.environ.get('direction_weight', 1))

        return mask_weight * mask_loss + direction_weight * direction_loss

