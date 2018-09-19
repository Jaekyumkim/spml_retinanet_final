
from __future__ import print_function

import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils import one_hot_embedding

class FocalLoss(nn.Module):
    def __init__(self, num_classes=20, loss_fn='sigmoid'):
        self.num_classes = num_classes
        super(FocalLoss, self).__init__()
        self.alpha = Variable(torch.Tensor([0.25])).cuda()
        self.gamma = Variable(torch.Tensor([2])).cuda()

    def focal_loss(self, pred, targ):
        '''Focal loss

        Args:
            pred: (tensor) sized [#anchors, #num_classes]
            targ: (tensor) sized [#anchors, ]

        Return:
            loss = -1*alpht*(1-p_t)**gamma*log(p_t)
        '''
        alpha = 0.25
        gamma = 2

        targ = one_hot_embedding(targ.data.cpu(), 1+self.num_classes)
        targ = targ[:,1:]                # exclude_background
        targ = Variable(targ).cuda()    # [#anchors, #num_clsasses]

        p = pred.sigmoid()
        # p_t = p if pred is targ,        p_t = 1-p if pred is not targ
        p_t = p*targ + (1-p)*(1-targ)          
        # coeff = alpha if pred is targ,  coeff = 1-alpha if pred is not targ
        coeff = alpha*targ + (1-alpha)*(1-targ) 
        coeff *= (1-p_t).pow(gamma)

        # within BCEwithlogits, pred and target go through sigmoid function
        return F.binary_cross_entropy_with_logits(pred, targ, coeff, size_average=False)

    def focal_loss_c(self, pred, targ):
        '''Focal loss

        Args:
            pred: (tensor) sized [#anchors, #num_classes]
            targ: (tensor) sized [#anchors, ]

        Return:
            loss = -1*alpht*(1-p_t)**gamma*log(p_t)
        '''
        alpha = 0.25
        gamma = 2

        targ = one_hot_embedding(targ.data.cpu(), 1+self.num_classes)
        targ = targ[:,1:]                # exclude_background
        targ = Variable(targ).cuda()    # [#anchors, #num_clsasses]

        if pred.is_cuda is False:
            pred = pred.cuda()
        p = pred.sigmoid()
        # p_t = p if pred is targ,        p_t = 1-p if pred is not targ
        p_t = p*targ + (1-p)*(1-targ)
        coeff1 = alpha*targ + (1-alpha)*(1-targ) 
        coeff2 = (1-p_t).pow(gamma) * coeff1

        sigmoid_p = pred.sigmoid()
        per_cross_ent = -1*coeff2*(p_t.log())

        return torch.sum(per_cross_ent)

    def focal_loss_softmax(self, pred, targ):
        '''Focal loss with softmax (paper: sigmoid)

        Args:
            pred: (tensor) sized [#anchors, #num_classes]
            targ: (tensor) sized [#anchors, ]

        Return:
            loss = -1*alpht*(1-p_t)**gamma*log(p_t)
        '''
        alpha = self.alpha.clone()
        gamma = self.gamma.clone()

        P = F.softmax(pred, dim=1)
        class_mask = Variable(pred.data.new(pred.size(0), pred.size(1)).fill_(0))
        ids = targ.view(-1, 1).type(torch.LongTensor).cuda()
        class_mask.scatter_(1, ids.data, 1.)

        alpha = alpha.repeat(pred.size(0), 1)
        zero_idx = (targ==0).nonzero().squeeze()
        alpha[zero_idx.data] = 1 - self.alpha          # ref: https://bit.ly/2nLBhlt
        probs = (P*class_mask).sum(1).view(-1,1)
        log_p = probs.log()
        batch_loss = -alpha*(torch.pow((1-probs), gamma))*log_p

        return batch_loss.sum()

    def forward(self, loc_preds, loc_targets, cls_preds, cls_targets):
        '''Compute loss between (locs_preds, loc_targets) and (cls_preds, cls_targets)
        Args:
            loc_preds:      (tensor) [#batch_size, #anchors, 4]
            loc_targets:    (tensor) [#batch_size, #anchors, 4]
            cls_preds:      (tensor) [#batch_size, #anchors, #num_classes]
            cls_targets:    (tensor) [#batch_size, #anchors, #num_classes]

        Return:
            loss: (tensor) smoothL1Loss(loc_preds, loc_targets) 
                + FocalLoss(cls_preds, cls_targets)
        '''
        batch_size, anchor_size = cls_targets.size()
        pos = cls_targets > 0   # [N, #anchors] exclude background
        num_pos = pos.data.long().sum().float()

        # loc_loss = smoothL1Loss(loc_preds, loc_targets)
        mask = pos.unsqueeze(2).expand_as(loc_preds)    # [batch, #anchors, 4] exclude background
        masked_loc_preds = loc_preds[mask].view(-1,4).float()   # [#pos.sum, 4] masked loc_preds coor
        masked_loc_targets = loc_targets[mask].view(-1,4).float()
        loc_loss = F.smooth_l1_loss(masked_loc_preds, masked_loc_targets, size_average=False)

        masked = masked_loc_targets/torch.Tensor([0.4,0.4,0.8,0.8]).cuda()
        regre_diff = torch.abs(masked - masked_loc_preds)
        regre_loss = torch.where(torch.le(regre_diff, 1./9.),
                0.5*9.*torch.pow(regre_diff, 2),
                regre_diff - 0.5/9.)

        # cls_loss = FocalLoss(cls_preds, cls_targets)
        pos_neg = cls_targets > -1   # exclude ignored anchros
        mask = pos_neg.unsqueeze(2).expand_as(cls_preds)
        masked_cls_preds = cls_preds[mask].view(-1,self.num_classes)
        cls_loss = self.focal_loss(masked_cls_preds, cls_targets[pos_neg])
        
        loss = (regre_loss.sum()+cls_loss)/num_pos
        return loss, regre_loss.sum()/num_pos, cls_loss/num_pos 
