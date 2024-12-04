import argparse
import logging
import os
import random
import sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from utils import DiceLoss, Focal_loss
from torchvision import transforms
from icecream import ic
from einops import repeat

def calc_loss(outputs, low_res_label_batch, ce_loss, dice_loss, dice_weight:float=0.8):
    low_res_logits = outputs['low_res_logits']
    low_res_label_batch = F.interpolate(low_res_label_batch, size=(128, 128), mode='bilinear', align_corners=False)
    # assert 1==0, print(low_res_label_batch.shape)
    low_res_label_batch=low_res_label_batch.squeeze(0)
    loss_ce = ce_loss(low_res_logits, low_res_label_batch[:].long())
    loss_dice = dice_loss(low_res_logits, low_res_label_batch, softmax=True)
    loss = (1 - dice_weight) * loss_ce + dice_weight * loss_dice
    return loss, loss_ce, loss_dice

def Minimize_domain_mean(features,targets):
    criterion_mse=torch.nn.MSELoss()
    criterion=torch.nn.L1Loss()
    # target= torch.normal(mean, std_dev)
    targets=targets.repeat(len(features),1,1,1)
    # assert 1==0,print(features.shape,targets.shape)
    l2=criterion_mse(features*0.9999, targets)#/torch.abs(targets).mean()
    l1=criterion(features*0.9999, targets)
    loss =1*l1+10*l2
    return loss 

def one_sample_train( model, sampled_batch, multimask_output,ind,targets,std_var):
    for i in range(1):
        base_lr = 0.1
        num_classes = 8
        model.train()
        ce_loss = CrossEntropyLoss()
        dice_loss = DiceLoss(num_classes + 1)

        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.sam.image_encoder.parameters()), lr=base_lr, momentum=0.9, weight_decay=0.0001)  # Even pass the model.parameters(), the `requires_grad=False` layers will not update

        image_batch, label_batch = sampled_batch['image'], sampled_batch['label']  # [b, c, h, w], [b, h, w]
        # image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
        input_image = image_batch[:,ind, :, :]
        input_image_torch =input_image.unsqueeze(0).float().cuda()
        input_image_torch = repeat(input_image_torch, 'b c h w -> b (repeat c) h w', repeat=3)
        # assert 1==0, print(input_image_torch.shape)
        label = label_batch[:,ind, :, :]
        label = label.unsqueeze(0).float().cuda()
        # image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
        # assert image_batch.max() <= 3, f'image_batch max: {image_batch.max()}'   # why image_batch have to be smaller than 3
        repeat_times=4
        perturb_img_noise=input_image_torch.repeat(repeat_times,1,1,1)
        perturb_img_noise = perturb_img_noise.clone() + torch.randn_like(torch.tensor(perturb_img_noise.clone()), device='cuda')*0.04
        # assert 1==0,print(perturb_img_noise.shape,input_image_torch.shape)
        perturb_img_noise=torch.cat((input_image_torch,perturb_img_noise),dim=0)
        features,features_layer = model.sam.image_encoder(perturb_img_noise)
        # assert 1==0, print(input_image_torch.shape)
        # loss, loss_ce, loss_dice = calc_loss(outputs, label, ce_loss, dice_loss, 0.8)
        loss =Minimize_domain_mean(features,targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('the domain loss is',loss)
    return model
