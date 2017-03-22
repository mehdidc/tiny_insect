#experimenting with ppgn generator
from __future__ import print_function
import numpy as np
from itertools import chain
import argparse
import sys
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision.models.resnet import resnet50
from torchvision.models.inception import inception_v3 
from torch.autograd import Variable

sys.path.append('/home/mcherti/work/code/external/ppgn')
sys.path.append('pytorch_pretrained')
 
from caffe_to_pytorch import Generator


def compute_objectness(v):
    marginal = v.mean(dim=0)
    marginal = marginal.repeat(v.size(0), 1)
    score = v * torch.log(v / (marginal))
    score = score.sum(dim=1).mean()
    return score


if __name__ == '__main__':
    HID = np.load('/home/mcherti/work/code/external/ppgn/hid.npz')['hid']
    
    gen = torch.load('/home/mcherti/work/code/external/ppgn/generator.th')
    gen = gen.cuda()

    
    #clf = torch.load('pytorch_pretrained/clf-3eef1ef7-9384-48aa-bce1-7dac309b0ee1.th')
    #clf = clf.cuda()
    clf = resnet50(pretrained=True)
    clf = clf.cuda()
    #clf = inception_v3(pretrained=True)
    #clf = clf.cuda()

    clf_mean = np.array([0.485, 0.456, 0.406], dtype='float32')
    clf_mean = clf_mean[np.newaxis, :, np.newaxis, np.newaxis]
    clf_mean = torch.from_numpy(clf_mean)
    clf_mean = Variable(clf_mean)
    clf_mean = clf_mean.cuda()
    clf_std = np.array([0.229, 0.224, 0.225], dtype='float32')
    clf_std = clf_std[np.newaxis, :, np.newaxis, np.newaxis]
    clf_std = torch.from_numpy(clf_std)
    clf_std = Variable(clf_std)
    clf_std = clf_std.cuda()

    def norm(x):
        min_val = -120
        max_val = 120
        x = torch.cat( (x[:, 2:3], x[:, 1:2], x[:, 0:1]),  1)
        x = (x - min_val) / (max_val - min_val)
        #x = nn.Sigmoid()(x)
        return x

    h = Variable(torch.randn(1, 4096), requires_grad=True).cuda()
    h_vel = torch.zeros(h.size(0), h.size(1)).cuda()
    h_grad = None
    def register_grad(g):
        global h_grad
        h_grad = g
    h.register_hook(register_grad)

    for i in range(10000):
        x = gen(h)
        x = norm(x)
        x.data.clamp_(0, 1)
        img = x
        x = x - clf_mean.repeat(x.size(0), 1, x.size(2), x.size(3))
        x = x / clf_std.repeat(x.size(0), 1, x.size(2), x.size(3))
        y = clf(x)
        y = nn.Softmax()(y)
        #obj = compute_objectness(y)
        obj = y[:, 0].mean()
        obj.backward()
        
        g = (h_grad.data / h_grad.data.abs().mean()) #+ 0.005 * h.data * 2.
        h_vel = h_vel * 0.9 + g * 0.1
        h.data += 0.01  * g
        h.data.clamp_(0, 30)
 
        if i % 10 == 0:
            vutils.save_image(img.data[0:16,:,16:-16,16:-16], 'sample.png')
        print(obj.data[0])
