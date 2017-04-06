from __future__ import print_function
from collections import defaultdict
from clize import run
from itertools import chain
import numpy as np
import sys
from skimage.io import imsave
import argparse
import os
import random
import math
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision.models import resnet50

from torch.autograd import Variable
import pandas as pd
import sys

nz = 100
nb_classes = 18
ngf = 64
ndf = 64
nc = 3

def compute_objectness(v):
    marginal = v.mean(dim=0)
    marginal = marginal.repeat(v.size(0), 1)
    score = v * torch.log(v / (marginal))
    score = math.exp(score.sum(dim=1).mean())
    return score

class _netG(nn.Module):
    def __init__(self, ngpu):
        super(_netG, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz+nb_classes , ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(ngf,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )
    def forward(self, input):
        return self.main(input)

def norm(x, mean, std):
    x = (x + 1) / 2.
    x += 1
    x /= 2.
    x -= mean.repeat(x.size(0), 1, x.size(2), x.size(3))
    x /= std.repeat(x.size(0), 1, x.size(2), x.size(3))
    return x

def objectness(classifier='../teachers/clf-128-vgg16/clf.th', 
               generator='../generators/samples/samples_128_cond_3/netG_epoch_7.pth', 
               batchSize=32,
               nc=18,
               n_samples=64000):
    if classifier == 'resnet50':
        clf = resnet50(pretrained=True).cuda()
    else:
        sys.path.append(os.path.dirname(classifier))
        clf = torch.load(classifier)

    clf_mean = np.array([0.485, 0.456, 0.406], dtype='float32')
    clf_mean = clf_mean[np.newaxis, :, np.newaxis, np.newaxis]
    clf_mean = torch.from_numpy(clf_mean)
    clf_mean = Variable(clf_mean)
    clf_std = np.array([0.229, 0.224, 0.225], dtype='float32')
    clf_std = clf_std[np.newaxis, :, np.newaxis, np.newaxis]
    clf_std = torch.from_numpy(clf_std)
    clf_std = Variable(clf_std)

    clf_mean = clf_mean.cuda()
    clf_std = clf_std.cuda()

    G = _netG(ngpu=1)
    G.load_state_dict(torch.load(generator))
    G = G.cuda()

    z = torch.randn(batchSize, nz, 1, 1)
    z = Variable(z)
    z = z.cuda()

    onehot = torch.zeros(batchSize, nc, 1, 1)
    onehot = Variable(onehot)
    onehot = onehot.cuda()

    u = torch.zeros(batchSize, nz, 1, 1)
    u = u.cuda()
    pred = []
    for i in range(0, n_samples, batchSize):
        z.data.normal_()
        onehot.data.zero_()
        u.uniform_()
        onehot.data.scatter_(1, u.max(1)[1], 1)
        g_input = torch.cat((z, onehot), 1)
        out = G(g_input)
        if classifier == 'resnet50':
            out = nn.UpsamplingBilinear2d(scale_factor=2)(out)
        y = clf(norm(out, clf_mean, clf_std))
        y = nn.Softmax()(y)
        y = y.data.cpu()
        pred.append(y)
    y = torch.cat(pred, 0)
    return compute_objectness(y)


def main(*,classifier='../teachers/clf-128-vgg16/clf.th', 
         generator='../generators/samples/samples_128_cond_3/netG_epoch_7.pth', 
         batchSize=32,
         nc=18,
         n_samples=64000):
    return objectness(classifier=classifier, generator=generator, batchSize=batchSize, nc=nc, n_samples=n_samples)
 
if __name__ == '__main__':
    run(main)
