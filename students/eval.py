from __future__ import print_function
from collections import defaultdict
import pandas as pd
import math
import sys
import time
from clize import run
from itertools import chain
import numpy as np
import sys
from skimage.io import imsave
import argparse
import os
import random
import torch
import torch.nn as nn
import random
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data.sampler import Sampler
from torch.autograd import Variable
from torch.nn.init import xavier_uniform
sys.path.append('../generators')
from loader import ImageFolder

from compressor import ConvStudent
from compressor import norm
from compressor import get_acc
from compressor import Gen
from compressor import GeneratorLoader

def student(*,student='student.th', classifier='/home/mcherti/work/code/external/densenet.pytorch/model/model.th',
            dataroot='/home/mcherti/work/data/cifar10', batchSize=32, imageSize=32, nb_classes=10):

    sys.path.append(os.path.dirname(classifier))
    sys.path.append(os.path.dirname(classifier) + '/..')
    clf = torch.load(classifier)

    if 'cifar10' in dataroot:
        mean = [0.49139968, 0.48215827, 0.44653124]
        std = [0.24703233, 0.24348505, 0.26158768]
    else:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    
    clf_mean = Variable(torch.FloatTensor(mean).view(1, -1, 1, 1)).cuda()
    clf_std = Variable(torch.FloatTensor(std).view(1, -1, 1, 1)).cuda()
    
    nc = 3
    w = imageSize
    h = imageSize
    no = nb_classes
    nbf = 512
    fc = 1200
    S = ConvStudent(nc, w, h, no, nbf=nbf, fc=fc)
    S.load_state_dict(torch.load(student))
    S = S.cuda()
    S.train(False)
    input = torch.zeros(batchSize, 3, imageSize, imageSize)
    input = Variable(input)
    input = input.cuda()
    
    if 'cifar10' in dataroot:
        transform = transforms.Compose([
           transforms.Scale(imageSize),
           transforms.CenterCrop(imageSize),
           transforms.ToTensor(),
           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        dataset = dset.CIFAR10(root=dataroot, download=True, transform=transform, train=False)
    else:
        transform = transforms.Compose([
               transforms.Scale(imageSize),
               transforms.CenterCrop(imageSize),
               transforms.ToTensor(),
               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        dataset = ImageFolder(root=dataroot, transform=transform)

    torch.cuda.manual_seed(42)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batchSize,
        num_workers=8)
    accs_student = []
    accs_teacher = []
    accs_student_teacher = []
    for b, (X, y) in enumerate(dataloader):
        t = time.time()
        batch_size = X.size(0)
        input.data.resize_(X.size()).copy_(X)
        y_true = torch.zeros(batch_size, nb_classes)
        y_true.scatter_(1, y.view(y.size(0), 1), 1)
        y_teacher = clf(norm(input, clf_mean, clf_std))
        y_student = S(input)
        acc_teacher = get_acc(y_true, y_teacher.data.cpu())
        acc_student = get_acc(y_true, y_student.data.cpu())
        acc_student_teacher = get_acc(y_teacher.data.cpu(), y_student.data.cpu())
        accs_student.append(acc_student)
        accs_teacher.append(acc_teacher)
        accs_student_teacher.append(acc_student_teacher)
        print('acc student : {:.3f}, acc teacher : {:.3f}, acc student on teacher : {:.3f}'.format(np.mean(accs_student), np.mean(accs_teacher), np.mean(accs_student_teacher)))
        dt = time.time() - t

def generator(*, classifier='/home/mcherti/work/code/external/densenet.pytorch/model/model.th',
              generator='../generators/samples/samples_pretrained_aux_dcgan_32/netG_epoch_35.pth',
              dataroot='/home/mcherti/work/data/cifar10', batchSize=32, imageSize=32, nz=100, nb_classes=10):

    sys.path.append(os.path.dirname(classifier))
    sys.path.append(os.path.dirname(classifier) + '/..')
    clf = torch.load(classifier)
        
    if 'cifar10' in dataroot:
        mean = [0.49139968, 0.48215827, 0.44653124]
        std = [0.24703233, 0.24348505, 0.26158768]
    else:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    
    clf_mean = Variable(torch.FloatTensor(mean).view(1, -1, 1, 1)).cuda()
    clf_std = Variable(torch.FloatTensor(std).view(1, -1, 1, 1)).cuda()
 
    input = torch.zeros(batchSize, 3, imageSize, imageSize)
    input = Variable(input)
    input = input.cuda()

    G = Gen(imageSize=imageSize, nb_classes=nb_classes)
    G.load_state_dict(torch.load(generator))
    G = G.cuda()

    z = torch.randn(batchSize, nz, 1, 1)
    z = Variable(z)
    z = z.cuda()

    onehot = torch.zeros(batchSize, nb_classes, 1, 1)
    onehot = Variable(onehot)
    onehot = onehot.cuda()

    u = torch.zeros(batchSize, nb_classes, 1, 1)
    u = u.cuda()
    
    nb_minibatches = 1000
    dataloader = GeneratorLoader(G, z, onehot, u, nb_minibatches)
    accs = []
    for X, y in dataloader:
        input.data.resize_(X.size()).copy_(X)
        y = y.view(-1, 1).cpu()
        y_true = torch.zeros(y.size(0), nb_classes)
        y_true.scatter_(1, y, 1)
        y_pred = clf(norm(input, clf_mean, clf_std)).data.cpu()
        acc = get_acc(y_true, y_pred)
        accs.append(acc)
        print(np.mean(accs))

if __name__ == '__main__':
    run(student, generator)
