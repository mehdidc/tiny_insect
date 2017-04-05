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
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import pandas as pd
import sys


nz = 100
nb_classes = 18
ngf = 64
ndf = 64
nc = 3
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

class MLPStudent(nn.Module):
    def __init__(self, nc, w, h, no):
        super(MLPStudent, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(nc * w * h, 250),
            nn.Linear(250, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(True),
            nn.Linear(1000, no),
        )

    def forward(self, input):
        input = input.view(input.size(0), -1)
        output = self.main(input)
        return output

class ConvStudent(nn.Module):
    def __init__(self, nc, w, h, no):
        super(ConvStudent, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(nc, 512, 5),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(14*14*512, no)
        )
        
    def forward(self, input):
        x = self.features(input)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def norm(x, mean, std):
    x = (x + 1) / 2.
    x += 1
    x /= 2.
    x -= mean.repeat(x.size(0), 1, x.size(2), x.size(3))
    x /= std.repeat(x.size(0), 1, x.size(2), x.size(3))
    return x

from torch.nn.init import xavier_uniform
def weights_init(m):
    import math
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        xavier_uniform(m.weight.data, gain=math.sqrt(2.))
        m.bias.data.fill_(0)

def get_acc(pred, true):
    _, pred_classes = pred.max(1)
    _, true_classes = true.max(1)
    return (pred_classes == true_classes).float().mean()

def main(*, classifier='pytorch_pretrained/clf-256-resnet/clf.th', 
        generator='samples_128_cond_3/netG_epoch_7.pth', 
        batchSize=32, 
        nz=100, 
        niter=100000, 
        no=18, 
        lr=1e-4, 
        dataroot='/home/mcherti/work/data/insects/img_classes'):
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
    
    S = ConvStudent(3, 32, 32, no)
    S.apply(weights_init)
    S = S.cuda()
    
    input = torch.zeros(batchSize, 3, 256, 256)
    input = Variable(input)
    input = input.cuda()

    z = torch.randn(batchSize, nz, 1, 1)
    z = Variable(z)
    z = z.cuda()
    onehot = torch.zeros(batchSize, no, 1, 1)
    onehot = Variable(onehot)
    onehot = onehot.cuda()

    u = torch.zeros(batchSize, nz, 1, 1)
    u = u.cuda()

    optimizer = optim.SGD(S.parameters(), lr=lr, momentum=0.9, nesterov=True)
    #optimizer = optim.Adam(S.parameters(), lr=lr)
    avg_loss = 0.
    avg_acc = 0.

    transform = transforms.Compose([
           transforms.Scale(256),
           transforms.CenterCrop(256),
           transforms.ToTensor(),
           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = dset.ImageFolder(root=dataroot, transform=transform)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batchSize,
        shuffle=True, 
        num_workers=1)
    j = 0
    stats = defaultdict(list)
    for i in range(niter):
        """
        z.data.normal_()
        onehot.data.zero_()
        u.uniform_()
        onehot.data.scatter_(1, u.max(1)[1], 1)
        g_input = torch.cat((z, onehot), 1)
        out = G(g_input)
        out = nn.UpsamplingBilinear2d(scale_factor=2)(out)
        input = out
        """
        for b, (X, y) in enumerate(dataloader):
            input.data.resize_(X.size()).copy_(X)
            S.zero_grad()
            y_true = clf(norm(input, clf_mean, clf_std))
            input_ = nn.AvgPool2d(8, 8)(input)
            y_pred = S(input_)
            
            loss = ((y_pred - y_true) ** 2).mean()
            loss.backward()
            optimizer.step()
            acc = get_acc(y_true, y_pred)
            stats['acc'].append(acc.data[0])
            stats['loss'].append(loss.data[0])
            
            avg_loss = avg_loss * 0.99 + loss.data[0] * 0.01
            avg_acc = avg_acc * 0.99 + acc.data[0] * 0.01
            if j % 100 == 0:
                pd.DataFrame(stats).to_csv('stats.csv', index=False)
                print('[{}/{}] batch {}/{}. moving_loss:{:.3f} moving_acc:{:.3f}'.format(i, niter, b, len(dataloader), avg_loss, avg_acc))
            j += 1
        torch.save(S, 'student.th')

if __name__ == '__main__':
    run(main)
