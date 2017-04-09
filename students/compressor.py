from __future__ import print_function
from collections import defaultdict
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
import pandas as pd
import sys
sys.path.append('../generators')
from loader import ImageFolder


class Rotation:

    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, img):
        angle = random.uniform(self.min_val, self.max_val)
        return img.rotate(angle)

class RandomSampler(Sampler):
    """Samples elements randomly, without replacement.
    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source):
        self.num_samples = len(data_source)
        self.perm = torch.randperm(self.num_samples).long()

    def __iter__(self):
        return iter(self.perm)

    def __len__(self):
        return self.num_samples

class LoaderDataAugmentation:

    def __init__(self, dataloader, nb_passes=1):
        self.dataloader = dataloader
        self.nb_passes = nb_passes

    def __iter__(self):
        for i in range(self.nb_passes):
            random.seed(i)
            self.dataloader.pass_idx = i
            yield from self.dataloader
    
    def __len__(self):
        return len(self.dataloader) * self.nb_passes


class _netG(nn.Module):
    def __init__(self, ngpu, imageSize=32, nz=100, nb_classes=18, nc=3, ngf=64):
        super(_netG, self).__init__()
        self.ngpu = ngpu

        if imageSize == 32:
            self.main = nn.Sequential(
                nn.ConvTranspose2d(     nz+nb_classes , ngf * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),
                
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
                
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
                
                nn.ConvTranspose2d(    ngf * 2,      nc, 4, 2, 1, bias=False),
                nn.Tanh()
            )

        else:
            self.main = nn.Sequential(
                nn.ConvTranspose2d(     nz+nb_classes , ngf * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),
                
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
                
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
                
                nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),
                
                nn.ConvTranspose2d(ngf,     ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),

                nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
                nn.Tanh()
            )

    def forward(self, input):
        x = self.main(input)
        return x

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
    def __init__(self, nc, w, h, no, nbf=512, fc=1000):
        super(ConvStudent, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(nc, nbf, 5),
            nn.BatchNorm2d(nbf),
            nn.ReLU(True),
            nn.MaxPool2d(2),
        )
        wout = (w - 5 + 1) // 2
        hout = (h - 5 + 1) // 2
        self.fc = nn.Sequential(
            nn.Linear(wout * hout * nbf, fc),
            nn.Linear(fc, no)
        )
        
    def forward(self, input):
        x = self.features(input)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def norm(x, mean, std):
    x = x + 1
    x = x / 2
    x = x - mean.repeat(x.size(0), 1, x.size(2), x.size(3))
    x = x / std.repeat(x.size(0), 1, x.size(2), x.size(3))
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

def train(*, classifier='/home/mcherti/work/code/external/densenet.pytorch/model/latest.pth', 
            generator='../generators/samples/samples_cond_dcgan_cifar10/netG_epoch_327.pth', 
            batchSize=32, 
            nz=100, 
            niter=200, 
            npasses=10,
            imageSize=32,
            nb_classes=10,
            dataroot='/home/mcherti/work/data/cifar10'):

    if not os.path.exists('{{folder}}'):
        os.mkdir('{{folder}}')

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

    G = _netG(ngpu=1, imageSize=imageSize, nb_classes=nb_classes)
    G.load_state_dict(torch.load(generator))
    G = G.cuda()
    
    #S = MLPStudent(3, imageSize, imageSize, nb_classes)
    nbf = {{'nb_filters'|choice(32, 64, 96, 128, 192, 256, 512, 800)}}
    fc = {{'fc'|choice(100, 200, 300, 400, 500, 600, 700, 800, 900, 1000)}}
    S = ConvStudent(3, imageSize, imageSize, nb_classes, nbf=nbf, fc=fc)
    S.apply(weights_init)
    S = S.cuda()
    
    input = torch.zeros(batchSize, 3, 256, 256)
    input = Variable(input)
    input = input.cuda()

    z = torch.randn(batchSize, nz, 1, 1)
    z = Variable(z)
    z = z.cuda()
    onehot = torch.zeros(batchSize, nb_classes, 1, 1)
    onehot = Variable(onehot)
    onehot = onehot.cuda()

    u = torch.zeros(batchSize, nz, 1, 1)
    u = u.cuda()


    algo = {{'algo'|choice(0, 1, 2)}}
    lr = {{'lr'|loguniform(-5, -1)}}
    if algo == 0:
        optimizer = torch.optim.Adam(S.parameters(), lr=lr)
    elif algo == 1:
        optimizer = torch.optim.SGD(S.parameters(), lr=lr, nesterov=True, momentum=0.9)
    elif algo == 2:
        optimizer = torch.optim.SGD(S.parameters(), lr=lr)

    avg_loss = 0.
    avg_acc = 0.
    
    if 'cifar10' in dataroot:
        transform = transforms.Compose([
            transforms.Scale(imageSize),
            transforms.CenterCrop(imageSize),
            Rotation(-10, 10),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        dataset = dset.CIFAR10(root=dataroot, download=True, transform=transform)
    else:
        transform = transforms.Compose([
               transforms.Scale(imageSize),
               transforms.CenterCrop(imageSize),
               #transforms.RandomHorizontalFlip(),
               transforms.ToTensor(),
               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        dataset = ImageFolder(root=dataroot, transform=transform)
   
    source = 'dataset'
    if source == 'dataset':
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batchSize,
            sampler=RandomSampler(dataset),
            num_workers=8)
    elif source == 'generator':
        dataloader_ = torch.utils.data.DataLoader(
            dataset, batch_size=batchSize,
            sampler=RandomSampler(dataset),
            num_workers=8)
        class Loader:
            def __iter__(self):
                torch.cuda.manual_seed(self.pass_idx)
                for X, y in dataloader_:
                    z.data.normal_()
                    onehot.data.zero_()
                    u.uniform_()
                    classes = u.max(1)[1]
                    onehot.data.scatter_(1, classes, 1)
                    g_input = torch.cat((z, onehot), 1)
                    out = G(g_input)
                    input = out.data
                    X = torch.cat((X, input), 0)
                    y = torch.cat((y, classes), 0)
                    yield X, y
            
            def __len__(self):
                return len(dataloader_)
        dataloader = Loader()
    

    dataloader = LoaderDataAugmentation(dataloader, nb_passes=npasses)

    stats = defaultdict(list)
    yt = torch.zeros(len(dataloader) * batchSize, nb_classes)
    j = 0
    print('Getting and storing predictions...')
        
    for b, (X, y) in enumerate(dataloader):
        if b % 100 == 0:
            print('batch {}/{}'.format(b, len(dataloader)))
        input.data.resize_(X.size()).copy_(X)
        y_true = clf(norm(input, clf_mean, clf_std))
        yt[b * batchSize:b * batchSize + input.size(0)].copy_(y_true.data)
    for i in range(niter):
        S.train(True)
        for b, (X, y) in enumerate(dataloader):
            t = time.time()
            input.data.resize_(X.size()).copy_(X)
            S.zero_grad()
            y_true = yt[b * batchSize:b * batchSize + input.size(0)]
            y_true = Variable(y_true).cuda()
            y_pred = S(input)
            loss = ((y_pred - y_true) ** 2).mean()
            loss.backward()
            optimizer.step()
            dt = time.time() - t
            acc = get_acc(y_true, y_pred)
            stats['acc'].append(acc.data[0])
            stats['loss'].append(loss.data[0])
            stats['time'].append(dt)
            
            avg_loss = avg_loss * 0.9 + loss.data[0] * 0.1
            avg_acc = avg_acc * 0.9 + acc.data[0] * 0.1
            if j % 100 == 0:
                pd.DataFrame(stats).to_csv('{{folder}}/stats.csv', index=False)
                print('[{}/{}] batch {}/{}. moving_loss:{:.3f} moving_acc:{:.3f}, time : {:.3f}'.format(i, niter, b, len(dataloader), avg_loss, avg_acc, dt))
            """
            if j % 10000 == 0:
                print('reducing lr')
                for param_group in optimizer.param_groups:
                    lr = param_group['lr']
                    lr /= 2.
                    param_group['lr'] = lr
            """
            j += 1
        S.train(False)
        accs = []
        for b, (X, y) in enumerate(dataloader):
            input.data.resize_(X.size()).copy_(X)
            y_true = yt[b * batchSize:b * batchSize + input.size(0)]
            y_true = Variable(y_true).cuda()
            y_pred = S(input)
            acc = get_acc(y_true, y_pred)
            accs.append(acc.data[0])
        train_acc = np.mean(accs)
        print('train acc : {}'.format(train_acc))
        torch.save(S.state_dict(), '{{folder}}/student.th')
    return train_acc

def eval(*,
         student='student.th', 
         classifier='../teachers/clf-256-resnet/clf.th', 
         dataroot='/home/mcherti/work/data/insects/train_img_classes', 
         batchSize=32, 
         imageSize=256,
         nb_classes=18):

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

    S = torch.load(student)
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
        dataset = dset.CIFAR10(root=dataroot, download=True, transform=transform, train=True)

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
        dataset, batch_size=batchSize,
        sampler=RandomSampler(dataset),
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

if __name__ == '__main__':
    result = train()
