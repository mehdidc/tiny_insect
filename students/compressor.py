from __future__ import print_function
from collections import defaultdict
import pandas as pd
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

from PIL import Image

class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """

    def __call__(self, img):
        if np.random.random() < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img

class Rotation:

    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, img):
        angle = np.random.uniform(low=self.min_val, high=self.max_val)
        return img.rotate(angle)


class SamplerFromIndices(Sampler):

    def __init__(self, data_source, indices):
        self.num_samples = len(data_source)
        self.perm = torch.LongTensor(indices)

    def __iter__(self):
        return iter(self.perm)

    def __len__(self):
        return len(self.perm)


class LoaderDataAugmentation:

    def __init__(self, dataloader, nb_epochs=1):
        self.dataloader = dataloader
        self.nb_epochs = nb_epochs

    def __iter__(self):
        for i in range(self.nb_epochs):
            yield from self.epoch(i)
    
    def epoch(self, i):
        random.seed(i)
        torch.manual_seed(i)
        torch.cuda.manual_seed(i)
        np.random.seed(i)
        yield from self.dataloader

    def __len__(self):
        return len(self.dataloader) * self.nb_epochs

class Gen(nn.Module):
    def __init__(self, imageSize=32, nz=100, nb_classes=18, nc=3, ngf=64):
        super(Gen, self).__init__()

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

class GeneratorLoader:

    def __init__(self, G, z, onehot, u, nb_minibatches):
        self.G = G
        self.z = z
        self.onehot = onehot
        self.u = u
        self.nb_minibatches = nb_minibatches

    def __iter__(self):
        G = self.G
        z = self.z
        onehot = self.onehot
        u = self.u
        nb_minibatches = self.nb_minibatches
        for i in range(nb_minibatches):
            z.data.normal_()
            onehot.data.zero_()
            u.uniform_()
            classes = u.max(1)[1]
            onehot.data.scatter_(1, classes, 1)
            g_input = torch.cat((z, onehot), 1)
            out = G(g_input)
            input = out.data
            yield input, classes
    
    def __len__(self):
        return self.nb_minibatches

def norm(x, mean, std):
    x = x + 1
    x = x / 2
    x = x - mean.repeat(x.size(0), 1, x.size(2), x.size(3))
    x = x / std.repeat(x.size(0), 1, x.size(2), x.size(3))
    return x

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

def train(*, data_source='generator'):
    classifier = '/home/mcherti/work/code/external/densenet.pytorch/model/model.th'
    generator = '../generators/samples/samples_pretrained_aux_dcgan_32/netG_epoch_35.pth'
    nb_passes = 10
    batchSize = 32 
    nz = 100 
    nb_epochs = 200
    imageSize = 32
    nb_classes = 10
    dataroot='/home/mcherti/work/data/cifar10'

    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

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
    
    nbf = {{'nb_filters'|choice(32, 64, 96, 128, 192, 256, 512, 600, 650, 700, 800)}}
    fc = {{'fc'|choice(100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1400, 1500, 1800, 2000)}}
    S = ConvStudent(3, imageSize, imageSize, nb_classes, nbf=nbf, fc=fc)
    S.apply(weights_init)
    S = S.cuda()
    
    input = torch.zeros(batchSize, 3, imageSize, imageSize)
    input = Variable(input)
    input = input.cuda()

    algo = {{'algo'|choice(0, 1, 2)}}
    lr = {{'lr'|loguniform(-5, -2)}}
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
            RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        transform_valid = transforms.Compose([
            transforms.Scale(imageSize),
            transforms.CenterCrop(imageSize),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        dataset = dset.CIFAR10(root=dataroot, download=True, transform=transform)
        dataset_valid = dset.CIFAR10(root=dataroot, download=True, transform=transform_valid)
    else:
        transform = transforms.Compose([
               transforms.Scale(imageSize),
               transforms.CenterCrop(imageSize),
               transforms.ToTensor(),
               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        dataset = ImageFolder(root=dataroot, transform=transform)
    
    if data_source == 'dataset':
        predictions_filename = 'yteacher_train.th'
        perm = np.arange(len(dataset))
        np.random.shuffle(perm)
        perm = torch.from_numpy(perm)
        perm_train = perm[0:40000]
        perm_valid = perm[40000:]
        nb_train_examples = len(perm_train)
        nb_valid_examples = len(perm_valid)
        dataloader_train = torch.utils.data.DataLoader(
            dataset, batch_size=batchSize,
            sampler=SamplerFromIndices(dataset, perm_train),
            num_workers=8)
        dataloader_valid = torch.utils.data.DataLoader(
            dataset_valid, batch_size=batchSize,
            sampler=SamplerFromIndices(dataset, perm_valid),
            num_workers=8)
    elif data_source == 'generator':
        predictions_filename = 'yteacher_train_generator.th'

        G = Gen(imageSize=imageSize, nb_classes=nb_classes)
        G.load_state_dict(torch.load(generator))
        G = G.cuda()

        z = torch.randn(batchSize, nz, 1, 1)
        z = Variable(z)
        z = z.cuda()

        onehot = torch.zeros(batchSize, nb_classes, 1, 1)
        onehot = Variable(onehot)
        onehot = onehot.cuda()

        u = torch.zeros(batchSize, nz, 1, 1)
        u = u.cuda()

        perm = np.arange(len(dataset))
        np.random.shuffle(perm)
        perm = torch.from_numpy(perm)
        perm_train = perm[0:40000]
        perm_valid = perm[40000:]
        #perm_train = perm_train[0:64]
        #perm_valid = perm_valid[0:64]
        nb_valid_examples = len(perm_valid)
        nb_minibatches = len(perm_train) // batchSize
        nb_train_examples = nb_minibatches * batchSize
        dataloader_train = GeneratorLoader(G, z, onehot, u, nb_minibatches)
        dataloader_valid = torch.utils.data.DataLoader(
            dataset_valid, batch_size=batchSize,
            sampler=SamplerFromIndices(dataset, perm_valid),
            num_workers=8)
    else:
        raise ValueError('Unknown data_source : {}'.format(data_source))

    dataloader_train = LoaderDataAugmentation(dataloader_train, nb_epochs=nb_epochs)
    batches_per_epoch = nb_train_examples // batchSize
    if not os.path.exists(predictions_filename):
        yteacher_train = torch.zeros(len(dataloader_train) * batchSize, nb_classes)
        print('Getting and storing predictions of train...')
        for b, (X, y) in enumerate(dataloader_train):
            if b % 100 == 0:
                print('batch {}/{}'.format(b, len(dataloader_train)))
            """
            # visualize samples (uncomment to execute)
            from machinedesign.viz import grid_of_images
            from skimage.io import imsave
            img = grid_of_images(X.cpu().numpy(), normalize=True)
            imsave('sample.png', img)
            """
            input.data.resize_(X.size()).copy_(X)
            y_true = clf(norm(input, clf_mean, clf_std))
            yteacher_train[b * batchSize:b * batchSize + input.size(0)].copy_(y_true.data)
        torch.save(yteacher_train, predictions_filename)
    else:
        yteacher_train = torch.load(predictions_filename)
    """
    # check if repassing through dataloader_train is determenistic
    # (uncomment to execute)
    for b, (X, y) in enumerate(dataloader_train):
        print(X.sum())

    for b, (X, y) in enumerate(dataloader_train):
        print(X.sum())
    sys.exit(0)
    """
    print('Start training...')
    nb_updates = 0
    nb_reduce_lr = 0
    max_valid_acc = 0.
    last_reduced = 0
    valid_accs = []
    stats = defaultdict(list)
    for epoch in range(nb_epochs * nb_passes):
        print('Start epoch : {}'.format(epoch + 1))
        ep = epoch % nb_epochs
        S.train(True)
        for b, (X, y) in enumerate(dataloader_train.epoch(ep)):
            t = time.time()
            input.data.resize_(X.size()).copy_(X)
            S.zero_grad()
            idx = ep * nb_train_examples + b * batchSize
            y_true = yteacher_train[idx:idx + input.size(0)]
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
            if b % 100 == 0:
                print('[{}/{}] batch {}/{}. moving_loss:{:.3f} moving_acc:{:.3f}, time : {:.3f}'.format(epoch, nb_epochs, b, batches_per_epoch, avg_loss, avg_acc, dt))
            nb_updates += 1
        S.train(False)
        accs = []

        for b, (X, y) in enumerate(dataloader_valid):
            input.data.resize_(X.size()).copy_(X)
            y_true = torch.zeros(y.size(0), nb_classes)
            y_true.scatter_(1, y.view(y.size(0), 1), 1)
            y_true = Variable(y_true).cuda()
            y_pred = S(input)
            acc = get_acc(y_true, y_pred)
            accs.append(acc.data[0])
        valid_acc = float(np.mean(accs))
        if valid_acc > max_valid_acc:
            print('improvement of validation acc : from {:.3f} to {:.3f}'.format(max_valid_acc, valid_acc))
            max_valid_acc = valid_acc
            filename = '{{folder}}//student.th'
            print('saving model to : {}'.format(filename))
            torch.save(S.state_dict(), filename)
        valid_accs.append(valid_acc)

        pd.DataFrame(stats).to_csv('{{folder}}/stats.csv', index=False)
        pd.DataFrame(valid_accs).to_csv('{{folder}}/valid.csv', index=False)

        print('valid acc : {}'.format(valid_acc))
        nb_epochs_before_reduce = 10
        if len(valid_accs) > nb_epochs_before_reduce and (epoch - last_reduced) > 8:
            up_to_last = valid_accs[0:-nb_epochs_before_reduce]
            max_valid = max(up_to_last)
            max_last = max(valid_accs[-nb_epochs_before_reduce:])
            if max_valid >= max_last:
                print('{} epochs without improvement : reduce lr'.format(nb_epochs_before_reduce))
                for param_group in optimizer.param_groups:
                    param_group['lr'] /= 2.
                nb_reduce_lr += 1
                last_reduced = epoch
                if nb_reduce_lr == 12:
                    print('reducing lr more 12 times, quit.')
                    break
    return max_valid_acc

if __name__ == '__main__':
    result = run(train)
