from __future__ import print_function
from collections import defaultdict
import time
import pandas as pd
import warnings
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

from PIL import Image

import traceback

from lightjob.cli import load_db
from lightjob.db import AVAILABLE, PENDING, RUNNING, SUCCESS, ERROR


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
    def __init__(self, nc, w, h, no, fc1=250, fc2=1000):
        super(MLPStudent, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(nc * w * h, fc1),
            nn.Linear(fc1, fc2),
            nn.BatchNorm1d(fc2),
            nn.ReLU(True),
            nn.Linear(fc2, no),
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


class ConvFcStudent(nn.Module):
    def __init__(self, nc, w, h, no, nbf=512, fc1=250, fc2=2000, sf=5):
        super(ConvFcStudent, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(nc, nbf, sf),
            nn.ReLU(True),
            nn.MaxPool2d(2),
        )
        wout = (w - sf + 1) // 2
        hout = (h - sf + 1) // 2
        self.fc = nn.Sequential(
            nn.Linear(wout * hout * nbf, fc1),
            nn.Linear(fc1, fc2),
            nn.ReLU(True),
            nn.Linear(fc2, no)
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

def insert(*, nb=1, where='dataset'):
    db = load_db()
    nb_inserted = 0
    for _ in range(nb):
        content = _sample()
        print(content)
        nb_inserted += db.safe_add_job(content, model=content['model'], where=where)
    print('Inserted {} row(s) in the db'.format(nb_inserted))

def _sample():
    model = 'convfc'
    rng = random
    if model == 'convfc':
        sf = rng.choice((3, 5))
        nbf = rng.choice((32, 64, 96, 128, 192, 256, 512, 600, 650, 700, 800))
        fc1 = rng.choice((50, 100, 200, 300, 400, 500))
        fc2 = rng.choice((500, 600, 700, 800, 900, 1000, 1200, 1400, 1500, 1800, 2000, 2200, 2300, 2500, 3000))
        hypers = {
            'nbf': nbf,
            'sf': sf,
            'fc1': fc1,
            'fc2': fc2,
        }
    else:
        raise ValueError(model)
    algo = rng.choice(('adam', 'nesterov', 'sgd'))
    momentum = rng.uniform(0.5, 0.95) if algo == 'nesterov' else None
    lr = _loguniform(rng, -5, -2)
    params = {
        'model': model,
        'hypers': hypers,
        'algo': algo,
        'lr': lr,
        'algo': algo,
        'momentum': momentum,
    }
    return params


def _loguniform(rng, low=0, high=1, base=10):
    return base ** rng.uniform(low, high)

def train(id, *, budget_secs=3600. * 6):
    job_summary = id
    db = load_db()
    job = db.get_job_by_summary(id)
    params = job['content']
    state = job['state']
    
    if state != AVAILABLE:
        warnings.warn('Job with id "{}" has a state : "{}", expecting it to be : "{}".Skip.'.format(job_summary, state, AVAILABLE))
        return

    db.modify_state_of(job_summary, RUNNING)
    params['folder'] = _get_outdir(job_summary)
    params['data_source'] = job['where']
    params['budget_secs'] = budget_secs
    try:
        result = _train_model(params)
    except Exception as ex:
        traceback = _get_traceback() 
        warnings.warn('Job with id "{}" raised an exception : {}. Putting state to "error" and saving traceback.'.format(job_summary, ex))
        db.job_update(job_summary, {'traceback': traceback})
        db.modify_state_of(job_summary, ERROR)
    else:
        db.modify_state_of(job_summary, SUCCESS)
        db.job_update(job_summary, {'stats': result})
        print('Job {} succesfully trained !'.format(job_summary))


def _get_traceback():
    lines  = traceback.format_exc().splitlines()
    lines = '\n'.join(lines)
    return lines


def _get_outdir(job_summary):
    return 'jobs/{}'.format(job_summary)


def _train_model(params):
    classifier = '/home/mcherti/work/code/external/densenet.pytorch/model/model.th'
    generator = '../generators/samples/samples_pretrained_aux_dcgan_32/netG_epoch_35.pth'
    nb_passes = 10
    batchSize = 32 
    nz = 100 
    nb_epochs = 200
    imageSize = 32
    nb_classes = 10
    nb_epochs_before_reduce = 10
    max_times_reduce_lr = 12
    gamma = 0.5
    reduce_wait = 8
    dataroot = '/home/mcherti/work/data/cifar10'
    data_source = params['data_source']
    
    hypers = params['hypers']
    nbf = hypers['nbf']
    fc1 = hypers['fc1']
    fc2 = hypers['fc2']
    algo = params['algo']
    lr = params['lr']
    momentum = params['momentum']

    budget_secs = float(params['budget_secs'])
    t0 = time.time()

    folder = params['folder']
    if not os.path.exists(folder):
        os.makedirs(folder)

    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    sys.path.append(os.path.dirname(classifier))
    sys.path.append(os.path.dirname(classifier) + '/..')
    clf = torch.load(classifier)
        
    mean = [0.49139968, 0.48215827, 0.44653124]
    std = [0.24703233, 0.24348505, 0.26158768]
    
    clf_mean = Variable(torch.FloatTensor(mean).view(1, -1, 1, 1)).cuda()
    clf_std = Variable(torch.FloatTensor(std).view(1, -1, 1, 1)).cuda()

    S = ConvFcStudent(3, imageSize, imageSize, nb_classes, nbf=nbf, fc1=fc1, fc2=fc2)
    S.apply(weights_init)
    S = S.cuda()
    
    input = torch.zeros(batchSize, 3, imageSize, imageSize)
    input = Variable(input)
    input = input.cuda()

    if algo == 'adam':
        optimizer = torch.optim.Adam(S.parameters(), lr=lr)
    elif algo == 'nesterov':
        optimizer = torch.optim.SGD(S.parameters(), lr=lr, nesterov=True, momentum=momentum)
    elif algo == 'sgd':
        optimizer = torch.optim.SGD(S.parameters(), lr=lr)

    avg_loss = 0.
    avg_acc = 0.
    
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

        u = torch.zeros(batchSize, nb_classes, 1, 1)
        u = u.cuda()

        perm = np.arange(len(dataset))
        np.random.shuffle(perm)
        perm = torch.from_numpy(perm)
        perm_train = perm[0:40000]
        perm_valid = perm[40000:]
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
    # Check the accuracy of the teacher on the generated data
    # good if high
    accs = []
    for X, y in dataloader_train:
        input.data.resize_(X.size()).copy_(X)
        y = y.view(-1, 1).cpu()
        y_true = torch.zeros(y.size(0), nb_classes)
        y_true.scatter_(1, y, 1)
        y_pred = clf(norm(input, clf_mean, clf_std)).data.cpu()
        acc = get_acc(y_true, y_pred)
        accs.append(acc)
        print(np.mean(accs))
    """

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
            filename = '{}/student.th'.format(folder)
            print('saving model to : {}'.format(filename))
            torch.save(S, filename)
        valid_accs.append(valid_acc)

        pd.DataFrame(stats).to_csv('{}/stats.csv'.format(folder), index=False)
        pd.DataFrame(valid_accs).to_csv('{}/valid.csv'.format(folder), index=False)

        print('valid acc : {}'.format(valid_acc))

        dt = time.time() - t0
        if dt > budget_secs:
            print('Budget finished. Quit')
            break

        if len(valid_accs) > nb_epochs_before_reduce and (epoch - last_reduced) > reduce_wait:
            up_to_last = valid_accs[0:-nb_epochs_before_reduce]
            max_valid = max(up_to_last)
            max_last = max(valid_accs[-nb_epochs_before_reduce:])
            if max_valid >= max_last:
                print('{} epochs without improvement : reduce lr'.format(nb_epochs_before_reduce))
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= gamma
                nb_reduce_lr += 1
                last_reduced = epoch
                if nb_reduce_lr == max_times_reduce_lr:
                    print('reduced lr {} times, quit.'.format(max_times_reduce_lr))
                    break
    result = {
            'valid_acc': valid_accs
    }
    return result

if __name__ == '__main__':
    result = run(train, insert)
