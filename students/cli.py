import math
import time
import warnings
import math
import sys
import time
import os
import random
import traceback
from collections import defaultdict
from itertools import chain
from functools import partial
from clize import run

import numpy as np
import pandas as pd
from skimage.io import imsave
from PIL import Image

import torch
import torch.nn as nn
import random
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data.sampler import Sampler
from torch.autograd import Variable
from torch.nn.init import xavier_uniform

from lightjob.cli import load_db
from lightjob.db import AVAILABLE, PENDING, RUNNING, SUCCESS, ERROR

from data import RandomHorizontalFlip
from data import Rotation
from data import HSV
from data import RandomSizedCrop
from data import hsv_augmentation
from data import SamplerFromIndices
from data import DataAugmentationLoader
from data import GeneratorLoader
from data import norm
from data import Tiny
from data import MergeLoader
from data import DataAugmentationLoaders
from data import GeneratorLoaderDomainTransfer

sys.path.append('../generators')


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


class GenDt(nn.Module):
    def __init__(self, nz=100, nc=3, nb_classes=10):
        super(GenDt, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nz + nc + nb_classes, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64 * 2, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(64 * 2,     64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(    64,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)


class MLPStudent(nn.Module):
    def __init__(self, nc, w, h, no, fc1=250, fc2=1000):
        super(MLPStudent, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(nc * w * h, fc1),
            nn.Linear(fc1, fc2),
            nn.ReLU(True),
            nn.Linear(fc2, no),
        )

    def forward(self, input):
        input = input.view(input.size(0), -1)
        output = self.main(input)
        return output


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

class Conv2FcStudent(nn.Module):
    def __init__(self, nc, w, h, no, nbf1=512, nbf2=512, fc=250, sf=5):
        super(Conv2FcStudent, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(nc, nbf1, sf),
            nn.ReLU(True),
            nn.MaxPool2d(2),

            nn.Conv2d(nbf1, nbf2, sf),
            nn.ReLU(True),
            nn.MaxPool2d(2),

        )
        wout = ((w - sf + 1) // 2 - sf + 1) // 2
        hout = ((h - sf + 1) // 2 - sf + 1) // 2
        self.fc = nn.Sequential(
            nn.Linear(wout * hout * nbf2, fc),
            nn.ReLU(True),
            nn.Linear(fc, no)
        )
        
    def forward(self, input):
        x = self.features(input)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Conv2FcBLStudent(nn.Module):
    def __init__(self, nc, w, h, no, nbf1=512, nbf2=512, fc1=250, fc2=250, sf=5):
        super(Conv2FcBLStudent, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(nc, nbf1, sf),
            nn.ReLU(True),
            nn.MaxPool2d(2),

            nn.Conv2d(nbf1, nbf2, sf),
            nn.ReLU(True),
            nn.MaxPool2d(2),

        )
        wout = ((w - sf + 1) // 2 - sf + 1) // 2
        hout = ((h - sf + 1) // 2 - sf + 1) // 2
        self.fc = nn.Sequential(
            nn.Linear(wout * hout * nbf2, fc1),
            nn.Linear(fc1, fc2),
            nn.ReLU(True),
            nn.Linear(fc2, no)
        )
        
    def forward(self, input):
        x = self.features(input)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Conv2FullyStudent(nn.Module):
    def __init__(self, nc, w, h, no, nbf1=512, nbf2=512, nbf3=512, sf=5):
        super(Conv2FullyStudent, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(nc, nbf1, sf),
            nn.ReLU(True),
            nn.MaxPool2d(2),

            nn.Conv2d(nbf1, nbf2, sf),
            nn.ReLU(True),
            nn.MaxPool2d(2),

            nn.Conv2d(nbf2, nbf3, sf),
            nn.ReLU(True),

        )
        self.fc = nn.Sequential(
            nn.Linear(nbf3, no)
        )
        
    def forward(self, input):
        x = self.features(input)
        x = nn.AvgPool2d(kernel_size=x.size()[2:])(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def weights_init(m, xavier=False):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        if xavier:
            xavier_uniform(m.weight.data, gain=math.sqrt(2.))
            m.bias.data.fill_(0)
        else:
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


def insert(*, nb=1, data_source=None, model=None, bayesopt=False, hypers_from=None):
    nb_inserted = 0
    db = load_db()
    params_list = _sample(nb=nb, data_source=data_source, model=model, bayesopt=bayesopt)

    if hypers_from:
        job = db.get_job_by_summary(hypers_from)
        hypers = job['content']['hypers']
    else:
        hypers = None

    for params in params_list:
        if hypers:
            params['hypers'] = hypers
        print(params)
        _check(params)
        nb_inserted += db.safe_add_job(params, model=params['model'])
    print('Inserted {} row(s) in the db'.format(nb_inserted))


def insert_best(*, data_source=None, model=None, top=1, data_source_from=None):
    db = load_db()
    jobs = db.jobs_with_state(SUCCESS)
    jobs = list(jobs)
    if data_source_from:
        jobs = [j for j in jobs if j['content']['data_source'] == data_source_from]
    X = [j['content'] for j in jobs]
    y = [np.max(j['stats']['valid_acc']) for j in jobs]
    indices = np.argsort(y)[::-1]
    nb_inserted = 0
    for ind in indices[0:top]:
        params = X[ind]
        value = y[ind]
        if data_source:
            params['data_source'] = data_source
        _check(params)
        print(params, value)
        nb_inserted += db.safe_add_job(params, model=params['model'])
    print('Inserted {} row(s) in the db'.format(nb_inserted))


def sample(*, nb=1, data_source=None, model=None, bayesopt=False):
    params_list = _sample(nb=nb, data_source=data_source, model=model, bayesopt=bayesopt)
    for params in params_list:
        print(params)


def _sample(nb=1, data_source=None, model=None, bayesopt=False):
    db = load_db()
    rng = random
    sample_func = partial(_sample_unif, data_source=data_source, model=model)
    if bayesopt:
        params_list = _sample_bayesopt(nb=nb, sample_func=sample_func, data_source=data_source)
    else:
        params_list = [sample_func(rng) for _ in range(nb)]
    return params_list


def _sample_unif(rng, model=None, data_source=None):
    if not model:
        model = rng.choice(('convfc', 'conv2fc', 'conv2fully', 'conv2fcbl'))
    if not data_source:
        data_source = rng.choice((
            'dataset', 
            'dataset_simple', 
            'dataset_raw',
            'tiny',
            'aux2', 
            'aux3', 
            'dataset_simple,aux2', 
            'dataset_simple,aux3', 
            'dataset_simple,dataset_simple',
            'aux2_big',
            'aux3_big'
        ))
    if model == 'convfc':
        sf = rng.choice((3, 5))
        nbf = rng.choice((32, 64, 96, 128, 192, 256, 512, 600, 650, 700, 800, 900, 1000))
        fc1 = rng.randint(1, 10) * 100
        fc2 = rng.randint(1, 100) * 100
        hypers = {
            'nbf': nbf,
            'sf': sf,
            'fc1': fc1,
            'fc2': fc2,
        }
    elif model == 'conv2fc':
        sf = rng.choice((3, 5))
        nbf1 = rng.choice((32, 64, 96, 128, 192, 256, 512))
        nbf2 = rng.choice((32, 64, 96, 128, 192, 256, 512))
        fc = rng.randint(1, 10) * 1000
        hypers = {
            'nbf1': nbf1,
            'nbf2': nbf2,
            'sf': sf,
            'fc': fc,
        }
    elif model == 'conv2fcbl':

        sf = rng.choice((3, 5))
        nbf1 = rng.choice((32, 64, 96, 128, 192, 256, 512))
        nbf2 = rng.choice((32, 64, 96, 128, 192, 256, 512))
        fc1 = rng.randint(1, 10) * 100
        fc2 = rng.randint(1, 10) * 100
        hypers = {
            'nbf1': nbf1,
            'nbf2': nbf2,
            'sf': sf,
            'fc1': fc1,
            'fc2': fc2
        }
    elif model == 'conv2fully':
        sf = rng.choice((3, 5))
        nbf1 = rng.choice((32, 64, 96, 128, 192, 256, 512))
        nbf2 = rng.choice((32, 64, 96, 128, 192, 256, 512))
        nbf3 = rng.choice((32, 64, 96, 128, 192, 256, 512))
        hypers = {
            'nbf1': nbf1,
            'nbf2': nbf2,
            'nbf3': nbf3,
            'sf': sf,
        }
    elif model == 'mlp':
        fc1 = rng.randint(1, 10) * 100
        fc2 = rng.randint(1, 100) * 100
        hypers = {
            'fc1': fc1,
            'fc2': fc2
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
    params['xavier'] = True
    params['data_source'] = data_source 
    return params


def _loguniform(rng, low=0, high=1, base=10):
    return base ** rng.uniform(low, high)


def _sample_bayesopt(*, nb=1, sample_func=_sample_unif, data_source=None):
    from fluentopt.bandit import Bandit
    from fluentopt.bandit import ucb_maximize
    from fluentopt.transformers import Wrapper
    from fluentopt.utils import RandomForestRegressorWithUncertainty
    from lightjob.db import SUCCESS
    from sklearn.ensemble import RandomForestRegressor
    import pandas as pd
    db = load_db()
    jobs = list(db.jobs_with_state(SUCCESS)) + list(db.jobs_with_state(ERROR))
    reg = RandomForestRegressorWithUncertainty(
        n_estimators=100, 
        min_samples_leaf=5,
        oob_score=True)
    extra = [j['content'] for j in jobs]
    opt = Bandit(
        sampler=sample_func, 
        score=ucb_maximize, 
        model=Wrapper(reg, transform_X=partial(_transform, extra=extra)),
        nb_suggestions=1000
    )
    if data_source:
        jobs = [j for j in jobs if j['content']['data_source'] == data_source]
    X = [j['content'] for j in jobs]
    y = [np.max(j['stats']['valid_acc']) if j['state'] == SUCCESS else 0 for j in jobs]
    print('{} examples from the surrogate to learn from'.format(len(X)))
    opt.update_many(X, y)
    print(reg.oob_score_)
    return [opt.suggest() for _ in range(nb)]


def _transform(dlist, extra=None):
    import copy
    from fluentopt.transformers import vectorize
    from fluentopt.utils import flatten_dict
    rng = np.random


    extra = copy.deepcopy(extra)
    extra.extend([_sample_unif(np.random) for _ in range(1000)])
    dlist = copy.deepcopy(dlist)
    dlist.extend(extra)
    for i, d in enumerate(dlist):
        dlist[i] = flatten_dict(d)
    df = pd.DataFrame(dlist)
    df = pd.get_dummies(df)
    df = df.fillna(-1)
    df = df[0:-len(extra)]
    return df.values


def _check(params):
    allowed = (
         'aux1', 
        'aux2', 
        'dataset', 
        'dataset_simple', 
        'aux3', 
        'tiny', 
        'dataset_raw', 
        'cond1',
        'tiny_dt'
    )
    data_source = params['data_source']
    if ',' in data_source:
        for ds in data_source.split(','):
            assert ds in allowed, 'Wrong data source : "{}"'.format(ds)
        return
    else:
        if '_big' in data_source:
            data_source = data_source.replace('_big', '')
        assert data_source in allowed, 'Wrong data source : "{}"'.format(data_source)
    

def clean():
    from shutil import rmtree
    db = load_db()
    # remove student.th form jobs with ERROR state
    jobs = db.jobs_with(state=ERROR)
    for job in jobs:
        folder = os.path.join('jobs', job['summary'])
        f = os.path.join(folder, 'student.th')
        if os.path.exists(f):
            os.remove(f)
            print(f)
    # remove job folders that do not exist in the DB
    jobs = db.all_jobs()
    summaries = set(j['summary'] for j in jobs)
    for dirname in os.listdir('jobs'):
        path = os.path.join('jobs', dirname)
        if dirname not in summaries and os.path.exists(path):
            print(path)
            try:
                rmtree(path)
            except OSError as ex:
                print(ex)
    # make jobs that stopped abruptly without chaning state to ERROR
    # to AVAILABLE again
    jobs = chain(db.jobs_with(state=RUNNING) , db.jobs_with(state=ERROR))
    for job in jobs:
        dirname = job['summary']
        output = os.path.join('jobs', dirname, 'output')
        if not os.path.exists(output):
            continue
        with open(output, 'r') as fd:
            s = fd.read()
        has_error = (
            ("all CUDA-capable devices are busy or unavailable" in s) or
            ("no CUDA-capable device is detected" in s) or 
            ("ValueError: nothing to open" in s)
        )
        if has_error:
            print('make the state of {} available'.format(job['summary']))
            db.modify_state_of(job['summary'], AVAILABLE)



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
    params['budget_secs'] = budget_secs
    try:
        result = _train_model(params)
    except Exception as ex:
        traceback = _get_traceback() 
        print(traceback)
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


def train_random(*, data_source=None, model=None, bayesopt=False):
    rng = random
    sample_func = partial(_sample_unif, data_source=data_source, model=model)
    if bayesopt:
        params, = _sample_bayesopt(nb=1, sample_func=sample_func, data_source=data_source)
    else:
        params = sample_func(rng)
    if data_source:
        params['data_source'] = data_source
    params['budget_secs'] = 3600
    params['folder'] = 'out/tmp'
    print(params)
    _train_model(params)


def resume(*, job, folder='out/tmp', budget_secs=3600., filename=None, lr=None):
    db = load_db()
    job = db.get_job_by_summary(job)
    params = job['content']
    params['folder'] = folder
    params['budget_secs'] = budget_secs
    if filename:
        params['student'] = filename
    else:
        params['student'] = 'jobs/{}/student.th'.format(job['summary'])
    if lr:
        lr = float(lr)
        print('previous lr : {}, new lr : {}'.format(params['lr'], lr))
        params['lr'] = lr
    _train_model(params)


def _get_data(data_source, batchSize=32, augment=True):
    np.random.seed(42)
    dataroot = '/home/mcherti/work/data/cifar10'
    imageSize = 32
    nb_classes = 10
    nz = 100
    if data_source.endswith('_big'):
        nb_epochs = 400
    else:
        nb_epochs = 200
    generators = {
        'aux1' : '../generators/samples/samples_pretrained_aux_dcgan_32/netG_epoch_35.pth',  #trained using pretrained_aux_dcgan_32.py
        'aux2' : '../generators/samples/samples_pretrained_aux_cifar/netG_epoch_72.pth',     #trained using pretrained_aux_dcgan_32.py
        'aux3' : '../generators/samples/samples_cond_dcgan_cls_32/netG_epoch_72.pth',        #trained using cond_dcgan_cls_32.py
        'cond1': '../generators/samples/samples_cond_dcgan_cifar10_bs32/netG_epoch_97.pth',      #trained using cond_dcgan_32.py
        'tiny_dt': '../generators/samples/samples_cond_dcgan_dtcls_3_32/netG_epoch_56.pth'
    }
    classifier = '/home/mcherti/work/code/external/densenet.pytorch/model/model.th'
    if ',' in data_source:
        data_sources = data_source.split(',')
        dl_trains = []
        yt_trains = []
        nb_trains = []
        bs = batchSize
        for data_source in data_sources:
            dl_train, dl_valid, yt_train, nb_train = _get_data(data_source, batchSize=bs, augment=False)
            dl_trains.append(dl_train)
            yt_trains.append(yt_train)
            nb_trains.append(nb_train)
        nb_train_examples = sum(nb_trains)
        dataloader_valid = dl_valid
        assert len(set(yt.size(0) for yt in yt_trains)) == 1
        per_epoch = yt_trains[0].size(0) // nb_epochs
        size = sum(yt.size(0) for yt in yt_trains)
        yteacher_train = torch.zeros(size, nb_classes)
        o = 0
        l = 0
        for epoch in range(nb_epochs):
            for yt in yt_trains:
                yteacher_train[l:l + per_epoch] = yt[o:o + per_epoch]
                l += per_epoch
            o += per_epoch
        dataloader_train = DataAugmentationLoaders(dl_trains, nb_epochs=nb_epochs)
        return dataloader_train, dataloader_valid, yteacher_train, nb_train_examples
    
    data_source_prefix = data_source.replace('_big', '')
    generator = generators.get(data_source_prefix)
    if data_source_prefix == 'dataset_simple':
        transform = transforms.Compose([
            transforms.Scale(imageSize),
            transforms.CenterCrop(imageSize),
            Rotation(-10, 10),
            RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    elif data_source_prefix == 'dataset_raw':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    else:
        transform = transforms.Compose([
            HSV(0.06, 0.26, 0.2, 0.21, 0.13),
            RandomHorizontalFlip(),
            RandomSizedCrop(24),
            transforms.Scale(32, interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    transform_valid = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = dset.CIFAR10(root=dataroot, download=True, transform=transform)
    dataset_valid = dset.CIFAR10(root=dataroot, download=True, transform=transform_valid)
    if data_source_prefix in ('dataset', 'dataset_old', 'dataset_simple', 'dataset_raw'):
        predictions_filename = 'predictions/{}.th'.format(data_source)
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
            num_workers=1)
        dataloader_valid = torch.utils.data.DataLoader(
            dataset_valid, batch_size=batchSize,
            sampler=SamplerFromIndices(dataset, perm_valid),
            num_workers=8)
    elif data_source_prefix == 'tiny':
        predictions_filename = 'predictions/{}.th'.format(data_source)
        perm = np.arange(len(dataset))
        np.random.shuffle(perm)
        perm = torch.from_numpy(perm)
        perm_train = perm[0:40000]
        perm_valid = perm[40000:]
        nb_train_examples = len(perm_train)
        nb_valid_examples = len(perm_valid)
        tiny = Tiny(
           '/home/mcherti/work/data/tiny_images/tiny_images_subset.bin', 
           transform=transforms.Compose([
               transforms.ToTensor(),
               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        )
        dataloader_train = torch.utils.data.DataLoader(
            tiny, batch_size=batchSize,
            num_workers=1)
        dataloader_valid = torch.utils.data.DataLoader(
            dataset_valid, batch_size=batchSize,
            sampler=SamplerFromIndices(dataset, perm_valid),
            num_workers=8)

    elif data_source_prefix == 'tiny_dt':
        predictions_filename = 'predictions/{}.th'.format(data_source)
        perm = np.arange(len(dataset))
        np.random.shuffle(perm)
        perm = torch.from_numpy(perm)
        perm_train = perm[0:40000]
        perm_valid = perm[40000:]
        nb_train_examples = len(perm_train)
        nb_valid_examples = len(perm_valid)
        tiny = Tiny(
           '/home/mcherti/work/data/tiny_images/tiny_images_subset.bin', 
           transform=transforms.Compose([
               transforms.ToTensor(),
               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        )
        source = torch.utils.data.DataLoader(
            tiny, batch_size=batchSize,
            num_workers=1)
        source = DataAugmentationLoader(source, nb_epochs=1)

        G = GenDt(nz=nz, nc=3, nb_classes=nb_classes)
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
        dataloader_train = GeneratorLoaderDomainTransfer(G=G, z=z, onehot=onehot, u=u, source=source) 
        dataloader_valid = torch.utils.data.DataLoader(
            dataset_valid, batch_size=batchSize,
            sampler=SamplerFromIndices(dataset, perm_valid),
            num_workers=8)
    else:
        predictions_filename = 'predictions/{}.th'.format(data_source)
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

    if augment:
        dataloader_train = DataAugmentationLoader(dataloader_train, nb_epochs=nb_epochs)

    if not os.path.exists(predictions_filename):
        sys.path.append(os.path.dirname(classifier))
        sys.path.append(os.path.dirname(classifier) + '/..')
        clf = torch.load(classifier)
            
        mean = [0.49139968, 0.48215827, 0.44653124]
        std = [0.24703233, 0.24348505, 0.26158768]
        
        clf_mean = Variable(torch.FloatTensor(mean).view(1, -1, 1, 1)).cuda()
        clf_std = Variable(torch.FloatTensor(std).view(1, -1, 1, 1)).cuda()
     
        yteacher_train = torch.zeros(len(dataloader_train) * batchSize, nb_classes)
        print('Getting and storing predictions of train...')
        input = torch.zeros(batchSize, 3, imageSize, imageSize)
        input = Variable(input)
        input = input.cuda()
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
    for b, (X, y) in enumerate(dataloader_train):
        # visualize samples (uncomment to execute)
        from machinedesign.viz import grid_of_images
        from skimage.io import imsave
        img = grid_of_images(X.cpu().numpy(), normalize=True)
        imsave('sample.png', img)
        break
    """
    """
    # check if repassing through dataloader_train is determenistic
    # (uncomment to execute)
    for b, (X, y) in enumerate(dataloader_train):
        print(X.sum())
        break
    for b, (X, y) in enumerate(dataloader_train):
        print(X.sum())
        break
    sys.exit(0)
    """
    return dataloader_train, dataloader_valid, yteacher_train, nb_train_examples

def _train_model(params):
    nb_passes = 10
    batchSize = 32 
    imageSize = 32
    nb_classes = 10
    nb_epochs_before_reduce = 10
    max_times_reduce_lr = 12
    gamma = 0.5
    reduce_wait = 8
    dataroot = '/home/mcherti/work/data/cifar10'
    data_source = params['data_source']
    classifier = '/home/mcherti/work/code/external/densenet.pytorch/model/model.th'
    print('data source : {}'.format(data_source))
    
    hypers = params['hypers']
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
    
    m = params['model']
    if m == 'convfc':
        nbf = hypers['nbf']
        fc1 = hypers['fc1']
        fc2 = hypers['fc2']
        S = ConvFcStudent(3, imageSize, imageSize, nb_classes, nbf=nbf, fc1=fc1, fc2=fc2)
    elif m == 'conv2fc':
        nbf1 = hypers['nbf1']
        nbf2 = hypers['nbf2']
        fc = hypers['fc']
        S = Conv2FcStudent(3, imageSize, imageSize, nb_classes, nbf1=nbf1, nbf2=nbf2, fc=fc)
    elif m == 'conv2fcbl':
        nbf1 = hypers['nbf1']
        nbf2 = hypers['nbf2']
        fc1 = hypers['fc1']
        fc2 = hypers['fc2']
        S = Conv2FcBLStudent(3, imageSize, imageSize, nb_classes, nbf1=nbf1, nbf2=nbf2, fc1=fc1, fc2=fc2)
    elif m == 'mlp':
        fc1 = hypers['fc1']
        fc2 = hypers['fc2']
        S = MLPStudent(3, imageSize, imageSize, nb_classes, fc1=fc1, fc2=fc2)
    elif m == 'conv2fully':
        nbf1 = hypers['nbf1']
        nbf2 = hypers['nbf2']
        nbf3 = hypers['nbf3']
        S = Conv2FullyStudent(3, imageSize, imageSize, nb_classes, nbf1=nbf1, nbf2=nbf2, nbf3=nbf3)
    else:
        raise ValueError('Wrong model : {}'.format(m))
    
    print('Nb of params : {}'.format(_get_nb_params(S)))

    if params.get('xavier') == True:
        S.apply(partial(weights_init, xavier=True))
    else:
        S.apply(weights_init)
    if 'student' in params:
        S = torch.load(params['student'])
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
    
    dataloader_train, dataloader_valid, yteacher_train, nb_train_examples = _get_data(data_source=data_source, batchSize=batchSize)
    batches_per_epoch = nb_train_examples // batchSize
    nb_epochs = yteacher_train.size(0) // nb_train_examples
    print('Nb epochs : {}'.format(nb_epochs))
    #check if predictions save are correect
    print('check if predictions are correct')
    for b, (X, y) in enumerate(dataloader_train):
        input.data.resize_(X.size()).copy_(X)
        ypred = clf(norm(input, clf_mean, clf_std)).data.cpu()
        ytrue = yteacher_train[b * batchSize:b * batchSize + input.size(0)]
        assert (torch.abs(ytrue - ypred) > 1e-3).sum() == 0
        #if b == batches_per_epoch:
        #    break
        if b == 10:
            break
    """
    # Check the accuracy of the teacher on the generated data
    # good if high
    accs = []
    input = torch.zeros(batchSize, 3, imageSize, imageSize)
    input = Variable(input)
    input = input.cuda()
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
    avg_loss = 0.
    avg_acc = 0.
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
            if np.isnan(loss.data[0]):
                raise ValueError('Nan detected')
            if np.isinf(loss.data[0]):
                raise ValueError('Inf detected')
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

def export(filename='jobs.json'):
    df = _build_jobs_df()
    df.to_json(filename)

def _build_jobs_df():
    import glob
    import os
    import numpy as np
    import json
    from bokeh.plotting import show, output_notebook, figure, ColumnDataSource
    from bokeh.charts import Scatter
    from bokeh.models import HoverTool, BoxSelectTool
    from bokeh.models import BoxZoomTool
    from lightjob.cli import load_db
    from lightjob.db import SUCCESS
    from lightjob.utils import summarize

    db = load_db()
    rows = []
    jobs = db.jobs_with(state=SUCCESS)
    for job in jobs:
        folder = '../students/jobs/{}'.format(job['summary'])
        stats = os.path.join(folder, 'stats.csv')
        if not os.path.exists(stats):
            continue
        df_stats = pd.read_csv(stats)
        df_valid = pd.read_csv(os.path.join(folder, 'valid.csv'))
        df_valid = df_valid[df_valid.columns[0]].values
        hypers = job['content']['hypers']
        start_time = job['life'][0]['dt']
        end_time = job['life'][0]['dt']    
        acc = _moving(df_stats['acc'])
        row = job['content'].copy()
        del row['hypers']
        row.update(hypers)

        row['train_acc'] = acc[::1250]
        row['valid_acc'] = df_valid.tolist()
        row['max_valid_acc'] = float(df_valid.max())
        row['last_train_acc'] = row['train_acc'][-1]
        row['max_train_acc'] = np.max(row['train_acc'])
        row['id'] = os.path.basename(folder)
        row['nb_updates'] = len(df_stats)
        row['start_time'] = start_time
        row['end_time'] = end_time
        row['n_epochs'] = (len(df_stats)*32)/40000
        row['hypers_id'] = summarize(hypers)
        filename = os.path.join(folder, 'student.th')
        row['nb_params'] = _get_nb_params_from_filename(filename) 
        rows.append(row)
    df = pd.DataFrame(rows)
    return df


def _get_nb_params_from_filename(filename):
    clf = torch.load(filename, map_location=lambda storage, loc: storage)
    return _get_nb_params(clf)


def _get_nb_params(clf):
    return sum(map(lambda p:p.view(-1).size(0), clf.parameters()))


def _moving(l):
    v = 0.
    lm = []
    for val in l:
        v = v * 0.9999 + val * 0.0001
        lm.append(v)
    return lm


def manual():
    nb_inserted = 0
    db = load_db()
    job = db.get_job_by_summary('91e2c05e100ae416e69b7e983c5be906')
    params = job['content']
    params['hypers']['fc'] = 1200
    nb_inserted += db.safe_add_job(params, model=params['model'])
    params['data_source'] = 'dataset'
    nb_inserted += db.safe_add_job(params, model=params['model'])
    params['data_source'] = 'dataset,aux2'
    nb_inserted += db.safe_add_job(params, model=params['model'])
    params['data_source'] = 'dataset,aux2,aux3'
    nb_inserted += db.safe_add_job(params, model=params['model'])
    params['data_source'] = 'dataset,aux2,tiny'
    nb_inserted += db.safe_add_job(params, model=params['model'])

    job = db.get_job_by_summary('585e3be7a561b047bd2b08ad6bd52e6a')
    params = job['content']
    params['data_source'] = 'dataset_simple,aux2'
    nb_inserted += db.safe_add_job(params, model=params['model'])
    params['data_source'] = 'dataset_raw,aux2'
    nb_inserted += db.safe_add_job(params, model=params['model'])
    print(nb_inserted)


if __name__ == '__main__':
    result = run(train, insert, clean, train_random, sample, insert_best, resume, export, manual)
