"""
autoencoder where the encoder part is frozen
and the encoder part is learned.
the encoder takes as input the image, and produces
a latent variable h which is a layer from alexnet, then that latent variable,
is used to reconstruct the image.
this is not a GAN, the reconstruction loss is MSE.
"""
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
from torch.autograd import Variable

sys.path.append('/home/mcherti/work/code/external/ppgn')
from caffe_to_pytorch import Generator

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='cifar10 | lsun | imagenet | folder | lfw ')
    parser.add_argument('--dataroot', required=True, help='path to dataset')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
    parser.add_argument('--ngpu'  , type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')

    opt = parser.parse_args()
    print(opt)

    try:
        os.makedirs(opt.outf)
    except OSError:
        pass
    opt.manualSeed = random.randint(1, 10000) # fix seed
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    cudnn.benchmark = True

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    if opt.dataset in ['imagenet', 'folder', 'lfw']:
        # folder dataset
        dataset = dset.ImageFolder(root=opt.dataroot,
                                   transform=transforms.Compose([
                                       transforms.Scale(opt.imageSize),
                                       transforms.CenterCrop(opt.imageSize),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]))
    elif opt.dataset == 'lsun':
        dataset = dset.LSUN(db_path=opt.dataroot, classes=['bedroom_train'],
                            transform=transforms.Compose([
                                transforms.Scale(opt.imageSize),
                                transforms.CenterCrop(opt.imageSize),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
    elif opt.dataset == 'cifar10':
        dataset = dset.CIFAR10(root=opt.dataroot, download=True,
                               transform=transforms.Compose([
                                   transforms.Scale(opt.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ])
        )
    assert dataset
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=opt.batchSize,
        shuffle=True, 
        num_workers=int(opt.workers))

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
        elif hasattr(m, 'weight') and hasattr(m, 'bias'):
            m.weight.data.normal_(0.0, 0.02)
            m.bias.data.fill_(0)


    class _netG(nn.Module):
        def __init__(self, ngpu):
            super(_netG, self).__init__()
            self.ngpu = ngpu
            self.fc = nn.Sequential(
                nn.Linear(4096, 4096),
                nn.BatchNorm1d(4096),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(4096, 4096),
                nn.BatchNorm1d(4096),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
            )
            self.conv = torch.load('/home/mcherti/work/code/external/ppgn/generator.th')
       
        def forward(self, input):
            y = self.fc(input)
            y = self.conv(y)
            return y

    class AlexNet(nn.Module):
        def __init__(self):
            super(AlexNet, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(64, 192, kernel_size=5, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(192, 384, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
            )
            self.classifier = nn.Sequential(
                nn.Linear(256 * 7 * 7, 4096),
            )

        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), 256 * 7 * 7)
            x = self.classifier(x)
            return x
    ngpu = 1
    netD = AlexNet()
    netD.apply(weights_init)
    netG = _netG(ngpu)
    netG.fc.apply(weights_init)
    criterion = nn.MSELoss()
    input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
    if opt.cuda:
        netG.cuda()
        netD.cuda()
        criterion.cuda()
        input = input.cuda()
    input = Variable(input)
    optimizer = optim.Adam([p for p in chain(netG.fc.parameters(), netD.parameters())], lr = opt.lr, betas = (opt.beta1, 0.999))

    def norm(fake):
        #convert caffe image to image between -1 and 1
        #min_val = min(-120., float(fake.min().data.cpu().numpy()))
        #max_val = max(120., float(fake.max().data.cpu().numpy()))
        min_val = -120
        max_val = 120
        fake = torch.cat( (fake[:, 2:3], fake[:, 1:2], fake[:, 0:1]),  1)
        fake = (fake - min_val) / (max_val - min_val) 
        fake = fake * 2 - 1
        return fake

    for epoch in range(opt.niter):
        for i, data in enumerate(dataloader):

            netD.zero_grad()
            netG.zero_grad()
            real_cpu, _ = data
            batch_size = real_cpu.size(0)
            input.data.resize_(real_cpu.size()).copy_(real_cpu)
            x = input
            h = netD(x)
            x_rec = netG(h)
            x_rec = norm(x_rec)
            err = criterion(x_rec, x)
            err.backward()
            optimizer.step()
            print('[%d/%d][%d/%d] MSE : %.4f' % (epoch, opt.niter, i, len(dataloader), err.data[0]))
            if i % 100 == 0:
                vutils.save_image((x_rec.data[0:64,:,:,:]+1)/2., '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch), nrow=8)
