# like aux_dcgan but where the classifier part
# is a pre-trained one
from __future__ import print_function
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


from loader import ImageFolder


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
        dataset = ImageFolder(root=opt.dataroot,
                              transform=transforms.Compose([
                                 transforms.Scale(opt.imageSize),
                                       transforms.CenterCrop(opt.imageSize),
                                       transforms.RandomHorizontalFlip(),
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

    ngpu = int(opt.ngpu)
    nz = int(opt.nz)
    ngf = int(opt.ngf)
    ndf = int(opt.ndf)
    nc = 3
    nb_classes = 18
    # custom weights initialization called on netG and netD
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    class _netG(nn.Module):
        def __init__(self, ngpu):
            super(_netG, self).__init__()
            self.ngpu = ngpu
            self.main = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(     nz + nb_classes, ngf * 8, 4, 1, 0, bias=False),
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
            gpu_ids = None
            if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
                gpu_ids = range(self.ngpu)
            return nn.parallel.data_parallel(self.main, input, gpu_ids)

    netG = _netG(ngpu)
    netG.apply(weights_init)
    if opt.netG != '':
        netG.load_state_dict(torch.load(opt.netG))
    print(netG)

    class _netD(nn.Module):
        def __init__(self, ngpu):
            super(_netD, self).__init__()
            self.ngpu = ngpu
            self.main = nn.Sequential(
                # input is (nc) x 64 x 64
                nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf) x 32 x 32
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*2) x 16 x 16
                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*4) x 8 x 8
                nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(ndf * 8, ndf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
 
                # state size. (ndf*8) x 4 x 4
                nn.Conv2d(ndf * 8,  1, 4, 1, 0, bias=False),
                nn.Sigmoid()
            )
        def forward(self, input):
            gpu_ids = None
            if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
                gpu_ids = range(self.ngpu)
            output = nn.parallel.data_parallel(self.main, input, gpu_ids)
            return output.view(-1, 1)

    netD = _netD(ngpu)
    netD.apply(weights_init)
    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD))
    print(netD)

    criterion = nn.BCELoss()

    input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
    noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)

    nb_rows = 10
    fixed_z = torch.randn   (nb_rows, nb_classes,        nz, 1, 1)
    #fixed_z = fixed_z.repeat(1,      nb_classes, 1, 1, 1)
    fixed_z = fixed_z.view(nb_rows * nb_classes, nz, 1, 1)
    fixed_onehot = torch.zeros(nb_rows, nb_classes, nb_classes, 1, 1)
    fixed_onehot = fixed_onehot.view(nb_rows * nb_classes, nb_classes, 1, 1)
    for i in range(fixed_onehot.size(0)):
        cl = i % nb_classes
        fixed_onehot[i, cl] = 1
    fixed_noise = torch.cat((fixed_z, fixed_onehot), 1).cuda()
    label = torch.FloatTensor(opt.batchSize)
    real_label = 1
    fake_label = 0

    if opt.cuda:
        netD.cuda()
        netG.cuda()
        criterion.cuda()
        input, label = input.cuda(), label.cuda()
        noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

    sys.path.append('pytorch_pretrained')
    clf = torch.load('pytorch_pretrained/clf-4cf42cbb-3c69-4d3f-86e1-c050330a7c7c.th')
    clf = clf.cuda()

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
        x = (x + 1) / 2.
        x = x - clf_mean.repeat(x.size(0), 1, x.size(2), x.size(3))
        x = x / clf_std.repeat(x.size(0), 1, x.size(2), x.size(3))
        return x

    aux_criterion = nn.CrossEntropyLoss().cuda()

    input = Variable(input)
    label = Variable(label)
    noise = Variable(noise)
    fixed_noise = Variable(fixed_noise)

    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr = opt.lr, betas = (opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr = opt.lr, betas = (opt.beta1, 0.999))
    optimizerC = optim.SGD(clf.parameters(), lr=1e-4)

    for epoch in range(opt.niter):
        for i, data in enumerate(dataloader):
            netD.zero_grad()
            clf.zero_grad()

            real_cpu, real_classes = data

            real_classes = real_classes.long().view(-1, 1)
            real_classes_var = Variable(real_classes[:, 0]).cuda()
            batch_size = real_cpu.size(0)
            
            y_onehot = torch.zeros(batch_size, nb_classes)
            y_onehot.scatter_(1, real_classes, 1)
            y_onehot_ = y_onehot
            y_onehot = y_onehot.view(y_onehot.size(0), y_onehot.size(1), 1, 1)
            y_onehot = y_onehot.repeat(1, 1, real_cpu.size(2), real_cpu.size(3))
            real_cpu_with_class = torch.cat((real_cpu, y_onehot), 1)

            input.data.resize_(real_cpu.size()).copy_(real_cpu)
            label.data.resize_(batch_size).fill_(real_label)
            
            clf_output = clf(norm(input))
            output = netD(input)
            errD_real = (
                criterion(output, label) + 
                aux_criterion(clf_output, real_classes_var)
            )
            
            _, pred = clf_output.max(1)
            acc_real = torch.mean((pred.data.cpu()[:, 0] == real_classes[:, 0]).float())

            errD_real.backward()
            D_x = output.data.mean()

            # train with fake
            z = torch.randn(batch_size, nz, 1, 1)
            z = torch.cat((z, y_onehot_), 1)
            noise.data.resize_(z.size()).copy_(z)
            fake = netG(noise)
            
            label.data.fill_(fake_label)
            output = netD(fake.detach())
            clf_output = clf(norm(fake))
            errD_fake = (
                criterion(output, label) + 
                aux_criterion(clf_output, real_classes_var)
            )

            _, pred = clf_output.max(1)
            acc_fake = torch.mean((pred.data.cpu()[:, 0] == real_classes[:, 0]).float())

            errD_fake.backward(retain_variables=True)
            D_G_z1 = output.data.mean()
            errD = errD_real + errD_fake
            optimizerD.step()
            #optimizerC.step()
            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            clf.zero_grad()
            label.data.fill_(real_label) # fake labels are real for generator cost
            output = netD(fake)
            clf_output = clf(norm(fake))
            errG = (
                criterion(output, label) + 
                aux_criterion(clf_output, real_classes_var)
            )
            errG.backward()
            D_G_z2 = output.data.mean()
            optimizerG.step()

            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f acc_real : %.4f acc_fake : %.4f '
                  % (epoch, opt.niter, i, len(dataloader),
                     errD.data[0], errG.data[0], acc_real, acc_fake))
            if i % 100 == 0:
                # the first 64 samples from the mini-batch are saved.
                vutils.save_image((real_cpu[0:64,:,:,:]+1)/2., '%s/real_samples.png' % opt.outf, nrow=8)
                fake = netG(fixed_noise)
                im = (fake.data + 1) / 2.
                fname = '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch)
                vutils.save_image(im, fname, nrow=nb_classes)

        # do checkpointing
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))
