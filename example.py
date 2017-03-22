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
from loader import  ImageFolder
from PIL import Image, ImageOps

class Scale(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        return img.resize((self.size, self.size), self.interpolation)

if __name__ == '__main__':
    transform = transforms.Compose([
      transforms.Scale(128),
      transforms.CenterCrop(128),
      #transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = ImageFolder(root='/home/mcherti/work/data/insects/img_classes', transform=transform)

    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=128,
        shuffle=True, 
        num_workers=1)

    sys.path.append('pytorch_pretrained')
    clf = torch.load('pytorch_pretrained/clf-4cf42cbb-3c69-4d3f-86e1-c050330a7c7c.th').cuda()
    clf_mean = np.array([0.485, 0.456, 0.406], dtype='float32')
    clf_mean = clf_mean[np.newaxis, :, np.newaxis, np.newaxis]
    clf_mean = torch.from_numpy(clf_mean)
    clf_mean = Variable(clf_mean).cuda()
    clf_std = np.array([0.229, 0.224, 0.225], dtype='float32')
    clf_std = clf_std[np.newaxis, :, np.newaxis, np.newaxis]
    clf_std = torch.from_numpy(clf_std)
    clf_std = Variable(clf_std).cuda()

    def norm(x):
        x = (x + 1) / 2.
        x = x - clf_mean.repeat(x.size(0), 1, x.size(2), x.size(3))
        x = x / clf_std.repeat(x.size(0), 1, x.size(2), x.size(3))
        return x

    for i, data in enumerate(dataloader):
        X, y = data
        X = Variable(X).cuda()
        X = norm(X)
        y_proba = clf(X).data.cpu()
        _, y_pred = y_proba.max(dim=1)
        acc = (y == y_pred[:, 0]).float()
        print(torch.mean(acc))
