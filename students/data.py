import math
import random

from skimage.color import rgb2hsv, hsv2rgb
from PIL import Image
import numpy as np
import torch.utils.data as data
from torch.autograd import Variable

import torch
from torch.utils.data.sampler import Sampler


class RandomHorizontalFlip:
    """
    transformer which randomy flips the image vertically.
    it alreadt exists in torchvision, but I had issues with
    reproducing experiments when using the torchvision version.
    I use np random here which works.
    """
    def __call__(self, img):
        if np.random.random() < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img


class Rotation:
    """
    transformer which rotates image using an angle (deg) sampled
    randomly from [min_val, max_val]
    """
    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, img):
        angle = np.random.uniform(low=self.min_val, high=self.max_val)
        return img.rotate(angle)


class HSV:
    """
    HSV data augmentation like the one in 'do deep convnets really need to be deep'
    """
    def __init__(self, hue_shift, saturation_shift, value_shift, saturation_scale, value_scale):
        self.hue_shift = hue_shift
        self.saturation_shift = saturation_shift
        self.value_shift = value_shift
        self.saturation_scale = saturation_scale
        self.value_scale = value_scale
        
    def __call__(self, img):
        return hsv_augmentation(img, self.hue_shift, self.saturation_shift, self.value_shift, self.saturation_scale, self.value_scale)


class RandomSizedCrop:
    """
    random crop form torchvision but using numpy random
    """
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = np.random.uniform(0.08, 1.0) * area
            aspect_ratio = np.random.uniform(3. / 4, 4. / 3)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if np.random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                x1 = np.random.randint(0, img.size[0] - w + 1)
                y1 = np.random.randint(0, img.size[1] - h + 1)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                assert(img.size == (w, h))

                return img.resize((self.size, self.size), self.interpolation)

        # Fallback
        scale = Scale(self.size, interpolation=self.interpolation)
        crop = CenterCrop(self.size)
        return crop(scale(img))


def hsv_augmentation(x, hue_shift, saturation_shift, value_shift, saturation_scale, value_scale):
    """Function to perform HSV data augmentation by shifting and rescaling the HSV color representations of a batch of images
       using values drawn from uniform distributions:
           H += U(-hue_shift, +hue_shift)
           S *= U(1/(1+saturation_scale), 1+saturation_scale)
           S += U(-saturation_shift, +saturation_shift)
           V *= U(1/(1+value_scale), 1+value_scale)
           V += U(-value_shift, value_shift)
    """
    x = np.array(x)
    x = x.astype(np.float32)
    x /= 255.
    # Assumes x has already been rescaled by (1./255)
    hsv = rgb2hsv(x)
    saturation_scale, value_scale = float(saturation_scale), float(value_scale)
    hsv[..., 0] += np.random.uniform(low=-hue_shift, high=hue_shift)
    hsv[..., 1] *= np.random.uniform(low=1/(1+saturation_scale), high=saturation_scale)
    hsv[..., 1] += np.random.uniform(low=-saturation_shift, high=saturation_shift)
    hsv[..., 2] *= np.random.uniform(low=1/(1+value_scale), high=1+value_scale)
    hsv[..., 2] += np.random.uniform(low=-value_shift, high=value_shift)
    # Note that it is necessary to rescale by 255 and convert to np.uint8 to obtain RGB image
    hsv = np.clip(hsv, 0, 1)
    x = hsv2rgb(hsv)
    x *= 255.
    x = x.astype(np.uint8)
    x = Image.fromarray(x)
    return x


class SamplerFromIndices(Sampler):

    def __init__(self, data_source, indices):
        self.num_samples = len(data_source)
        self.perm = torch.LongTensor(indices)

    def __iter__(self):
        return iter(self.perm)

    def __len__(self):
        return len(self.perm)


class DataAugmentationLoader:

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
        if hasattr(self.dataloader, 'dataset'):
            self.dataloader.dataset.epoch = i
        yield from self.dataloader

    def __len__(self):
        return len(self.dataloader) * self.nb_epochs


class DataAugmentationLoaders:

    def __init__(self, dataloaders, nb_epochs=1):
        self.dataloaders = dataloaders
        self.nb_epochs = nb_epochs

    def __iter__(self):
        for i in range(self.nb_epochs):
            yield from self.epoch(i)
    
    def epoch(self, i):
        for dataloader in self.dataloaders:
            random.seed(i)
            torch.manual_seed(i)
            torch.cuda.manual_seed(i)
            np.random.seed(i)
            if hasattr(dataloader, 'dataset'):
                dataloader.dataset.epoch = i
            yield from dataloader

    def __len__(self):
        return sum(map(len, self.dataloaders)) * self.nb_epochs

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

class GeneratorLoaderDomainTransfer:

    def __init__(self, G, z, onehot, u, source):
        self.G = G
        self.z = z
        self.onehot = onehot
        self.u = u
        self.source = source

    def __iter__(self):
        G = self.G
        z = self.z
        onehot = self.onehot
        u = self.u
        for X, _ in self.source:
            z.data.normal_()
            onehot.data.zero_()
            u.uniform_()
            classes = u.max(1)[1]
            onehot.data.scatter_(1, classes, 1)
            z_ = z.repeat(1, 1, X.size(2), X.size(3))
            onehot_ = onehot.repeat(1, 1, X.size(2), X.size(3))
            X = Variable(X).cuda()
            g_input = torch.cat((X, z_, onehot_), 1)
            out = G(g_input)
            input = out.data
            yield input, classes
    
    def __len__(self):
        return len(self.source)


class MergeLoader:

    def __init__(self, loaders):
        self.loaders = loaders

    def __iter__(self):

        for data in zip(*self.loaders):
            xlist = [x.cuda() for x, y  in data]
            ylist = [y.cuda() for x, y in data]
            X = torch.cat(xlist, 0)
            y = torch.cat(ylist, 0)
            yield X, y
    
    def __len__(self):
        return min(map(len, self.loaders))


def norm(x, mean, std):
    x = x + 1
    x = x / 2
    x = x - mean.repeat(x.size(0), 1, x.size(2), x.size(3))
    x = x / std.repeat(x.size(0), 1, x.size(2), x.size(3))
    return x


class Tiny(data.Dataset):

    def __init__(self, path, transform=None, chunk_size=1024):
        self.fd = open(path, 'rb')
        self.transform = transform

    def __getitem__(self, index):
        # 40000 size of training data of cifar
        offset = self.epoch * 40000 + index
        self.fd.seek(offset * 3072)
        data = self.fd.read(3072)
        data = np.fromstring(data, dtype='uint8')
        data = data.reshape(32, 32, 3, order="F")
        input = Image.fromarray(data)
        target =  0
        if self.transform:
            input = self.transform(input)
        return input, target

    def __delete__(self):
        self.fd.close()
    
    def __len__(self):
        return 40000
