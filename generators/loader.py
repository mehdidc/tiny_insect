import torch.utils.data as data
import numpy as np
from collections import defaultdict
from PIL import Image
import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes =  sorted(classes, key=lambda i:int(i))
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx):
    images = []
    for target in os.listdir(dir):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in fnames:
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images

def uniformize_dataset(images):
    freq = defaultdict(int)
    for path, class_idx in images:
        freq[class_idx] += 1
    total = len(images)
    max_freq = max(freq.values())

    resampled_images = []
    for path, class_idx in images:
        nb = int(max_freq / freq[class_idx])
        resampled_images.extend([(path, class_idx)] * nb)
    return resampled_images

def default_loader(path):
    return Image.open(path).convert('RGB')


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader, uniformize=False):
        classes, class_to_idx = find_classes(root)
        imgs = make_dataset(root, class_to_idx)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
        if uniformize:
            imgs = uniformize_dataset(imgs)
        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.imgs)

class Tiny(data.Dataset):

    def __init__(self, path, transform=None, chunk_size=1024):
        self.fd = open(path, 'rb')
        self.transform = transform

    def __getitem__(self, index):
        self.fd.seek(index * 3072)
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
        return 200 * 40000
