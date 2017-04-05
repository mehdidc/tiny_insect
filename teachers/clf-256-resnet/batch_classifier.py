from __future__ import division
import time
from itertools import islice
import math
import random
 
import numpy as np
 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision.models import resnet101
import uuid

class BatchClassifier(object):
 
    def __init__(self):
        self.net = Net()
 
    def fit(self, gen_builder):
        batch_size = 9
        nb_epochs = 3
        lr = 1e-3
        gen_train, gen_valid, nb_train, nb_valid = gen_builder.get_train_valid_generators(batch_size=batch_size, valid_ratio=0.1)
        self.net = self.net.cuda()
        net = self.net
        optimizer = optim.SGD(net.parameters(), lr=lr)
        nb_train_minibatches = _get_nb_minibatches(nb_train, batch_size)
        nb_valid_minibatches = _get_nb_minibatches(nb_valid, batch_size)
        criterion = nn.CrossEntropyLoss().cuda()

        filename = 'clf-{}.th'.format(uuid.uuid4())
        print(filename)
 
        for epoch in range(nb_epochs):
            t0 = time.time()
            net.train() # train mode
            nb_trained = 0
            nb_updates = 0
            train_loss = []
            train_acc = []
            for X, y in islice(gen_train, nb_train_minibatches):
                y = y.argmax(axis=1) # convert onehot to integers, pytorch require the class indices.
                X = _make_variable(X)
                y = _make_variable(y)
                optimizer.zero_grad() # zero-out the gradients because they accumulate by default
                y_pred = net(X)
                loss = criterion(y_pred, y)
                loss.backward() # compute gradients
                optimizer.step() # update params
 
                # Loss and accuracy
                train_acc.extend(self._get_acc(y_pred, y))
                train_loss.append(loss.data[0])
                nb_trained += X.size(0)
                nb_updates += 1
                if nb_updates % 100 == 0 or nb_updates == nb_train_minibatches:
                    print('Epoch [{}/{}], [trained {}/{}], avg_loss: {:.4f}, avg_train_acc: {:.4f}'.format(epoch+1, nb_epochs, nb_trained, nb_train, np.mean(train_loss), np.mean(train_acc)))
 
            net.eval() # eval mode
            nb_valid = 0
            valid_acc = []
            for X, y in islice(gen_valid, nb_valid_minibatches):
                y = y.argmax(axis=1)
                X = _make_variable(X)
                y = _make_variable(y)
                y_pred = net(X)
                valid_acc.extend(self._get_acc(y_pred, y))
                nb_valid += y.size(0)
 
            delta_t = time.time() - t0
            print('Finished epoch {}'.format(epoch + 1))
            print('Time spent : {:.4f}'.format(delta_t))
            print('Train acc : {:.4f}'.format(np.mean(train_acc)))            
            print('Valid acc : {:.4f}'.format(np.mean(valid_acc)))
            torch.save(net, filename)
 
    def _get_acc(self, y_pred, y_true):
        y_pred = y_pred.cpu().data.numpy().argmax(axis=1)
        y_true = y_true.cpu().data.numpy()
        return (y_pred==y_true)
 
    def predict_proba(self, X):
        X = X.astype(np.float32)
        X = _make_variable(X)
        y_proba = nn.Softmax()(self.net(X)).cpu().data.numpy()
        return y_proba
 
 
def _make_variable(X):
    return Variable(torch.from_numpy(X)).cuda()
 
class Net(nn.Module):
 
    def __init__(self, pretrained=True):
        super(Net, self).__init__()
        self.model = resnet101(pretrained = True)
        self.model.fc = nn.Sequential(
            nn.Linear(2048, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(True),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(True),
            nn.Linear(4096, 18),
        )
 
    def forward(self, x):
        x = self.model(x)
        return x
 
 
def _flatten(x):
    return x.view(x.size(0), -1)
 
 
def _get_nb_minibatches(nb_examples, batch_size):
    nb = nb_examples // batch_size
    if nb_examples % batch_size != 0:
        nb += 1
    return nb
