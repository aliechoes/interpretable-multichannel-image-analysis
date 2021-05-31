import torch
import torchvision
from imblearn.over_sampling import RandomOverSampler
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import argparse
from datetime import datetime
import os
import time
import multiprocessing
import psutil
import json
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py
from sklearn.model_selection import train_test_split
from skimage.util import crop, random_noise
from skimage.transform import rescale, resize, rotate, AffineTransform, warp
import torch.optim as optim
from tqdm import tqdm

from custom_transforms import AddGaussianNoise
from resnet18 import resnet18
from collections import Counter
from util import get_statistics
from dataset import Dataset_Generator, train_validation_test_split, get_classes_map, number_of_classes, \
    number_of_channels, get_all_object_numbers_labels
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
import logging
import sys

JCD_CLASS_NAMES = ['Anaphase', 'G1', 'G2', 'Metaphase', 'Prophase', 'S', 'Telophase']
WBC_CLASS_NAMES = [' unknown',
                   ' CD4+ T',
                   ' CD8+ T',
                   ' CD15+ neutrophil',
                   ' CD14+ monocyte',
                   ' CD19+ B',
                   ' CD56+ NK',
                   ' NKT',
                   ' eosinophil']

sys.path.append("..")

parser = argparse.ArgumentParser()
parser.add_argument('--h5_file', default="data/WBC/Lyse fix sample_1_Focused & Singlets & CD45 pos.h5",
                    help="dataset root dir")
parser.add_argument('--batch_size', default=300, help="batch size", type=int)
parser.add_argument('--n_epochs', default=100, help="epochs to train", type=int)
parser.add_argument('--lr', default=0.001, help="learning rate", type=float)
parser.add_argument('--n_splits', default=5, type=int)
parser.add_argument('--model_save_path', default='models/', help="path to save models")
parser.add_argument('--log_dir', default='logs/', help="path to save logs")
parser.add_argument('--only_channels', default=[], help="the channels to be used for the model training", nargs='+',
                    type=int)
parser.add_argument('--only_classes', default=None, help="the classes to be used for the model training", nargs='+',
                    type=int)
parser.add_argument('--dev', default='cpu', help="cpu or cuda")
parser.add_argument('--class_names', default=WBC_CLASS_NAMES, help="name of the classes", nargs='+', type=str)
opt = parser.parse_args()

WBC_CLASS_NAMES = [' unknown',
                   ' CD4+ T',
                   ' CD8+ T',
                   ' CD15+ neutrophil',
                   ' CD14+ monocyte',
                   ' CD19+ B',
                   ' CD56+ NK',
                   ' NKT',
                   ' eosinophil']


class oversampled_Kfold():
    def __init__(self, n_splits, ros, n_repeats=1):
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.ros = ros

    def get_n_splits(self, X, y, groups=None):
        return self.n_splits * self.n_repeats

    def split(self, X, y, groups=None):
        splits = np.array_split(np.random.choice(len(X), len(X), replace=False), self.n_splits)
        train, test = [], []
        for repeat in range(self.n_repeats):
            for idx in range(len(splits)):
                trainingIdx = np.delete(splits, idx, 0)
                Xidx_r, y_r = self.ros.fit_resample(np.hstack(trainingIdx).reshape((-1, 1)),
                                                    np.asarray(y[np.hstack(trainingIdx)]))
                train.append(Xidx_r.flatten())
                test.append(splits[idx])
        return list(zip(train, test))


if __name__ == '__main__':
    if opt.dev != 'cpu':
        opt.dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    now = datetime.now()
    timestamp = datetime.timestamp(now)

    logging.basicConfig(filename=os.path.join(opt.log_dir, 'cross_validation_h5_{}.txt'.format(timestamp)),
                        level=logging.DEBUG)
    logging.info("the deviced being used is {}".format(opt.dev))

    transform = transforms.Compose(
        [transforms.RandomVerticalFlip(),
         transforms.RandomHorizontalFlip(),
         transforms.RandomRotation(45),
         AddGaussianNoise(0., 1., 0.3)])

    ros = RandomOverSampler(random_state=42, sampling_strategy='all')

    num_classes = number_of_classes(opt.h5_file, only_classes=opt.only_classes)
    num_channels = number_of_channels(opt.h5_file, only_channels=opt.only_channels)

    rkf_search = oversampled_Kfold(n_splits=opt.n_splits, n_repeats=1, ros=ros)
    best_accuracy = 0.0
    criterion = nn.BCEWithLogitsLoss()
    X, y = get_all_object_numbers_labels(opt.h5_file, opt.only_classes)

    logging.info("Start validation")
    print("Start validation")
    for train_indx, test_indx in rkf_search.split(X, y):
        train_dataset = Dataset_Generator(opt.h5_file, train_indx, reshape_size=64, transform=transform,
                                          only_channels=opt.only_channels, only_classes=opt.only_classes)
        trainloader = DataLoader(train_dataset,
                                 batch_size=opt.batch_size,
                                 shuffle=False,
                                 num_workers=0)
        statistics = get_statistics(trainloader, opt.only_channels)
        logging.info('statistics used: %s' % (str(statistics)))
        logging.info('length of the dataloader is: %s' % (str(len(trainloader))))
        print('statistics used: %s' % (str(statistics)))
        print('length of the dataloader is: %s' % (str(len(trainloader))))
        train_dataset = Dataset_Generator(opt.h5_file, train_indx, reshape_size=64, transform=transform,
                                          means=statistics["mean"].div_(len(trainloader)),
                                          stds=statistics["std"].div_(len(trainloader)),
                                          only_channels=opt.only_channels,
                                          only_classes=opt.only_classes)
        test_dataset = Dataset_Generator(opt.h5_file, test_indx, reshape_size=64,
                                         means=statistics["mean"].div_(len(trainloader)),
                                         stds=statistics["std"].div_(len(trainloader)), only_channels=opt.only_channels,
                                         only_classes=opt.only_classes)
        trainloader = DataLoader(train_dataset,
                                 batch_size=opt.batch_size,
                                 shuffle=False,
                                 num_workers=0)
        testloader = DataLoader(test_dataset,
                                batch_size=opt.batch_size,
                                shuffle=False,
                                num_workers=0)
        model = resnet18(pretrained=True)

        # loading the imagenet weights in case it is possible
        if num_channels != 3:
            model.conv1 = nn.Conv2d(num_channels, 64, kernel_size=(7, 7),
                                    stride=(2, 2), padding=(3, 3), bias=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

        model = model.to(opt.dev)
        optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9)
        for epoch in range(opt.n_epochs):
            running_loss = 0.0
            print('epoch%d' % epoch)
            for i, data in enumerate(trainloader, 0):
                indx = (data["object_number"] != -1).reshape(-1)
                if indx.sum() > 0:
                    inputs, labels = data["image"][indx], data["label"][indx]

                    inputs, labels = inputs.to(opt.dev), labels.to(opt.dev)
                    inputs = inputs.float()
                    labels = labels.reshape(-1)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward + backward + optimize
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)

                    loss = criterion(outputs, F.one_hot(labels.long(), num_classes).type_as(outputs))
                    loss.backward()
                    optimizer.step()

                    # print statistics
                    running_loss += loss.item()
                if i % 100 == 99:  # print every 2000 mini-batches
                    print('[%d, %5d] training loss: %.8f' % (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0
        correct = 0.
        total = 0.
        y_true = list()
        y_pred = list()

        with torch.no_grad():
            for data in testloader:
                indx = (data["object_number"] != -1).reshape(-1)
                if indx.sum() > 0:
                    inputs, labels = data["image"][indx], data["label"][indx]

                    inputs, labels = inputs.to(opt.dev), labels.to(opt.dev)
                    inputs = inputs.float()
                    labels = labels.reshape(-1)

                    outputs = model(inputs)
                    pred = outputs.argmax(dim=1)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (labels.reshape(-1) == predicted).sum().item()
                    for i in range(len(pred)):
                        y_true.append(labels[i].item())
                        y_pred.append(pred[i].item())

        print('Accuracy of the network on the %d test images: %d %%' % (len(test_dataset),
                                                                        100 * correct / total))
        if 100 * correct / total > best_accuracy:
            torch.save(model.state_dict(), os.path.join("models/final_model_dict_best_metrics_h5.pth"))
            logging.info('test_indx used: %s' % (', '.join(str(x) for x in np.unique(test_indx))))
        print(classification_report(y_true, y_pred, target_names=opt.class_names, digits=4))
        f1_score_original = f1_score(y_true, y_pred, average=None, labels=np.arange(num_classes))
        df = pd.DataFrame(np.atleast_2d(f1_score_original), columns=opt.class_names)
        print(df.to_string())
        torch.cuda.empty_cache()
