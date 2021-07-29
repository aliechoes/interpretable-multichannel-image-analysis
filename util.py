import logging
import os

from custom_transforms import LoadTensor
from test_dataset import TestDataset
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import torch
import matplotlib.pyplot as plt
import h5py

seed_value = 42

os.environ['PYTHONHASHSEED'] = str(seed_value)
import random

random.seed(seed_value)
import numpy as np

np.random.seed(seed_value)

torch.manual_seed(42)


def preprocess_image(path_to_images, batch, num_workers, test_data=[], statistics={}):
    files_to_interpret = []
    for file in os.listdir(path_to_images):
        if file.endswith(".pt") and (len(test_data) == 0 or int(os.path.splitext(file)[0]) in test_data):
            files_to_interpret.append(os.path.join(path_to_images, file))
    logging.info("The samples to interpret are in: {} and the number of samples is {}".format(path_to_images,
                                                                                              len(files_to_interpret)))
    test_loader = load_test_data(files_to_interpret, batch, num_workers)
    files_names = [os.path.basename(os.path.normpath(file)) for file in files_to_interpret]
    return files_names, test_loader


def load_test_data(files_to_interpret, batch, num_workers):
    test_transforms = transforms.Compose([
        LoadTensor()
    ])
    test_ds = TestDataset(files_to_interpret, test_transforms)
    return DataLoader(test_ds, batch_size=batch, shuffle=False, num_workers=num_workers)


def get_statistics_2(dataloader, only_channels, logging, num_channels):
    nmb_channels = 0
    if len(only_channels) == 0:
        nmb_channels = num_channels
    else:
        nmb_channels = len(only_channels)

    statistics = dict()
    statistics["mean"] = torch.zeros(nmb_channels)
    statistics["std"] = torch.zeros(nmb_channels)

    for j, data_l in enumerate(dataloader, 0):
        breakpoint()
        data_l = data_l[0]
        for n in range(nmb_channels):
            statistics["mean"][n] += data_l[:, n, :, :].mean()
            statistics["std"][n] += data_l[:, n, :, :].std()
    statistics["mean"] = statistics["mean"].div_(len(dataloader))
    statistics["std"] = statistics["std"].div_(len(dataloader))
    if logging is not None:
        logging.info('statistics used: %s' % (str(statistics)))
    return statistics


def get_statistics_h5(dataloader, only_channels, logging, num_channels):
    nmb_channels = 0
    if len(only_channels) == 0:
        nmb_channels = num_channels
    else:
        nmb_channels = len(only_channels)

    statistics = dict()
    statistics["mean"] = torch.zeros(nmb_channels)
    statistics["std"] = torch.zeros(nmb_channels)

    for j, data_l in enumerate(dataloader, 0):
        data_l = data_l["image"]
        for n in range(nmb_channels):
            statistics["mean"][n] += data_l[:, n, :, :].mean()
            statistics["std"][n] += data_l[:, n, :, :].std()
    statistics["mean"] = statistics["mean"].div_(len(dataloader))
    statistics["std"] = statistics["std"].div_(len(dataloader))
    if logging is not None:
        logging.info('statistics used: %s' % (str(statistics)))
    return statistics


def get_statistics(dataloader, only_channels):
    nmb_channels = 0
    if len(only_channels) == 0:
        nmb_channels = 12
    else:
        nmb_channels = len(only_channels)

    statistics = dict()
    statistics["mean"] = torch.zeros(nmb_channels)
    statistics["std"] = torch.zeros(nmb_channels)

    for j, data_l in enumerate(dataloader, 0):
        data_l = data_l["image"]
        for n in range(nmb_channels):
            statistics["mean"][n] += data_l[:, n, :, :].mean()
            statistics["std"][n] += data_l[:, n, :, :].std()
    print('statistics used: %s' % (str(statistics)))
    # logging.info('statistics used: %s' % (str(statistics)))
    # logging.info('length of the dataloader is: %s' % (str(len(dataloader))))
    return statistics


def plot_heatmap_3_channels(heatmap):
    fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3, figsize=(15, 15))
    ax1.axes.get_xaxis().set_visible(False)
    ax1.axes.get_yaxis().set_visible(False)
    ax1.imshow(heatmap[0])
    ax2.axes.get_xaxis().set_visible(False)
    ax2.axes.get_yaxis().set_visible(False)
    ax2.imshow(heatmap[1])
    ax3.axes.get_xaxis().set_visible(False)
    ax3.axes.get_yaxis().set_visible(False)
    ax3.imshow(heatmap[2])


def load_h5_file(file):
    f = h5py.File(file, 'r')
    return os.path.splitext(file)[0], f.get("label")[()]


import math
import matplotlib.pyplot as plt
from pylab import *


def plot_n_images(heatmap, number_channels):
    # fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9), (ax10, ax11, ax12)) = plt.subplots(4, 3, figsize=(15,15))
    # fig, ax = plt.subplots(math.ceil(number_channels / 3), 3, figsize=(15,15))
    number_of_subplots = math.ceil(number_channels / 3)
    for i in range(number_channels):
        i = i + 1
        ax1 = subplot(number_of_subplots, 3, i)
        ax1.axes.get_xaxis().set_visible(False)
        ax1.axes.get_yaxis().set_visible(False)
        ax1.imshow(heatmap[i - 1])
    plt.show()


def read_data(path_to_data):
    X = []
    y = []
    for image_name in os.listdir(path_to_data):
        o_n = os.path.splitext(image_name)[0]
        r = h5py.File(os.path.join(path_to_data, image_name), 'r')
        X.append(int(o_n))
        y.append(r["label"][()])

    class_names = list(set(y))
    data_map = dict(zip(sorted(set(y)), np.arange(len(set(y)))))
    return X, y, class_names, data_map


def calculate_weights(y_train):
    class_sample_count = np.array([len(np.where(np.asarray(y_train) == t)[0]) for t in np.unique(y_train)])
    weights = len(y_train) / class_sample_count
    return weights
