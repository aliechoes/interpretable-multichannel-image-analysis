import logging
import os

from custom_transforms import LoadTensor
from test_dataset import TestDataset
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import torch


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


def get_statistics_2(dataloader, only_channels, logging):
    nmb_channels = 0
    if len(only_channels) == 0:
        nmb_channels = 12
    else:
        nmb_channels = len(only_channels)

    statistics = dict()
    statistics["mean"] = torch.zeros(nmb_channels)
    statistics["std"] = torch.zeros(nmb_channels)

    for j, data_l in enumerate(dataloader, 0):
        data_l = data_l[0]
        for n in range(nmb_channels):
            statistics["mean"][n] += data_l[:, n, :, :].mean()
            statistics["std"][n] += data_l[:, n, :, :].std()
    print('statistics used: %s' % (str(statistics)))
    logging.info('statistics used: %s' % (str(statistics)))
    logging.info('length of the dataloader is: %s' % (str(len(dataloader))))
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
