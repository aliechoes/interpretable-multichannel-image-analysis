import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
from sklearn.model_selection import train_test_split
from skimage.util import crop, random_noise
import copy
import sys
from imblearn.over_sampling import RandomOverSampler
import os

sys.path.append("..")


def crop_pad_h_w(image_dummy, reshape_size):
    if image_dummy.shape[0] < reshape_size:
        h1_pad = (reshape_size - image_dummy.shape[0]) / 2
        h1_pad = int(h1_pad)
        h2_pad = reshape_size - h1_pad - image_dummy.shape[0]
        h1_crop = 0
        h2_crop = 0
    else:
        h1_pad = 0
        h2_pad = 0
        h1_crop = (reshape_size - image_dummy.shape[0]) / 2
        h1_crop = abs(int(h1_crop))
        h2_crop = image_dummy.shape[0] - reshape_size - h1_crop

    if image_dummy.shape[1] < reshape_size:
        w1_pad = (reshape_size - image_dummy.shape[1]) / 2
        w1_pad = int(w1_pad)
        w2_pad = reshape_size - w1_pad - image_dummy.shape[1]
        w1_crop = 0
        w2_crop = 0
    else:
        w1_pad = 0
        w2_pad = 0
        w1_crop = (reshape_size - image_dummy.shape[1]) / 2
        w1_crop = abs(int(w1_crop))
        w2_crop = image_dummy.shape[1] - reshape_size - w1_crop

    h = [h1_crop, h2_crop, h1_pad, h2_pad]
    w = [w1_crop, w2_crop, w1_pad, w2_pad]
    return h, w


def get_all_object_numbers_labels(h5_file, only_classes=None):
    data = h5py.File(h5_file, "r")
    if only_classes is None:
        only_classes = data.get("labels")[()]
    object_numbers = data.get("object_number")[()][np.isin(data.get("labels")[()], only_classes)]
    labels = data.get("labels")[()][object_numbers]
    data.close()
    return object_numbers, labels


def train_validation_test_split(h5_file, validation_size=0.15, test_size=0.20, only_classes=None):
    data = h5py.File(h5_file, "r")
    # object_numbers = data.get("object_number")[()]

    oversample = RandomOverSampler(random_state=42, sampling_strategy='all')

    if only_classes is None:
        only_classes = data.get("labels")[()]
    object_numbers = data.get("object_number")[()][np.isin(data.get("labels")[()], only_classes)]
    train, test = train_test_split(object_numbers,
                                   test_size=test_size, stratify=data.get("labels")[()][object_numbers],
                                   random_state=314)

    train, validation = train_test_split(train,
                                         test_size=validation_size,
                                         random_state=314)

    train_lbl = data.get("labels")[()][train]
    val_lbl = data.get("labels")[()][validation]
    train, _ = oversample.fit_resample(np.asarray(train).reshape(-1, 1), np.asarray(train_lbl))
    validation, _ = oversample.fit_resample(np.asarray(validation).reshape(-1, 1), np.asarray(val_lbl))
    # breakpoint()
    data.close()
    return train.T[0], validation.T[0], test


def get_classes_map(h5_file):
    data = h5py.File(h5_file, "r")
    label_map = data.get("label_map")[()]
    data.close()
    return eval(label_map)


def number_of_channels(h5_file, only_channels=[]):
    if len(only_channels) == 0:
        data = h5py.File(h5_file, "r")
        object_numbers = data.get("object_number")[()]
        o_n = object_numbers[0]
        num_channels = data.get(str(o_n) + "_image")[()].shape[2]
        data.close()
        return num_channels
    else:
        return len(only_channels)


def number_of_classes(h5_file, only_classes=None):
    if only_classes is not None:
        return len(only_classes)
    else:
        data = h5py.File(h5_file, "r")
        labels = data.get("labels")[()]
        num_classes = len(set(labels))
        data.close()
        return num_classes


class Dataset_Generator(Dataset):

    def __init__(self, h5_file, set_indx, scaling_factor=4095.,
                 reshape_size=64, data_map=[], statistics=None,
                 transform=None, means=None, stds=None,
                 only_channels=[], only_classes=None):

        self.data = h5py.File(h5_file, "r")

        object_numbers = self.data.get("object_number")[()]

        try:
            labels = self.data.get("labels")[()]
        except TypeError as TE:
            labels = np.array(len(object_numbers) * [-1])

        self.only_channels = only_channels
        self.only_classes = only_classes
        self.object_numbers = object_numbers[set_indx]
        self.labels = labels[set_indx]

        self.num_channels = number_of_channels(h5_file, self.only_channels)
        self.scaling_factor = scaling_factor
        self.reshape_size = reshape_size
        self.data_map = data_map
        self.statistics = statistics
        self.transform = transform
        if means is None:
            self.means = torch.zeros(self.num_channels)
        else:
            self.means = means
        if stds is None:
            self.stds = torch.ones(self.num_channels)
        else:
            self.stds = stds

    def __len__(self):
        return len(self.object_numbers)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        o_n = self.object_numbers[idx]
        lbl = self.labels[idx]

        try:
            image_original = self.data.get(str(o_n) + "_image")[()] / self.scaling_factor
            # image_original = (image_original - self.means) / self.stds
            # creating the image
            h, w = crop_pad_h_w(image_original, self.reshape_size)
            h1_crop, h2_crop, h1_pad, h2_pad = h
            w1_crop, w2_crop, w1_pad, w2_pad = w
            image = np.zeros((self.num_channels, self.reshape_size, self.reshape_size), dtype=np.float64)
            nmb_of_channels = 0
            # filling the image with different channels
            for ch in range(image_original.shape[2]):
                if len(self.only_channels) == 0 or ch in self.only_channels:
                    image_dummy = crop(image_original[:, :, ch], ((h1_crop, h2_crop), (w1_crop, w2_crop)))
                    image_dummy = np.pad(image_dummy, ((h1_pad, h2_pad), (w1_pad, w2_pad)), "edge")
                    image[nmb_of_channels, :, :] = image_dummy
                    nmb_of_channels += 1

            image_original = None
            # map numpy array to tensor
            image = torch.from_numpy(copy.deepcopy(image))

            for i in range(self.num_channels):
                image[i] = (image[i] - self.means[i]) / self.stds[i]

            if self.transform:
                image = self.transform(image)

            if self.only_classes is not None:
                lbl = self.only_classes.index(lbl)
            label = np.array([lbl])

            object_number = np.array([o_n])

            sample = {'image': image, 'label': torch.from_numpy(label), "idx": idx, "object_number": object_number}
        except:
            sample = {'image': torch.from_numpy(
                np.zeros((self.num_channels, self.reshape_size, self.reshape_size), dtype=np.float64)),
                'label': torch.from_numpy(np.array([-1])), "idx": idx, "object_number": np.array([-1])}

        return sample


class Dataset_Generator_Preprocessed(Dataset):

    def __init__(self, path_to_data, set_indx,
                 transform=None, means=None, stds=None,
                 only_channels=[], only_classes=None, num_channels=12):

        self.path_to_data = path_to_data
        self.only_channels = only_channels
        self.only_classes = only_classes
        self.object_numbers = set_indx

        self.num_channels = num_channels
        self.transform = transform
        if means is None:
            self.means = torch.zeros(self.num_channels)
        else:
            self.means = means
        if stds is None:
            self.stds = torch.ones(self.num_channels)
        else:
            self.stds = stds

    def __len__(self):
        return len(self.object_numbers)

    def __getitem__(self, idx):
        tensor = torch.load(os.path.join(self.path_to_data, "{}.pt".format(idx)))
        image, label = tensor[0], tensor[1]
        if len(self.only_channels) > 0:
            image = image[self.only_channels, :, :]
        for i in range(self.num_channels):
            image[i] = (image[i] - self.means[i]) / self.stds[i]
        if self.transform:
            image = self.transform(image)
        return image, label, idx
