import logging
import os

from custom_transforms import LoadTensor
from test_dataset import TestDataset
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader


def preprocess_image(path_to_images, batch, num_workers):
    files_to_interpret = []
    for file in os.listdir(path_to_images):
        if file.endswith(".pt"):
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