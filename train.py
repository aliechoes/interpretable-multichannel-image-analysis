import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import torch.optim as optim
import sys

sys.path.append("..")
import logging

from resnet18 import resnet18
from dataset import Dataset_Generator, train_validation_test_split, get_classes_map, number_of_classes, \
    number_of_channels

parser = argparse.ArgumentParser()
parser.add_argument('--h5_file', default="data/Lyse fix sample_1_Focused & Singlets & CD45 pos.h5",
                    help="dataset root dir")
parser.add_argument('--batch_size', default=64, help="batch size", type=int)
parser.add_argument('--n_epochs', default=10, help="epochs to train", type=int)
parser.add_argument('--num_workers', type=int, default=2, help='number of data loading workers')
parser.add_argument('--lr', default=0.001, help="learning rate", type=float)
parser.add_argument('--model_save_path', default='models/', help="path to save models")
parser.add_argument('--log_dir', default='logs/', help="path to save logs")
parser.add_argument('--resume_model', default='', help="resume model name")
parser.add_argument('--only_channels', default=[], help="the channels to be used for the model training", nargs='+', type=int)
parser.add_argument('--only_classes', default=None, help="the classes to be used for the model training", nargs='+', type=int)
opt = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_statistics(dataloader):
    nmb_channels = 0
    if len(opt.only_channels) == 0:
        nmb_channels = 12
    else:
        nmb_channels = len(opt.only_channels)

    statistics = dict()
    statistics["mean"] = torch.zeros(nmb_channels)
    statistics["std"] = torch.zeros(nmb_channels)

    for j, data in enumerate(dataloader, 1):

        data = data["image"]
        for i in range(nmb_channels):
            statistics["mean"][i] += data[:, i, :, :].mean()
            statistics["std"][i] += data[:, i, :, :].std()
    return statistics


if __name__ == '__main__':

    logging.basicConfig(filename=os.path.join(opt.log_dir, 'output.txt'), level=logging.DEBUG)

    train_indx, validation_indx, test_indx = train_validation_test_split(h5_file=opt.h5_file, only_classes=opt.only_classes)

    label_map = get_classes_map(opt.h5_file)

    transform = transforms.Compose(
        [transforms.RandomVerticalFlip(),
         transforms.RandomHorizontalFlip(),
         transforms.RandomRotation(45)])

    train_dataset = Dataset_Generator(opt.h5_file, train_indx, reshape_size=64, transform=transform,
                                      only_channels=opt.only_channels, only_classes=opt.only_classes)
    trainloader = DataLoader(train_dataset,
                             batch_size=opt.batch_size,
                             shuffle=False,
                             num_workers=opt.num_workers)

    # collect statistics of the train data (mean & standard deviation) to normalize the data
    statistics = get_statistics(trainloader)

    # create a new normalized train_dataset
    train_dataset = Dataset_Generator(opt.h5_file, train_indx, reshape_size=64, transform=transform,
                                      means=statistics["mean"].div_(len(trainloader)),
                                      stds=statistics["std"].div_(len(trainloader)), only_channels=opt.only_channels,
                                      only_classes=opt.only_classes)
    validation_dataset = Dataset_Generator(opt.h5_file, validation_indx, reshape_size=64,
                                           means=statistics["mean"].div_(len(trainloader)),
                                           stds=statistics["std"].div_(len(trainloader)),
                                           only_channels=opt.only_channels, only_classes=opt.only_classes)
    test_dataset = Dataset_Generator(opt.h5_file, test_indx, reshape_size=64,
                                     means=statistics["mean"].div_(len(trainloader)),
                                     stds=statistics["std"].div_(len(trainloader)), only_channels=opt.only_channels,
                                     only_classes=opt.only_classes)

    trainloader = DataLoader(train_dataset,
                             batch_size=opt.batch_size,
                             shuffle=False,
                             num_workers=opt.num_workers)
    validationloader = DataLoader(validation_dataset,
                                  batch_size=opt.batch_size,
                                  shuffle=False,
                                  num_workers=1)
    testloader = DataLoader(test_dataset,
                            batch_size=opt.batch_size,
                            shuffle=False,
                            num_workers=1)

    logging.info('train dataset: %d, validation dataset: %d, test dataset: %d' % (
        len(train_dataset), len(validation_dataset), len(test_dataset)))

    num_classes = number_of_classes(opt.h5_file, only_classes=opt.only_classes)
    num_channels = number_of_channels(opt.h5_file, only_channels=opt.only_channels)
    logging.info('used only channels: %s; only classes: %s' % (str(opt.only_channels), str(opt.only_classes)))

    model = resnet18(pretrained=True)

    # loading the imagenet weights in case it is possible
    if num_channels != 3:
        model.conv1 = nn.Conv2d(num_channels, 64, kernel_size=(7, 7),
                                stride=(2, 2), padding=(3, 3), bias=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9)

    if opt.resume_model != '':
        checkpoint = torch.load('{0}/{1}'.format(opt.model_save_path, opt.resume_model))
        model.load_state_dict(checkpoint)
        for log in os.listdir(opt.log_dir):
            os.remove(os.path.join(opt.log_dir, log))

    for epoch in range(opt.n_epochs):
        running_loss = 0.0
        logging.info('epoch%d' % epoch)
        for i, data in enumerate(trainloader, 0):

            # get the inputs; data is a list of [inputs, labels]
            indx = (data["object_number"] != -1).reshape(-1)
            if indx.sum() > 0:
                inputs, labels = data["image"][indx], data["label"][indx]

                inputs, labels = inputs.to(device), labels.to(device)
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
            if i % 500 == 499:  # print every 2000 mini-batches
                logging.info('[%d, %5d] training loss: %.8f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for i, data in enumerate(validationloader, 0):
                #breakpoint()
                indx = (data["object_number"] != -1).reshape(-1)
                if indx.sum() > 0:
                    inputs, labels = data["image"][indx], data["label"][indx]

                    inputs, labels = inputs.to(device), labels.to(device)
                    inputs = inputs.float()
                    labels = labels.reshape(-1)

                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (labels.reshape(-1) == predicted).sum().item()

        logging.info('Accuracy of the network on the %d validation images: %d %%' % (
            len(validation_dataset), 100 * correct / total))

    logging.info('Finished Training')

    correct = 0.
    total = 0.

    with torch.no_grad():
        for data in testloader:
            indx = (data["object_number"] != -1).reshape(-1)
            if indx.sum() > 0:
                inputs, labels = data["image"][indx], data["label"][indx]

                inputs, labels = inputs.to(device), labels.to(device)
                inputs = inputs.float()
                labels = labels.reshape(-1)

                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (labels.reshape(-1) == predicted).sum().item()

    logging.info('Accuracy of the network on the %d test images: %d %%' % (len(test_dataset),
                                                                           100 * correct / total))

    torch.save(model.state_dict(), os.path.join(opt.model_save_path, "final_model_dict.pth"))
