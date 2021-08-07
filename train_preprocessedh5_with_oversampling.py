import torch
import logging
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.nn as nn
import argparse
import os
import sys
from datetime import datetime
import numpy as np
from sklearn.metrics import classification_report, f1_score
import pandas as pd
from resnet18 import resnet18
from dataset import Dataset_Generator, train_validation_test_split, get_classes_map, number_of_classes, \
    number_of_channels, Dataset_Generator_Preprocessed, train_validation_test_split_wth_augmentation, \
    train_validation_test_split_undersample, Dataset_Generator_Preprocessed_h5
from util import get_statistics_h5, read_data, calculate_weights
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from torch.optim import lr_scheduler
from custom_transforms import AddGaussianNoise
from sklearn.metrics import roc_auc_score

seed_value = 0

os.environ['PYTHONHASHSEED'] = str(seed_value)
import random
import h5py

random.seed(seed_value)

np.random.seed(seed_value)

torch.manual_seed(42)

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
parser.add_argument('--path_to_data', default="data/JurkatCells/PreprocessedDatah5",
                    help="dataset root dir")
parser.add_argument('--batch_size', default=64, help="batch size", type=int)
parser.add_argument('--n_epochs', default=100, help="epochs to train", type=int)
parser.add_argument('--num_workers', type=int, default=2, help='number of data loading workers')
parser.add_argument('--scaling_factor', type=float, default=255., help='scaling factor')
parser.add_argument('--reshape_size', type=int, default=64, help='reshape size of the image')
parser.add_argument('--lr', default=1e-3, help="learning rate", type=float)
parser.add_argument('--model_save_path', default='models/', help="path to save models")
parser.add_argument('--model_name', default='best_metrics', help="model name")
parser.add_argument('--log_dir', default='logs/', help="path to save logs")
parser.add_argument('--resume_model', default='', help="resume model name")
parser.add_argument('--only_channels', default=[], help="the channels to be used for the model training", nargs='+',
                    type=int)
parser.add_argument('--only_classes', default=None, help="the classes to be used for the model training", nargs='+',
                    type=int)
parser.add_argument('--dev', default='cpu', help="cpu or cuda")
parser.add_argument('--num_channels', type=int, default=3, help='number of channels')
parser.add_argument('--num_classes', type=int, default=7, help='number of classes')
opt = parser.parse_args()

if __name__ == '__main__':

    # set device
    if opt.dev != 'cpu':
        opt.dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    now = datetime.now()
    timestamp = datetime.timestamp(now)

    # initialize logging
    logging.basicConfig(
        filename=os.path.join(opt.log_dir, 'output_preprocessedh5_ovs_{}_{}.txt'.format(opt.model_name, timestamp)),
        level=logging.DEBUG)
    logging.info("the deviced being used is {}".format(opt.dev))

    # load X and y
    X, y, class_names, data_map = read_data(opt.path_to_data)

    # split data without augmentation
    train_indx, validation_indx, test_indx = train_validation_test_split_wth_augmentation(X, y,
                                                                                          only_classes=opt.only_classes)

    transform = transforms.Compose(
        [transforms.RandomVerticalFlip(),
         transforms.RandomHorizontalFlip(),
         transforms.RandomRotation(45)])
    # Gaussian Blur
    train_transform = transforms.Compose([
        transforms.RandomCrop(64),
        transforms.RandomAffine(degrees=90),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomErasing(scale=(0.02, 0.2), ratio=(0.1, 0.3)),
        AddGaussianNoise(0., 0.01, 0.5)
    ])
    test_transform = transforms.Compose([
        transforms.CenterCrop(64),
    ])

    # initialize train_dataset and trainloader to calculate the train distribution

    train_dataset = Dataset_Generator_Preprocessed_h5(path_to_data=opt.path_to_data,
                                                      set_indx=train_indx,
                                                      scaling_factor=opt.scaling_factor,
                                                      reshape_size=opt.reshape_size,
                                                      transform=train_transform,
                                                      data_map=data_map,
                                                      only_channels=opt.only_channels,
                                                      num_channels=opt.num_channels)

    trainloader = DataLoader(train_dataset,
                             batch_size=opt.batch_size,
                             shuffle=False,
                             num_workers=opt.num_workers)

    # get statistics to normalize data
    statistics = get_statistics_h5(trainloader, opt.only_channels, logging, opt.num_channels)

    # use oversampling to cope with unbalance data
    y_train = [y[i] for i in train_indx]
    weights = calculate_weights(y_train)
    oversample = RandomOverSampler(random_state=seed_value, sampling_strategy='all')

    train_indx, y_train = oversample.fit_resample(np.asarray(train_indx).reshape(-1, 1), np.asarray(y_train))
    train_indx = train_indx.T[0]
    y_train = [y[i] for i in train_indx]
    # create a new normalized datasets and loaders
    train_dataset = Dataset_Generator_Preprocessed_h5(path_to_data=opt.path_to_data,
                                                      set_indx=train_indx,
                                                      scaling_factor=opt.scaling_factor,
                                                      reshape_size=opt.reshape_size,
                                                      transform=train_transform,
                                                      data_map=data_map,
                                                      only_channels=opt.only_channels,
                                                      num_channels=opt.num_channels,
                                                      means=statistics["mean"],
                                                      stds=statistics["std"]
                                                      )

    validation_dataset = Dataset_Generator_Preprocessed_h5(path_to_data=opt.path_to_data,
                                                           set_indx=validation_indx,
                                                           scaling_factor=opt.scaling_factor,
                                                           reshape_size=opt.reshape_size,
                                                           transform=test_transform,
                                                           data_map=data_map,
                                                           only_channels=opt.only_channels,
                                                           num_channels=opt.num_channels,
                                                           means=statistics["mean"],
                                                           stds=statistics["std"]
                                                           )

    test_dataset = Dataset_Generator_Preprocessed_h5(path_to_data=opt.path_to_data,
                                                     set_indx=test_indx,
                                                     scaling_factor=opt.scaling_factor,
                                                     reshape_size=opt.reshape_size,
                                                     transform=test_transform,
                                                     data_map=data_map,
                                                     only_channels=opt.only_channels,
                                                     num_channels=opt.num_channels,
                                                     means=statistics["mean"],
                                                     stds=statistics["std"]
                                                     )

    trainloader = DataLoader(train_dataset,
                             batch_size=opt.batch_size,
                             shuffle=False,
                             num_workers=opt.num_workers)
    validationloader = DataLoader(validation_dataset,
                                  batch_size=opt.batch_size,
                                  shuffle=False,
                                  num_workers=opt.num_workers)
    testloader = DataLoader(test_dataset,
                            batch_size=opt.batch_size,
                            shuffle=False,
                            num_workers=opt.num_workers)

    logging.info('test_indx used: %s' % (', '.join(str(x) for x in test_indx)))

    logging.info('train dataset: %d, validation dataset: %d, test dataset: %d' % (
        len(train_dataset), len(validation_dataset), len(test_dataset)))

    logging.info('used only channels: %s; only classes: %s' % (str(opt.only_channels), str(opt.only_classes)))

    # loading the model
    model = resnet18(pretrained=True)
    if opt.num_channels != 3:
        model.conv1 = nn.Conv2d(opt.num_channels, 64, kernel_size=(7, 7),
                                stride=(2, 2), padding=(3, 3), bias=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, opt.num_classes)

    model = model.to(opt.dev)
    class_weights = torch.FloatTensor(weights).to(opt.dev)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    if opt.resume_model != '':
        checkpoint = torch.load('{0}/{1}'.format(opt.model_save_path, opt.resume_model))
        model.load_state_dict(checkpoint)

    # start training
    best_metric = -1
    best_metric_epoch = -1
    for epoch in range(opt.n_epochs):
        # y_pred = torch.tensor([], dtype=torch.float32, device=opt.dev)
        # y = torch.tensor([], dtype=torch.long, device=opt.dev)
        running_loss = 0.0
        logging.info('epoch%d' % epoch)
        for i, data in enumerate(trainloader, 0):

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data["image"].to(opt.dev).float(), data["label"].to(opt.dev).reshape(-1).long()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # _, predicted = torch.max(outputs.data, 1)
            # y_pred = torch.cat([y_pred, predicted], dim=0)
            # y = torch.cat([y, labels.reshape(-1)], dim=0)

            # print statistics
            running_loss += loss.item()
            if i % 50 == 49:  # print every 2000 mini-batches
                logging.info('[%d, %5d] training loss: %.8f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
        # if scheduler is not None:
        # scheduler.step()
        # cr = classification_report(y.detach().cpu(), y_pred.detach().cpu(), target_names=class_names, digits=4)
        # logging.info(cr)
        correct = 0
        total = 0
        with torch.no_grad():
            y_pred = torch.tensor([], dtype=torch.float32, device=opt.dev)
            y = torch.tensor([], dtype=torch.long, device=opt.dev)
            epoch_val_loss = 0
            step_val = 0
            for i, data in enumerate(validationloader, 0):
                step_val += 1
                inputs, labels = data["image"].to(opt.dev).float(), data["label"].to(opt.dev).reshape(-1).long()
                outputs = model(inputs)
                val_loss = criterion(outputs, labels)
                epoch_val_loss += val_loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (labels.reshape(-1) == predicted).sum().item()
                y_pred = torch.cat([y_pred, predicted], dim=0)
                y = torch.cat([y, labels.reshape(-1)], dim=0)
            epoch_val_loss /= step_val
            scheduler.step(epoch_val_loss)
            f1_sc = f1_score(y.cpu(), y_pred.cpu(), average='macro')
            if f1_sc > best_metric:
                best_metric = f1_sc
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(),
                           os.path.join(opt.model_save_path, "intermediate_model_dict_{}.pth".format(opt.model_name)))
                print('saved new best metric model')
            cr = classification_report(y.detach().cpu(), y_pred.detach().cpu(), target_names=class_names, digits=4)
            logging.info(cr)

        logging.info('Accuracy of the network on the %d validation images: %d %%' % (
            len(validation_dataset), 100 * correct / total))

    logging.info('Finished Training')

    # evaluate data on test dataset
    correct = 0.
    total = 0.
    y_true = list()
    y_pred = list()

    with torch.no_grad():
        for data in testloader:
            inputs, labels = data["image"].to(opt.dev).float(), data["label"].to(opt.dev).reshape(-1).long()

            outputs = model(inputs)
            pred = outputs.argmax(dim=1)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (labels.reshape(-1) == predicted).sum().item()
            for i in range(len(pred)):
                y_true.append(labels[i].item())
                y_pred.append(pred[i].item())

    logging.info('Accuracy of the network on the %d test images: %d %%' % (len(test_dataset),
                                                                           100 * correct / total))

    logging.info("The model saved: %s" % "final_model_dict_{}.pth".format(opt.model_name))

    # save model and output classification report + f1 scores per class
    torch.save(model.state_dict(), os.path.join(opt.model_save_path, "final_model_dict_{}.pth".format(opt.model_name)))
    cr = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    logging.info(cr)
    f1_score_original = f1_score(y_true, y_pred, average=None, labels=np.arange(opt.num_classes))
    df = pd.DataFrame(np.atleast_2d(f1_score_original), columns=class_names)
    logging.info(df.to_string())
