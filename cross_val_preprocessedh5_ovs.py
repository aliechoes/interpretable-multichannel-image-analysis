import torch
import logging
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import argparse
import os
import sys
from datetime import datetime
import numpy as np
from sklearn.metrics import classification_report, f1_score
import pandas as pd
from resnet18 import resnet18
from dataset import Dataset_Generator_Preprocessed_h5
from util import get_statistics_h5, read_data, calculate_weights
from collections import Counter
from sklearn.model_selection import KFold
from imblearn.over_sampling import RandomOverSampler
from torch.optim import lr_scheduler
from custom_transforms import AddGaussianNoise
from sklearn.model_selection import train_test_split

seed_value = 42

os.environ['PYTHONHASHSEED'] = str(seed_value)
import random

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
parser.add_argument('--n_epochs', default=300, help="epochs to train", type=int)
parser.add_argument('--num_workers', type=int, default=2, help='number of data loading workers')
parser.add_argument('--model_save_path', default='models/cv', help="path to save models")
parser.add_argument('--model_name', default='best_metrics', help="model name")
parser.add_argument('--lr', default=1e-3, help="learning rate", type=float)
parser.add_argument('--log_dir', default='logs/', help="path to save logs")
parser.add_argument('--scaling_factor', type=float, default=255., help='scaling factor')
parser.add_argument('--reshape_size', type=int, default=66, help='reshape size of the image')
parser.add_argument('--only_channels', default=[], help="the channels to be used for the model training", nargs='+',
                    type=int)
parser.add_argument('--only_classes', default=None, help="the classes to be used for the model training", nargs='+',
                    type=int)
parser.add_argument('--dev', default='cpu', help="cpu or cuda")
parser.add_argument('--num_channels', type=int, default=3, help='number of channels')
parser.add_argument('--num_classes', type=int, default=7, help='number of classes')
parser.add_argument('--n_splits', default=5, type=int)
opt = parser.parse_args()

if __name__ == '__main__':

    # set device
    if opt.dev != 'cpu':
        opt.dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    now = datetime.now()
    timestamp = datetime.timestamp(now)

    # initialize logging
    logs_file_name = 'cross_val_preprocessed_ovs_{}.txt'.format(timestamp)
    logging.basicConfig(filename=os.path.join(opt.log_dir, logs_file_name),
                        level=logging.DEBUG)
    print("logs are saved in {}".format(logs_file_name))
    logging.info("the deviced being used is {}".format(opt.dev))

    # load X and y
    X, y, class_names, data_map = read_data(opt.path_to_data)

    #initialize cross validation
    kf = KFold(n_splits=opt.n_splits, shuffle=True)

    logging.info("Start validation")

    transform = transforms.Compose(
        [transforms.RandomVerticalFlip(),
         transforms.RandomHorizontalFlip(),
         transforms.RandomRotation(45)])

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

    oversample = RandomOverSampler(random_state=seed_value, sampling_strategy='all')
    fold = 0
    for train_indx, test_indx in kf.split(X):
        y_train = [y[i] for i in train_indx]
        train_indx, val_indx, y_train, _ = train_test_split(train_indx,y_train, test_size=0.15, stratify=y_train, random_state=42)

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
        #y_train = [y[i] for i in train_indx]
        weights = calculate_weights(y_train)

        train_indx, _ = oversample.fit_resample(np.asarray(train_indx).reshape(-1, 1), np.asarray(y_train))
        train_indx = train_indx.T[0]

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
                                                               set_indx=val_indx,
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

        logging.info('train dataset: %d, test dataset: %d' % (len(train_dataset), len(test_dataset)))

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

        optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        best_metric = -1
        best_metric_epoch = -1
        for epoch in range(opt.n_epochs):
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

                # print statistics
                running_loss += loss.item()
                if i % 50 == 49:  # print every 2000 mini-batches
                    logging.info('[%d, %5d] training loss: %.8f' % (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0

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
                cr = classification_report(y.detach().cpu(), y_pred.detach().cpu(), target_names=class_names, digits=4)
                logging.info(cr)
                if f1_sc > best_metric:
                    best_metric = f1_sc
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(),
                               os.path.join(opt.model_save_path,
                                            "cv_model_fold_{}_dict_{}.pth".format(fold, opt.model_name)))
                    print('saved new best metric model')


        logging.info('Finished Training')

        correct = 0.
        total = 0.
        y_true = list()
        y_pred = list()
        best_saved_model = torch.load(os.path.join(opt.model_save_path,
                                            "cv_model_fold_{}_dict_{}.pth".format(fold, opt.model_name)))
        model.load_state_dict(best_saved_model)

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

        cr = classification_report(y_true, y_pred, target_names=opt.class_names, digits=4)
        logging.info(cr)
        f1_score_original = f1_score(y_true, y_pred, average=None, labels=np.arange(opt.num_classes))
        df = pd.DataFrame(np.atleast_2d(f1_score_original), columns=opt.class_names)
        logging.info(df.to_string())
        torch.cuda.empty_cache()