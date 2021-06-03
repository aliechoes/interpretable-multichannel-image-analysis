import torch
import logging
from torchvision import transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import torch.optim as optim
import sys
from datetime import datetime
import numpy as np
from sklearn.metrics import classification_report, f1_score
import pandas as pd
from resnet18 import resnet18
from dataset import get_classes_map, Dataset_Generator_Preprocessed
from util import get_statistics_2
from collections import Counter
from sklearn.model_selection import KFold


seed_value = 42

os.environ['PYTHONHASHSEED']=str(seed_value)
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
parser.add_argument('--h5_file', default="data/WBC/Lyse fix sample_1_Focused & Singlets & CD45 pos.h5",
                    help="dataset root dir")
parser.add_argument('--path_to_data', default="data/WBC/PreprocessedData",
                    help="dataset root dir")
parser.add_argument('--batch_size', default=64, help="batch size", type=int)
parser.add_argument('--n_epochs', default=50, help="epochs to train", type=int)
parser.add_argument('--num_workers', type=int, default=2, help='number of data loading workers')
parser.add_argument('--lr', default=0.001, help="learning rate", type=float)
parser.add_argument('--model_save_path', default='models/', help="path to save models")
parser.add_argument('--model_name', default='best_metrics', help="path to save models")
parser.add_argument('--log_dir', default='logs/', help="path to save logs")
parser.add_argument('--only_channels', default=[], help="the channels to be used for the model training", nargs='+',
                    type=int)
parser.add_argument('--channels_to_shuffle', default=[], help="the channels to be shuffled for the model training", nargs='+',
                    type=int)
parser.add_argument('--only_classes', default=None, help="the classes to be used for the model training", nargs='+',
                    type=int)
parser.add_argument('--dev', default='cpu', help="cpu or cuda")
parser.add_argument('--num_channels', type=int, default=12, help='number of channels')
parser.add_argument('--num_classes', type=int, default=9, help='number of classes')
parser.add_argument('--class_names', default=WBC_CLASS_NAMES, help="name of the classes", nargs='+', type=str)
parser.add_argument('--n_splits', default=5, type=int)
opt = parser.parse_args()

if __name__ == '__main__':
    if opt.dev != 'cpu':
        opt.dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    now = datetime.now()
    timestamp = datetime.timestamp(now)

    logging.basicConfig(filename=os.path.join(opt.log_dir, 'cross_val_preprocessed{}.txt'.format(timestamp)),
                        level=logging.DEBUG)
    logging.info("the deviced being used is {}".format(opt.dev))

    # load X and y
    X, y = np.loadtxt(os.path.join(opt.path_to_data, "X.txt"), dtype=int), np.loadtxt(
        os.path.join(opt.path_to_data, "y.txt"), dtype=int)

    kf = KFold(n_splits=opt.n_splits, shuffle=True)

    criterion = nn.BCEWithLogitsLoss()

    logging.info("Start validation")
    print("Start Validation")

    label_map = get_classes_map(opt.h5_file)

    transform = transforms.Compose(
        [transforms.RandomVerticalFlip(),
         transforms.RandomHorizontalFlip(),
         transforms.RandomRotation(45)])

    for train_indx, test_indx in kf.split(X):

        train_dataset = Dataset_Generator_Preprocessed(path_to_data=opt.path_to_data,
                                                       set_indx=train_indx,
                                                       transform=transform,
                                                       channels_to_shuffle=opt.channels_to_shuffle,
                                                       num_channels=opt.num_channels)

        trainloader = DataLoader(train_dataset,
                                 batch_size=opt.batch_size,
                                 shuffle=False,
                                 num_workers=opt.num_workers)

        # get statistics to normalize data
        statistics = get_statistics_2(trainloader, [], logging)


        # get weights per class and initialize weighted_sampler
        y_train = y[train_indx]
        class_sample_count = np.array([len(np.where(y_train == t)[0]) for t in np.unique(y_train)])
        weights = 1. / torch.tensor(class_sample_count, dtype=torch.float).to(opt.dev)
        class_weights = weights.double()

        class_weights_all = class_weights[y_train]
        weighted_sampler = WeightedRandomSampler(
            weights=class_weights_all,
            num_samples=len(class_weights_all),
            replacement=False
        )

        # create a new normalized datasets and loaders
        train_dataset = Dataset_Generator_Preprocessed(path_to_data=opt.path_to_data,
                                                       set_indx=train_indx, transform=transform,
                                                       means=statistics["mean"].div_(len(trainloader)),
                                                       stds=statistics["std"].div_(len(trainloader)),
                                                       channels_to_shuffle=opt.channels_to_shuffle,
                                                       num_channels=opt.num_channels)

        test_dataset = Dataset_Generator_Preprocessed(path_to_data=opt.path_to_data,
                                                      set_indx=test_indx,
                                                      means=statistics["mean"].div_(len(trainloader)),
                                                      stds=statistics["std"].div_(len(trainloader)),
                                                      channels_to_shuffle=opt.channels_to_shuffle,
                                                      num_channels=opt.num_channels)


        trainloader = DataLoader(train_dataset,
                                 batch_size=opt.batch_size,
                                 shuffle=False,
                                 num_workers=opt.num_workers,
                                 sampler=weighted_sampler)

        testloader = DataLoader(test_dataset,
                                batch_size=opt.batch_size,
                                shuffle=False,
                                num_workers=opt.num_workers)

        logging.info('label_map used: %s' % (str(label_map)))

        logging.info('train dataset: %d, test dataset: %d' % (len(train_dataset), len(test_dataset)))

        logging.info('shuffled channels: %s; only classes: %s' % (str(opt.channels_to_shuffle), str(opt.only_classes)))

        # loading the model
        model = resnet18(pretrained=True)
        if opt.num_channels != 3:
            model.conv1 = nn.Conv2d(opt.num_channels, 64, kernel_size=(7, 7),
                                stride=(2, 2), padding=(3, 3), bias=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, opt.num_classes)

        model = model.to(opt.dev)

        optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9)

        for epoch in range(opt.n_epochs):
            running_loss = 0.0
            logging.info('epoch%d' % epoch)
            for i, data in enumerate(trainloader, 0):

                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(opt.dev).float(), data[1].to(opt.dev)
                labels = labels.reshape(-1)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)

                loss = criterion(outputs, F.one_hot(labels.long(), opt.num_classes).type_as(outputs))
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 50 == 49:  # print every 2000 mini-batches
                    logging.info('[%d, %5d] training loss: %.8f' % (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0

        logging.info('Finished Training')

        correct = 0.
        total = 0.
        y_true = list()
        y_pred = list()

        with torch.no_grad():
            for data in testloader:
                inputs, labels = data[0].to(opt.dev).float(), data[1].to(opt.dev)
                labels = labels.reshape(-1)

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

