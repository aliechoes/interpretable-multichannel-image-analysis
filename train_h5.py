import torch
import logging
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
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
from dataset import Dataset_Generator, train_validation_test_split, get_classes_map, number_of_classes, \
    number_of_channels
from util import get_statistics
from multiprocessing import Pool
from util import load_h5_file

sys.path.append("..")

parser = argparse.ArgumentParser()
parser.add_argument('--path_to_data', default="data/JurkatCells/PreprocessedDatah5",
                    help="dataset root dir")
parser.add_argument('--batch_size', default=256, help="batch size", type=int)
parser.add_argument('--n_epochs', default=30, help="epochs to train", type=int)
parser.add_argument('--num_workers', type=int, default=2, help='number of data loading workers')
parser.add_argument('--lr', default=0.001, help="learning rate", type=float)
parser.add_argument('--model_save_path', default='models/', help="path to save models")
parser.add_argument('--model_name', default='best_metrics', help="path to save models")
parser.add_argument('--log_dir', default='logs/', help="path to save logs")
parser.add_argument('--resume_model', default='', help="resume model name")
parser.add_argument('--only_channels', default=[], help="the channels to be used for the model training", nargs='+',
                    type=int)
parser.add_argument('--only_classes', default=None, help="the classes to be used for the model training", nargs='+',
                    type=int)
parser.add_argument('--dev', default='cpu', help="cpu or cuda")
opt = parser.parse_args()


if __name__ == '__main__':
    if opt.dev != 'cpu':
        opt.dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    now = datetime.now()

    timestamp = datetime.timestamp(now)

    ### create a log file
    logging.basicConfig(filename=os.path.join(opt.log_dir, 'output_{}.txt'.format(timestamp)), level=logging.DEBUG)
    logging.info("the deviced being used is {}".format(opt.dev))

    postfix = 'h5'
    file_list = [os.path.join(opt.path_to_data, f) for f in os.listdir(opt.path_to_data) if postfix in f]
    breakpoint()
    p = Pool(5)
    X, y = p.map(load_h5_file, file_list)
    breakpoint()
    train_indx, validation_indx, test_indx = train_validation_test_split(h5_file=opt.path_to_data,
                                                                         only_classes=opt.only_classes)

    label_map = get_classes_map(opt.h5_file)

    #logging.info('train_indx used: %s' % (', '.join(str(x) for x in train_indx)))
    #logging.info('validation_indx used: %s' % (', '.join(str(x) for x in validation_indx)))
    logging.info('test_indx used: %s' % (', '.join(str(x) for x in test_indx)))
    logging.info('label_map used: %s' % (str(label_map)))

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

    logging.info('the length of the trainloader is: %s' % (str(len(trainloader))))
    # collect statistics of the train data (mean & standard deviation) to normalize the data
    statistics = get_statistics(trainloader, opt.only_channels)
    logging.info('statistics used: %s' % (str(statistics)))

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

    if opt.save_test:
        # data_dir = "/home/aleksandra/PycharmProjects/interpretable-multichannel-image-analysis/data/WBC/PreprocessedTestData"
        for i, x in zip(np.arange(len(test_dataset)), test_dataset):
            torch.save((x['image'], int(x['label'])), os.path.join(opt.save_test, 'test_sample_{}.pt'.format(i)))
        print("All test Data preprocessed and saved")

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

    model = model.to(opt.dev)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9)

    if opt.resume_model != '':
        checkpoint = torch.load('{0}/{1}'.format(opt.model_save_path, opt.resume_model))
        model.load_state_dict(checkpoint)
        for log in os.listdir(opt.log_dir):
            os.remove(os.path.join(opt.log_dir, log))
    # breakpoint()
    for epoch in range(opt.n_epochs):
        running_loss = 0.0
        logging.info('epoch%d' % epoch)
        for i, data in enumerate(trainloader, 0):

            # get the inputs; data is a list of [inputs, labels]
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
            if i % 500 == 499:  # print every 2000 mini-batches
                logging.info('[%d, %5d] training loss: %.8f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for i, data in enumerate(validationloader, 0):
                indx = (data["object_number"] != -1).reshape(-1)
                if indx.sum() > 0:
                    inputs, labels = data["image"][indx], data["label"][indx]

                    inputs, labels = inputs.to(opt.dev), labels.to(opt.dev)
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

    logging.info('Accuracy of the network on the %d test images: %d %%' % (len(test_dataset),
                                                                           100 * correct / total))

    logging.info("The model saved: %s" % "final_model_dict_{}.pth".format(opt.model_name))
    torch.save(model.state_dict(), os.path.join(opt.model_save_path, "final_model_dict_{}.pth".format(opt.model_name)))
    cr = classification_report(y_true, y_pred, target_names=opt.class_names, digits=4)
    logging.info(cr)
    f1_score_original = f1_score(y_true, y_pred, average=None, labels=np.arange(opt.num_classes))
    df = pd.DataFrame(np.atleast_2d(f1_score_original), columns=opt.class_names)
    logging.info(df.to_string())

    # python train.py --n_epochs 100 --only_channels 0 2 3 4 5 6 7 8 9 10 11 --dev cuda --save_test data\WBC\test_samples_without_1_ch
