import argparse
import torch
import logging
import os
import torch.nn as nn
from matplotlib import pyplot as plt
import numpy as np
from resnet18 import resnet18
from interpretation_methods import shuffle_pixel_interpretation
from util import preprocess_image
from sklearn.metrics import f1_score
import pandas as pd
from datetime import datetime
from timeit import default_timer as timer

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

parser = argparse.ArgumentParser()
parser.add_argument('--path_to_model', default="models/best_metric_model_jurkat_oversampling_more_augm.pth",
                    help="path to the model to interpret")
parser.add_argument('--path_to_save_results', default="results/",
                    help="path to the folder to save results")
parser.add_argument('--path_to_images', default='images_to_interpret/',
                    help="path to the images to interpret", type=str)
parser.add_argument('--num_class', default=7, help="number of the classes to load the model", type=int)
parser.add_argument('--num_channels', default=3, help="number of the channels", type=int)
parser.add_argument('--class_names', default=WBC_CLASS_NAMES,
                    help="mapped names of classes", nargs='+',
                    type=str)
parser.add_argument('--shuffle_times', default=5, help="number of the channels", type=int)
parser.add_argument('--log_dir', default='logs/', help="path to save logs")
parser.add_argument('--dev', default='cpu', help="cpu or cuda")
parser.add_argument('--cmap', default='gray', help="colormap for visualizing saliency maps")
parser.add_argument('--batch', default=64, type=int)
parser.add_argument('--num_workers', default=1, type=int)
opt = parser.parse_args()

now = datetime.now()
timestamp = datetime.timestamp(now)
logging.basicConfig(filename=os.path.join(opt.log_dir, 'interpretation_metrics_output_{}.txt'.format(timestamp)),
                    level=logging.DEBUG)

if __name__ == '__main__':
    # define device
    if opt.dev != 'cpu':
        opt.dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info("the deviced being used is {}".format(opt.dev))
    print("the deviced being used is {}".format(opt.dev))
    ### measure runtime

    # load model
    model = resnet18(pretrained=True)
    if opt.num_channels != 3:
        model.conv1 = nn.Conv2d(opt.num_channels, 64, kernel_size=(7, 7),
                                stride=(2, 2), padding=(3, 3), bias=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, opt.num_class)
    model = model.to(opt.dev)
    model.load_state_dict(
        torch.load(opt.path_to_model, map_location=torch.device(opt.dev)))
    logging.info('The model is loaded')
    print('The model is loaded')

    files_to_interpret, data_loader = preprocess_image(opt.path_to_images, opt.batch, opt.num_workers)

    start = timer()
    print("Start shuffling")
    y_true, y_pred, y_pred_per_channel = shuffle_pixel_interpretation(model, data_loader, opt.num_channels, opt.dev,
                                                                      opt.shuffle_times)
    f1_scores_per_channel = {}
    f1_score_original = f1_score(y_true, y_pred, average=None, labels=np.arange(opt.num_class))
    df = pd.DataFrame(np.atleast_2d(f1_score_original), columns=opt.class_names)
    logging.info("the f1 score for original data is \n {}".format(df))
    for channel, y_pred_per_ch in y_pred_per_channel.items():
        f1_scores_per_shuffle = []
        '''torch.stack(y_pred_ch, dim=0): tensor([[1, 1, 1, 1, 1], [1, 3, 1, 1, 1], [1, 1, 1, 1, 1]]), where 5 is # 
        of shuffle times and 3 is # of samples. Transpose (y_pred_per_im) returns the tensor, where the values in one 
        row are predicted labels for all samples and column are predicted labels for one sample shuffled n times '''
        y_pred_per_im = torch.stack(y_pred_per_ch, dim=0).T
        for y_pred_per_shuffle in y_pred_per_im:
            f1_scores_per_shuffle.append(
                f1_score(y_true, y_pred_per_shuffle, average=None, labels=np.arange(opt.num_class)))
        f1_score_all_per_channel = np.stack(f1_scores_per_shuffle)

        f1_diff_from_original = f1_score_original - f1_score_all_per_channel
        df_diff = pd.DataFrame(np.atleast_2d(f1_diff_from_original), columns=opt.class_names)
        # df_diff = df_diff[['G1', 'G2', 'S', 'Prophase', 'Metaphase', 'Anaphase', 'Telophase']]
        fig = plt.figure(figsize=(10, 5))
        bp = df_diff.boxplot()
        fig.savefig(os.path.join(opt.path_to_save_results,
                                 "bp-shuffle_method-model-{}-channel-{}.png".format(
                                     str(os.path.basename(os.path.normpath(opt.path_to_model))), str(channel))))
        # breakpoint()
        # f1_mean = np.mean(f1_score_all_per_channel, axis=0)
        # f1_std = np.std(f1_score_all_per_channel, axis=0)
        # df_mean = pd.DataFrame(np.atleast_2d(f1_mean), columns=opt.class_names)
        # df_std = pd.DataFrame(np.atleast_2d(f1_std), columns=opt.class_names)
        # logging.info("the mean of the f1 score for the channel {} is \n {}".format(channel, df_mean))
        # logging.info("the standard deviation of the f1 score for the channel {} is \n {}".format(channel, df_std))
        """
        python interpretation_metrics.py --dev cpu --shuffle_times 100 --path_to_images /home/aleksandra/PycharmProjects/interpretable-multichannel-image-analysis/data/WBC/PreprocessedTestData --path_to_model /home/aleksandra/PycharmProjects/interpretable-multichannel-image-analysis/models/final_model_dict_wbc_all.pth --num_class 9 --num_channels 12"""
    end = timer()
    logging.info("runtime of the shuffle pixels method is: {}".format(end - start))
