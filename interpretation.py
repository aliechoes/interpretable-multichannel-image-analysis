import argparse
import torch
import logging
import os
import torch.nn as nn
from matplotlib import pyplot as plt
from interpretation_methods import *
import numpy as np
from resnet18 import resnet18
from torchvision import datasets, transforms
from custom_transforms import LoadTensor
from test_dataset import TestDataset
from torch.utils.data import Dataset, DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--path_to_model', default="models/best_metric_model_jurkat_oversampling_more_augm.pth",
                    help="path to the model to interpret")
parser.add_argument('--path_to_save_results', default="results/",
                    help="path to the folder to save results")
parser.add_argument('--only_methods', default=None,
                    help="the methods to be conducted on the model, if None then all methods are applied", nargs='+',
                    type=str)
parser.add_argument('--path_to_images', default='images_to_interpret/',
                    help="path to the images to interpret", type=str)
parser.add_argument('--num_class', default=7, help="number of the classes to load the model", type=int)
parser.add_argument('--log_dir', default='logs/', help="path to save logs")
parser.add_argument('--dev', default='cpu', help="cpu or cuda")
parser.add_argument('--cmap', default='gray', help="colormap for visualizing saliency maps")
parser.add_argument('--batch', default=64, type=int)
parser.add_argument('--num_workers', default=1, type=int)
opt = parser.parse_args()

logging.basicConfig(filename=os.path.join(opt.log_dir, 'interpretation_output.txt'), level=logging.DEBUG)

__all_interpret_methods__ = ['deep_lift', 'guided_grad_cam', 'saliency', 'gradient_shap', 'input_x_gradient',
                             'deconvolution', 'guided_backprop', 'intergrated_gradients']



def load_test_data(files_to_interpret):
    test_transforms = transforms.Compose([
        LoadTensor()
    ])
    test_ds = TestDataset(files_to_interpret, test_transforms)
    return DataLoader(test_ds, batch_size = opt.batch, shuffle=False, num_workers=opt.num_workers)


def preprocess_image(path_to_images=None):
    files_to_interpret = []
    for file in os.listdir(path_to_images):
        if file.endswith(".pt"):
            files_to_interpret.append(os.path.join(path_to_images, file))
    logging.info("The samples to interpret are in: {} and the number of samples is {}".format(path_to_images,
                                                                                              len(files_to_interpret)))
    test_loader = load_test_data(files_to_interpret)
    files_names = [os.path.basename(os.path.normpath(file)) for file in files_to_interpret]
    return files_names, test_loader


def run_interpretation_methods(model, test_loader, methods_to_run):
    logging.info('Interpretation methods to be applied: {}'.format(methods_to_run))
    heatmaps = {}
    with torch.no_grad():
        for test_data in test_loader:
            test_images, test_labels = test_data[0].to(opt.dev).float(), test_data[1].to(opt.dev)
            if 'original' not in heatmaps:
                heatmaps['original'] = torch.empty(0, dtype=torch.float32, device=opt.dev)
            heatmaps['original'] = torch.cat((heatmaps['original'], test_images))
            for method in methods_to_run:
                if method not in heatmaps:
                    heatmaps[method] = torch.empty(0, dtype=torch.float32, device=opt.dev)
                heatmaps[method] = globals()[method](model, test_images, test_labels)
    logging.info("Interpretation stage is over")
    return heatmaps


def save_heatmaps(heatmaps=[], files_to_interpret=[], model_name=""):
    for method,val in heatmaps.items():
        for (file,heatmap) in zip(files_to_interpret,val):
            # breakpoint()
            fig, axs = plt.subplots(2, len(heatmap), figsize=(15, 10))
            axs = axs.ravel()
            for (i, h) in zip(np.arange(len(heatmap)), heatmap):
                h = h.detach().numpy()
                if method != 'original':
                    #heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
                    axs[i].imshow(h, cmap=opt.cmap, aspect='auto')
                else:
                    axs[i].imshow(h)
                binwidth = 0.025
                lim = np.ceil(np.abs(h).max() / binwidth) * binwidth
                bins = np.arange(-lim, lim + binwidth, binwidth)
                axs[i + len(heatmap)].hist(h, bins=bins, color=['blue' for i in range(66)])
            fig.savefig(os.path.join(opt.path_to_save_results, "{}-method-{}-model-{}.png".format(file, method, str(model_name))))
    logging.info("Interpretation results are saved in {}".format(opt.path_to_save_results))


if __name__ == '__main__':
    # define device
    if opt.dev != 'cpu':
        opt.dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info("the deviced being used is {}".format(opt.dev))

    # load model
    model = resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, opt.num_class)
    model = model.to(opt.dev)
    model.load_state_dict(
        torch.load(opt.path_to_model, map_location=torch.device(opt.dev)))
    logging.info('The model is loaded')

    files_to_interpret, data_loader = preprocess_image(opt.path_to_images)
    if opt.only_methods is not None:
        methods_to_run = opt.only_methods
    else:
        methods_to_run = __all_interpret_methods__
    heatmaps = run_interpretation_methods(model, data_loader, methods_to_run)
    save_heatmaps(heatmaps, files_to_interpret, os.path.basename(os.path.normpath(opt.path_to_model)))
