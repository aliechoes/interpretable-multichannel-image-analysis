import argparse
import torch
import logging
import os
import torch.nn as nn
from matplotlib import pyplot as plt
from interpretation_methods import *
import numpy as np
from resnet18 import resnet18

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
parser.add_argument('--num_class', default=7, help="number of the classes to load the model", nargs='+', type=int)
parser.add_argument('--log_dir', default='logs/', help="path to save logs")
parser.add_argument('--dev', default='cpu', help="cpu or cuda")

opt = parser.parse_args()
"""
logger = logging.getLogger()
fhandler = logging.FileHandler(filename=os.path.join(opt.log_dir, 'interpretation_output.txt'), mode='a')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fhandler.setFormatter(formatter)
logger.addHandler(fhandler)
logger.setLevel(logging.DEBUG)
"""
logging.basicConfig(filename=os.path.join(opt.log_dir, 'interpretation_output.txt'), level=logging.DEBUG)

__all_interpret_methods__ = ['deep_lift', 'guided_grad_cam', 'saliency', 'gradient_shap', 'input_x_gradient',
                             'deconvolution', 'guided_backprop', 'intergrated_gradients']


def preprocess_image(path_to_images=None):
    files_to_interpret = []
    for file in os.listdir(path_to_images):
        if file.endswith(".pt"):
            files_to_interpret.append(os.path.join(path_to_images, file))
    logging.info("The samples to interpret are in: {} and the number of samples is {}".format(path_to_images,
                                                                                              len(files_to_interpret)))
    images = []
    labels = []
    files_names = []
    for file in files_to_interpret:
        sample_tensor = torch.load(file)
        images.append(sample_tensor[0].unsqueeze(0))
        labels.append(int(sample_tensor[1]))
        files_names.append(os.path.basename(os.path.normpath(file)))
    return files_names, images, labels


def run_interpretation_methods(model=None, files_to_interpret=[], images=[], labels=[]):
    if opt.only_methods is not None:
        methods_to_run = opt.only_methods
    else:
        methods_to_run = __all_interpret_methods__
    logging.info('Interpretation methods to be applied: {}'.format(methods_to_run))
    heatmaps = {}
    for (file,image,label) in zip(files_to_interpret,images,labels):
        heatmaps_per_image = {}
        heatmaps_per_image['original'] = image
        for method in methods_to_run:
            heatmaps_per_image[method] = globals()[method](model, image.float(), label)
        heatmaps[file] = heatmaps_per_image
    logging.info("Interpretation stage is over")
    return heatmaps


def save_heatmaps(heatmaps=None, model_name=""):
    for key, val in heatmaps.items():
        for method, res in val.items():
            value = res[0]
            if method != 'original':
                value = (value - value.min()) / (value.max() - value.min())
            fig, axs = plt.subplots(1, len(value), figsize=(15, 6))
            axs = axs.ravel()
            for (i, h) in zip(np.arange(len(value)), value):
                if method != 'original':
                    axs[i].imshow(h.detach().numpy(), vmin=0, vmax=2, cmap='gray', aspect='auto')
                else:
                    axs[i].imshow(h.detach().numpy())
            fig.savefig(os.path.join(opt.path_to_save_results, "{}-method-{}-model-{}.png".format(key, method, str(model_name))))
    logging.info("Interpretation results are saved in {}".format(opt.path_to_save_results))


if __name__ == '__main__':
    # define device
    if opt.dev is not 'cpu':
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

    files_to_interpret, images, labels = preprocess_image(opt.path_to_images)
    heatmaps = run_interpretation_methods(model, files_to_interpret, images, labels)
    save_heatmaps(heatmaps, os.path.basename(os.path.normpath(opt.path_to_model)))
