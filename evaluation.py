import argparse
import torch
import logging
import os
import torch.nn as nn
from matplotlib import pyplot as plt
from interpretation_methods import *
import numpy as np
from resnet18 import resnet18
from interpretation import preprocess_image, run_interpretation_methods
import copy
from scipy.special import softmax

parser = argparse.ArgumentParser()
parser.add_argument('--path_to_model', default="models/best_metric_model_jurkat_oversampling_more_augm.pth",
                    help="path to the model to interpret")
parser.add_argument('--path_to_images', default='images_to_interpret/',
                    help="path to the images to interpret", type=str)
parser.add_argument('--only_methods', default=None, help="the methods to be evaluated", nargs='+', type=str)
parser.add_argument('--dev', default='cpu', help="cpu or cuda")
parser.add_argument('--num_class', default=7, help="number of the classes to load the model", type=int)
parser.add_argument('--batch', default=64, type=int)
parser.add_argument('--path_to_save_results', default="results/",
                    help="path to the folder to save results")
opt = parser.parse_args()

__all_interpret_methods__ = ['deep_lift', 'guided_grad_cam', 'saliency', 'gradient_shap', 'input_x_gradient',
                             'deconvolution', 'guided_backprop', 'intergrated_gradients']


def calculate_aops(model, data_loader, number_of_samples, method, L = 100):
    with torch.no_grad():
        aopc_per_iteration = []
        for test_data in data_loader:
            inputs, labels = test_data[0].to(opt.dev).float(), test_data[1].to(opt.dev)
            input_size = inputs.shape
            labels = labels.reshape(-1)
            baselines = softmax(model(inputs), axis=1).detach()[np.arange(len(labels.tolist())), labels.tolist()]
            heatmaps = globals()[method](model, inputs, labels)
            image_size = heatmaps.shape[1]*heatmaps.shape[2]*heatmaps.shape[3]
            sorted_regions_vals, sorted_regions_ids = torch.sort(heatmaps.reshape(heatmaps.shape[0],image_size),
                                                                     dim=1, descending=True)
            inputs_flatten = copy.deepcopy(inputs.reshape(inputs.shape[0],image_size))
            # saves the results of the previous column
            intermediate_sum = []
            # AOPC for the ith iteration for the
            aopc_results_per_batch_per_iteration = []
            for i in range(min(L,image_size)):
                region_ids = sorted_regions_ids[:, :i+1]
                for j in range(len(region_ids)):
                    inputs_flatten[j,region_ids[j]] = 100.0
                prob_occl = softmax(model(inputs_flatten.reshape(input_size)), axis=1).detach()[np.arange(len(labels.tolist())), labels.tolist()]
                if i != 0:
                    intermediate_sum.append(intermediate_sum[-1] + baselines - prob_occl)
                else:
                    intermediate_sum.append(baselines - prob_occl)
                aopc_results_per_batch_per_iteration.append(
                    (torch.sum(intermediate_sum[i]) / number_of_samples) / (i + 1))
            aopc_per_iteration.append(aopc_results_per_batch_per_iteration)
        # plt.bar(range(1, len(aopc) + 1), aopc)
        return np.asarray(aopc_per_iteration).sum(axis=0)


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

    # evaluate the method
    files_to_interpret, data_loader = preprocess_image(opt.path_to_images)
    number_of_samples = len(files_to_interpret)
    if opt.only_methods is not None:
        methods_to_run = opt.only_methods
    else:
        methods_to_run = __all_interpret_methods__
    for method in methods_to_run:
        aopc_per_method = calculate_aops(model, data_loader, number_of_samples, method)
        plt.plot(range(1, len(aopc_per_method) + 1), aopc_per_method, label="aopc for {}".format(method))
    plt.xlabel('L (# sorted pixels occluded)')
    plt.ylabel("AOPC")
    plt.title('area over the perturbation curve (AOPC)')
    plt.legend()
    plt.savefig(os.path.join(opt.path_to_save_results, "AOPC-for-methods-{}-model-{}.png".format(''.join(methods_to_run), str(os.path.basename(os.path.normpath(opt.path_to_model))))))
    plt.clf()

# example to run: python evaluation.py --only_methods deep_lift guided_grad_cam deconvolution saliency input_x_gradient
