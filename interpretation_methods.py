import torch
from torch import Tensor
import numpy as np
import copy
from scipy.special import softmax
from captum.attr import (
    GuidedGradCam,
    DeepLift,
    Saliency,
    DeepLiftShap,
    GradientShap,
    InputXGradient,
    IntegratedGradients,
    GuidedBackprop,
    Deconvolution,
    Occlusion,
    FeaturePermutation,
    ShapleyValueSampling,
    Lime,
    KernelShap
)

__all__ = ['deep_lift', 'guided_grad_cam', 'saliency', 'gradient_shap', 'input_x_gradient', 'shapley_value_sampling', 'feature_permutation', 'occlusion', 'deconvolution', 'guided_backprop',
           'intergrated_gradients']
"""
Interpretation Methods from Library Captum
"""


def kernel_shap(model, input, label):
    ks = KernelShap(model)
    return ks.attribute(input, target=label, n_samples=20)


def similarity_kernel(
        original_input: Tensor,
        perturbed_input: Tensor,
        perturbed_interpretable_input: Tensor,
        **kwargs) -> Tensor:
    # kernel_width will be provided to attribute as a kwarg
    kernel_width = kwargs["kernel_width"]
    l2_dist = torch.norm(original_input - perturbed_input)
    return torch.exp(- (l2_dist ** 2) / (kernel_width ** 2))


def perturb_func(
        original_input: Tensor,
        **kwargs) -> Tensor:
    return original_input + torch.randn_like(original_input)


def lime(model, input, label):
    lm = Lime(model)
    return lm.attribute(input, target=label)


def shapley_value_sampling(model, input, label):
    svs = ShapleyValueSampling(model)
    return svs.attribute(input, target=label, baselines=input * 0, n_samples=20)


def feature_permutation(model, input, label):
    fp = FeaturePermutation(model)
    return fp.attribute(input, target=label)


def occlusion(model, input, label):
    occ = GuidedBackprop(model)
    return occ.attribute(input, (4, 4), target=label)


def deconvolution(model, input, label):
    dc = GuidedBackprop(model)
    return dc.attribute(input, target=label)


def guided_backprop(model, input, label):
    gb = GuidedBackprop(model)
    return gb.attribute(input, target=label)


def intergrated_gradients(model, input, label):
    ig = IntegratedGradients(model)
    return ig.attribute(input, target=label)


def input_x_gradient(model, input, label):
    ixg = InputXGradient(model)
    return ixg.attribute(input, target=label)


def gradient_shap(model, input, label):
    gs = GradientShap(model)
    return gs.attribute(input, target=label, baselines=input * 0.0)


"""
# If baselines are provided in shape of scalars or with a single baseline example, `DeepLift` approach can be used instead.
def deep_lift_shap(model, input, label):
    dls = DeepLiftShap(model)
    background = input * 0 + input.min()
    return dls.attribute(input, target=label, baselines=background)
"""


def deep_lift(model, input, label):
    dl = DeepLift(model)
    return dl.attribute(input, target=label)


def guided_grad_cam(model, input, label, layer=None):
    if layer is None:
        layer = model.layer1
    guided_gc = GuidedGradCam(model, layer)
    return guided_gc.attribute(input, target=label)


def saliency(model, input, label):
    saliency = Saliency(model)
    return saliency.attribute(input, target=label)


### Shuffle interpretation methods

def shuffle_pixel_interpretation(model, test_loader, num_channels, device, shuffle_times):
    model.eval()
    y_true = list()
    y_pred = list()
    y_pred_per_channel = {}
    for n in range(num_channels):
        y_pred_per_channel["y_pred_{}".format(n)] = list()
    with torch.no_grad():
        for test_data in test_loader:
            test_images, test_labels = test_data[0].to(device).float(), test_data[1].to(device)
            pred = model(test_images).argmax(dim=1)
            ### True and Predicted labels of original data
            for i_or in range(len(pred)):
                y_true.append(test_labels[i_or].item())
                y_pred.append(pred[i_or].item())
            ### Predited labels of shuffled data
            for i in range(test_images[0].shape[0]):
                pred_images = []
                for image in test_images:
                    image_shuffled_times = []
                    for t in range(shuffle_times):
                        im = shuffle_pixels_in_channel(i, image)
                        image_shuffled_times.append(im)
                    image_shuffled_times = torch.stack(image_shuffled_times, dim=0)
                    pred_image_shuffled_times = model(image_shuffled_times).argmax(dim=1)
                    pred_images.append(pred_image_shuffled_times)
                for i_ch in range(len(pred_images)):
                    y_pred_per_channel["y_pred_{}".format(i)].append(pred_images[i_ch])
        return y_true, y_pred, y_pred_per_channel


def shuffle_pixels_in_channel(channel, image):
    im = copy.deepcopy(image)
    channel_shape = im[channel].shape
    arr = np.asarray(im[channel].flatten())
    np.random.shuffle(arr)
    im[channel] = torch.Tensor(arr.reshape(channel_shape))
    return im
