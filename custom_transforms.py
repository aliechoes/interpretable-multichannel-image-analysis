import torch
from pathlib import Path
from typing import Sequence, Union


class LoadTensor(object):
    """
    Load the test samples from already preprocessed dataset
    """

    def __init__(self, part=1.0) -> None:
        self.part = part

    def __call__(self, name: Union[Sequence[Union[Path, str]], Path, str]):
        sample_tensor = torch.load(name)
        return sample_tensor[0], sample_tensor[1]


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1., p=0.5):
        self.std = std
        self.mean = mean
        self.p = p

    def __call__(self, tensor):
        if torch.rand(1) < self.p:
            return tensor + torch.randn(tensor.size()) * self.std + self.mean
        return tensor

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
