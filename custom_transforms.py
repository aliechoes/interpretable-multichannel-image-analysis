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