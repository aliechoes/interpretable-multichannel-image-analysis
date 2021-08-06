import torch
from pathlib import Path
from typing import Sequence, Union
import numpy as np
from monai.transforms.compose import Transform
from PIL import Image
import os
from sklearn.base import BaseEstimator, TransformerMixin
import scipy.sparse as sp
from scipy.stats import kurtosis, skew
from scipy.spatial import distance as dist
from skimage.feature import greycomatrix, greycoprops
from skimage.measure import shannon_entropy


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


class LoadImage(Transform):
    """
    Load common 2D image format (PNG, JPG, etc. using PIL) file or files from provided path.
    If loading a list of files, stack them together and add a new dimension as first dimension,
    and use the meta data of the first image to represent the stacked result.
    It's based on the Image module in PIL library:
    https://pillow.readthedocs.io/en/stable/reference/Image.html
    """

    def __init__(self, only_channels=[]) -> None:
        self.only_channels = only_channels

    def __call__(self, name: Union[Sequence[Union[Path, str]], Path, str]):
        if isinstance(name, (np.ndarray, np.generic)):
            name = name[0]
        img_array = list()
        class_dir = os.path.dirname(name)
        sample_id = os.path.basename(name) + "_Ch"
        channels = [np.asarray(Image.open(os.path.join(class_dir, file))) / 255. for file in os.listdir(class_dir) if
                    file.startswith(sample_id)]
        image = np.stack(channels, axis=2)
        image_RGB = image * 255.
        return Image.fromarray(np.uint8(image_RGB))


class ToTensorCustom(Transform):
    """
    Converts the input image to a tensor without applying any other transformations.
    """

    def __call__(self, img: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Apply the transform to `img` and make it contiguous.
        """
        # img = np.asarray(img)
        # image = torch.from_numpy(copy.deepcopy(img))
        if torch.is_tensor(img):
            return img.contiguous().T
        # Tracer()()
        return torch.as_tensor(np.ascontiguousarray(img) / 255.).T


class RandomCrop3D():
    def __init__(self, img_sz, crop_sz):
        c, h, w, d = img_sz
        assert (h, w, d) > crop_sz
        self.img_sz = tuple((h, w, d))
        self.crop_sz = tuple(crop_sz)

    def __call__(self, x):
        slice_hwd = [self._get_slice(i, k) for i, k in zip(self.img_sz, self.crop_sz)]
        return self._crop(x, *slice_hwd)

    @staticmethod
    def _get_slice(sz, crop_sz):
        try:
            lower_bound = torch.randint(sz - crop_sz, (1,)).item()
            return lower_bound, lower_bound + crop_sz
        except:
            return (None, None)

    @staticmethod
    def _crop(x, slice_h, slice_w, slice_d):
        return x[:, slice_h[0]:slice_h[1], slice_w[0]:slice_w[1], slice_d[0]:slice_d[1]]


class BestStatisticalFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.features_ = []

    def fit(self, X=None, y=None, **fit_params):
        return self

    def transform(self, X):
        for image in X:
            features = []
            for ch in range(image.shape[0]):
                # percentiles
                features.append(image[ch, :, :].min())
                features.append(np.percentile(image[ch, :, :], 0.1))
                features.append(np.percentile(image[ch, :, :], 0.2))
                features.append(np.percentile(image[ch, :, :], 0.3))
                features.append(np.percentile(image[ch, :, :], 0.4))
                features.append(np.percentile(image[ch, :, :], 0.5))
                features.append(np.percentile(image[ch, :, :], 0.6))
                features.append(np.percentile(image[ch, :, :], 0.7))
                features.append(np.percentile(image[ch, :, :], 0.8))
                features.append(np.percentile(image[ch, :, :], 0.9))
                features.append(image[ch, :, :].max())

                # pixel sum
                features.append(image[ch, :, :].sum())

                # moments
                features.append(image[ch, :, :].mean())
                features.append(image[ch, :, :].std())
                features.append(kurtosis(image[ch, :, :].ravel()))
                features.append(skew(image[ch, :, :].ravel()))

                features.append(shannon_entropy(image[ch, :, :]))
            self.features_.append(features)

        return sp.csr_matrix(np.asarray(self.features_), dtype=np.float64)


class GLSMFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.features_ = []

    def fit(self, X=None, y=None, **fit_params):
        return self

    def transform(self, X):
        for image in X:
            features = []
            for ch in range(image.shape[0]):
                # create a 2D temp image
                temp_image = image[ch, :, :].copy()
                temp_image = (temp_image / temp_image.max()) * 255  # use 8bit pixel values for GLCM
                temp_image = temp_image.astype('uint8')  # convert to unsigned for GLCM

                # calculating glcm
                glcm = greycomatrix(temp_image, distances=[5], angles=[0], levels=256)

                # storing the glcm values
                features.append(greycoprops(glcm, prop='contrast')[0, 0])
                features.append(greycoprops(glcm, prop='dissimilarity')[0, 0])
                features.append(greycoprops(glcm, prop='homogeneity')[0, 0])
                features.append(greycoprops(glcm, prop='ASM')[0, 0])
                features.append(greycoprops(glcm, prop='energy')[0, 0])
                features.append(greycoprops(glcm, prop='correlation')[0, 0])
            self.features_.append(features)

        return sp.csr_matrix(np.asarray(self.features_), dtype=np.float64)


class ItemSelector(BaseEstimator, TransformerMixin):
    """For data grouped by feature, select subset of data at a provided key.

    The data is expected to be stored in a 2D data structure, where the first
    index is over features and the second is over samples.  i.e.

    >> len(data[key]) == n_samples

    Please note that this is the opposite convention to scikit-learn feature
    matrixes (where the first index corresponds to sample).

    ItemSelector only requires that the collection implement getitem
    (data[key]).  Examples include: a dict of lists, 2D numpy array, Pandas
    DataFrame, numpy record array, etc.

    >> data = {'a': [1, 5, 2, 5, 2, 8],
               'b': [9, 4, 1, 4, 1, 3]}
    >> ds = ItemSelector(key='a')
    >> data['a'] == ds.transform(data)

    ItemSelector is not designed to handle data grouped by sample.  (e.g. a
    list of dicts).  If your data is structured this way, consider a
    transformer along the lines of `sklearn.feature_extraction.DictVectorizer`.

    Parameters
    ----------
    key : hashable, required
        The key corresponding to the desired value in a mappable.
    """

    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        images = []
        for image in X:
            images.append(image[self.key].cpu().numpy())
        return np.asarray(images)
