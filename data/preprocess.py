import cv2
import torch
import random
import numpy as np

from torchvision.transforms import transforms
from typing import Sequence, Union, Tuple, List, Any


class SaltPepperNoise:

    """
    Add Salt & Pepper noise to data.

    * Args:
        amount (float):
            Amount of salt and pepper with respect to entire pixels.
            Must be in range (0.0, 1.0)
            Default value is 0.05 (5%).

        upper_value (float):
            Value of salt.
            Default is 255.0.

        lower_value (float):
            Value of pepper.
            Default is 0.0
    """

    def __init__(self,
                 amount: float = 0.05,
                 upper_value: float = 255.0,
                 lower_value: float = 0.0):

        assert amount is not None, "Argument 'amount' cannot be None."
        assert 0 < amount < 1, "Argument `amount` should be in range (0, 1)."

        self.amount = amount
        self.upper_value = upper_value
        self.lower_value = lower_value

    def __call__(self, data: torch.Tensor):

        assert data.dim() >= 2, \
            "Dimension of 'data' must be equal to or larger than 2."

        rand_matrix = torch.Tensor(np.random.random(data.shape))
        data[rand_matrix >= (1 - (self.amount / 2))] = self.upper_value
        data[rand_matrix <= (self.amount / 2)] = self.lower_value

        return data


class GaussianNoise:

    """
    Add Gaussian noise to data.

    * Args:
        std (Union[List[int], int]):
            Standard deviation of Gaussian Noise.
            If 'std' is entered in the form of a list,
            one of the elements of the list is randomly selected and applied.

        clip (bool):
            Whether to clip the noise-added data values between 'min_value' and 'max_value'.
            Default is True.

        min_value, max_value (float):
            Minimum and maximum values of the noise-added data.
            Default is 0.0 and 255.0, respectively.
    """

    def __init__(self,
                 std: Union[List[int], int],
                 clip: bool = True,
                 min_value: float = 0.0,
                 max_value: float = 255.0):

        assert std is not None, "Argument 'std' cannot be None."
        if isinstance(std, int):
            std = [std]

        self.std = std
        self.clip = clip
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, data: torch.Tensor):
        std = random.choice(self.std)
        noise = torch.randn(*data.shape) * std
        noise_added = data + noise

        if self.clip:
            return torch.clip(noise_added, self.min_value, self.max_value)
        return noise_added


class NoiseFree:

    """
    Identity class that sends input as it is to output again.
    """

    def __init__(self, params: Any):
        pass

    def __call__(self, data: torch.Tensor):
        return data


class RandomCrop:

    """
    Randomly crop patches from given image.
    [EX]
        (H, W, C) --> (Patch_h, Patch_w, C)
        (H, W)    --> (Patch_h, Patch,w)

    * Args:
        output_size (Union[Tuple[int, int], int]):
            Size of cropped patches.

    * Returns
        np.ndarray
    """

    def __init__(self, output_size: Union[Tuple[int, int], int]):
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        self.output_size = output_size

    def __call__(self, image: np.ndarray):
        assert image.ndim >= 2, \
            "Dimension of 'image' must be equal to or larger than 2."
        height, width = image.shape[:2]
        new_height, new_width = self.output_size

        top = np.random.randint(0, height - new_height)
        left = np.random.randint(0, width - new_width)

        id_y = np.arange(top, top + new_height, 1)[:, np.newaxis].astype(np.int32)
        id_x = np.arange(left, left + new_width, 1).astype(np.int32)

        return image[id_y, id_x]


class Resize:

    """
    Resize given image array to desired size.

    * Args:
        desired_size (Union[Tuple[int, int], int]):
            Desired resized shape

    * Returns:
        np.ndarray
    """

    def __init__(self, desired_size: Union[Tuple[int, int], int]):
        if isinstance(desired_size, int):
            desired_size = (desired_size, desired_size)
        self.desired_size = desired_size

    def __call__(self, image: np.ndarray):
        return cv2.resize(image, dsize=self.desired_size)


class Normalizer:

    """
    Normalize given input tensor with 'mean' and 'std'.

    * Args:
        mean, std (Union[Sequence[float], float]):
            Mean and Standard deviation value needed for normalization.
            When these parameters are input in a sequence format,
            the length of the sequence must be the same as the number of channels of the input tensor.
    """

    def __init__(self,
                 mean: Union[Sequence[float], float],
                 std: Union[Sequence[float], float]):

        self.mean = mean
        self.std = std

    def __call__(self, tensor: torch.Tensor):
        if tensor.dim() == 3:
            return transforms.Normalize(mean=self.mean, std=self.std)(tensor)
        elif tensor.dim() == 4:
            return torch.cat([transforms.Normalize(mean=self.mean, std=self.std)(t).unsqueeze(0)
                              for t in tensor], dim=0)
        else:
            raise Exception("Dimension of `tensor` should be equal to or larger than 3.")


class Denormalizer:

    """
    Denormalize given input tensor with 'mean' and 'std'.
    Inverse transform of :Class: 'Normalizer'.

    * Args:
        mean, std (Union[Sequence[float], float]):
            Mean and Standard deviation value needed for denormalization.
            When these parameters are input in a sequence format,
            the length of the sequence must be the same as the number of channels of the input tensor.
    """

    def __init__(self, mean, std):

        self.mean = mean
        self.std = std

    def __call__(self, tensor: torch.Tensor):

        if tensor.dim() == 3:
            num_ch = tensor.shape[0]
            self.mean = self.convert(self.mean, num_ch)
            self.std = self.convert(self.std, num_ch)

            return transforms.Normalize(mean=[-m / s for m, s in zip(self.mean, self.std)],
                                        std=[1 / s for s in self.std])(tensor)

        elif tensor.dim() == 4:
            num_ch = tensor.shape[1]
            self.mean = self.convert(self.mean, num_ch)
            self.std = self.convert(self.std, num_ch)

            return torch.cat([transforms.Normalize(mean=[-m / s for m, s in zip(self.mean, self.std)],
                                                   std=[1 / s for s in self.std])(t).unsqueeze(0) for t in tensor], dim=0)

        else:
            raise Exception("Dimension of 'tensor' should be 3 or 4.")

    def convert(self, input_param: Any, desired_dim: int):
        if isinstance(input_param, Sequence):
            assert len(input_param) == desired_dim
            return input_param

        else:
            return [input_param for _ in range(desired_dim)]
