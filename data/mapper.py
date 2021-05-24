import copy
import torch
import detectron2.data.transforms as T
import detectron2.data.detection_utils as utils

from yacs.config import CfgNode
from typing import Any, Union, Tuple
from data.preprocess import GaussianNoise, SaltPepperNoise, NoiseFree


def get_noiser(noise_type: str, noise_params: Any = None):

    """
    Build instance of class which adds noise to data.

    * Args:
        noise_type (str):
            Type of noise.
            Must be one in ['none, 'gaussian', 'snp'].
            ('snp' represents 'Salt & Pepper'.)
            If a noise type that is not supported is input, noise is not added.

        noise_params (Any):
            - When 'noise_type' is 'none',
                This argument will be ignored.

            - When 'noise_type' is 'gaussian',
                Please enter the standard deviation of Gaussian noise.
                The input type must be Union[int, List[int]].

            - When 'noise_type' is 'snp',
                Please enter the amount of salt & pepper noise.
                The input type must be float which range in (0.0, 1.0).
    """

    MAP_DICT = {
        'none': NoiseFree,
        'gaussian': GaussianNoise,
        'snp': SaltPepperNoise
    }

    assert noise_type in MAP_DICT.keys(), \
        f"Argument 'noise_type' must be one in {list(MAP_DICT.keys())}."
    return MAP_DICT.get(noise_type, 'none')(noise_params)


def generate_mapper(image_size: Union[Tuple[int, int], int],
                    noise_type: str,
                    noise_param: Any,
                    apply_flip: bool = True):

    """
    Generate mapper that is used in training.
    The mapper transforms the raw dataset so it is suitable for training.

    * Args:
        image_size (Union[Tuple[int, int], int]):
            Desired size of training images.

        noise_type (str), noise_param (Any):
            Please check the comment of :Method: 'get_noiser'.

        apply_flip (bool):
            Whether to apply horizontal flip as data augmentation.
            Probability of flip is set to 0.5.
    """

    if isinstance(image_size, int):
        image_size = (image_size, image_size)

    noiser = get_noiser(noise_type, noise_param)
    default_aug = [T.Resize(shape=image_size)]
    if apply_flip:
        default_aug.append(T.RandomFlip(prob=0.50, horizontal=True, vertical=False),)

    def mapper(dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(dataset_dict['file_name'], format='BGR')
        image, transforms = T.apply_transform_gens(default_aug, image)

        dataset_dict['image'] = noiser(torch.as_tensor(image.transpose(2, 0, 1).astype('float32')))
        annos = [utils.transform_instance_annotations(obj, transforms, image.shape[:2])
                 for obj in dataset_dict.pop("annotations")
                 if obj.get("iscrowd", 0) == 0]
        instances = utils.annotations_to_instances(annos, image.shape[:2])
        dataset_dict['instances'] = utils.filter_empty_instances(instances)
        return dataset_dict

    return mapper


def generate_mapper_using_cfg(cfg: CfgNode):

    """
    This method and :Method: 'generate_mapper' are exactly the same, only arguments are different.

    * Args:
        cfg (CfgNode):
            Configuration
    """

    image_size = cfg.DATASETS.RESIZE
    noise_type = cfg.DATASETS.NOISE_TYPE
    noise_params = cfg.DATASETS.NOISE_PARAMS

    if isinstance(image_size, int):
        image_size = (image_size, image_size)

    noiser = get_noiser(noise_type, noise_params)
    default_aug = [T.Resize(shape=image_size),
                   T.RandomFlip(prob=0.50, horizontal=True, vertical=False)]

    def mapper(dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(dataset_dict['file_name'], format='BGR')
        image, transforms = T.apply_transform_gens(default_aug, image)

        dataset_dict['image'] = noiser(torch.as_tensor(image.transpose(2, 0, 1).astype('float32')))
        annos = [utils.transform_instance_annotations(obj, transforms, image.shape[:2])
                 for obj in dataset_dict.pop("annotations")
                 if obj.get("iscrowd", 0) == 0]
        instances = utils.annotations_to_instances(annos, image.shape[:2])
        dataset_dict['instances'] = utils.filter_empty_instances(instances)
        return dataset_dict

    return mapper