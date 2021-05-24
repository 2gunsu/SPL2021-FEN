import os
import yaml
import numpy as np

from yacs.config import CfgNode
from typing import Any, List, Union, Tuple

from detectron2.config import get_cfg
from detectron2.model_zoo import model_zoo
from detectron2.data import DatasetCatalog
from detectron2.data.datasets import register_coco_instances


__all__ = ['save_config',
           'set_cfg_params',
           'load_cfg_arch',
           'add_fen_params',
           'register_datasets',
           ]


def save_config(cfg: CfgNode, save_path: str):
    with open(save_path, "w") as f:
        yaml.dump(yaml.safe_load(cfg.dump()), f)
    print(f'Config file is saved to "{save_path}".')


def set_cfg_params(cfg: CfgNode,

                   num_epoch: int,
                   class_num: int,
                   lr: float,
                   num_worker: int,
                   batch_size: int,
                   data_root: str,

                   image_size: Union[Tuple[int, int], int],
                   noise_type: str,
                   noise_params: Any,

                   print_period: int = None,
                   checkpoint_period: int = None,
                   validation_period: int = None,
                   output_dir: str = '.'):

    # Checkpoint Directory
    if output_dir is not None:
        cfg.OUTPUT_DIR = os.path.join('output', output_dir)

    # Fixed Params
    cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 2000
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 2000
    cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 2000
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 2000
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = \
        (np.array(cfg.MODEL.ANCHOR_GENERATOR.SIZES) * 1.5625).astype('int').tolist()

    # Number of classes
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = class_num
    cfg.MODEL.META_ARCHITECTURE = 'RCNN'

    # Data Parameters
    register_datasets(data_root=data_root)
    cfg.DATASETS.RESIZE = image_size
    cfg.DATASETS.NOISE_TYPE = noise_type
    cfg.DATASETS.NOISE_PARAMS = noise_params
    cfg.DATASETS.TRAIN = ('train',)
    cfg.DATASETS.TEST = ('val',)

    # Hyper Parameters
    cfg.SOLVER.BASE_LR = lr
    cfg.DATALOADER.NUM_WORKERS = num_worker
    cfg.SOLVER.IMS_PER_BATCH = batch_size
    cfg.SOLVER.MAX_ITER = int(len(DatasetCatalog.get('train')) / batch_size) * num_epoch

    # Period Parameters
    cfg.TEST.EVAL_PERIOD = int(len(DatasetCatalog.get('train')) / batch_size) \
        if validation_period is None else validation_period
    cfg.SOLVER.LOG_PERIOD = 20 if print_period is None else print_period
    cfg.SOLVER.CHECKPOINT_PERIOD = cfg.TEST.EVAL_PERIOD \
        if checkpoint_period is None else checkpoint_period

    return cfg


def add_fen_params(cfg: CfgNode,
                   levels: List[str] = None,
                   patch_per_img: int = 4,
                   patch_size: int = 32,
                   min_patch_size: int = 20,
                   erase_ratio: float = 0.5,
                   soften_ratio: float = 0.6):

    cfg.MODEL.FEN = CfgNode()
    cfg.MODEL.FEN.LEVELS = levels
    cfg.MODEL.FEN.PATCH_PER_IMG = patch_per_img
    cfg.MODEL.FEN.PATCH_SIZE = patch_size
    cfg.MODEL.FEN.MIN_PATCH_SIZE = min_patch_size
    cfg.MODEL.FEN.ERASE_RATIO = erase_ratio
    cfg.MODEL.FEN.SOFTEN_RATIO = soften_ratio

    return cfg


def load_cfg_arch(cfg: CfgNode, arch_name: str):

    _PATH_PREFIX = "COCO-Detection/faster_rcnn_"
    _ARCH_DICT = {'R50-C4': 'R_50_C4_3x.yaml',
                  'R50-DC5': 'R_50_DC5_3x.yaml',
                  'R50-FPN': 'R_50_FPN_3x.yaml',
                  'R101-C4': 'R_101_C4_3x.yaml',
                  'R101-DC5': 'R_101_DC5_3x.yaml',
                  'R101-FPN': 'R_101_FPN_3x.yaml',
                  'X101-FPN': 'X_101_32x8d_FPN_3x.yaml'}

    if arch_name in _ARCH_DICT.keys():
        cfg.merge_from_file(model_zoo.get_config_file(config_path=_PATH_PREFIX + _ARCH_DICT[arch_name]))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_path=_PATH_PREFIX + _ARCH_DICT[arch_name])
        print(f'Config and Weights for "{arch_name}" is loaded.')

    else:
        print('Cannot find "arch_name" you entered.')
        print(f'"arch_name" must be one in {list(_ARCH_DICT.keys())}.')

    return cfg


def register_datasets(data_root: str):
    assert os.path.isdir(data_root), "Cannot find 'data_root' you entered."

    # Register Train Data
    train_dir = os.path.join(data_root, 'Train')
    if os.path.isdir(train_dir):
        register_coco_instances(name='train',
                                metadata={},
                                json_file=os.path.join(train_dir, 'Label.json'),
                                image_root=os.path.join(train_dir, 'Image'))

    # Register Test Data
    test_dir = os.path.join(data_root, 'Test')
    if os.path.isdir(test_dir):
        register_coco_instances(name='test',
                                metadata={},
                                json_file=os.path.join(test_dir, 'Label.json'),
                                image_root=os.path.join(test_dir, 'Image'))

    # Register Validation Data
    val_dir = os.path.join(data_root, 'Val')
    if os.path.isdir(val_dir):
        register_coco_instances(name='val',
                                metadata={},
                                json_file=os.path.join(val_dir, 'Label.json'),
                                image_root=os.path.join(val_dir, 'Image'))