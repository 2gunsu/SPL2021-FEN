import os
import warnings
from detectron2.config import get_cfg
from typing import Any, List, Union, Tuple

from module import RCNN
from engine.trainer import FENTrainer
from utils.config import save_config, load_cfg_arch, set_cfg_params, add_fen_params
warnings.filterwarnings('ignore')


if __name__ == '__main__':

    # Parameters

    #
    # 'backbone_arch':
    #           Select one in ['R50-FPN', 'R101-FPN', 'X101-FPN'].
    #
    # 'data_root':
    #           See the 'Datasets and Preparation' section in the following link for the data structure.
    #           https://github.com/2gunsu/SPL2021-FEN
    #
    # 'output_dir':
    #           Output directory where training results are saved to.
    #
    # 'input_size':
    #           Determine the size of the image to be used for training.
    #
    # 'noise_type':
    #           Decide what kind of noise to add to the data batch.
    #           Select one in ['none', 'gaussian', 'snp'].
    #           ('snp' represents 'Salt & Pepper')
    #
    # 'noise_params':
    #           This parameter controls noise, and the input type is different depending on the type of noise.
    #             - When 'noise_type' is 'none',
    #                 This argument will be ignored.
    #
    #             - When 'noise_type' is 'gaussian',
    #                 Please enter the standard deviation of Gaussian noise.
    #                 The input type must be Union[int, List[int]].
    #                   [EX] 50, [50], [25, 50] are all possible.
    #
    #             - When 'noise_type' is 'snp',
    #                 Please enter the amount of salt & pepper noise.
    #                 The input type must be float which range in (0.0, 1.0).
    #                 It is recommended to use a value of 0.15 or less.
    #

    backbone_arch: str = 'X101-FPN'
    data_root: str = ""
    output_dir: str = ""

    input_size: Union[Tuple[int, int], int] = 800
    noise_type: str = ''
    noise_params: Any = []

    num_epoch = 10
    num_class = 1
    num_workers = 8
    batch_size = 4
    base_lr = 2.0E-03

    log_period: int = 1
    val_period: int = 5
    checkpoint_period: int = 100

    #
    # 'levels':
    #           The levels of the feature map to be enhanced.
    #           Available feature levels are ['p2', 'p3', 'p4', 'p5', 'p6'].
    #           Several combinations can be made using these elements.
    #               [EX] ['p2', 'p5'] or ['p4', 'p5'] or ['p5']
    #           Please note that using up all levels will require large V-RAM.
    #

    use_fen: bool = False
    levels: List[str] = []
    patch_per_img: int = 4
    patch_size: int = 32
    min_patch_size: int = 20
    erase_ratio: float = 0.5
    soften_ratio: float = 0.6

    # Set Configuration
    cfg = get_cfg()
    cfg = load_cfg_arch(cfg, arch_name=backbone_arch)
    cfg = set_cfg_params(
        cfg, num_epoch, num_class, base_lr, num_workers, batch_size,
        data_root, input_size, noise_type, noise_params, log_period,
        checkpoint_period, val_period, output_dir
    )
    cfg = add_fen_params(cfg, levels, patch_per_img, patch_size, min_patch_size, erase_ratio, soften_ratio)
    cfg.MODEL.FEN.USE_FEN = use_fen

    # Start Training
    trainer = FENTrainer(cfg=cfg)
    trainer.resume_or_load(False)
    trainer.train()

    # Save configuration file to 'cfg.OUTPUT_DIR'.
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, 'model_final.pth')
    save_config(cfg, save_path=os.path.join(cfg.OUTPUT_DIR, 'config.yaml'))
