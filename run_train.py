import os
import warnings
from argparse import ArgumentParser
from detectron2.config import get_cfg
from typing import Any, List, Union, Tuple

from module import RCNN
from engine.trainer import FENTrainer
from utils.config import save_config, load_cfg_arch, set_cfg_params, add_fen_params
warnings.filterwarnings('ignore')


parser = ArgumentParser(description="Training for FEN")

parser.add_argument('--data_root', type=str, help="Directory of dataset", required=True)
parser.add_argument('--noise_type', type=str, default='none', help="What kind of noise to be added.", choices=['none', 'gaussian', 'snp'])
parser.add_argument('--noise_params', nargs="+", default=[], help="Parameters for controlling the noise.")
parser.add_argument('--output_dir', type=str, help="Output directory where training results are saved to.", required=True)

# Whether to use FEN.
parser.add_argument('use_fen', action='store_true')

parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs.")
parser.add_argument('--batch', type=int, default=4, help="Batch Size")
parser.add_argument('--num_workers', type=int, default=8, help="Number of workers")
parser.add_argument('--num_classes', type=int, default=1, help="Number of classes in dataset.")

parser.add_argument('--arch', type=str, default='X101-FPN', help="Architecture of backbone network.",
                    choices=['R50-FPN', 'R101-FPN', 'X101-FPN'])
parser.add_argument('--input_size', type=int, default=800, help="Determinte the size of the image to be used for training.")
parser.add_argument('--fen_levels', nargs="+", default=['p4', 'p5'], help="The levels of the feature map to be enhanced.",
                    choices=['p2', 'p3', 'p4', 'p5', 'p6'])


if __name__ == '__main__':

    args = parser.parse_args()

    backbone_arch: str = args.arch
    data_root: str = args.data_root
    output_dir: str = args.output_dir

    input_size: Union[Tuple[int, int], int] = args.input_size
    noise_type: str = args.noise_type
    noise_params: Any = args.noise_params
    if isinstance(noise_params, list) and len(noise_params) != 0:
        if noise_type == "gaussian":
            noise_params = [int(p) for p in noise_params]
        elif noise_type == "snp":
            noise_params = float(noise_params[0])

    num_epoch = args.epochs
    num_class = args.num_classes
    num_workers = args.num_workers
    batch_size = args.batch
    base_lr = 2.0E-03

    use_fen: bool = args.use_fen
    levels: List[str] = args.fen_levels
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
        data_root, input_size, noise_type, noise_params,
        None, None, None, output_dir
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
