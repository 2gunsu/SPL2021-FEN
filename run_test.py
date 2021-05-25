import os
import yaml
import warnings

from argparse import ArgumentParser
from typing import Any, Union, Tuple

from detectron2.config import get_cfg, CfgNode
from detectron2.data import build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.checkpoint import DetectionCheckpointer

from module.rcnn import RCNN
from engine.trainer import FENTrainer
from data.mapper import generate_mapper
warnings.filterwarnings('ignore')


parser = ArgumentParser(description="Evaulation for FEN")

parser.add_argument('--data_root', type=str, help="Directory of test dataset")
parser.add_argument('--ckpt_root', type=str, help="Directory of checkpoint")
parser.add_argument('--noise_type', type=str, default='none', help="What kind of noise to be added.", choices=['none', 'gaussian', 'snp'])
parser.add_argument('--noise_params', nargs="+", default=[], help="Parameters for controlling the noise.")
parser.add_argument('--input_size', type=int, default=800, help="Determinte the size of the image to be used for evaluation.")


if __name__ == "__main__":

    args = parser.parse_args()

    # Parameters
    checkpoint_path: str = args.ckpt_root
    test_data_root: str = args.data_root

    test_input_size: Union[Tuple[int, int], int] = args.input_size
    test_noise_type: str = args.noise_type
    test_noise_params: Any = args.noise_params
    if isinstance(test_noise_params, list) and len(test_noise_params) != 0:
        if test_noise_type == "gaussian":
            noise_params = [int(p) for p in test_noise_params]
        elif test_noise_type == "snp":
            noise_params = float(test_noise_params[0])

    # Register test dataset.
    register_coco_instances(name='test',
                            metadata={},
                            json_file=os.path.join(test_data_root, 'Label.json'),
                            image_root=os.path.join(test_data_root, 'Image'))

    # Load configuration from checkpoint.
    with open(os.path.join(checkpoint_path, 'config.yaml'), 'r') as f:
        data = yaml.safe_load(f)
    cfg = CfgNode(data)
    cfg.DATASETS.TRAIN, cfg.DATASETS.TEST = ('test',), ('test',)

    # Build trained model, evaluator, data loader for test.
    trainer = FENTrainer(cfg)
    mapper = generate_mapper(test_input_size, test_noise_type, test_noise_params)
    DetectionCheckpointer(trainer.model).load(cfg.MODEL.WEIGHTS)
    print(f'Weight "{cfg.MODEL.WEIGHTS}" is loaded for evaluation.')

    evaluator = COCOEvaluator(dataset_name='test',
                              tasks=('bbox', ),
                              distributed=False,
                              output_dir=os.path.join(cfg.OUTPUT_DIR, 'evaluation'))

    test_loader = build_detection_test_loader(cfg,
                                              dataset_name='test',
                                              mapper=mapper)

    # Evaluate the trained model.
    result = inference_on_dataset(model=trainer.model,
                                  data_loader=test_loader,
                                  evaluator=evaluator)
