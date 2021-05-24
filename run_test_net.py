import os
import yaml
import warnings

from yacs.config import CfgNode
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


if __name__ == "__main__":

    # Parameters
    checkpoint_path: str = ""
    test_data_root: str = ""

    test_input_size: Union[Tuple[int, int], int] = 800
    test_noise_type: str = ''
    test_noise_params: Any = []

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
