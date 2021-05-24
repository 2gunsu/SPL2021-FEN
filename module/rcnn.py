import torch
import torch.nn as nn

from collections import OrderedDict
from typing import Tuple, List, Dict
from detectron2.config import configurable, CfgNode
from detectron2.structures import ImageList, Instances
from detectron2.modeling.backbone import Backbone, build_backbone
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.modeling.roi_heads import build_roi_heads
from detectron2.modeling import META_ARCH_REGISTRY

from module import FENs
from data.preprocess import Normalizer


def extract_weight(pth_file: str, search_word: str, remove_word: List[str] = None):
    dict_ = torch.load(pth_file)['model']
    new_dict_ = OrderedDict()

    for k, v in dict_.items():
        if k.find(search_word) != -1:
            if remove_word is not None:
                for rw in remove_word:
                    k = k.replace(rw, '')
            new_dict_.update({k: v})
    return new_dict_


@META_ARCH_REGISTRY.register()
class RCNN(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        cfg: CfgNode,
        backbone: Backbone,
        denoiser: nn.Module,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
    ):

        super().__init__()
        self.cfg = cfg
        self.backbone = backbone                            # Backbone
        self.denoiser = denoiser                            # Denoiser (Enhancer)
        self.proposal_generator = proposal_generator        # RPN
        self.roi_heads = roi_heads                          # ROI

        self.max_iter = self.cfg.SOLVER.MAX_ITER

        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1))

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        return {
            "cfg": cfg.clone(),
            "backbone": backbone,
            "denoiser": FENs(cfg, init_freeze=True),
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs: List[Dict]):
        if not self.training:
            return self.inference(batched_inputs)

        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

        # Normalize input and extract features from backbone.
        input_images = torch.cat([x['image'].unsqueeze(0).to('cuda') for x in batched_inputs], dim=0)
        input_images = Normalizer(mean=self.pixel_mean, std=self.pixel_std)(input_images)
        input_list = ImageList.from_tensors([tensor for tensor in input_images])
        features = self.backbone(input_list.tensor)
        cloned_features = {k: v.clone().detach() for k, v in features.items()}

        # Enhance specific layers of features.
        features = self.denoiser(features)

        # Pass enhanced features into following modules. (Proposal Generator, ROI Heads)
        proposals, proposal_losses = self.proposal_generator(input_list, features, gt_instances)
        _, detector_losses = self.roi_heads(input_list, features, proposals, gt_instances)

        # Update Losses
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses, cloned_features

    def inference(self, batched_inputs: List[Dict], do_postprocess: bool = True):
        assert not self.training, \
            "Model is currently in train mode."

        # Normalize input and extract features from backbone.
        input_images = torch.cat([x['image'].unsqueeze(0).to('cuda') for x in batched_inputs], dim=0)
        input_images = Normalizer(mean=self.pixel_mean, std=self.pixel_std)(input_images)
        input_list = ImageList.from_tensors([tensor for tensor in input_images])
        features = self.backbone(input_list.tensor)

        # Enhance specific layers of features.
        features = self.denoiser(features)

        # Pass enhanced features into following modules. (Proposal Generator, ROI Heads)
        # Loss is not used.
        proposals, _ = self.proposal_generator(input_list, features, None)
        results, _ = self.roi_heads(input_list, features, proposals, None)

        # Size Calibration for predicted instances.
        if do_postprocess:
            return RCNN._postprocess(instances=results,
                                     batched_inputs=batched_inputs,
                                     image_sizes=input_list.image_sizes)
        else:
            return results

    @staticmethod
    def _postprocess(instances: Instances,
                     batched_inputs: List[Dict],
                     image_sizes: List[Tuple[int, int]]):
        processed_results = []
        for instance_per_img, single_img, image_size in zip(instances, batched_inputs, image_sizes):
            height = single_img.get("height", image_size[0])
            width = single_img.get("width", image_size[1])
            processed_results.append({"instances": detector_postprocess(instance_per_img, height, width)})
        return processed_results
