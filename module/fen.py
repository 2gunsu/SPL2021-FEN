import copy
import torch
import torch.nn as nn
import numpy as np
from typing import Dict
from yacs.config import CfgNode

from module.unet import UNet
from data.preprocess import RandomCrop, Resize


class FENs(nn.Module):
    def __init__(self,
                 cfg: CfgNode,
                 init_freeze: bool = False):

        super(FENs, self).__init__()
        self.module_list = self._build_modules(cfg, all_identity=(not cfg.MODEL.FEN.USE_FEN))
        if init_freeze:
            self.freeze()

    def forward(self, cloned_features: Dict[str, torch.Tensor]):
        forward_dict = {}
        for k, m in zip(cloned_features.keys(), self.module_list):
            if isinstance(m, nn.Identity):
                forward_dict.update({k: m(cloned_features[k])})
            else:
                forward_dict.update({k: m(cloned_features, level=k)})
        return forward_dict

    def get_loss(self, cloned_features: Dict[str, torch.Tensor]):
        losses = []

        for key, module in zip(cloned_features.keys(), self.module_list):
            if isinstance(module, nn.Identity):
                pass

            else:
                losses.append(module.get_loss(cloned_features, key))
        return sum(losses)

    def _build_modules(self,
                       cfg: CfgNode,
                       all_identity: bool = False):

        full_levels = cfg.MODEL.RPN.IN_FEATURES     # ['p2', 'p3', 'p4', 'p5', 'p6']
        fen_levels = cfg.MODEL.FEN.LEVELS

        if isinstance(fen_levels, str) and (fen_levels in full_levels):
            fen_levels = [fen_levels, ]
        assert all([(level in full_levels) for level in fen_levels]), \
            f"'cfg.MODEL.FEN.LEVELS' must be subset of {full_levels}."

        modules = []
        for level in full_levels:
            if not all_identity:
                modules.append(FEN(cfg) if level in fen_levels else nn.Identity())
            else:
                modules.append(nn.Identity())

        return nn.ModuleList(modules)

    def freeze(self):
        for m in self.module_list:
            if not isinstance(m, nn.Identity):
                m.freeze()

    def unfreeze(self):
        for m in self.module_list:
            if not isinstance(m, nn.Identity):
                m.unfreeze()


class FEN(nn.Module):

    """
    Class of FEN (Feature Enhancement Network)

    * Args:
        cfg (CfgNode):
            Configuration
    """

    def __init__(self, cfg: CfgNode):

        super().__init__()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cfg = cfg
        self.model = UNet(in_channels=cfg.MODEL.FPN.OUT_CHANNELS).to(device)

    def forward(self, cloned_features: Dict[str, torch.Tensor], level: str):
        return self.model(cloned_features[level])

    def get_loss(self, cloned_features: Dict[str, torch.Tensor], level: str):
        input_patch, mask_patch, label_patch = self.extract_patches(
            features=cloned_features[level],
            patch_size=self.cfg.MODEL.FEN.PATCH_SIZE,
            patch_per_img=self.cfg.MODEL.FEN.PATCH_PER_IMG,
            erase_ratio=self.cfg.MODEL.FEN.ERASE_RATIO,
            soften_ratio=self.cfg.MODEL.FEN.SOFTEN_RATIO)

        output = self.model(input_patch)
        loss = nn.L1Loss()(output * (1 - mask_patch), label_patch * (1 - mask_patch))
        return loss

    def extract_patches(self,
                        features: torch.Tensor,
                        patch_size: int = 32,
                        min_patch_size: int = 20,
                        patch_per_img: int = 4,
                        erase_ratio: float = 0.50,
                        soften_ratio: float = 0.60,
                        device: str = 'cuda'):

        output_list, mask_list, label_list = [], [], []
        SIZE_WIN = (5, 5)

        arr_features = features.permute(0, 2, 3, 1).detach().cpu().numpy()
        for arr_feature in arr_features:
            for p_idx in range(patch_per_img):

                cropped_arr = None
                changed_size = None

                if patch_size < arr_feature.shape[0]:
                    cropped_arr = RandomCrop(patch_size)(arr_feature)
                    changed_size = patch_size

                elif patch_size == arr_feature.shape[0]:
                    cropped_arr = RandomCrop(min_patch_size)(arr_feature)
                    changed_size = min_patch_size

                elif patch_size > arr_feature.shape[0]:
                    arr_feature = Resize(patch_size)(arr_feature)
                    cropped_arr = RandomCrop(min_patch_size)(arr_feature)
                    changed_size = min_patch_size

                ch = arr_feature.shape[-1]
                num_spots = int((changed_size ** 2) * erase_ratio)
                mask = np.ones((changed_size, changed_size, ch)) * soften_ratio
                output = copy.deepcopy(cropped_arr)

                idy_msk = np.random.randint(0, changed_size, num_spots)
                idx_msk = np.random.randint(0, changed_size, num_spots)

                idy_neigh = np.random.randint(-SIZE_WIN[0] // 2 + SIZE_WIN[0] % 2,
                                              SIZE_WIN[0] // 2 + SIZE_WIN[0] % 2,
                                              num_spots)
                idx_neigh = np.random.randint(-SIZE_WIN[1] // 2 + SIZE_WIN[1] % 2,
                                              SIZE_WIN[1] // 2 + SIZE_WIN[1] % 2,
                                              num_spots)

                idy_msk_neigh = idy_msk + idy_neigh
                idx_msk_neigh = idx_msk + idx_neigh

                idy_msk_neigh = idy_msk_neigh + (idy_msk_neigh < 0) * changed_size - (
                        idy_msk_neigh >= changed_size) * changed_size
                idx_msk_neigh = idx_msk_neigh + (idx_msk_neigh < 0) * changed_size - (
                        idx_msk_neigh >= changed_size) * changed_size

                id_msk = (idy_msk, idx_msk)
                id_msk_neigh = (idy_msk_neigh, idx_msk_neigh)

                output[id_msk] = cropped_arr[id_msk_neigh]
                mask[id_msk] = soften_ratio

                output_list.append(torch.from_numpy(output).permute(2, 0, 1).unsqueeze(dim=0))
                mask_list.append(torch.from_numpy(mask).permute(2, 0, 1).unsqueeze(dim=0))
                label_list.append(torch.from_numpy(cropped_arr).permute(2, 0, 1).unsqueeze(dim=0))

        output = torch.cat(output_list, dim=0).to(device)
        mask = torch.cat(mask_list, dim=0).to(device)
        label = torch.cat(label_list, dim=0).to(device)

        return output, mask, label

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad = True