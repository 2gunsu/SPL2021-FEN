import torch.nn as nn


def transfer_weight(src_model: nn.Module,
                    dst_model: nn.Module):
    src_dict = src_model.state_dict()
    dst_dict = dst_model.state_dict()
    assert len(src_dict.keys()) == len(dst_dict.keys()), \
        "`src_model` and `dst_model` seems different."

    for src_key, dst_key in zip(src_dict.keys(), dst_dict.keys()):
        dst_dict[dst_key] = src_dict[src_key]
    dst_model.load_state_dict(dst_dict)


def freeze(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False


def unfreeze(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = True