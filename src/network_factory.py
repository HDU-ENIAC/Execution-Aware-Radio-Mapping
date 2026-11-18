from typing import Dict
import torch
from torch import nn

def build_model_from_cfg(cfg: Dict[str, object]):
    from model import UNet3DCoarseFine
    model = UNet3DCoarseFine(in_channels=int(cfg.get('in_channels', 1)), out_channels=int(cfg.get('out_channels', 1)), base_channels=int(cfg.get('base_channels', 16)), use_wavelet=bool(cfg.get('use_wavelet', True)), use_attn=bool(cfg.get('use_attn', False)), group_conv=bool(cfg.get('group_conv', False)), bottleneck_mode=cfg.get('bottleneck_mode', None), append_obs_mask=bool(cfg.get('append_obs_mask', False)), conv_block_type=cfg.get('conv_block_type', 'standard'), norm_type=cfg.get('norm_type', 'batch'), prior_fusion_mode=cfg.get('prior_fusion_mode', 'first'), prior_channels=int(cfg.get('prior_channels', 1)), route_context_mode=cfg.get('route_context_mode', 'none'), route_confidence_channel=int(cfg.get('route_confidence_channel', 3)), route_observation_channel=int(cfg.get('route_observation_channel', 4)))
    return model

def freeze_batchnorm_stats(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.eval()
