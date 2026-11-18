"""GP--neural residual reconstruction with the experiment checkpoint layout."""
from pathlib import Path
import json

import numpy as np
import torch
from torch import nn

from features import make_model_input_np
from network_factory import build_model_from_cfg, freeze_batchnorm_stats
from model import RouteHybridResidualHead as OGAC
from windowing import tile_windows

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / 'configs/default.json'


class GPResidualReconstructor(nn.Module):
    """The final fixed-scale arm; net/gate names preserve saved state keys."""
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = json.loads(DEFAULT_CONFIG.read_text()) if cfg is None else dict(cfg)
        if self.cfg['in_channels'] != 10:
            raise ValueError('The current reconstructor expects ten input channels.')
        self.net = build_model_from_cfg(self.cfg)
        self.gate = nn.Sequential(nn.Conv3d(6,8,1),nn.SiLU(),nn.Conv3d(8,1,1))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        self.gate.requires_grad_(False)
        nn.init.zeros_(self.net.final_fuse_conv.weight)
        nn.init.zeros_(self.net.final_fuse_conv.bias)
        for name, parameter in self.net.route_residual_head.named_parameters():
            if name.endswith('weight') or name.endswith('bias'):
                nn.init.zeros_(parameter)

    def forward(self,x):
        observed, building = x[:,4:5]>.5, x[:,5:6]>.5
        missing = ~(observed | building)
        output = self.net(x,mask=missing.float())
        base = (torch.softmax(self.net.prior_gate(x),dim=1)*x[:,:2]).sum(1,keepdim=True)
        gate_input = x[:,[8,9,3,5,6,7]]
        output = x[:,8:9]+torch.sigmoid(self.gate(gate_input))*(output-base)
        return torch.where(building,-torch.ones_like(output),torch.where(observed,x[:,:1],output))

    @classmethod
    def from_checkpoint(cls,path,device='cpu',cfg=None):
        # Only load checkpoints obtained from a trusted source.
        checkpoint = torch.load(path,map_location='cpu',weights_only=True)
        if float(checkpoint.get('fixed_gate',.5)) != .5:
            raise ValueError('Checkpoint does not use the final fixed 0.5 residual scale.')
        model = cls(cfg)
        model.load_state_dict(checkpoint['model'],strict=True)
        if torch.count_nonzero(model.gate[-1].weight) or torch.count_nonzero(model.gate[-1].bias):
            raise ValueError('Expected a constant, untrained residual gate.')
        return model.to(device).eval()


def reconstruction_loss(prediction,target,inputs):
    missing = (inputs[:,4:5]<.5) & (inputs[:,5:6]<.5)
    if not missing.any():
        raise ValueError('No unobserved free-space voxels in this batch.')
    difference = (prediction-target)[missing]
    return difference.abs().mean()+.5*difference.square().mean()


def prepare_inputs(signal,observed,building,gp_mean,variance_proxy):
    features = make_model_input_np(signal,observed.astype(np.float32),False,
        building_mask=building,feature_mode='partial_prior_conf',prior_tau=16.)
    mean = np.asarray(gp_mean,np.float32).copy()
    mean[building] = -1.
    mean[observed] = signal[observed]
    return np.concatenate((features,mean[None],np.asarray(variance_proxy,np.float32)[None]),axis=0)


@torch.inference_mode()
def reconstruct_volume(model,inputs,output_path,device='cpu'):
    if inputs.shape != (10,32,512,512):
        raise ValueError('Published tiled inference expects a (10,32,512,512) volume.')
    result = np.lib.format.open_memmap(output_path,mode='w+',dtype=np.float32,shape=(32,512,512))
    model.eval()
    for r0,r1,row,lr0,lr1 in tile_windows(512):
        for c0,c1,col,lc0,lc1 in tile_windows(512):
            x = torch.from_numpy(np.array(inputs[:,:,row:row+96,col:col+96],copy=True)[None]).to(device)
            with torch.autocast(device_type=torch.device(device).type,enabled=torch.device(device).type=='cuda'):
                p = model(x).float().cpu().numpy()[0,0]
            result[:,r0:r1,c0:c1] = p[:,lr0:lr1,lc0:lc1]
    result.flush()
    return result
