"""Observation/geometry input channels from the experiment implementation."""
from typing import Optional, Tuple
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

def geometry_feature_stack(building_mask: np.ndarray, tau: float=3.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    b = building_mask.astype(bool)
    if ndi is not None:
        dist = ndi.distance_transform_edt(~b).astype(np.float32)
        edge = np.exp(-dist / max(1e-06, float(tau))).astype(np.float32)
    else:
        near = np.zeros_like(b, dtype=bool)
        near[:, 1:, :] |= b[:, :-1, :]
        near[:, :-1, :] |= b[:, 1:, :]
        near[:, :, 1:] |= b[:, :, :-1]
        near[:, :, :-1] |= b[:, :, 1:]
        edge = near.astype(np.float32)
    if b.shape[0] <= 1:
        z = np.zeros_like(edge, dtype=np.float32)
    else:
        z_vals = np.linspace(-1.0, 1.0, b.shape[0], dtype=np.float32)
        z = np.broadcast_to(z_vals[:, None, None], b.shape).astype(np.float32, copy=False)
    return (b.astype(np.float32), edge.astype(np.float32), z.astype(np.float32))

def make_model_input_np(signal_np: np.ndarray, obs_np: np.ndarray, append_obs_mask: bool, building_mask: Optional[np.ndarray]=None, feature_mode: str='', prior_tau: float=3.0) -> np.ndarray:
    mode = str(feature_mode or ('signal_obs' if append_obs_mask else 'signal')).lower()
    signal = signal_np.astype(np.float32, copy=False)
    obs = obs_np.astype(np.float32, copy=False)
    if mode in {'', 'signal', 'legacy'}:
        return signal
    if mode in {'obs', 'signal_obs'}:
        return np.stack([signal, obs], axis=0)
    if building_mask is None:
        raise ValueError(f'input_feature_mode={mode} requires building_mask')
    b, edge, z = geometry_feature_stack(building_mask)
    if mode == 'geom':
        return np.stack([signal, obs, b, edge, z], axis=0)
    if mode in {'dual_prior', 'nearest_rbf_geom'}:
        rbf = apply_prior_fill(signal, building_mask, obs_mask=obs.astype(bool), mode='rbf', tau=prior_tau)
        return np.stack([signal, rbf, signal - rbf, obs, b, edge, z], axis=0)
    if mode in {'multi_prior_conf', 'nearest_rbf_conf'}:
        if ndi is None:
            raise RuntimeError(f'input_feature_mode={mode} requires scipy.ndimage')
        obs_bool = obs.astype(bool)
        dist, inds = ndi.distance_transform_edt(~obs_bool, return_indices=True)
        dist = dist.astype(np.float32)
        nearest = signal[tuple(inds)].astype(np.float32, copy=True)
        nearest[building_mask.astype(bool)] = signal[building_mask.astype(bool)]
        confidence = np.exp(-dist / max(1e-06, float(prior_tau))).astype(np.float32)
        rbf = apply_prior_fill(signal, building_mask, obs_mask=obs_bool, mode='rbf', tau=prior_tau)
        return np.stack([nearest, rbf, nearest - rbf, confidence, obs, b, edge, z], axis=0)
    if mode in {'partial_prior_conf', 'nearest_decay_conf'}:
        if ndi is None:
            raise RuntimeError(f'input_feature_mode={mode} requires scipy.ndimage')
        dist, inds = ndi.distance_transform_edt(~obs.astype(bool), return_indices=True)
        dist = dist.astype(np.float32)
        nearest = signal[tuple(inds)].astype(np.float32, copy=True)
        nearest[building_mask.astype(bool)] = signal[building_mask.astype(bool)]
        confidence = np.exp(-dist / max(1e-06, float(prior_tau))).astype(np.float32)
        partial = nearest * confidence
        return np.stack([nearest, partial, nearest - partial, confidence, obs, b, edge, z], axis=0)
    raise ValueError(f'unknown input_feature_mode: {feature_mode}')

def observation_mask_from_input(x: np.ndarray, building_mask: np.ndarray) -> np.ndarray:
    return ((x != 0) & ~building_mask.astype(bool)).astype(bool)

def apply_prior_fill(x: np.ndarray, building_mask: np.ndarray, obs_mask: Optional[np.ndarray]=None, mode: str='none', tau: float=8.0) -> np.ndarray:
    mode = str(mode).lower()
    if mode in {'', 'none', 'raw'}:
        return x.astype(np.float32, copy=True)
    if obs_mask is None:
        obs_mask = observation_mask_from_input(x, building_mask)
    valid = ~building_mask.astype(bool)
    if int(obs_mask.sum()) <= 0:
        return x.astype(np.float32, copy=True)
    if mode in {'idw', 'knn_idw', 'kernel', 'gaussian_kernel', 'local_gp'}:
        if cKDTree is None:
            raise RuntimeError(f'prior_fill mode={mode!r} requires scipy.spatial.cKDTree')
        obs_pts = np.argwhere(obs_mask & valid)
        if len(obs_pts) == 0:
            return x.astype(np.float32, copy=True)
        fill_mask = valid & ~obs_mask
        fill_pts = np.argwhere(fill_mask)
        out = x.astype(np.float32, copy=True)
        if len(fill_pts) == 0:
            return out
        k = max(1, min(int(round(float(tau))), len(obs_pts)))
        power = 2.0
        tree = cKDTree(obs_pts.astype(np.float32, copy=False))
        obs_vals = x[obs_mask & valid].astype(np.float32, copy=False)
        chunk = 200000
        for start in range(0, len(fill_pts), chunk):
            pts = fill_pts[start:start + chunk].astype(np.float32, copy=False)
            dist, ind = tree.query(pts, k=k, workers=-1)
            dist = np.asarray(dist, dtype=np.float32)
            ind = np.asarray(ind)
            if k == 1:
                vals = obs_vals[ind]
            else:
                zero = dist <= 1e-06
                if mode in {'kernel', 'gaussian_kernel', 'local_gp'}:
                    sigma = max(1e-06, float(tau))
                    weights = np.exp(-0.5 * (dist / sigma) ** 2)
                else:
                    weights = 1.0 / np.maximum(dist, 1e-06) ** power
                neigh_vals = obs_vals[ind]
                vals = np.sum(weights * neigh_vals, axis=1) / np.maximum(np.sum(weights, axis=1), 1e-12)
                if np.any(zero):
                    rows = np.where(zero.any(axis=1))[0]
                    cols = zero[rows].argmax(axis=1)
                    vals[rows] = neigh_vals[rows, cols]
            out[tuple(fill_pts[start:start + chunk].T)] = vals.astype(np.float32, copy=False)
        return out.astype(np.float32, copy=False)
    if ndi is None:
        raise RuntimeError('prior_fill requires scipy.ndimage')
    dist, inds = ndi.distance_transform_edt(~obs_mask, return_indices=True)
    nearest = x[tuple(inds)].astype(np.float32, copy=False)
    out = x.astype(np.float32, copy=True)
    fill = valid & ~obs_mask
    if mode == 'nearest':
        out[fill] = nearest[fill]
    elif mode in {'decay', 'nearest_decay'}:
        weight = np.exp(-dist.astype(np.float32) / max(1e-06, float(tau)))
        out[fill] = nearest[fill] * weight[fill]
    elif mode in {'rbf', 'rbf_nearest'}:
        sigma = max(1e-06, float(tau))
        obs_float = obs_mask.astype(np.float32, copy=False)
        numerator = ndi.gaussian_filter(x.astype(np.float32, copy=False) * obs_float, sigma=sigma, mode='nearest')
        denominator = ndi.gaussian_filter(obs_float, sigma=sigma, mode='nearest')
        smooth = np.divide(numerator, np.maximum(denominator, 1e-06), out=np.zeros_like(numerator, dtype=np.float32), where=denominator > 1e-06)
        supported = denominator > 1e-05
        out[fill & supported] = smooth[fill & supported]
        out[fill & ~supported] = nearest[fill & ~supported]
    else:
        raise ValueError(f'unknown prior_fill mode: {mode}')
    return out.astype(np.float32, copy=False)
