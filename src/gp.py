"""Latent-grid GP used by acquisition and reconstruction."""
import numpy as np
import torch
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.distance import cdist
SEED = 20261025

class GridBayesianGP:
    """Shrestha Sec.III-C latent-grid Bayesian filter, rank-one implementation.

    This implements the Gaussian conditioning equations, not the DRUE network.
    Grid resolution is explicit; no full-resolution covariance is allocated.
    """

    def __init__(self, grid, length=40.0, variance=0.05, noise=0.0001, mean=0.0):
        self.grid = np.asarray(grid, dtype=float)
        self.length, self.variance, self.noise = (float(length), float(variance), float(noise))
        self.offset = float(mean)
        kernel = self.kernel(self.grid, self.grid)
        self.factor = cho_factor(kernel + np.eye(len(grid)) * 1e-12, lower=True)
        self.mean = np.zeros(len(grid))
        self.cov = kernel
        self.n = 0
        self.coeff = None

    def kernel(self, x, y):
        return self.variance * np.exp(-cdist(np.atleast_2d(x), np.atleast_2d(y)) / self.length)

    def update(self, point, value):
        k = self.kernel(self.grid, point)[:, 0]
        a = cho_solve(self.factor, k, check_finite=False)
        conditional_var = self.variance - k @ a + self.noise
        if conditional_var < -1e-09:
            raise FloatingPointError('Negative observation conditional variance')
        cov_a = self.cov @ a
        innovation_var = max(conditional_var, 0) + a @ cov_a
        if innovation_var <= 0:
            raise FloatingPointError('Nonpositive Bayesian innovation variance')
        self.mean += cov_a * ((float(value) - self.offset - a @ self.mean) / innovation_var)
        self.cov -= np.outer(cov_a, cov_a) / innovation_var
        self.n += 1
        self.coeff = None

    def fit(self, points, values):
        for p, y in zip(points, values):
            self.update(p, y)
        return self

    def predict(self, points, variance=False):
        if variance:
            raise ValueError('Use stored diagonal for planner grid; full cross-grid variance is not requested')
        if self.coeff is None:
            self.coeff = cho_solve(self.factor, self.mean, check_finite=False)
        return self.offset + self.kernel(points, self.grid) @ self.coeff

class AnisotropicBayesian(GridBayesianGP):

    def __init__(self, grid, lengths=(40.0, 40.0, 40.0), **kwargs):
        self.lengths = np.asarray(lengths, dtype=float)
        if self.lengths.shape != (3,) or np.any(self.lengths <= 0):
            raise ValueError('Three positive coordinate lengths in z,y,x order are required')
        super().__init__(grid, length=1.0, **kwargs)

    def kernel(self, x, y):
        return self.variance * np.exp(-cdist(np.atleast_2d(x) / self.lengths, np.atleast_2d(y) / self.lengths))

class CachedPosterior:

    def __init__(self, path):
        with np.load(path) as z:
            self.grid, self.coeff = (z['grid'].copy(), z['coeff'].copy())
            self.lengths, self.posterior_diagonal = (z['lengths'].copy(), z['diagonal'].copy())
            self.offset, self.variance = (float(z['offset']), float(z['variance']))

    def predict(self, points, variance=False):
        if variance:
            raise ValueError('Cached state exposes mean and latent diagonal only')
        kernel = self.variance * np.exp(-cdist(np.atleast_2d(points) / self.lengths, self.grid / self.lengths))
        return self.offset + kernel @ self.coeff

def signal_from_readings(shape, building, points, values):
    mask = np.zeros(shape, bool)
    points = np.asarray(points, dtype=np.int64)
    assert len(np.unique(points, axis=0)) == len(points)
    assert (points >= 0).all() and (points < np.asarray(shape)).all()
    assert not building[tuple(points.T)].any()
    mask[tuple(points.T)] = True
    signal = np.zeros(shape, np.float32)
    signal[building] = -1.0
    signal[tuple(points.T)] = np.asarray(values, np.float32)
    return (signal, mask)

def batch_posterior(model, points, values):
    """Batch conditioning for the exact linear observation model in GridBayesianGP.update."""
    if model.n:
        raise ValueError('Batch conditioner requires an unconditioned prior')
    kernel = model.kernel(model.grid, points)
    h = cho_solve(model.factor, kernel, check_finite=False)
    noise = model.variance - np.sum(kernel * h, axis=0) + model.noise
    if noise.min() < -1e-09:
        raise FloatingPointError('Negative conditional observation variance')
    noise = np.maximum(noise, 0)
    cross = model.cov @ h
    system = h.T @ cross + np.diag(noise)
    factor = cho_factor((system + system.T) * 0.5, lower=True, check_finite=False)
    weights = cho_solve(factor, np.asarray(values) - model.offset, check_finite=False)
    mean = cross @ weights
    coeff = cho_solve(model.factor, mean, check_finite=False)
    inverse_cross = cho_solve(factor, cross.T, check_finite=False)
    diagonal = np.diag(model.cov) - np.sum(cross * inverse_cross.T, axis=1)
    if diagonal.min() < -1e-08:
        raise FloatingPointError('Negative batch posterior variance')
    return {'grid': model.grid, 'coeff': coeff, 'lengths': model.lengths, 'diagonal': diagonal, 'offset': model.offset, 'variance': model.variance}

def save_posterior(path, model, lengths):
    model.predict(model.grid[:1], variance=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, grid=model.grid, coeff=model.coeff, lengths=lengths, diagonal=np.diag(model.cov), offset=model.offset, variance=model.variance)

class GPMean:

    def __init__(self, model):
        model.predict(model.grid[:1], variance=False)
        self.grid = torch.tensor(model.grid / model.lengths, dtype=torch.float32, device='cuda')
        self.coeff = torch.tensor(model.coeff, dtype=torch.float32, device='cuda')
        self.lengths = torch.tensor(model.lengths, dtype=torch.float32, device='cuda')
        self.offset, self.variance = (model.offset, model.variance)
        self.double = False

    @torch.inference_mode()
    def __call__(self, q):
        q = torch.as_tensor(q, dtype=self.grid.dtype, device='cuda') / self.lengths
        d = torch.cdist(q, self.grid, compute_mode='donot_use_mm_for_euclid_dist')
        return (self.offset + self.variance * torch.exp(-d) @ self.coeff).cpu().numpy()

    def check(self, model, shape):
        q = np.random.default_rng(SEED).uniform(0, np.asarray(shape) - 1, (257, 3))
        cpu = model.predict(q, variance=False)
        gpu = self(q)
        if not np.allclose(cpu, gpu, rtol=2e-05, atol=2e-06):
            self.grid = torch.tensor(model.grid / model.lengths, dtype=torch.float64, device='cuda')
            self.coeff = torch.tensor(model.coeff, dtype=torch.float64, device='cuda')
            self.lengths = self.lengths.double()
            self.double = True
            gpu = self(q)
            np.testing.assert_allclose(cpu, gpu, rtol=1e-08, atol=1e-09)
        return {'max_absolute_error': float(np.max(np.abs(cpu - gpu))), 'float64_fallback': self.double, 'queries': len(q)}
