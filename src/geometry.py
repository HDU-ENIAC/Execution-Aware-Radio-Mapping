"""Original free-space navigation and corner-clearance checks."""
from functools import lru_cache
from itertools import product
import numpy as np
from scipy.sparse import coo_matrix
SUBSETS = np.asarray(list(product((0, 1), repeat=3))[1:], dtype=np.int32)

def validate_grid_trace(trace, valid):
    p = np.asarray(trace, dtype=np.int32)
    if p.ndim != 2 or p.shape[1] != 3 or (not len(p)):
        raise ValueError('Invalid trace shape')
    if np.any(p < 0) or np.any(p >= valid.shape) or (not valid[tuple(p.T)].all()):
        raise ValueError('Trace outside execution free space')
    delta = np.diff(p, axis=0)
    if len(delta) and (not np.all(np.max(np.abs(delta), axis=1) == 1)):
        raise ValueError('Trace is not unit 26-neighbor continuous')
    diagonal = np.count_nonzero(delta, axis=1) > 1
    for mask in SUBSETS:
        touched = p[:-1][diagonal] + delta[diagonal] * mask
        if not valid[tuple(touched.T)].all():
            raise ValueError('Trace cuts an occupied voxel corner')

def raster_segment(first, last):
    a, b = (np.asarray(first, dtype=np.int32), np.asarray(last, dtype=np.int32))
    reverse = tuple(a) > tuple(b)
    if reverse:
        a, b = (b, a)
    steps = int(np.max(np.abs(b - a)))
    if not steps:
        return a[None, :]
    points = np.rint(a + (b - a) * np.arange(steps + 1)[:, None] / steps).astype(np.int32)
    return points[::-1] if reverse else points

def cadence_indices(points, spacing=10.0):
    p = np.asarray(points)
    length = np.r_[0.0, np.linalg.norm(np.diff(p, axis=0), axis=1).cumsum()]
    m = max(1, int(length[-1] // spacing))
    requested = np.linspace(0.0, length[-1], m + 1)
    hi = np.minimum(np.searchsorted(length, requested), len(p) - 1)
    lo = np.maximum(hi - 1, 0)
    ids = np.where(requested - length[lo] <= length[hi] - requested, lo, hi)
    return (np.unique(ids), m)

class DiagonalGeometryGraph:

    def __init__(self, valid, required, stride=(4, 8, 8)):
        self.valid = valid
        self.axes = [np.unique(np.r_[np.arange(0, n, s), n - 1, [p[a] for p in required]]) for a, (n, s) in enumerate(zip(valid.shape, stride))]
        self.shape = tuple(map(len, self.axes))
        self.coords = np.stack(np.meshgrid(*self.axes, indexing='ij'), axis=-1).reshape(-1, 3).astype(np.int32)
        self.free = valid[tuple(self.coords.T)]
        grid = np.arange(len(self.coords)).reshape(self.shape)
        sources, dests, distances = ([], [], [])
        for offset in product((-1, 0, 1), repeat=3):
            nonzero = [v for v in offset if v]
            if not nonzero or nonzero[0] < 0:
                continue
            lo = tuple((slice(0, n - d) if d > 0 else slice(-d, n) if d < 0 else slice(None) for n, d in zip(self.shape, offset)))
            hi = tuple((slice(d, n) if d > 0 else slice(0, n + d) if d < 0 else slice(None) for n, d in zip(self.shape, offset)))
            start, end = (grid[lo].ravel(), grid[hi].ravel())
            endpoints_free = self.free[start] & self.free[end]
            start, end = (start[endpoints_free], end[endpoints_free])
            if not len(start):
                continue
            a, b = (self.coords[start], self.coords[end])
            steps = np.max(np.abs(b - a), axis=1)
            clear, length, previous = (np.ones(len(a), bool), np.zeros(len(a)), a)
            for t in range(1, int(steps.max()) + 1):
                current = np.rint(a + (b - a) * (np.minimum(t, steps) / steps)[:, None]).astype(np.int32)
                delta = current - previous
                for mask in SUBSETS:
                    corner = previous + delta * mask
                    clear &= valid[tuple(corner.T)]
                length += np.linalg.norm(delta, axis=1)
                previous = current
            start, end, length = (start[clear], end[clear], length[clear])
            sources.extend((start, end))
            dests.extend((end, start))
            distances.extend((length, length))
        if not sources:
            self.graph = coo_matrix((len(self.coords), len(self.coords))).tocsr()
        else:
            self.graph = coo_matrix((np.concatenate(distances), (np.concatenate(sources), np.concatenate(dests))), shape=(len(self.coords), len(self.coords))).tocsr()

    def index(self, point):
        pos = tuple((int(np.searchsorted(a, v)) for a, v in zip(self.axes, point)))
        if any((v >= len(self.axes[k]) or self.axes[k][v] != point[k] for k, v in enumerate(pos))):
            raise ValueError('Point not in navigation lattice')
        return int(np.ravel_multi_index(pos, self.shape))

    @lru_cache(maxsize=16384)
    def edge(self, low, high):
        return raster_segment(self.coords[low], self.coords[high])

    def expand_nodes(self, nodes):
        parts = [self.coords[int(nodes[0])][None, :]]
        for a, b in zip(nodes[:-1], nodes[1:]):
            edge = self.edge(min(int(a), int(b)), max(int(a), int(b)))
            parts.append((edge if a < b else edge[::-1])[1:])
        return np.concatenate(parts)

    def route(self, start_id, end_id, predecessors):
        nodes = [int(end_id)]
        while nodes[-1] != start_id:
            p = int(predecessors[nodes[-1]])
            if p < 0 or len(nodes) > len(self.coords):
                raise ValueError('Unreachable or cyclic graph route')
            nodes.append(p)
        points = self.expand_nodes(nodes[::-1])
        validate_grid_trace(points, self.valid)
        return list(map(tuple, points.tolist()))

class GeometryGraph:
    """Axis edges checked at every original 1-m voxel; no assumed roof plane."""

    def __init__(self, valid, required, stride=(4, 8, 8)):
        self.valid = valid
        self.axes = [np.unique(np.r_[np.arange(0, n, s), n - 1, [p[a] for p in required]]) for a, (n, s) in enumerate(zip(valid.shape, stride))]
        self.shape = tuple(map(len, self.axes))
        self.coords = np.stack(np.meshgrid(*self.axes, indexing='ij'), axis=-1).reshape(-1, 3)
        self.free = valid[tuple(self.coords.T)]
        idx = np.arange(len(self.coords)).reshape(self.shape)
        sources, dests, distances = ([], [], [])
        for axis in range(3):
            for j in range(self.shape[axis] - 1):
                region = list(self.axes)
                region[axis] = np.arange(self.axes[axis][j], self.axes[axis][j + 1] + 1)
                clear = valid[np.ix_(*region)].all(axis=axis)
                lo, hi = ([slice(None)] * 3, [slice(None)] * 3)
                lo[axis], hi[axis] = (j, j + 1)
                start, end = (idx[tuple(lo)][clear], idx[tuple(hi)][clear])
                sources.extend((start, end))
                dests.extend((end, start))
                length = float(self.axes[axis][j + 1] - self.axes[axis][j])
                distances.extend((np.full(len(start), length), np.full(len(start), length)))
        self.graph = coo_matrix((np.concatenate(distances), (np.concatenate(sources), np.concatenate(dests))), shape=(len(self.coords), len(self.coords))).tocsr()

    def index(self, point):
        pos = tuple((int(np.searchsorted(a, v)) for a, v in zip(self.axes, point)))
        if any((self.axes[k][v] != point[k] for k, v in enumerate(pos))):
            raise ValueError('Point not included in navigation graph')
        return int(np.ravel_multi_index(pos, self.shape))

    def route(self, start_id, end_id, predecessors):
        nodes = [int(end_id)]
        while nodes[-1] != start_id:
            p = int(predecessors[nodes[-1]])
            if p < 0 or len(nodes) > len(self.coords):
                raise ValueError('Unreachable or cyclic graph route')
            nodes.append(p)
        nodes.reverse()
        points = [tuple(map(int, self.coords[nodes[0]]))]
        for first, last in zip(nodes[:-1], nodes[1:]):
            a, b = (self.coords[first], self.coords[last])
            active = np.flatnonzero(a != b)
            assert len(active) == 1
            axis = int(active[0])
            for v in range(int(a[axis] + np.sign(b[axis] - a[axis])), int(b[axis] + np.sign(b[axis] - a[axis])), int(np.sign(b[axis] - a[axis]))):
                p = a.copy()
                p[axis] = v
                points.append(tuple(map(int, p)))
        validate_trace(points, self.valid)
        return points

def validate_trace(trace, valid):
    p = np.asarray(trace, dtype=int)
    if np.any(p < 0) or np.any(p >= np.asarray(valid.shape)) or (not np.all(valid[tuple(p.T)])):
        raise ValueError('Trace outside known free-space volume')
    if len(p) > 1 and (not np.all(np.abs(np.diff(p, axis=0)).sum(axis=1) == 1)):
        raise ValueError('Trace is not six-neighbor continuous')

def workload(trace, measurements):
    p = np.asarray(trace, dtype=float)
    d = np.diff(p, axis=0)
    length = np.linalg.norm(d[:, 1:], axis=1).sum() + 1.5 * np.abs(d[:, 0]).sum()
    if len(d) > 1:
        unit = d / np.linalg.norm(d, axis=1)[:, None]
        length += 0.05 * np.arccos(np.clip(np.sum(unit[:-1] * unit[1:], axis=1), -1, 1)).sum()
    return float(length + 0.01 * measurements)

def route_nodes(start, end, predecessors):
    nodes = [int(end)]
    while nodes[-1] != start:
        p = int(predecessors[nodes[-1]])
        if p < 0 or len(nodes) > len(predecessors):
            raise RuntimeError('Broken predecessor chain')
        nodes.append(p)
    return np.asarray(nodes[::-1])
