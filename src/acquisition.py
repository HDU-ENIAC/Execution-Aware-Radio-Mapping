"""Executable-prefix selection from the current experiment implementation.

Historical class names below are retained for source-level correspondence.
Use ExecutionAwarePolicy (the short-prefix mode) for the proposed method.
"""
import numpy as np
from scipy.ndimage import convolve
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.linalg import cho_factor, cho_solve
from gp import GridBayesianGP, AnisotropicBayesian
from geometry import GeometryGraph, DiagonalGeometryGraph, workload, route_nodes, validate_grid_trace
from execution import MeasurementMission, capped_segment
LENGTHS = (12.,40.,40.)

def navigation_graph(mission, stride=(4, 8, 8)):
    required = [t[-1] for t in mission.traces] + [(mission.sensing_valid.shape[0] - 1, 0, 0)]
    graph_type = DiagonalGeometryGraph if mission.allow_diagonal else GeometryGraph
    return graph_type(mission.valid, required, stride=stride)

def observable_nodes(graph, mission):
    allowed = graph.free & (graph.coords[:, 0] < mission.sensing_valid.shape[0])
    return allowed

class ShresthaBayesianPolicy:

    def __init__(self, mission, seed):
        self.mission = mission
        self.graph = navigation_graph(mission, stride=(8, 32, 32))
        self.observable = observable_nodes(self.graph, mission)
        self.ids = np.flatnonzero(self.observable)
        if len(self.ids) > 5000:
            raise ValueError('Declared Bayesian latent-node memory limit exceeded')
        self.model = GridBayesianGP(self.graph.coords[self.ids], mean=float(np.mean(mission.values)))
        self.model.fit(mission.observations, mission.values)
        self.visits = np.zeros(len(self.graph.coords))
        self.smoothed = None
        self.edges = self.graph.graph.tocoo()
        self.meta = {'branch': 'Bayesian, not DRUE', 'latent_nodes': len(self.ids), 'prior_mean_from_initial_observations': self.model.offset}

    def plan(self, agent):
        start = self.graph.index(self.mission.traces[agent][-1])
        raw = np.zeros(len(self.graph.coords))
        raw[self.ids] = np.maximum(np.diag(self.model.cov), 0) / self.model.variance
        self.smoothed = raw if self.smoothed is None else 0.25 * raw + 0.75 * self.smoothed
        utility = convolve(self.smoothed.reshape(self.graph.shape), np.ones((3, 3, 3)) / 27, mode='reflect').ravel()
        utility /= 1 + self.visits
        utility[start] = 0
        reachable = np.isfinite(dijkstra(self.graph.graph, indices=start, directed=False))
        eligible = self.observable & reachable
        eligible[start] = False
        if not eligible.any():
            return None
        end = int(np.argmax(np.where(eligible, utility, -np.inf)))
        self.visits[end] += 1
        cost = 0.25 + 0.75 / (utility + 0.01)
        weighted = coo_matrix((self.edges.data * (cost[self.edges.row] + cost[self.edges.col]) / 2, (self.edges.row, self.edges.col)), shape=self.graph.graph.shape).tocsr()
        _, previous = dijkstra(weighted, indices=start, directed=False, return_predecessors=True)
        nodes = route_nodes(start, end, previous)[:8]
        suffix = self.graph.route(start, int(nodes[-1]), previous)
        scheduled = [tuple(map(int, self.graph.coords[n])) for n in nodes[1:] if self.observable[n]]
        return (suffix, scheduled, {'proposed_destination': list(map(int, self.graph.coords[end])), 'planned_waypoints': len(nodes) - 1})

    def update(self, points, values):
        for p, value in zip(points, values):
            self.model.update(p, value)

    def finalize(self):
        pass

def utility_for_next(policy, agent):
    raw = np.zeros(len(policy.graph.coords))
    raw[policy.ids] = np.maximum(np.diag(policy.model.cov), 0) / policy.model.variance
    smooth = raw if policy.smoothed is None else 0.25 * raw + 0.75 * policy.smoothed
    utility = convolve(smooth.reshape(policy.graph.shape), np.ones((3, 3, 3)) / 27, mode='reflect').ravel() / (1 + policy.visits)
    start = policy.graph.index(policy.mission.traces[agent][-1])
    utility[start] = 0
    return (start, utility)

def probe_points(valid):
    flat = np.flatnonzero(valid)
    return np.column_stack(np.unravel_index(flat[np.linspace(0, len(flat) - 1, 2048).astype(int)], valid.shape))

def variance_gain(model, points, probe_h):
    if not len(points):
        return 0.0
    k = model.kernel(model.grid, points)
    h = cho_solve(model.factor, k, check_finite=False)
    noise = model.variance - np.sum(k * h, axis=0) + model.noise
    cross = model.cov @ h
    s = h.T @ cross + np.diag(np.maximum(noise, 0))
    factor = cho_factor((s + s.T) * 0.5, lower=True, check_finite=False)
    projected = probe_h.T @ cross
    gain = np.sum(projected * cho_solve(factor, projected.T, check_finite=False).T) / len(probe_h.T)
    assert gain >= -1e-12
    return float(max(gain, 0))

class ProjectedVariance:
    """Cache H_probe^T Cov; update by the exact same rank-one conditioning."""

    def __init__(self, model, probes):
        self.model = model
        self.h = cho_solve(model.factor, model.kernel(model.grid, probes), check_finite=False)
        self.projected = self.h.T @ model.cov

    def gain(self, points):
        if not len(points):
            return 0.0
        model = self.model
        k = model.kernel(model.grid, points)
        a = cho_solve(model.factor, k, check_finite=False)
        noise = np.maximum(model.variance - np.sum(k * a, axis=0) + model.noise, 0)
        cross = model.cov @ a
        s = a.T @ cross + np.diag(noise)
        factor = cho_factor((s + s.T) * 0.5, lower=True, check_finite=False)
        projected = self.projected @ a
        gain = np.sum(projected * cho_solve(factor, projected.T, check_finite=False).T) / len(projected)
        assert gain >= -1e-12
        return float(max(0.0, gain))

    def update(self, points, values):
        model = self.model
        for point, value in zip(points, values):
            k = model.kernel(model.grid, point)[:, 0]
            a = cho_solve(model.factor, k, check_finite=False)
            cross = model.cov @ a
            noise = max(model.variance - k @ a + model.noise, 0)
            innovation = noise + a @ cross
            self.projected -= np.outer(self.projected @ a, cross) / innovation
            model.update(point, value)

    def check(self):
        ids = np.linspace(0, len(self.model.grid) - 1, 17).astype(int)
        exact = self.h.T @ self.model.cov[:, ids]
        error = float(np.max(np.abs(self.projected[:, ids] - exact)))
        assert error < 2e-10, error
        return error

class CalibratedNative(ShresthaBayesianPolicy):

    def __init__(self, mission, seed):
        super().__init__(mission, seed)
        self.model = AnisotropicBayesian(self.model.grid, LENGTHS, mean=float(np.mean(mission.values)))
        self.model.fit(mission.observations, mission.values)
        self.meta.update(lengths_zyx=LENGTHS, branch='calibrated native uncertainty control')

class RoutePolicy(CalibratedNative):

    def __init__(self, mission, seed, mode):
        super().__init__(mission, seed)
        assert mode in ('count_per_cost', 'route_gain')
        self.mode = mode
        self.projector = ProjectedVariance(self.model, probe_points(mission.sensing_valid)) if mode == 'route_gain' else None
        self.decisions = 0
        self.meta.update(branch=mode, max_goals=8, separation=64, max_coarse_edges=7, probes=2048 if self.projector else 0, neural_feedback=False)

    def pool(self, agent):
        m, graph = (self.mission, self.graph)
        visits = self.visits.copy()
        start, utility = utility_for_next(self, agent)
        native = super().plan(agent)
        self.visits = visits
        if native is None:
            return []
        suffix, scheduled, details = native
        goal = graph.index(details['proposed_destination'])
        edge_cost = 0.25 + 0.75 / (utility + 0.01)
        edges = self.edges
        weighted = coo_matrix((edges.data * (edge_cost[edges.row] + edge_cost[edges.col]) / 2, (edges.row, edges.col)), shape=graph.graph.shape).tocsr()
        distance, previous = dijkstra(weighted, indices=start, directed=False, return_predecessors=True)
        possible = self.observable & np.isfinite(distance)
        possible[start] = False
        for point in m.seen:
            try:
                possible[graph.index(point)] = False
            except ValueError:
                pass
        goals = [goal]
        while len(goals) < 8:
            possible &= np.linalg.norm(graph.coords - graph.coords[goals[-1]], axis=1) >= 64.0
            if not possible.any():
                break
            goals.append(int(np.argmax(np.where(possible, utility, -np.inf))))
        base_cost = workload(m.traces[agent], m.counts[agent])
        candidates, unique = ([], set())
        for rank, end in enumerate(goals):
            if rank == 0:
                path, points, detail = (suffix, scheduled, details)
            else:
                nodes = route_nodes(start, end, previous)[:8]
                path = graph.route(start, int(nodes[-1]), previous)
                points = [tuple(map(int, graph.coords[n])) for n in nodes[1:] if self.observable[n]]
                detail = dict(proposed_destination=list(map(int, graph.coords[end])), planned_waypoints=len(nodes) - 1)
            full, acquired, audit = capped_segment(m.traces[agent], path, points, m.seen, m.counts[agent], m.cap, True)
            executed = full[len(m.traces[agent]) - 1:]
            key = tuple(map(tuple, executed))
            if key in unique:
                continue
            unique.add(key)
            validate_grid_trace(executed, m.valid)
            if not audit['truncated']:
                graph.index(executed[-1])
            candidates.append(dict(path=path, samples=points, details=detail, goal=end, increment=audit['cost'] - base_cost, acquired=[list(p) for p in acquired], execution=audit, native=rank == 0))
        return candidates

    def plan(self, agent):
        candidates = self.pool(agent)
        if not candidates:
            return None
        for c in candidates:
            gain = self.projector.gain(c['acquired']) if self.projector else float(len(c['acquired']))
            c.update(gain=gain, score=gain / c['increment'] if c['increment'] > 0 else 0.0)
        if self.projector and self.decisions % 32 == 0:
            first = candidates[0]
            np.testing.assert_allclose(first['gain'], variance_gain(self.model, first['acquired'], self.projector.h), atol=2e-12, rtol=1e-08)
            self.projector.check()
        chosen = max(candidates, key=lambda c: c['score'])
        self.visits[chosen['goal']] += 1
        self.decisions += 1
        detail = dict(chosen['details'], selection_rule=self.mode, selected_is_native=chosen['native'], candidates=[dict(destination=c['details']['proposed_destination'], scheduled_new_points=c['acquired'], incremental_workload=c['increment'], numerator=c['gain'], score=c['score'], truncated=c['execution']['truncated'], native=c['native'], selected=c is chosen) for c in candidates])
        return (chosen['path'], chosen['samples'], detail)

    def update(self, points, values):
        if self.projector:
            self.projector.update(points, values)
        else:
            super().update(points, values)

def lookahead_account(trace, suffix, scheduled, seen, count, cap):
    """Vectorized counterpart of capped_segment, including boundary turn and unique reads."""
    t = np.asarray(trace, dtype=np.int64)
    p = np.asarray(suffix, dtype=np.int64)
    if len(t) == 0 or len(p) == 0 or (not np.array_equal(t[-1], p[0])):
        raise ValueError('Invalid lookahead connection')
    delta = np.diff(p, axis=0)
    if len(delta) and (not (np.abs(delta).max(axis=1) == 1).all()):
        raise ValueError('Noncontinuous lookahead')
    proposed = set(map(tuple, scheduled))
    if not proposed.issubset(set(map(tuple, p))):
        raise ValueError('Off-route scheduled reading')
    if tuple(p[0]) in proposed and tuple(p[0]) not in seen:
        raise ValueError('Unsupported reading at current position')
    base = workload(t, count)
    if base > cap + 1e-09:
        raise ValueError('Initialization exceeds budget')
    charged = np.zeros(len(delta), dtype=bool)
    already = set(seen)
    for i, point in enumerate(p[1:]):
        pt = tuple(point)
        if pt in proposed and pt not in already:
            charged[i] = True
            already.add(pt)
    steps = np.linalg.norm(delta[:, 1:], axis=1) + 1.5 * np.abs(delta[:, 0])
    if len(delta):
        previous = np.vstack((t[-1] - t[-2] if len(t) > 1 else delta[0], delta[:-1]))
        cosine = (previous * delta).sum(axis=1) / (np.linalg.norm(previous, axis=1) * np.linalg.norm(delta, axis=1))
        turns = 0.05 * np.arccos(np.clip(cosine, -1.0, 1.0))
        if len(t) == 1:
            turns[0] = 0.0
        steps += turns + 0.01 * charged
    cumulative = base + np.cumsum(steps)
    rejected = np.flatnonzero(cumulative > cap + 1e-09)
    kept = int(rejected[0]) if len(rejected) else len(delta)
    acquired = p[1:kept + 1][charged[:kept]].tolist()
    cost = float(cumulative[kept - 1]) if kept else float(base)
    return dict(acquired=acquired, increment=cost - base, cost=cost, steps=kept, truncated=kept < len(delta), proposed_steps=len(delta))

class HorizonPolicy(RoutePolicy):

    def __init__(self, mission, seed, horizon):
        super().__init__(mission, seed, 'count_per_cost')
        if horizon not in ('short', 'long'):
            raise ValueError(horizon)
        self.horizon = horizon
        self.meta.update(branch='count_cost_horizon_control', score_horizon=horizon, execute_coarse_edges=7, full_route_budget_capped=True)

    def pool(self, agent):
        start, utility = utility_for_next(self, agent)
        candidates = super().pool(agent)
        if not candidates:
            return []
        graph, m, edges = (self.graph, self.mission, self.edges)
        costs = 0.25 + 0.75 / (utility + 0.01)
        weighted = coo_matrix((edges.data * (costs[edges.row] + costs[edges.col]) / 2, (edges.row, edges.col)), shape=graph.graph.shape).tocsr()
        _, previous = dijkstra(weighted, indices=start, directed=False, return_predecessors=True)
        for c in candidates:
            nodes = route_nodes(start, c['goal'], previous)
            full_path = graph.route(start, int(nodes[-1]), previous)
            np.testing.assert_array_equal(np.asarray(full_path)[:len(c['path'])], c['path'])
            scheduled = [tuple(map(int, graph.coords[n])) for n in nodes[1:] if self.observable[n]]
            long = lookahead_account(m.traces[agent], full_path, scheduled, m.seen, m.counts[agent], m.cap)
            if self.decisions % 64 == 0:
                exact, points, audit = capped_segment(m.traces[agent], full_path, scheduled, m.seen, m.counts[agent], m.cap, True)
                np.testing.assert_array_equal(np.asarray(points).reshape(-1, 3), np.asarray(long['acquired']).reshape(-1, 3))
                np.testing.assert_allclose(long['cost'], audit['cost'], atol=2e-07, rtol=0)
                assert long['steps'] == audit['executed_steps']
                validate_grid_trace(exact[len(m.traces[agent]) - 1:], m.valid)
            c['long'] = long
            c['short_score'] = len(c['acquired']) / c['increment'] if c['increment'] > 0 else 0.0
            c['long_score'] = len(long['acquired']) / long['increment'] if long['increment'] > 0 else 0.0
        return candidates

    def plan(self, agent):
        candidates = self.pool(agent)
        if not candidates:
            return None
        short = max(candidates, key=lambda c: c['short_score'])
        long = max(candidates, key=lambda c: c['long_score'])
        chosen = short if self.horizon == 'short' else long
        self.visits[chosen['goal']] += 1
        self.decisions += 1
        details = dict(chosen['details'], selection_rule=self.horizon, selected_is_native=chosen['native'], horizon_choices_differ=short is not long, predicted_new_points=chosen['acquired'], predicted_increment=chosen['increment'], candidates=[])
        for c in candidates:
            details['candidates'].append(dict(destination=c['details']['proposed_destination'], short_new=c['acquired'], short_increment=c['increment'], short_score=c['short_score'], long_new_count=len(c['long']['acquired']), long_increment=c['long']['increment'], long_score=c['long_score'], long_budget_truncated=c['long']['truncated'], short_budget_truncated=c['execution']['truncated'], native=c['native'], selected=c is chosen))
        return (chosen['path'], chosen['samples'], details)

def mission_from_saved(y, valid, saved, cap):
    n = int(saved['initial_count'])
    mission = MeasurementMission(y, valid, [saved[f'initial_trace_{a}'] for a in range(4)], saved['observations'][:n], saved['owners'][:n], cap, valid, True)
    np.testing.assert_array_equal(np.asarray(mission.values, np.float32), saved['values'].astype(np.float32))
    return mission

class ExecutionAwarePolicy(HorizonPolicy):
    def __init__(self, mission, seed=20261025):
        super().__init__(mission, seed, "short")
