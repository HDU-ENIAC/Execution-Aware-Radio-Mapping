"""Independent physical budget accounting; no learned route-selection code."""
import numpy as np

from geometry import validate_trace, workload
from geometry import validate_grid_trace


def capped_segment(trace, suffix, sample_points, measured, count, cap, allow_diagonal=False):
    """Keep complete move/acquire events, charging the boundary turn once."""
    trace = [tuple(map(int, p)) for p in trace]
    suffix = [tuple(map(int, p)) for p in suffix]
    if not trace or not suffix or trace[-1] != suffix[0]:
        raise ValueError("Suffix must start at current UAV position")
    if len(suffix) > 1:
        delta = np.abs(np.diff(suffix, axis=0))
        local = delta.max(axis=1) == 1 if allow_diagonal else delta.sum(axis=1) == 1
        if not local.all():
            raise ValueError("Noncontinuous proposed motion")
    proposed = set(map(tuple, sample_points))
    if not proposed.issubset(set(suffix)):
        raise ValueError("A scheduled measurement is not on the proposed route")
    if suffix[0] in proposed and suffix[0] not in measured:
        raise ValueError("Measure-at-current-position must be scheduled separately")
    cost = workload(trace, count)
    if cost > cap + 1e-9:
        raise ValueError("Initial acquisition exceeds budget")
    direction = np.asarray(trace[-1])-np.asarray(trace[-2]) if len(trace) > 1 else None
    kept, acquired, seen = [suffix[0]], [], set(measured)
    truncated = False
    next_increment = None
    for point in suffix[1:]:
        step = np.asarray(point)-np.asarray(kept[-1])
        charge = float(np.linalg.norm(step[1:]) + 1.5*abs(step[0]))
        if direction is not None:
            charge += .05*float(np.arccos(np.clip(np.dot(direction, step) /
                (np.linalg.norm(direction)*np.linalg.norm(step)), -1., 1.)))
        read = point in proposed and point not in seen
        charge += .01*int(read)
        if cost+charge > cap+1e-9:
            truncated, next_increment = True, charge
            break
        cost += charge
        kept.append(point)
        direction = step
        if read:
            acquired.append(point)
            seen.add(point)
    result = trace+kept[1:]
    expected = workload(result, count+len(acquired))
    np.testing.assert_allclose(cost, expected, atol=1e-7, rtol=0)
    return result, acquired, {"truncated": truncated, "cost": expected,
        "remaining_budget": cap-expected, "next_rejected_increment": next_increment,
        "proposed_steps": len(suffix)-1, "executed_steps": len(kept)-1}


class MeasurementMission:
    """Only execute() exposes new readings to acquisition models."""

    def __init__(self, target, valid, traces, observations, owners, cap, execution_valid=None,
                 allow_diagonal=False):
        self._target = target
        self.sensing_valid = valid
        self.valid = valid if execution_valid is None else execution_valid
        self.allow_diagonal = bool(allow_diagonal)
        self.validate = validate_grid_trace if allow_diagonal else validate_trace
        self.cap = float(cap)
        self.traces = [list(map(tuple, t)) for t in traces]
        self.observations = list(map(tuple, observations))
        self.owners = list(owners)
        self.seen = set(self.observations)
        self.counts = np.bincount(owners, minlength=4).tolist()
        self.closed = [False]*4
        self.stops = [None]*4
        if len(self.seen) != len(self.observations):
            raise ValueError("Duplicate initialization observation")
        if not all(self.observable(p) for p in self.observations):
            raise ValueError("Initialization outside sensing domain")
        for i, trace in enumerate(self.traces):
            self.validate(trace, self.valid)
            owned = [p for p, owner in zip(self.observations, owners) if owner == i]
            if not set(owned).issubset(set(trace)):
                raise ValueError("Initialization contains unvisited readings")
        if max(self.costs()) > cap+1e-8:
            raise ValueError("Initialization does not fit the common budget")
        self.values = [float(target[p]) for p in self.observations]
        self.initial_costs = self.costs()

    def costs(self):
        return [workload(t, n) for t, n in zip(self.traces, self.counts)]

    def observable(self, point):
        p = np.asarray(point, dtype=int)
        return bool(np.all(p >= 0) and np.all(p < self.sensing_valid.shape)
            and self.sensing_valid[tuple(p)])

    def execute(self, agent, suffix, sample_points):
        if self.closed[agent]:
            raise ValueError("Closed agent cannot execute another move")
        self.validate(suffix, self.valid)
        if not all(self.observable(p) for p in sample_points):
            raise ValueError("Scheduled observation outside sensing domain")
        trace, new, audit = capped_segment(self.traces[agent], suffix, sample_points,
            self.seen, self.counts[agent], self.cap, self.allow_diagonal)
        self.validate(trace, self.valid)
        self.traces[agent] = trace
        self.counts[agent] += len(new)
        self.observations.extend(new)
        self.owners.extend([agent]*len(new))
        self.seen.update(new)
        values = [float(self._target[p]) for p in new]
        self.values.extend(values)
        if audit["truncated"]:
            self.closed[agent] = True
            self.stops[agent] = "next_move_plus_measurement_exceeds_cap"
        assert max(self.costs()) <= self.cap+1e-7
        return new, values, audit

    def exhaust_candidates(self, agent):
        self.closed[agent] = True
        self.stops[agent] = "no_reachable_unmeasured_candidate"
