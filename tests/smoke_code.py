"""CPU checks with artificial inputs; no dataset or learned weights required."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import copy
import json
import numpy as np
import torch
torch.set_num_threads(2)
from gp import AnisotropicBayesian,batch_posterior,signal_from_readings
from geometry import validate_grid_trace
from acquisition import ExecutionAwarePolicy,lookahead_account
from execution import MeasurementMission,capped_segment
from reconstruction import GPResidualReconstructor,reconstruction_loss

rng = np.random.default_rng(11)
grid = rng.uniform(0,20,(24,3))
points = rng.uniform(0,20,(8,3))
values = rng.random(8)
sequential = AnisotropicBayesian(grid,(12,40,40))
batch = copy.deepcopy(sequential)
sequential.fit(points,values)
state = batch_posterior(batch,points,values)
q = rng.uniform(0,20,(13,3))
prediction = state['offset']+batch.kernel(q,grid)@state['coeff']
np.testing.assert_allclose(prediction,sequential.predict(q),atol=1e-8,rtol=0)
valid = np.ones((8,65,65),bool)
target = rng.random(valid.shape).astype(np.float32)
traces = [[(z,0,0),(z,0,1)] for z in range(4)]
observations = [p for t in traces for p in t]
mission = MeasurementMission(target,valid,traces,observations,np.repeat(np.arange(4),2),300,valid,True)
policy = ExecutionAwarePolicy(mission,11)
plan = policy.plan(0)
assert plan is not None
path,samples,detail = plan
predicted = lookahead_account(mission.traces[0],path,samples,mission.seen,mission.counts[0],mission.cap)
_,new,audit = capped_segment(mission.traces[0],path,samples,mission.seen,mission.counts[0],mission.cap,True)
assert predicted['acquired']==[list(p) for p in new]
np.testing.assert_allclose(predicted['cost'],audit['cost'],atol=1e-8)
new,measurements,_ = mission.execute(0,path,samples)
policy.update(new,measurements)
validate_grid_trace(mission.traces[0],valid)
blocked = np.ones((2,2,2),bool)
blocked[0,1,0] = False
try:
    validate_grid_trace([(0,0,0),(0,1,1)],blocked)
except ValueError:
    pass
else:
    raise AssertionError('Occupied-corner cutting was allowed.')
model = GPResidualReconstructor().eval()
x = torch.rand(1,10,32,32,32)
x[:,4:6] = 0
x[:,4,0,0,0] = 1
x[:,5,1,1,1] = 1
with torch.no_grad():
    result = model(x)
assert result.shape==(1,1,32,32,32)
assert result[0,0,0,0,0]==x[0,0,0,0,0]
assert result[0,0,1,1,1]==-1
assert torch.isfinite(reconstruction_loss(result,torch.rand_like(result),x))
print(json.dumps(dict(status='PASS',gp_batch_sequential=True,prefix_accounting=True,
    planner_execution=True,corner_clearance=True,network_forward=True,
    observed_values_retained=True,device='cpu')))
