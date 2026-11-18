"""Run current acquisition from an explicit initial state and reconstruct a map."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial import cKDTree

from acquisition import ExecutionAwarePolicy, mission_from_saved
from gp import signal_from_readings, save_posterior, GPMean
from metrics import metrics


def gp_fields(model,shape,chunk=2048,device='cpu'):
    mean = np.empty(shape,np.float32)
    proxy = np.empty(shape,np.float32)
    tree = cKDTree(model.grid/model.lengths)
    diagonal = np.maximum(np.diag(model.cov),0)/model.variance
    predictor = GPMean(model) if str(device).startswith('cuda') else model.predict
    if str(device).startswith('cuda'):
        predictor.check(model,shape)
    for lo in range(0,mean.size,chunk):
        hi = min(mean.size,lo+chunk)
        queries = np.column_stack(np.unravel_index(np.arange(lo,hi),shape))
        mean.ravel()[lo:hi] = predictor(queries)
        _,ids = tree.query(queries/model.lengths)
        proxy.ravel()[lo:hi] = np.clip(diagonal[ids],0,1)
    return mean,proxy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--target',type=int,required=True)
    parser.add_argument('--initial-state',type=Path,required=True)
    parser.add_argument('--cap',type=float,required=True)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--seed',type=int,default=20261025)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--max-turns',type=int,default=16000)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    with h5py.File(args.data,'r') as data:
        target = data['target'][args.target].astype(np.float32)
        building = data['building_mask'][args.target].astype(bool)
    with np.load(args.initial_state,allow_pickle=False) as data:
        saved = {key:data[key].copy() for key in data.files}
    mission = mission_from_saved(target,~building,saved,args.cap)
    policy = ExecutionAwarePolicy(mission,args.seed+args.target)
    records = []
    for cursor in range(args.max_turns):
        if all(mission.closed):
            break
        agent = cursor % 4
        if mission.closed[agent]:
            continue
        plan = policy.plan(agent)
        if plan is None:
            mission.exhaust_candidates(agent)
            continue
        path,scheduled,details = plan
        points,values,audit = mission.execute(agent,path,scheduled)
        policy.update(points,values)
        records.append(dict(agent=agent,observed_points=[list(p) for p in points],execution=audit,**details))
    if not all(mission.closed):
        raise RuntimeError('Turn limit reached; no completed result is reported.')
    points = np.asarray(mission.observations,np.int32)
    values = np.asarray(mission.values,np.float32)
    np.savez_compressed(args.output/'observations_and_traces.npz',observations=points,values=values,
        owners=np.asarray(mission.owners),**{f'trace_{a}':np.asarray(t,np.int32) for a,t in enumerate(mission.traces)})
    save_posterior(args.output/'posterior.npz',policy.model,policy.model.lengths)
    summary = dict(status='COMPLETE',cap=args.cap,Cmax=max(mission.costs()),
        Csum=sum(mission.costs()),Nobs=len(points),stop_reasons=mission.stops)
    (args.output/'acquisition.json').write_text(json.dumps(dict(summary=summary,records=records),indent=2),encoding='utf-8')
    mean,proxy = gp_fields(policy.model,target.shape,device=args.device)
    signal,observed = signal_from_readings(target.shape,building,points,values)
    if args.checkpoint:
        from reconstruction import GPResidualReconstructor,prepare_inputs,reconstruct_volume
        net = GPResidualReconstructor.from_checkpoint(args.checkpoint,args.device)
        inputs = prepare_inputs(signal,observed,building,mean,proxy)
        prediction = reconstruct_volume(net,inputs,args.output/'prediction.npy',args.device)
    else:
        prediction = mean
        prediction[building] = -1.
        prediction[observed] = signal[observed]
        np.save(args.output/'gp_mean.npy',prediction)
    summary['mae_free'] = float(np.abs(prediction[~building]-target[~building]).mean(dtype=np.float64))
    summary['metrics'] = metrics(prediction,target,~building,observed)
    summary['reconstruction'] = 'GP--neural residual' if args.checkpoint else 'GP only'
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary))


if __name__=='__main__':
    main()
