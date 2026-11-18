"""Read one real HDF5 input and its target; inspect metadata without loading a corpus."""
import argparse
import json
from pathlib import Path
import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('file',type=Path)
    parser.add_argument('--input-id',type=int,default=0)
    args = parser.parse_args()
    with h5py.File(args.file,'r') as f:
        if not 0 <= args.input_id < len(f['input']):
            raise ValueError('Input ID is outside the HDF5 dataset.')
        target_id = int(f['target_index'][args.input_id])
        target = f['target'][target_id]
        building = f['building_mask'][target_id].astype(bool)
        sparse = f['input'][args.input_id]
        free = ~building
        assert target.shape==building.shape==sparse.shape==(32,512,512)
        assert np.isfinite(target[free]).all()
        assert (target[building]==-1).all()
        print(json.dumps(dict(file=args.file.name,data_version=str(f.attrs.get('data_version','unknown')),
            input_id=args.input_id,target_id=target_id,shape_zyx=list(target.shape),
            crop_origin_xy=f['target_positions'][target_id].tolist(),building_voxels=int(building.sum()),
            free_power_min=float(target[free].min()),free_power_max=float(target[free].max())),indent=2))


if __name__=='__main__':
    main()
