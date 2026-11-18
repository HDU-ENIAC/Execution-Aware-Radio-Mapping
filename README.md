# Execution-Aware Radio Mapping

Core research code for multi-UAV sensing and GP--neural residual reconstruction
of 3D urban radio maps. Acquisition uses a shared GP to select executable path
prefixes; neural residual correction with observation-geometry-adaptive context
(OGAC) is applied after acquisition.

## Contents

- `src/`: acquisition, obstacle-aware execution, GP estimation and reconstruction.
- `configs/default.json`: reconstruction architecture configuration.
- `tests/smoke_code.py`: self-contained CPU checks using synthetic inputs.
- `examples/read_dataset.py`: HDF5 reading example.

## Installation

Use Python 3.11 or later and a PyTorch build appropriate for your hardware.

```sh
python -m pip install -r requirements.txt
python tests/smoke_code.py
```

## Minimal Usage

With a local map, initialization file and compatible trained checkpoint:

```sh
python src/run_mapping.py \
  --data local_assets/maps.h5 \
  --target 0 \
  --initial-state local_assets/initial_state.npz \
  --cap 20000 \
  --checkpoint local_assets/reconstruction.pth \
  --device cuda \
  --output results/example
```

The workload cap is illustrative. Without `--checkpoint`, the command produces
a GP-only map. Use a new output directory for each run. See
[Usage](docs/USAGE.md) for the required local inputs.

## Data

The source maps are normalized 3D received-power volumes generated using
WinProp. [Dataset Format](docs/DATASET.md) describes the basic array structure.
This limited release contains code and format documentation; map arrays,
trained weights and detailed experiment records are not included. It is not
a complete reproduction package for the paper's tables.

## Citation and Reuse

Use of the authors' code or any subsequently released dataset requires citation
of the associated paper. Bibliographic details will be added after publication.
Reuse terms and excluded third-party assets are described in [NOTICE.txt](NOTICE.txt).
