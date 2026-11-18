# Local Inputs and Outputs

`src/run_mapping.py` runs four-UAV acquisition followed by GP estimation and,
when a compatible checkpoint is supplied, neural residual reconstruction.
Neural predictions do not participate in path decisions.

## Map and Initialization

The map file must provide `target` and `building_mask` arrays in the
[documented HDF5 format](DATASET.md). The `--target` argument selects one volume.

The initialization NPZ must contain:

| Key | Meaning |
| --- | --- |
| `observations` | Initial measurement coordinates `(N, 3)`, in `(z, y, x)` order |
| `values` | Corresponding readings, length N |
| `owners` | Measuring UAV index, 0 through 3 |
| `initial_count` | Number of initial readings, N |
| `initial_trace_0` through `initial_trace_3` | Unit-step routes ending at the four current UAV positions |

Measurements and route vertices must lie in recorded free space. Supplied
readings must match the selected map at their coordinates. The `--cap`
argument sets the cumulative workload limit for each UAV.

## Reconstruction and Results

`--checkpoint` must point to a compatible trusted checkpoint containing a
`model` state dictionary. The architecture is specified in
`configs/default.json`; weights are not included. The source class name
`RouteHybridResidualHead` denotes OGAC and is retained for checkpoint compatibility.

The output directory contains acquired measurements, flight traces, the GP
posterior, a reconstructed volume and JSON summaries. Measured values are
retained in the final map. The command refuses an existing output directory.

Use `python src/run_mapping.py --help` for command-line options. The CPU smoke
test can run without datasets or pretrained weights.
