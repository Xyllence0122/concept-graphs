# RTAB-Map to ConceptGraphs offline tools

These tools bridge the ROS 2 D455 mapping prototype and the preserved upstream
ConceptGraphs pipeline. Runtime datasets stay outside Git because RGB-D images,
RTAB-Map databases, PLY files, and model weights are generated artifacts.

## Dependencies

- `rtabmap-export` from the RTAB-Map command-line tools
- Python 3.10 or newer
- NumPy, Pillow, and PyYAML
- CloudCompare or another PLY viewer for visual inspection

The converter itself does not need ROS 2 or PyTorch. Run the upstream
ConceptGraphs model in its separate Conda environment; do not install its old
CUDA/Python dependencies into the ROS 2 Jazzy environment.

## 1. Export optimized RGB-D keyframes from RTAB-Map

```bash
DB="$HOME/Concept Graph/concept_graph_data/rtabmap/lab_smoke_2026-10-29_01.db"
EXPORT_ROOT="$HOME/Concept Graph/concept_graph_data/offline_exports/lab_smoke_2026-10-29_01"
RUN_NAME="lab_smoke_2026-10-29_01"

mkdir -p "$EXPORT_ROOT"
rtabmap-export \
  --images \
  --images_id \
  --poses_camera \
  --poses_format 11 \
  --output "$RUN_NAME" \
  --output_dir "$EXPORT_ROOT" \
  "$DB"
```

`--poses_camera` is required: every exported pose must describe the camera
optical frame used to back-project its RGB-D pair, not the robot base frame.

## 2. Convert the export to `RealsenseDataset` layout

```bash
SCENE="$HOME/Concept Graph/concept_graph_data/conceptgraphs_datasets/lab_smoke_2026-10-29_01"

python3 offline_tools/export_rtabmap_dataset.py \
  --export-dir "$EXPORT_ROOT" \
  --run-name "$RUN_NAME" \
  --output-dir "$SCENE" \
  --max-depth-m 5.0
```

The output contains:

```text
SCENE/
  camera_config.yaml
  manifest.json
  rgb/000000.jpg
  depth/000000.png
  poses/000000.npy
```

The script requires matching RGB, depth, calibration, and pose node IDs. It
clips invalid or out-of-range depth to zero and refuses to overwrite an
existing scene directory.

The preserved upstream `RealsenseDataset` conjugates each stored pose with
`P = diag(1, -1, -1, 1)`. The converter stores the inverse-conjugated
RTAB-Map `map_T_camera_optical` matrix so the loader reconstructs the original
camera-to-map transform.

## 3. Validate geometry before running the model

```bash
python3 offline_tools/validate_posed_rgbd.py \
  --scene-dir "$SCENE" \
  --pixel-step 8 \
  --frame-step 1
```

This creates `SCENE/validation/` containing:

- `fused_preview.ply`: sampled colored point cloud;
- `topdown_xy.png`: top-down XY raster preview;
- `side_xz.png`: side-view XZ raster preview;
- `report.json`: frame count, point count, bounds, and sampling settings.

Open the PLY file on Ubuntu with:

```bash
CloudCompare "$SCENE/validation/fused_preview.ply"
```

Do not continue to the expensive ConceptGraphs inference stage if adjacent
frames are visibly separated, mirrored, upside down, or scaled incorrectly.

## Upstream PyTorch compatibility

The preserved upstream stack uses PyTorch 2.0.1 with CUDA 11.8. If a manually
created environment resolves MKL 2025 or NumPy 2, `import torch` can fail with
`undefined symbol: iJIT_NotifyEvent` or later NumPy ABI errors. The verified
compatibility pins are:

```bash
conda install --override-channels \
  -c pytorch -c nvidia -c defaults -c conda-forge \
  "mkl=2021.4.0" \
  "intel-openmp=2021.4.0" \
  "numpy=1.24.4"
```

The original full environment remains in
[`../upstream_concept_graphs/environment.yml`](../upstream_concept_graphs/environment.yml).
