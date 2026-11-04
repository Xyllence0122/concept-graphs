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
  camera_config_conceptgraphs.yaml
  manifest.json
  rgb/000000.jpg
  depth/000000.png
  poses/000000.npy
```

`camera_config.yaml` preserves the exported RealSense distortion
coefficients. `camera_config_conceptgraphs.yaml` omits them because the
preserved upstream loader would undistort RGB without applying the same warp
to aligned depth; use the latter for GSA and cfslam commands.

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

## 4. Query a generated object map

After the upstream mapping pipeline creates a post-processed `pkl.gz` map,
activate its Conda environment. Query by exact label without loading CLIP:

```bash
conda activate conceptgraphs-official
python offline_tools/query_object_map.py \
  --map-file "$SCENE/pcd_saves/full_pcd_custom_targets_full_smoke_targeted_v1_post.pkl.gz" \
  --query "whiteboard eraser" \
  --mode label
```

For open-vocabulary retrieval using the same CLIP model as ConceptGraphs,
activate the upstream Conda environment and use the default `clip` mode:

```bash
conda activate conceptgraphs-official
python offline_tools/query_object_map.py \
  --map-file "$SCENE/pcd_saves/full_pcd_custom_targets_full_smoke_targeted_v1_post.pkl.gz" \
  --query "the object used to erase a whiteboard" \
  --mode clip \
  --device cuda \
  --robot-x 0.0 \
  --robot-y 0.0 \
  --standoff-m 0.30 \
  --output-json "$SCENE/queries/whiteboard_eraser.json"
```

The returned `candidate_goal.pose_stamped` faces the matched object from the
requested standoff distance. It is only a geometric candidate: an AMR must
still validate or adjust it against the Nav2 occupancy map and costmap before
execution.

## 4a. Build an automatic open-vocabulary scene map

For official ConceptGraphs-Detect-style maps, do not use a two-class custom
list. RAM proposes visible classes on every keyframe, accumulated classes keep
earlier discoveries active, and optional extra classes guarantee important
scene-specific terms that RAM may not name precisely:

```bash
conda activate conceptgraphs-official
export GSA_PATH="$HOME/conceptgraphs-deps/Grounded-Segment-Anything"

DATASET_ROOT="$HOME/Concept Graph/concept_graph_data/conceptgraphs_datasets"
SCENE_ID="rubiks_table_2026-11-04_01"
SCENE="$DATASET_ROOT/$SCENE_ID"

cd "$HOME/concept-graphs/upstream_concept_graphs/conceptgraph"

python scripts/generate_gsa_results.py \
  --dataset_root "$DATASET_ROOT" \
  --dataset_config "$SCENE/camera_config_conceptgraphs.yaml" \
  --scene_id "$SCENE_ID" \
  --stride 1 \
  --class_set ram \
  --extra_classes "Rubik's cube" table \
  --detector dino \
  --add_bg_classes \
  --accumu_classes \
  --box_threshold 0.25 \
  --text_threshold 0.25 \
  --nms_threshold 0.5 \
  --device cuda \
  --low_vram \
  --exp_suffix open_vocab_v1
```

The annotated JPEG files and compressed detection files are the inputs needed
for inspection and 3D fusion.  `--save_video` is optional and requires an
ImageIO FFmpeg or PyAV backend; omit it on minimal reproducible environments so
missing video codecs cannot turn otherwise successful inference into a failing
exit status.

Fuse all discovered instances, including background surfaces, into the 3D
map:

```bash
python slam/cfslam_pipeline_batch.py \
  dataset_root="$DATASET_ROOT" \
  dataset_config="$SCENE/camera_config_conceptgraphs.yaml" \
  scene_id="$SCENE_ID" \
  stride=1 \
  device=cuda \
  gsa_variant=ram_open_vocab_v1 \
  spatial_sim_type=overlap \
  match_method=sim_sum \
  sim_threshold=1.2 \
  mask_conf_threshold=0.25 \
  max_bbox_area_ratio=0.95 \
  dbscan_eps=0.1 \
  obj_min_detections=3 \
  skip_bg=False \
  merge_interval=20 \
  merge_overlap_thresh=0.7 \
  merge_visual_sim_thresh=0.8 \
  merge_text_sim_thresh=0.8 \
  save_suffix=open_vocab_v1
```

Automatic tagging is broad but not infallible. Keep `extra_classes` limited to
important or queried terms and require repeated, geometrically consistent
observations during 3D fusion; a single prompted 2D detection is not proof that
an object exists.

## 5. Infer a geometric support relation

For queries such as "the panda plush toy on the table", first establish the
directed `on` edge instead of relying on text similarity alone. The support
tool fits the dominant plane of the support object's point cloud, measures the
subject's contact gap and height relative to that plane, and checks that the
subject footprint overlaps the support footprint:

```bash
conda activate conceptgraphs-official
python offline_tools/infer_support_relations.py \
  --map-file "$SCENE/pcd_saves/full_pcd_custom_panda_table_full_v1_panda_table_v1_post.pkl.gz" \
  --subject-label "panda plush toy" \
  --support-label "table" \
  --require-match \
  --output-json "$SCENE/relations/panda_on_table.json"
```

An accepted relation is emitted as a directed
`panda plush toy --on--> table` edge with the fitted plane, contact gap,
relative height, footprint overlap, and thresholds preserved as evidence.
This geometric relation assumes that the support object contains a dominant
supporting plane; it is not appropriate for arbitrary non-planar containers.

## 6. Visualize text-query results

The local query viewer combines the reconstructed RGB point cloud, semantic
object nodes, stored relation edges, and text-query highlighting in one Open3D
window:

```bash
conda activate conceptgraphs-official
python offline_tools/visualize_query_map.py \
  --map-file "$SCENE/pcd_saves/full_pcd_custom_panda_table_full_v1_panda_table_v1_post.pkl.gz" \
  --scene-cloud "$SCENE/validation/fused_preview.ply" \
  --relations-file "$SCENE/relations/panda_on_table.json" \
  --query "find the panda plush toy on the table" \
  --mode auto \
  --screenshot-path "$SCENE/visualizations/panda_on_table.png"
```

The matched object is yellow, its supporting object is cyan, other semantic
objects use distinct instance colors, and the reconstructed RGB scene is
dimmed in the background.
`auto` mode resolves stored relation queries first and falls back to label
matching. Use `--mode clip` for open-vocabulary CLIP retrieval when the query
does not require an explicit graph edge. CLIP mode colors every object by
similarity from purple/blue (low) to yellow (high), matching the paper demo's
query heatmap. The screenshot button stores the current 3D view at the
requested path.

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
