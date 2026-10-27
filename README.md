# ROS 2 D455 Semantic Mapping for AMR

This fork is organized around the ROS 2 prototype currently under development:
RealSense D455 RGB-D input, RTAB-Map localization and mapping, YOLO instance
segmentation, 3D object projection, and a persistent semantic object graph.

The research repository that this work started from is preserved separately in
[`upstream_concept_graphs/`](upstream_concept_graphs/). Keeping it in one
directory prevents its Conda environment, scripts, assets, and documentation
from being confused with the current ROS 2 Jazzy workspace.

## Repository layout

| Path | Purpose |
| --- | --- |
| [`ros2_ws/`](ros2_ws/) | Current ROS 2 D455 semantic-mapping implementation |
| [`log/`](log/) | One dated Markdown file per development session |
| [`.github/workflows/ros2-jazzy.yml`](.github/workflows/ros2-jazzy.yml) | Ubuntu 24.04 / ROS 2 Jazzy build and test workflow |
| [`upstream_concept_graphs/`](upstream_concept_graphs/) | Preserved original ConceptGraphs research code and documentation |

The two codebases intentionally have separate dependency setups:

- Use [`ros2_ws/README.md`](ros2_ws/README.md) for the current D455 and AMR work.
- Use [`upstream_concept_graphs/README.md`](upstream_concept_graphs/README.md)
  only when reproducing the original paper pipeline.
- After each development session, create a new `log/YYYY-MM-DD.md` file so
  that results, evidence, limitations, and next steps remain traceable. The
  existing entries are listed in [`log/`](log/).

Do not install the original `environment.yml` into the ROS 2 virtual
environment. The original pipeline targets an older Python/CUDA stack, while
the current workspace has been validated on Ubuntu 24.04, ROS 2 Jazzy, and
Python 3.12.

## Current pipeline

```text
RealSense D455 RGB + aligned depth
  -> RTAB-Map visual odometry and map transform
  -> YOLO instance masks
  -> depth back-projection into camera-frame 3D objects
  -> transform into the map frame
  -> persistent semantic object graph and RViz markers
```

Current ROS outputs include:

- `/concept_graph/objects/detections_3d_json`
- `/concept_graph/objects/detections_map_json`
- `/concept_graph/graph/json`
- `/concept_graph/graph/markers`
- `/rtabmap/cloud_map`

The graph uses schema version 2. Its `edges` contain deterministic map-frame
spatial relations: `near`, `left_of/right_of`, and `above/below`. Directional
relations explicitly follow the map X and Z axes rather than a viewer's
temporary camera perspective.

## Quick start

The tested native target is x86_64 Ubuntu 24.04 with ROS 2 Jazzy. For the full
compatibility table, CUDA/CPU choices, RealSense requirements, Docker option,
DDS configuration, and verification commands, read
[`ros2_ws/README.md`](ros2_ws/README.md).

```bash
cd ros2_ws
./scripts/bootstrap_native.sh

source scripts/ros_env.sh
source install/setup.bash
export CG_AI_SITE_PACKAGES="$PWD/.venv/lib/python3.12/site-packages"
export CG_MODEL_PATH="$HOME/concept_graph_models/yolo26n-seg.pt"
./scripts/check_environment.sh

./scripts/run_d455_mapping.sh \
  database_path:="$HOME/concept_graph_data/rtabmap/d455_lab.db" \
  delete_db_on_start:=true \
  rtabmap_viz:=true
```

Local testing uses CycloneDDS with `ROS_DOMAIN_ID=73` and localhost-only
discovery. Before deploying across the camera computer and AMR, change the
discovery range to `SUBNET` on both machines as documented in the ROS workspace
guide.

## Current limitations

- Spatial relation edges currently use axis-aligned 3D bounding boxes and fixed
  thresholds; semantic support relations such as `on` and `inside` are not yet
  generated.
- Duplicate object nodes may appear after drift, occlusion, or loop closure.
- Dynamic objects such as people are not filtered from the persistent graph.
- The graph is not yet exported to a durable graph database.
- Nav2 `PoseStamped` goals are not generated yet.

## Original project attribution

The original ConceptGraphs source, authorship, paper links, and MIT license are
kept in [`upstream_concept_graphs/`](upstream_concept_graphs/). The upstream
project is available at <https://github.com/concept-graphs/concept-graphs>.
