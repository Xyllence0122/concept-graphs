# ROS 2 D455 semantic-mapping prototype

This directory packages the ROS 2 work used to turn Intel RealSense D455 RGB-D
frames into persistent, map-frame semantic object nodes.

The original paper implementation is kept separately in
[`../upstream_concept_graphs/`](../upstream_concept_graphs/). It is not a build
dependency of this ROS 2 workspace and uses a different Python/CUDA environment.

The current data flow is:

```text
D455 RGB + aligned depth
  -> RTAB-Map visual odometry / map TF
  -> YOLO instance masks
  -> mask + depth back-projection
  -> camera-frame 3D objects
  -> map-frame 3D objects
  -> persistent JSON graph + RViz markers
```

This is a reproducible prototype, not yet the complete ConceptGraphs paper
pipeline. It generates deterministic geometric relations (`near`,
`left_of/right_of`, and `above/below`), but not learned or language-derived
relations.

## Tested compatibility baseline

The exact machine snapshot is recorded in
[`environment/versions-tested.yaml`](environment/versions-tested.yaml).

| Layer | Tested value |
| --- | --- |
| CPU architecture | x86_64 |
| OS | Ubuntu 24.04 |
| ROS | ROS 2 Jazzy |
| DDS | CycloneDDS, `ROS_DOMAIN_ID=73` |
| Camera | Intel RealSense D455, firmware 5.17.3.10 |
| librealsense | 2.58.4 |
| realsense-ros | 4.58.1 |
| RTAB-Map ROS | 0.23.13 |
| Python | 3.12.3 |
| PyTorch | 2.11.0 + CUDA 12.8 wheel |
| Ultralytics | 8.4.154 |
| GPU used for validation | RTX 4060 Laptop, driver 595.99.02 |

The lock files target x86_64. Do not use the CUDA wheel lock unchanged on an
ARM64/Jetson target. A target-specific PyTorch build is required there.

## What is and is not stored in Git

Git contains source code, tests, launch files, dependency manifests, a Docker
recipe, environment checks, and the tested version snapshot.

Git intentionally excludes generated `build/`, `install/`, and `log/`
directories, virtual environments, rosbag files, RTAB-Map databases, model
weights, and map data. These are machine-specific or large runtime artifacts;
copying a virtual environment between devices is not a valid installation
method because its interpreter paths and compiled wheels are platform-bound.

The tested model artifact is:

```text
yolo26n-seg.pt
size:   6719965 bytes
sha256: 361fbfabab285c3237700b6bb91d7ecfa602cd945fffda8dbe1242829b71e73f
```

Place that file at `~/concept_graph_models/yolo26n-seg.pt`, or set
`CG_MODEL_PATH` to an absolute path.

## Native installation

Host requirements:

- Ubuntu 24.04 on x86_64
- a working D455 host installation and USB permissions
- an NVIDIA driver compatible with the CUDA 12.8 PyTorch wheel, or CPU mode
- network access while installing packages

From the repository root:

```bash
cd ros2_ws
./scripts/bootstrap_native.sh
```

For a CPU-only target:

```bash
cd ros2_ws
CG_TORCH_FLAVOR=cpu ./scripts/bootstrap_native.sh
```

The bootstrap script installs the package names in
`environment/apt-packages.txt`, builds an isolated `.venv`, installs the pinned
Python packages, resolves package.xml dependencies with rosdep, and runs every
unit test.

Before each run:

```bash
cd ros2_ws
source scripts/ros_env.sh
source install/setup.bash
export CG_AI_SITE_PACKAGES="$PWD/.venv/lib/python3.12/site-packages"
export CG_MODEL_PATH="$HOME/concept_graph_models/yolo26n-seg.pt"
./scripts/check_environment.sh
```

Start the complete D455 + RTAB-Map + semantic pipeline:

```bash
./scripts/run_d455_mapping.sh \
  database_path:="$HOME/concept_graph_data/rtabmap/d455_lab.db" \
  delete_db_on_start:=true \
  rtabmap_viz:=true
```

Set `delete_db_on_start:=false` after the initial mapping run when the existing
database must be preserved. To replay a rosbag instead of starting the camera,
pass `start_camera:=false` and play the bag in another terminal with the same
ROS environment.

## Docker installation

Docker improves user-space reproducibility but cannot package the host kernel,
NVIDIA driver, RealSense USB permissions, or firmware. Install Docker Engine
and the NVIDIA Container Toolkit on the host first. The Compose configuration
uses host networking for DDS and privileged device access for RealSense; only
use it on a trusted machine.

```bash
cd ros2_ws
mkdir -p models data
cp /absolute/path/to/yolo26n-seg.pt models/
docker compose build
docker compose run --rm concept-graph
```

Inside the container:

```bash
ros2 launch cg_object_segmentation d455_mapping.launch.py \
  yolo_model_path:=/models/yolo26n-seg.pt \
  yolo_ai_site_packages:=/opt/cg-ai/lib/python3.12/site-packages \
  database_path:=/data/d455_lab.db
```

The official ROS image tag used by the Dockerfile is
`ros:jazzy-ros-base-noble`.

## Single-machine and AMR networking

For local testing, `scripts/ros_env.sh` defaults to:

```bash
ROS_DOMAIN_ID=73
RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
```

When the camera computer and AMR are different machines, use the same domain
and RMW implementation on both, but change discovery to `SUBNET` before
sourcing the script:

```bash
export ROS_DOMAIN_ID=73
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET
source scripts/ros_env.sh
```

Both machines must permit ROS 2 DDS multicast/UDP traffic. `LOCALHOST` is
deliberately unsuitable for communication between two computers.

## Outputs and verification

Important outputs:

| Topic | Meaning |
| --- | --- |
| `/concept_graph/debug/segmentation_image` | YOLO visualization |
| `/concept_graph/objects/detections_3d_json` | camera-frame 3D detections |
| `/concept_graph/objects/detections_map_json` | map-frame 3D detections |
| `/concept_graph/graph/json` | persistent semantic object graph |
| `/concept_graph/graph/markers` | graph markers for RViz |
| `/rtabmap/cloud_map` | RTAB-Map point cloud map |

Spatial relation thresholds are configured in
`src/cg_object_segmentation/config/object_graph.yaml`. Directional relations
use the map X axis for `left_of/right_of` and the map Z axis for
`above/below`; they are not relative to the current camera view.

Useful checks:

```bash
ros2 topic hz /cg/d455/color/image_raw
ros2 topic hz /cg/d455/aligned_depth_to_color/image_raw
ros2 topic hz /rtabmap/odom
ros2 run tf2_ros tf2_echo map d455_link
ros2 topic echo /concept_graph/objects/detections_map_json --once --full-length
ros2 topic echo /concept_graph/graph/json --once --full-length
```

## Known limitations

- Spatial edges are based on axis-aligned bounding boxes and fixed thresholds;
  `on`, `inside`, and learned semantic relations are not built yet.
- Association is greedy label-plus-distance tracking, so duplicate object nodes
  can appear after drift, occlusion, or loop closure.
- Dynamic classes such as `person` are not filtered from the persistent map.
- The map transform currently composes the latest `map -> odom` and
  `odom -> sensor` transforms. It avoids timestamp extrapolation failures but
  is not yet a motion-compensated transform at the image timestamp.
- The graph is published in memory and is not yet exported to a durable graph
  database.
- No Nav2 `PoseStamped` goal is produced yet. That requires object-to-safe-goal
  projection on the navigation costmap.

## Upstream references

- [ROS 2 Jazzy documentation](https://docs.ros.org/en/jazzy/)
- [Official ROS Docker image](https://hub.docker.com/_/ros)
- [RealSense ROS wrapper](https://github.com/realsenseai/realsense-ros)
- [RTAB-Map ROS](https://github.com/introlab/rtabmap_ros)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- [PyTorch version installation matrix](https://pytorch.org/get-started/previous-versions/)
