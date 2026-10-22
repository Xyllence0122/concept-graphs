ARG ROS_IMAGE=ros:jazzy-ros-base-noble
FROM ${ROS_IMAGE}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=jazzy
ENV ROS_DOMAIN_ID=73
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ENV ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST

COPY environment/apt-packages.txt /tmp/apt-packages.txt
RUN apt-get update \
    && sed -e 's/[[:space:]]*#.*$//' -e '/^[[:space:]]*$/d' \
        /tmp/apt-packages.txt \
        | xargs apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

COPY environment/requirements-torch-cu128.txt /tmp/requirements-torch.txt
COPY environment/requirements-ai.txt /tmp/requirements-ai.txt
RUN python3 -m venv --system-site-packages /opt/cg-ai \
    && /opt/cg-ai/bin/python -m pip install \
        pip==26.2.1 setuptools==78.1.0 wheel==0.48.0 \
    && /opt/cg-ai/bin/python -m pip install -r /tmp/requirements-torch.txt \
    && /opt/cg-ai/bin/python -m pip install -r /tmp/requirements-ai.txt

ENV CG_AI_SITE_PACKAGES=/opt/cg-ai/lib/python3.12/site-packages
ENV PYTHONPATH=/opt/cg-ai/lib/python3.12/site-packages
ENV CG_MODEL_PATH=/models/yolo26n-seg.pt

WORKDIR /workspace/ros2_ws
COPY src /workspace/ros2_ws/src
RUN source /opt/ros/jazzy/setup.bash \
    && (rosdep init 2>/dev/null || true) \
    && rosdep update --rosdistro jazzy \
    && rosdep install \
        --from-paths src --ignore-src --rosdistro jazzy -r -y \
    && colcon build --symlink-install \
        --packages-select cg_depth_projection cg_object_segmentation \
    && colcon test \
        --packages-select cg_depth_projection cg_object_segmentation \
    && colcon test-result --verbose

COPY docker/entrypoint.sh /ros_entrypoint.sh
RUN chmod +x /ros_entrypoint.sh
ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
