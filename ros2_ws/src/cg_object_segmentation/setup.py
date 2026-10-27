from glob import glob
import os

from setuptools import find_packages, setup


package_name = "cg_object_segmentation"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        ("share/" + package_name, ["package.xml"]),
        (
            os.path.join("share", package_name, "launch"),
            glob("launch/*.launch.py"),
        ),
        (
            os.path.join("share", package_name, "config"),
            glob("config/*.yaml"),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Concept Graph Developer",
    maintainer_email="user@example.com",
    description="GPU instance segmentation for the Concept Graph prototype.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "grounded_sam_segmentation_node = "
            "cg_object_segmentation.grounded_sam_segmentation_node:main",
            "yolo_segmentation_node = "
            "cg_object_segmentation.yolo_segmentation_node:main",
            "object_3d_projection_node = "
            "cg_object_segmentation.object_3d_projection_node:main",
            "object_map_transform_node = "
            "cg_object_segmentation.object_map_transform_node:main",
            "object_graph_node = "
            "cg_object_segmentation.object_graph_node:main",
        ],
    },
)
