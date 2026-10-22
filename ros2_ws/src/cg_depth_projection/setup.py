from glob import glob
from setuptools import find_packages, setup


package_name = "cg_depth_projection"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Concept Graph Developer",
    maintainer_email="user@example.com",
    description="Aligned-depth ROI to 3D point-cloud smoke test.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "depth_roi_node = cg_depth_projection.depth_roi_node:main",
        ],
    },
)
