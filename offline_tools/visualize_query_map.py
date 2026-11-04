#!/usr/bin/env python3
"""Interactive ConceptGraphs-style text-query viewer for local object maps."""

from __future__ import annotations

import argparse
import colorsys
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from query_object_map import (
    clip_similarities,
    label_similarity,
    load_objects,
    most_common_label,
)


HIGHLIGHT_COLOR = np.array([1.0, 0.82, 0.05])
SUPPORT_COLOR = np.array([0.05, 0.82, 0.78])


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Show a reconstructed scene and highlight object-map results for "
            "label, CLIP, or stored relation queries."
        )
    )
    parser.add_argument("--map-file", type=Path, required=True)
    parser.add_argument("--scene-cloud", type=Path)
    parser.add_argument("--relations-file", type=Path)
    parser.add_argument(
        "--mode",
        choices=("auto", "label", "clip"),
        default="auto",
        help="Auto resolves stored relations first, then uses label matching.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--query",
        default="find the panda plush toy on the table",
    )
    parser.add_argument("--point-size", type=float, default=4.0)
    parser.add_argument("--scene-point-size", type=float, default=2.0)
    parser.add_argument("--scene-voxel-size", type=float, default=0.025)
    parser.add_argument("--width", type=int, default=1440)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--screenshot-path", type=Path)
    return parser.parse_args()


def normalize_text(value: str) -> str:
    return " ".join(value.lower().split())


def load_relations(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"relations file does not exist: {resolved}")
    data = json.loads(resolved.read_text(encoding="utf-8"))
    relations = data.get("relations", data) if isinstance(data, dict) else data
    if not isinstance(relations, list):
        raise ValueError("relations file must contain a relation list")
    return [relation for relation in relations if relation.get("accepted", True)]


def object_index(value: Any) -> int:
    text = str(value)
    if text.startswith("object_"):
        return int(text.split("_", 1)[1])
    return int(text)


def label_is_in_query(label: str, query: str) -> bool:
    label_normalized = normalize_text(label)
    query_normalized = normalize_text(query)
    if label_normalized in query_normalized:
        return True
    aliases = {
        "panda plush toy": ("panda", "panda toy", "熊貓玩偶", "熊猫玩偶"),
        "table": ("table", "desk", "桌子", "桌上"),
    }
    return any(alias in query_normalized for alias in aliases.get(label_normalized, ()))


def relation_is_in_query(relation: str, query: str) -> bool:
    query_normalized = normalize_text(query)
    aliases = {
        "on": (" on ", "on top of", "桌上", "上面的", "上的"),
    }
    padded_query = f" {query_normalized} "
    return any(alias in padded_query for alias in aliases.get(relation, (relation,)))


@dataclass
class QueryResult:
    object_id: int
    label: str
    score: float
    method: str
    support_id: int | None = None
    support_label: str | None = None
    relation: dict[str, Any] | None = None
    scores: list[float] | None = None


class QueryEngine:
    def __init__(
        self,
        objects: list[dict[str, Any]],
        relations: list[dict[str, Any]],
        mode: str,
        device: str,
    ) -> None:
        self.objects = objects
        self.labels = [most_common_label(obj) for obj in objects]
        self.relations = relations
        self.mode = mode
        self.device = device

    def _relation_result(self, query: str) -> QueryResult | None:
        for relation in self.relations:
            source_label = str(relation.get("source_label", ""))
            target_label = str(relation.get("target_label", ""))
            relation_name = str(relation.get("relation", ""))
            if (
                label_is_in_query(source_label, query)
                and label_is_in_query(target_label, query)
                and relation_is_in_query(relation_name, query)
            ):
                return QueryResult(
                    object_id=object_index(relation["source_id"]),
                    label=source_label,
                    score=1.0,
                    method="relation graph",
                    support_id=object_index(relation["target_id"]),
                    support_label=target_label,
                    relation=relation,
                )
        return None

    def query(self, text: str) -> QueryResult:
        relation_result = self._relation_result(text)
        if relation_result is not None:
            return relation_result

        if self.mode == "clip":
            scores = clip_similarities(
                text,
                self.objects,
                self.device,
                "ViT-H-14",
                "laion2b_s32b_b79k",
            )
            method = "CLIP"
        else:
            scores = [label_similarity(text, label) for label in self.labels]
            method = "label"
        best_index = int(np.argmax(scores))
        return QueryResult(
            object_id=best_index,
            label=self.labels[best_index],
            score=float(scores[best_index]),
            method=method,
            scores=[float(score) for score in scores],
        )


class QueryMapWindow:
    def __init__(self, args: argparse.Namespace) -> None:
        try:
            import open3d as o3d
            from open3d.visualization import gui, rendering
        except ImportError as error:
            raise RuntimeError(
                "the query viewer requires Open3D; activate the "
                "conceptgraphs-official environment"
            ) from error

        self.o3d = o3d
        self.gui = gui
        self.rendering = rendering
        self.args = args
        self.objects = load_objects(args.map_file.expanduser().resolve())
        self.labels = [most_common_label(obj) for obj in self.objects]
        self.relations = load_relations(args.relations_file)
        self.engine = QueryEngine(self.objects, self.relations, args.mode, args.device)
        self.object_clouds: list[Any] = []
        self.object_boxes: list[Any] = []
        self.active_relation_name = "active_relation"

        app = gui.Application.instance
        app.initialize()
        self.window = app.create_window(
            "ConceptGraphs text-query viewer",
            args.width,
            args.height,
        )
        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = rendering.Open3DScene(self.window.renderer)
        self.scene_widget.scene.set_background([0.025, 0.03, 0.045, 1.0])

        em = self.window.theme.font_size
        self.panel = gui.Vert(0.6 * em, gui.Margins(em, em, em, em))
        self.panel.add_child(gui.Label("Text queries via CLIP or relation graph"))
        self.query_edit = gui.TextEdit()
        self.query_edit.text_value = args.query
        self.panel.add_child(self.query_edit)
        self.search_button = gui.Button("Find object")
        self.search_button.set_on_clicked(self._on_search)
        self.panel.add_child(self.search_button)
        self.result_label = gui.Label("Ready")
        self.panel.add_child(self.result_label)
        self.evidence_label = gui.Label("")
        self.panel.add_child(self.evidence_label)
        self.screenshot_button = gui.Button("Save screenshot")
        self.screenshot_button.set_on_clicked(self._save_screenshot)
        self.panel.add_child(self.screenshot_button)

        self.window.add_child(self.scene_widget)
        self.window.add_child(self.panel)
        self.window.set_on_layout(self._on_layout)

        self._add_scene_cloud()
        self._build_object_geometries()
        initial_result = self.engine.query(args.query)
        self._render_result(initial_result)
        bounds = self.scene_widget.scene.bounding_box
        self.scene_widget.setup_camera(60.0, bounds, bounds.get_center())

    def _point_material(self, size: float) -> Any:
        material = self.rendering.MaterialRecord()
        material.shader = "defaultUnlit"
        material.point_size = size
        return material

    def _line_material(self, width: float) -> Any:
        material = self.rendering.MaterialRecord()
        material.shader = "unlitLine"
        material.line_width = width
        return material

    def _add_scene_cloud(self) -> None:
        if self.args.scene_cloud is None:
            return
        path = self.args.scene_cloud.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"scene cloud does not exist: {path}")
        cloud = self.o3d.io.read_point_cloud(str(path))
        if self.args.scene_voxel_size > 0:
            cloud = cloud.voxel_down_sample(self.args.scene_voxel_size)
        colors = np.asarray(cloud.colors)
        if len(colors) == len(cloud.points):
            cloud.colors = self.o3d.utility.Vector3dVector(
                np.clip(colors * 0.42 + 0.04, 0.0, 1.0)
            )
        self.scene_widget.scene.add_geometry(
            "rgb_scene",
            cloud,
            self._point_material(self.args.scene_point_size),
        )

    def _build_object_geometries(self) -> None:
        for obj in self.objects:
            points = np.asarray(obj["pcd_np"], dtype=np.float64)
            cloud = self.o3d.geometry.PointCloud(
                self.o3d.utility.Vector3dVector(points)
            )
            self.object_clouds.append(cloud)
            self.object_boxes.append(cloud.get_axis_aligned_bounding_box())

    def _colored_cloud(self, index: int, color: np.ndarray) -> Any:
        cloud = self.o3d.geometry.PointCloud(self.object_clouds[index])
        cloud.paint_uniform_color(color.tolist())
        return cloud

    def _instance_color(self, index: int) -> np.ndarray:
        # A golden-ratio hue step keeps adjacent object ids visually distinct
        # without requiring a fixed class palette.
        hue = (0.58 + index * 0.61803398875) % 1.0
        return np.asarray(colorsys.hsv_to_rgb(hue, 0.68, 0.82))

    def _colored_box(self, index: int, color: np.ndarray) -> Any:
        box = self.o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(
            self.object_boxes[index]
        )
        box.paint_uniform_color(color.tolist())
        return box

    def _replace_object(self, index: int, color: np.ndarray) -> None:
        cloud_name = f"object_{index:04d}"
        box_name = f"bbox_{index:04d}"
        for name in (cloud_name, box_name):
            if self.scene_widget.scene.has_geometry(name):
                self.scene_widget.scene.remove_geometry(name)
        self.scene_widget.scene.add_geometry(
            cloud_name,
            self._colored_cloud(index, color),
            self._point_material(self.args.point_size),
        )
        self.scene_widget.scene.add_geometry(
            box_name,
            self._colored_box(index, color),
            self._line_material(2.0),
        )

    def _render_relation(self, result: QueryResult) -> None:
        if self.scene_widget.scene.has_geometry(self.active_relation_name):
            self.scene_widget.scene.remove_geometry(self.active_relation_name)
        if result.support_id is None:
            return
        source_center = np.asarray(
            self.object_clouds[result.object_id].get_center(), dtype=np.float64
        )
        target_center = np.asarray(
            self.object_clouds[result.support_id].get_center(), dtype=np.float64
        )
        line = self.o3d.geometry.LineSet(
            points=self.o3d.utility.Vector3dVector([source_center, target_center]),
            lines=self.o3d.utility.Vector2iVector([[0, 1]]),
        )
        line.paint_uniform_color(HIGHLIGHT_COLOR.tolist())
        self.scene_widget.scene.add_geometry(
            self.active_relation_name,
            line,
            self._line_material(4.0),
        )

    def _render_result(self, result: QueryResult) -> None:
        similarity_colors = None
        if result.method == "CLIP" and result.scores is not None:
            import matplotlib

            scores = np.asarray(result.scores, dtype=np.float64)
            score_range = float(np.ptp(scores))
            if score_range > 1e-12:
                normalized = (scores - scores.min()) / score_range
            else:
                normalized = np.ones_like(scores)
            similarity_colors = matplotlib.colormaps["viridis"](normalized)[:, :3]

        for index in range(len(self.objects)):
            color = (
                similarity_colors[index]
                if similarity_colors is not None
                else self._instance_color(index)
            )
            if index == result.support_id:
                color = SUPPORT_COLOR
            if index == result.object_id and similarity_colors is None:
                color = HIGHLIGHT_COLOR
            self._replace_object(index, color)
        self._render_relation(result)

        if result.relation is None:
            self.result_label.text = (
                f"Match: {result.label}  |  {result.method} score {result.score:.3f}"
            )
            self.evidence_label.text = "No stored spatial relation was required."
            return

        evidence = result.relation.get("evidence", {})
        distances = evidence.get("subject_plane_distance_m", {})
        gap_mm = 1000.0 * float(distances.get("q05_contact_gap", 0.0))
        overlap = 100.0 * float(evidence.get("footprint_overlap_ratio", 0.0))
        self.result_label.text = (
            f"Match: {result.label} --on--> {result.support_label}"
        )
        self.evidence_label.text = (
            f"Relation graph | contact gap {gap_mm:.1f} mm | "
            f"support overlap {overlap:.1f}%"
        )

    def _on_search(self) -> None:
        query = self.query_edit.text_value.strip()
        if not query:
            self.result_label.text = "Enter a query."
            return
        try:
            self._render_result(self.engine.query(query))
        except Exception as error:
            self.result_label.text = f"Query failed: {error}"

    def _default_screenshot_path(self) -> Path:
        if self.args.screenshot_path is not None:
            return self.args.screenshot_path.expanduser().resolve()
        return (
            self.args.map_file.expanduser().resolve().parent.parent
            / "visualizations"
            / "text_query.png"
        )

    def _save_screenshot(self) -> None:
        output_path = self._default_screenshot_path()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        def save_image(image: Any) -> None:
            self.o3d.io.write_image(str(output_path), image)
            self.result_label.text = f"Saved: {output_path}"

        self.scene_widget.scene.scene.render_to_image(save_image)

    def _on_layout(self, _layout_context: Any) -> None:
        content = self.window.content_rect
        panel_width = min(int(30 * self.window.theme.font_size), content.width // 2)
        self.scene_widget.frame = self.gui.Rect(
            content.x,
            content.y,
            content.width - panel_width,
            content.height,
        )
        self.panel.frame = self.gui.Rect(
            content.get_right() - panel_width,
            content.y,
            panel_width,
            content.height,
        )

    def run(self) -> None:
        self.gui.Application.instance.run()


def main() -> None:
    args = parse_arguments()
    if args.point_size <= 0 or args.scene_point_size <= 0:
        raise ValueError("point sizes must be positive")
    QueryMapWindow(args).run()


if __name__ == "__main__":
    main()
