#!/usr/bin/env python
"""STEP/STP geometry analyzer using OpenCascade via cadquery-ocp.

The analyzer reads exact B-Rep geometry from STEP files and produces JSON,
CSV, and HTML reports with part dimensions and inferred feature candidates.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

BUILD123D_PYTHON = Path(r"D:\miniconda\envs\build123d_env\python.exe")

try:
    from OCP.Bnd import Bnd_Box
    from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
    from OCP.BRepBndLib import BRepBndLib
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.BRepGProp import BRepGProp
    from OCP.BRep import BRep_Tool
    from OCP.BRepTools import BRepTools
    from OCP.BRepTools import BRepTools_WireExplorer
    from OCP.GProp import GProp_GProps
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPControl import STEPControl_Reader
    from OCP.TopAbs import (
        TopAbs_COMPOUND,
        TopAbs_COMPSOLID,
        TopAbs_EDGE,
        TopAbs_FACE,
        TopAbs_REVERSED,
        TopAbs_SHELL,
        TopAbs_SOLID,
        TopAbs_VERTEX,
        TopAbs_WIRE,
    )
    from OCP.TopExp import TopExp, TopExp_Explorer
    from OCP.TopTools import TopTools_IndexedMapOfShape
    from OCP.TopoDS import TopoDS, TopoDS_Shape
    from OCP.gp import gp_Pnt, gp_Vec
except ImportError as exc:  # pragma: no cover - exercised by users without OCP.
    print(
        "This tool needs OpenCascade Python bindings from cadquery-ocp.\n"
        "Run it with the existing conda environment:\n"
        "  conda run -n build123d_env python step_analyzer.py your_file.step\n"
        f"\nOriginal import error: {exc}",
        file=sys.stderr,
    )
    raise SystemExit(2) from exc


ROUND_DIGITS = 6
FULL_CIRCLE = math.tau
FULL_CIRCLE_TOLERANCE = math.radians(1.0)


@dataclass
class BoundingBox:
    min_x: float
    min_y: float
    min_z: float
    max_x: float
    max_y: float
    max_z: float
    length_x: float
    length_y: float
    length_z: float
    diagonal: float
    center: dict[str, float]


@dataclass
class MassProperties:
    volume: float
    surface_area: float
    center_of_mass: dict[str, float]


@dataclass
class FaceInfo:
    face_id: str
    surface_type: str
    area: float
    orientation: str
    center: dict[str, float] | None = None
    u_span_degrees: float | None = None
    v_span: float | None = None
    radius: float | None = None
    diameter: float | None = None
    height: float | None = None
    axis_origin: dict[str, float] | None = None
    axis_direction: dict[str, float] | None = None
    normal: dict[str, float] | None = None
    plane_origin: dict[str, float] | None = None
    plane_x_axis: dict[str, float] | None = None
    plane_y_axis: dict[str, float] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class EdgeInfo:
    edge_id: str
    curve_type: str
    length: float
    radius: float | None = None
    diameter: float | None = None
    angle_degrees: float | None = None


@dataclass
class PlanarProfileEdge:
    edge_index: int
    curve_type: str
    length: float
    start_2d: dict[str, float]
    end_2d: dict[str, float]
    start_3d: dict[str, float]
    end_3d: dict[str, float]
    angle_degrees: float | None = None
    radius: float | None = None
    diameter: float | None = None
    center_2d: dict[str, float] | None = None
    center_3d: dict[str, float] | None = None
    angular_span_degrees: float | None = None


@dataclass
class PlanarLoop:
    loop_id: str
    shape_guess: str
    edge_count: int
    vertex_count: int
    closed: bool
    signed_area_2d: float | None
    area_2d: float | None
    perimeter: float
    bbox_2d: dict[str, float]
    vertices_2d: list[dict[str, float]]
    vertices_3d: list[dict[str, float]]
    edges: list[PlanarProfileEdge]


@dataclass
class PlanarProfile:
    face_id: str
    surface_type: str
    area: float
    center_3d: dict[str, float]
    normal: dict[str, float]
    x_axis: dict[str, float]
    y_axis: dict[str, float]
    loop_count: int
    outer_loop_id: str | None
    outer_shape_guess: str | None
    outer_width: float | None
    outer_height: float | None
    outer_area_2d: float | None
    loops: list[PlanarLoop]
    notes: str = ""


@dataclass
class FeatureInfo:
    part_id: str
    feature_id: str
    source_face_id: str | None
    category: str
    confidence: str
    radius: float | None = None
    diameter: float | None = None
    height: float | None = None
    area: float | None = None
    orientation: str | None = None
    axis_origin: dict[str, float] | None = None
    axis_direction: dict[str, float] | None = None
    notes: str = ""


@dataclass
class PartReport:
    part_id: str
    name: str
    is_valid: bool
    bounding_box: BoundingBox
    mass_properties: MassProperties
    topology_counts: dict[str, int]
    face_type_counts: dict[str, int]
    edge_type_counts: dict[str, int]
    faces: list[FaceInfo]
    edges: list[EdgeInfo]
    planar_profiles: list[PlanarProfile]
    features: list[FeatureInfo]


@dataclass
class AnalysisReport:
    source_file: str
    detected_step_units: list[str]
    step_metadata: dict[str, Any]
    import_system_length_unit: float
    topology_counts: dict[str, int]
    part_count: int
    parts: list[PartReport]
    notes: list[str]


def rounded(value: float | None, digits: int = ROUND_DIGITS) -> float | None:
    if value is None:
        return None
    if abs(value) < 10 ** (-(digits + 1)):
        value = 0.0
    return round(float(value), digits)


def xyz(point_or_dir: Any) -> dict[str, float]:
    return {
        "x": rounded(point_or_dir.X()),
        "y": rounded(point_or_dir.Y()),
        "z": rounded(point_or_dir.Z()),
    }


def scaled_xyz(point_or_dir: Any, scale: float = 1.0) -> dict[str, float]:
    return {
        "x": rounded(point_or_dir.X() * scale),
        "y": rounded(point_or_dir.Y() * scale),
        "z": rounded(point_or_dir.Z() * scale),
    }


def xy(u: float, v: float) -> dict[str, float]:
    return {"x": rounded(u), "y": rounded(v)}


def enum_name(value: Any) -> str:
    return getattr(value, "name", str(value))


def unique_shapes(shape: TopoDS_Shape, shape_type: Any) -> list[TopoDS_Shape]:
    indexed = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(shape, shape_type, indexed)
    return [indexed.FindKey(i) for i in range(1, indexed.Extent() + 1)]


def count_topology(shape: TopoDS_Shape) -> dict[str, int]:
    entries = [
        ("compounds", TopAbs_COMPOUND),
        ("compsolids", TopAbs_COMPSOLID),
        ("solids", TopAbs_SOLID),
        ("shells", TopAbs_SHELL),
        ("faces", TopAbs_FACE),
        ("wires", TopAbs_WIRE),
        ("edges", TopAbs_EDGE),
        ("vertices", TopAbs_VERTEX),
    ]
    return {name: len(unique_shapes(shape, shape_type)) for name, shape_type in entries}


def shape_bounding_box(shape: TopoDS_Shape) -> BoundingBox:
    box = Bnd_Box()
    BRepBndLib.AddOptimal_s(shape, box)
    min_x, min_y, min_z, max_x, max_y, max_z = box.Get()
    length_x = max_x - min_x
    length_y = max_y - min_y
    length_z = max_z - min_z
    diagonal = math.sqrt(length_x**2 + length_y**2 + length_z**2)
    center = {
        "x": rounded((min_x + max_x) / 2),
        "y": rounded((min_y + max_y) / 2),
        "z": rounded((min_z + max_z) / 2),
    }
    return BoundingBox(
        min_x=rounded(min_x),
        min_y=rounded(min_y),
        min_z=rounded(min_z),
        max_x=rounded(max_x),
        max_y=rounded(max_y),
        max_z=rounded(max_z),
        length_x=rounded(length_x),
        length_y=rounded(length_y),
        length_z=rounded(length_z),
        diagonal=rounded(diagonal),
        center=center,
    )


def mass_properties(shape: TopoDS_Shape) -> MassProperties:
    volume_props = GProp_GProps()
    surface_props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, volume_props)
    BRepGProp.SurfaceProperties_s(shape, surface_props)
    center = volume_props.CentreOfMass()
    return MassProperties(
        volume=rounded(volume_props.Mass()),
        surface_area=rounded(surface_props.Mass()),
        center_of_mass=xyz(center),
    )


def face_area(face: TopoDS_Shape) -> float:
    props = GProp_GProps()
    BRepGProp.SurfaceProperties_s(face, props)
    return float(props.Mass())


def face_area_center(face: TopoDS_Shape) -> tuple[float, dict[str, float]]:
    props = GProp_GProps()
    BRepGProp.SurfaceProperties_s(face, props)
    return float(props.Mass()), xyz(props.CentreOfMass())


def edge_length(edge: TopoDS_Shape) -> float:
    props = GProp_GProps()
    BRepGProp.LinearProperties_s(edge, props)
    return float(props.Mass())


def parameter_bounds(face: TopoDS_Shape) -> tuple[float, float, float, float]:
    try:
        return BRepTools.UVBounds_s(TopoDS.Face_s(face))
    except Exception:
        return (0.0, 0.0, 0.0, 0.0)


def classify_cylindrical_face(face: FaceInfo) -> tuple[str, str, str]:
    angle = math.radians(face.u_span_degrees or 0.0)
    is_full_cylinder = abs(angle - FULL_CIRCLE) <= FULL_CIRCLE_TOLERANCE
    is_reversed = face.orientation == enum_name(TopAbs_REVERSED)

    if is_full_cylinder and is_reversed:
        return (
            "cylindrical_hole_candidate",
            "high",
            "Full cylindrical face with reversed orientation; likely a cut/hole wall.",
        )
    if is_full_cylinder:
        return (
            "outer_cylinder_or_boss_candidate",
            "high",
            "Full cylindrical face with outward orientation; likely an outside cylinder or boss.",
        )
    if is_reversed:
        return (
            "internal_fillet_or_round_candidate",
            "medium",
            "Partial cylindrical face with reversed orientation; likely an internal round.",
        )
    return (
        "external_fillet_or_round_candidate",
        "medium",
        "Partial cylindrical face; likely a fillet, round, or partial cylindrical feature.",
    )


def analyze_face(face: TopoDS_Shape, face_id: str) -> FaceInfo:
    topo_face = TopoDS.Face_s(face)
    surface = BRepAdaptor_Surface(topo_face)
    surface_type = enum_name(surface.GetType()).replace("GeomAbs_", "")
    area, center = face_area_center(topo_face)
    u_min, u_max, v_min, v_max = parameter_bounds(topo_face)
    u_span = abs(u_max - u_min)
    v_span = abs(v_max - v_min)
    orientation = enum_name(topo_face.Orientation())

    info = FaceInfo(
        face_id=face_id,
        surface_type=surface_type,
        area=rounded(area),
        orientation=orientation,
        center=center,
        u_span_degrees=rounded(math.degrees(u_span)),
        v_span=rounded(v_span),
    )

    try:
        if surface_type == "Plane":
            plane = surface.Plane()
            normal_scale = -1.0 if orientation == enum_name(TopAbs_REVERSED) else 1.0
            info.normal = scaled_xyz(plane.Axis().Direction(), normal_scale)
            info.plane_origin = xyz(plane.Location())
            info.plane_x_axis = xyz(plane.XAxis().Direction())
            info.plane_y_axis = xyz(plane.YAxis().Direction())
        elif surface_type == "Cylinder":
            cylinder = surface.Cylinder()
            radius = float(cylinder.Radius())
            info.radius = rounded(radius)
            info.diameter = rounded(radius * 2)
            info.height = rounded(v_span)
            info.axis_origin = xyz(cylinder.Axis().Location())
            info.axis_direction = xyz(cylinder.Axis().Direction())
        elif surface_type == "Cone":
            cone = surface.Cone()
            radius = float(cone.RefRadius())
            info.radius = rounded(radius)
            info.diameter = rounded(radius * 2)
            info.axis_origin = xyz(cone.Axis().Location())
            info.axis_direction = xyz(cone.Axis().Direction())
            info.extra["semi_angle_degrees"] = rounded(math.degrees(cone.SemiAngle()))
        elif surface_type == "Sphere":
            sphere = surface.Sphere()
            radius = float(sphere.Radius())
            info.radius = rounded(radius)
            info.diameter = rounded(radius * 2)
            info.axis_origin = xyz(sphere.Location())
        elif surface_type == "Torus":
            torus = surface.Torus()
            info.axis_origin = xyz(torus.Axis().Location())
            info.axis_direction = xyz(torus.Axis().Direction())
            info.extra["major_radius"] = rounded(torus.MajorRadius())
            info.extra["minor_radius"] = rounded(torus.MinorRadius())
    except Exception as exc:
        info.extra["surface_detail_error"] = str(exc)

    return info


def analyze_edge(edge: TopoDS_Shape, edge_id: str) -> EdgeInfo:
    topo_edge = TopoDS.Edge_s(edge)
    curve = BRepAdaptor_Curve(topo_edge)
    curve_type = enum_name(curve.GetType()).replace("GeomAbs_", "")
    length = edge_length(topo_edge)
    first = float(curve.FirstParameter())
    last = float(curve.LastParameter())
    angle = abs(last - first)
    info = EdgeInfo(
        edge_id=edge_id,
        curve_type=curve_type,
        length=rounded(length),
    )

    try:
        if curve_type == "Circle":
            radius = float(curve.Circle().Radius())
            info.radius = rounded(radius)
            info.diameter = rounded(radius * 2)
            info.angle_degrees = rounded(math.degrees(angle))
        elif curve_type == "Ellipse":
            ellipse = curve.Ellipse()
            info.radius = rounded(ellipse.MajorRadius())
            info.diameter = rounded(ellipse.MinorRadius() * 2)
            info.angle_degrees = rounded(math.degrees(angle))
    except Exception:
        pass

    return info


def point_from_dict(point: dict[str, float]) -> gp_Pnt:
    return gp_Pnt(point["x"], point["y"], point["z"])


def direction_vector(direction: Any) -> gp_Vec:
    return gp_Vec(direction.X(), direction.Y(), direction.Z())


def project_point_2d(point: gp_Pnt, origin: gp_Pnt, x_axis: gp_Vec, y_axis: gp_Vec) -> dict[str, float]:
    vector = gp_Vec(origin, point)
    return xy(vector.Dot(x_axis), vector.Dot(y_axis))


def bbox_2d(points: list[dict[str, float]]) -> dict[str, float]:
    if not points:
        return {
            "min_x": 0.0,
            "min_y": 0.0,
            "max_x": 0.0,
            "max_y": 0.0,
            "width": 0.0,
            "height": 0.0,
            "center_x": 0.0,
            "center_y": 0.0,
        }
    xs = [point["x"] for point in points]
    ys = [point["y"] for point in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    return {
        "min_x": rounded(min_x),
        "min_y": rounded(min_y),
        "max_x": rounded(max_x),
        "max_y": rounded(max_y),
        "width": rounded(max_x - min_x),
        "height": rounded(max_y - min_y),
        "center_x": rounded((min_x + max_x) / 2),
        "center_y": rounded((min_y + max_y) / 2),
    }


def signed_polygon_area(points: list[dict[str, float]]) -> float | None:
    if len(points) < 3:
        return None
    total = 0.0
    for index, point in enumerate(points):
        next_point = points[(index + 1) % len(points)]
        total += point["x"] * next_point["y"] - next_point["x"] * point["y"]
    return total / 2


def vector_2d(start: dict[str, float], end: dict[str, float]) -> tuple[float, float]:
    return end["x"] - start["x"], end["y"] - start["y"]


def length_2d(vector: tuple[float, float]) -> float:
    return math.hypot(vector[0], vector[1])


def almost_equal(a: float, b: float, rel_tol: float = 0.02, abs_tol: float = 1e-6) -> bool:
    return abs(a - b) <= max(abs_tol, rel_tol * max(abs(a), abs(b), 1.0))


def are_parallel(v1: tuple[float, float], v2: tuple[float, float], tolerance_degrees: float = 2.0) -> bool:
    l1 = length_2d(v1)
    l2 = length_2d(v2)
    if l1 <= 1e-9 or l2 <= 1e-9:
        return False
    cross = abs(v1[0] * v2[1] - v1[1] * v2[0]) / (l1 * l2)
    return cross <= math.sin(math.radians(tolerance_degrees))


def are_perpendicular(
    v1: tuple[float, float], v2: tuple[float, float], tolerance_degrees: float = 2.0
) -> bool:
    l1 = length_2d(v1)
    l2 = length_2d(v2)
    if l1 <= 1e-9 or l2 <= 1e-9:
        return False
    cosine = abs(v1[0] * v2[0] + v1[1] * v2[1]) / (l1 * l2)
    return cosine <= math.sin(math.radians(tolerance_degrees))


def classify_planar_loop(vertices_2d: list[dict[str, float]], edges: list[PlanarProfileEdge]) -> str:
    if not edges:
        return "empty_loop"

    if (
        len(edges) == 1
        and edges[0].curve_type == "Circle"
        and edges[0].angular_span_degrees is not None
        and abs(edges[0].angular_span_degrees - 360) <= 1
    ):
        return "circle"

    if any(edge.curve_type != "Line" for edge in edges):
        return "curved_loop"

    side_count = len(vertices_2d)
    if side_count == 3:
        return "triangle"
    if side_count != 4:
        return f"polygon_{side_count}_sides"

    vectors = [
        vector_2d(vertices_2d[index], vertices_2d[(index + 1) % side_count])
        for index in range(side_count)
    ]
    lengths = [length_2d(vector) for vector in vectors]
    opposite_parallel = [
        are_parallel(vectors[0], vectors[2]),
        are_parallel(vectors[1], vectors[3]),
    ]
    adjacent_perpendicular = all(
        are_perpendicular(vectors[index], vectors[(index + 1) % side_count])
        for index in range(side_count)
    )
    opposite_lengths_equal = almost_equal(lengths[0], lengths[2]) and almost_equal(
        lengths[1], lengths[3]
    )

    if all(opposite_parallel) and adjacent_perpendicular and opposite_lengths_equal:
        return "rectangle"
    if all(opposite_parallel):
        return "parallelogram"
    if opposite_parallel.count(True) == 1:
        return "trapezium_candidate"
    return "quadrilateral"


def circle_edge_area(edge: PlanarProfileEdge) -> float | None:
    if (
        edge.curve_type == "Circle"
        and edge.radius is not None
        and edge.angular_span_degrees is not None
        and abs(edge.angular_span_degrees - 360) <= 1
    ):
        return math.pi * edge.radius**2
    return None


def edge_profile_info(
    edge: TopoDS_Shape,
    edge_index: int,
    start_point: gp_Pnt,
    end_point: gp_Pnt,
    origin: gp_Pnt,
    x_axis: gp_Vec,
    y_axis: gp_Vec,
) -> PlanarProfileEdge:
    topo_edge = TopoDS.Edge_s(edge)
    curve = BRepAdaptor_Curve(topo_edge)
    curve_type = enum_name(curve.GetType()).replace("GeomAbs_", "")
    start_2d = project_point_2d(start_point, origin, x_axis, y_axis)
    end_2d = project_point_2d(end_point, origin, x_axis, y_axis)
    dx, dy = vector_2d(start_2d, end_2d)
    angle = math.degrees(math.atan2(dy, dx)) if math.hypot(dx, dy) > 1e-9 else None
    first = float(curve.FirstParameter())
    last = float(curve.LastParameter())

    info = PlanarProfileEdge(
        edge_index=edge_index,
        curve_type=curve_type,
        length=rounded(edge_length(topo_edge)),
        start_2d=start_2d,
        end_2d=end_2d,
        start_3d=xyz(start_point),
        end_3d=xyz(end_point),
        angle_degrees=rounded(angle),
    )

    try:
        if curve_type == "Circle":
            circle = curve.Circle()
            center = circle.Location()
            radius = float(circle.Radius())
            info.radius = rounded(radius)
            info.diameter = rounded(radius * 2)
            info.center_3d = xyz(center)
            info.center_2d = project_point_2d(center, origin, x_axis, y_axis)
            info.angular_span_degrees = rounded(math.degrees(abs(last - first)))
        elif curve_type == "Ellipse":
            ellipse = curve.Ellipse()
            center = ellipse.Location()
            info.radius = rounded(ellipse.MajorRadius())
            info.diameter = rounded(ellipse.MinorRadius() * 2)
            info.center_3d = xyz(center)
            info.center_2d = project_point_2d(center, origin, x_axis, y_axis)
            info.angular_span_degrees = rounded(math.degrees(abs(last - first)))
    except Exception:
        pass

    return info


def analyze_planar_loop(
    wire: TopoDS_Shape,
    face: TopoDS_Shape,
    loop_id: str,
    origin: gp_Pnt,
    x_axis: gp_Vec,
    y_axis: gp_Vec,
) -> PlanarLoop:
    wire_explorer = BRepTools_WireExplorer(TopoDS.Wire_s(wire), TopoDS.Face_s(face))
    edges: list[TopoDS_Shape] = []
    start_points: list[gp_Pnt] = []

    while wire_explorer.More():
        edge = TopoDS.Edge_s(wire_explorer.Current())
        vertex = wire_explorer.CurrentVertex()
        edges.append(edge)
        start_points.append(BRep_Tool.Pnt_s(vertex))
        wire_explorer.Next()

    vertices_2d = [project_point_2d(point, origin, x_axis, y_axis) for point in start_points]
    vertices_3d = [xyz(point) for point in start_points]

    profile_edges: list[PlanarProfileEdge] = []
    for index, edge in enumerate(edges):
        start_point = start_points[index]
        end_point = start_points[(index + 1) % len(start_points)] if start_points else start_point
        profile_edges.append(
            edge_profile_info(edge, index + 1, start_point, end_point, origin, x_axis, y_axis)
        )

    shape_guess = classify_planar_loop(vertices_2d, profile_edges)
    signed_area = signed_polygon_area(vertices_2d)
    circle_area = circle_edge_area(profile_edges[0]) if len(profile_edges) == 1 else None
    area = circle_area if circle_area is not None else abs(signed_area) if signed_area is not None else None
    perimeter = sum(edge.length for edge in profile_edges)

    if circle_area is not None and profile_edges[0].center_2d and profile_edges[0].radius:
        center = profile_edges[0].center_2d
        radius = profile_edges[0].radius
        loop_bbox = {
            "min_x": rounded(center["x"] - radius),
            "min_y": rounded(center["y"] - radius),
            "max_x": rounded(center["x"] + radius),
            "max_y": rounded(center["y"] + radius),
            "width": rounded(radius * 2),
            "height": rounded(radius * 2),
            "center_x": center["x"],
            "center_y": center["y"],
        }
    else:
        loop_bbox = bbox_2d(vertices_2d)

    return PlanarLoop(
        loop_id=loop_id,
        shape_guess=shape_guess,
        edge_count=len(profile_edges),
        vertex_count=len(vertices_2d),
        closed=bool(profile_edges),
        signed_area_2d=rounded(signed_area),
        area_2d=rounded(area),
        perimeter=rounded(perimeter),
        bbox_2d=loop_bbox,
        vertices_2d=vertices_2d,
        vertices_3d=vertices_3d,
        edges=profile_edges,
    )


def analyze_planar_profile(face: TopoDS_Shape, face_info: FaceInfo) -> PlanarProfile | None:
    if face_info.surface_type != "Plane" or face_info.center is None:
        return None

    topo_face = TopoDS.Face_s(face)
    surface = BRepAdaptor_Surface(topo_face)
    try:
        plane = surface.Plane()
    except Exception:
        return None

    origin = point_from_dict(face_info.center)
    x_axis = direction_vector(plane.XAxis().Direction())
    y_axis = direction_vector(plane.YAxis().Direction())

    loops: list[PlanarLoop] = []
    wire_explorer = TopExp_Explorer(topo_face, TopAbs_WIRE)
    while wire_explorer.More():
        wire = TopoDS.Wire_s(wire_explorer.Current())
        loops.append(
            analyze_planar_loop(
                wire,
                topo_face,
                f"{face_info.face_id}_L{len(loops) + 1:02d}",
                origin,
                x_axis,
                y_axis,
            )
        )
        wire_explorer.Next()

    outer_loop = None
    if loops:
        outer_loop = max(
            loops,
            key=lambda loop: (
                loop.area_2d if loop.area_2d is not None else -1,
                loop.bbox_2d.get("width", 0) * loop.bbox_2d.get("height", 0),
            ),
        )

    return PlanarProfile(
        face_id=face_info.face_id,
        surface_type=face_info.surface_type,
        area=face_info.area,
        center_3d=face_info.center,
        normal=face_info.normal or {"x": 0.0, "y": 0.0, "z": 0.0},
        x_axis=face_info.plane_x_axis or xyz(plane.XAxis().Direction()),
        y_axis=face_info.plane_y_axis or xyz(plane.YAxis().Direction()),
        loop_count=len(loops),
        outer_loop_id=outer_loop.loop_id if outer_loop else None,
        outer_shape_guess=outer_loop.shape_guess if outer_loop else None,
        outer_width=outer_loop.bbox_2d.get("width") if outer_loop else None,
        outer_height=outer_loop.bbox_2d.get("height") if outer_loop else None,
        outer_area_2d=outer_loop.area_2d if outer_loop else None,
        loops=loops,
        notes=(
            "2D vertices are measured in this face plane from center_3d using x_axis/y_axis. "
            "Use 3D vertices when you need global arrangement."
        ),
    )


def build_features(part_id: str, faces: list[FaceInfo]) -> list[FeatureInfo]:
    features: list[FeatureInfo] = []
    counters: defaultdict[str, int] = defaultdict(int)

    def next_id(prefix: str) -> str:
        counters[prefix] += 1
        return f"{prefix}{counters[prefix]:03d}"

    for face in faces:
        if face.surface_type == "Cylinder":
            category, confidence, notes = classify_cylindrical_face(face)
            prefix = {
                "cylindrical_hole_candidate": "H",
                "outer_cylinder_or_boss_candidate": "B",
                "internal_fillet_or_round_candidate": "R",
                "external_fillet_or_round_candidate": "R",
            }.get(category, "C")
            features.append(
                FeatureInfo(
                    part_id=part_id,
                    feature_id=next_id(prefix),
                    source_face_id=face.face_id,
                    category=category,
                    confidence=confidence,
                    radius=face.radius,
                    diameter=face.diameter,
                    height=face.height,
                    area=face.area,
                    orientation=face.orientation,
                    axis_origin=face.axis_origin,
                    axis_direction=face.axis_direction,
                    notes=notes,
                )
            )
        elif face.surface_type == "Torus":
            features.append(
                FeatureInfo(
                    part_id=part_id,
                    feature_id=next_id("R"),
                    source_face_id=face.face_id,
                    category="toroidal_fillet_or_round_candidate",
                    confidence="medium",
                    radius=face.extra.get("minor_radius"),
                    diameter=rounded((face.extra.get("minor_radius") or 0) * 2)
                    if face.extra.get("minor_radius") is not None
                    else None,
                    area=face.area,
                    orientation=face.orientation,
                    axis_origin=face.axis_origin,
                    axis_direction=face.axis_direction,
                    notes="Toroidal face; often produced by rounded blends or revolved features.",
                )
            )
        elif face.surface_type == "Cone":
            features.append(
                FeatureInfo(
                    part_id=part_id,
                    feature_id=next_id("K"),
                    source_face_id=face.face_id,
                    category="cone_taper_chamfer_or_countersink_candidate",
                    confidence="medium",
                    radius=face.radius,
                    diameter=face.diameter,
                    area=face.area,
                    orientation=face.orientation,
                    axis_origin=face.axis_origin,
                    axis_direction=face.axis_direction,
                    notes=(
                        "Conical face; inspect as taper, countersink, drafted wall, "
                        "or circular chamfer."
                    ),
                )
            )

    return features


def analyze_part(shape: TopoDS_Shape, part_id: str, name: str) -> PartReport:
    valid = bool(BRepCheck_Analyzer(shape).IsValid())
    face_shapes = unique_shapes(shape, TopAbs_FACE)
    faces = [
        analyze_face(face, f"F{i:04d}")
        for i, face in enumerate(face_shapes, start=1)
    ]
    edges = [
        analyze_edge(edge, f"E{i:04d}")
        for i, edge in enumerate(unique_shapes(shape, TopAbs_EDGE), start=1)
    ]
    planar_profiles = [
        profile
        for profile in (
            analyze_planar_profile(face_shape, face_info)
            for face_shape, face_info in zip(face_shapes, faces)
        )
        if profile is not None
    ]

    return PartReport(
        part_id=part_id,
        name=name,
        is_valid=valid,
        bounding_box=shape_bounding_box(shape),
        mass_properties=mass_properties(shape),
        topology_counts=count_topology(shape),
        face_type_counts=dict(Counter(face.surface_type for face in faces)),
        edge_type_counts=dict(Counter(edge.curve_type for edge in edges)),
        faces=faces,
        edges=edges,
        planar_profiles=planar_profiles,
        features=build_features(part_id, faces),
    )


def parse_step_metadata(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="ignore")

    def step_unescape(value: str) -> str:
        return value.replace("''", "'").strip()

    products = sorted(
        {
            step_unescape(match.group(1))
            for match in re.finditer(r"PRODUCT\s*\(\s*'((?:[^']|'')*)'", text, re.I)
        }
    )
    file_names = [
        step_unescape(match.group(1))
        for match in re.finditer(r"FILE_NAME\s*\(\s*'((?:[^']|'')*)'", text, re.I)
    ]
    schemas = [
        step_unescape(match.group(1))
        for match in re.finditer(r"FILE_SCHEMA\s*\(\s*\(\s*'((?:[^']|'')*)'", text, re.I)
    ]

    unit_tokens = sorted(
        {
            " ".join(token for token in match if token)
            for match in re.findall(
                r"SI_UNIT\s*\(\s*(?:\.([A-Z]+)\.)?\s*,\s*\.([A-Z]+)\.\s*\)",
                text,
                re.I,
            )
        }
    )
    conversion_units = sorted(
        {
            step_unescape(match.group(1))
            for match in re.finditer(
                r"CONVERSION_BASED_UNIT\s*\(\s*'((?:[^']|'')*)'", text, re.I
            )
        }
    )

    return {
        "file_names": file_names,
        "schemas": schemas,
        "products": products,
        "unit_tokens": unit_tokens,
        "conversion_based_units": conversion_units,
    }


def read_step_file(path: Path) -> tuple[TopoDS_Shape, list[str], float]:
    reader = STEPControl_Reader()
    # Keep one OpenCascade unit equal to one millimeter for practical shop reports.
    # STEP files with different units are scaled during transfer by the reader.
    reader.SetSystemLengthUnit(1.0)
    status = reader.ReadFile(str(path))
    if status != IFSelect_RetDone:
        raise RuntimeError(f"OpenCascade could not read STEP file: status={status}")

    units: list[str] = []
    try:
        sequence = reader.FileUnits()
        units = [sequence.Value(i) for i in range(1, sequence.Length() + 1)]
    except Exception:
        units = []

    roots = reader.NbRootsForTransfer()
    if roots == 0:
        raise RuntimeError("STEP file has no transferable roots.")
    transferred = reader.TransferRoots()
    if transferred == 0:
        raise RuntimeError("STEP transfer completed but produced no shapes.")
    return reader.OneShape(), units, float(reader.SystemLengthUnit())


def analyze_step(path: Path) -> AnalysisReport:
    shape, detected_units, system_unit = read_step_file(path)
    metadata = parse_step_metadata(path)
    solids = unique_shapes(shape, TopAbs_SOLID)

    notes = [
        "Measurements come from exact OpenCascade B-Rep geometry, not STL/mesh approximation.",
        "Feature names are inferred from geometry because STEP usually does not preserve the native CAD feature tree.",
        "Cylindrical cut/hole detection uses face orientation and angular span; verify critical manufacturing decisions against the CAD model.",
        "Planar profile loops list face-local 2D vertices plus global 3D vertices for reconstructing rectangles, trapeziums, pads, pockets, and arrangements in build123d.",
    ]

    parts: list[PartReport] = []
    if solids:
        product_names = metadata.get("products") or []
        for index, solid in enumerate(solids, start=1):
            name = choose_part_name(product_names, index)
            parts.append(analyze_part(solid, f"P{index:03d}", name))
    else:
        notes.append("No solid bodies were found; the top-level STEP shape was analyzed as one shape.")
        parts.append(analyze_part(shape, "P001", "TopLevelShape"))

    return AnalysisReport(
        source_file=str(path.resolve()),
        detected_step_units=detected_units,
        step_metadata=metadata,
        import_system_length_unit=rounded(system_unit),
        topology_counts=count_topology(shape),
        part_count=len(parts),
        parts=parts,
        notes=notes,
    )


def choose_part_name(product_names: list[str], index: int) -> str:
    fallback = f"Solid_{index:03d}"
    if index - 1 >= len(product_names):
        return fallback
    candidate = product_names[index - 1].strip()
    if not candidate:
        return fallback
    generic_markers = ("open cascade step translator", "opencascade step translator")
    if any(marker in candidate.lower() for marker in generic_markers):
        return fallback
    return candidate


def dataclass_json(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return {key: dataclass_json(item) for key, item in asdict(value).items()}
    if isinstance(value, list):
        return [dataclass_json(item) for item in value]
    if isinstance(value, dict):
        return {key: dataclass_json(item) for key, item in value.items()}
    return value


def write_json(report: AnalysisReport, path: Path) -> None:
    path.write_text(json.dumps(dataclass_json(report), indent=2), encoding="utf-8")


def write_part_summary_csv(report: AnalysisReport, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "part_id",
                "name",
                "is_valid",
                "center_x",
                "center_y",
                "center_z",
                "bbox_x",
                "bbox_y",
                "bbox_z",
                "volume",
                "surface_area",
                "faces",
                "edges",
                "features",
            ],
        )
        writer.writeheader()
        for part in report.parts:
            writer.writerow(
                {
                    "part_id": part.part_id,
                    "name": part.name,
                    "is_valid": part.is_valid,
                    "center_x": part.bounding_box.center["x"],
                    "center_y": part.bounding_box.center["y"],
                    "center_z": part.bounding_box.center["z"],
                    "bbox_x": part.bounding_box.length_x,
                    "bbox_y": part.bounding_box.length_y,
                    "bbox_z": part.bounding_box.length_z,
                    "volume": part.mass_properties.volume,
                    "surface_area": part.mass_properties.surface_area,
                    "faces": part.topology_counts.get("faces", 0),
                    "edges": part.topology_counts.get("edges", 0),
                    "features": len(part.features),
                }
            )


def compact_points(points: list[dict[str, float]]) -> str:
    return "; ".join(f"({point['x']}, {point['y']})" for point in points)


def write_planar_profiles_csv(report: AnalysisReport, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "part_id",
                "part_name",
                "face_id",
                "face_area",
                "center_x",
                "center_y",
                "center_z",
                "normal",
                "x_axis",
                "y_axis",
                "loop_count",
                "outer_loop_id",
                "outer_shape_guess",
                "outer_width",
                "outer_height",
                "outer_area_2d",
            ],
        )
        writer.writeheader()
        for part in report.parts:
            for profile in part.planar_profiles:
                writer.writerow(
                    {
                        "part_id": part.part_id,
                        "part_name": part.name,
                        "face_id": profile.face_id,
                        "face_area": profile.area,
                        "center_x": profile.center_3d["x"],
                        "center_y": profile.center_3d["y"],
                        "center_z": profile.center_3d["z"],
                        "normal": flatten_vector(profile.normal),
                        "x_axis": flatten_vector(profile.x_axis),
                        "y_axis": flatten_vector(profile.y_axis),
                        "loop_count": profile.loop_count,
                        "outer_loop_id": profile.outer_loop_id,
                        "outer_shape_guess": profile.outer_shape_guess,
                        "outer_width": profile.outer_width,
                        "outer_height": profile.outer_height,
                        "outer_area_2d": profile.outer_area_2d,
                    }
                )


def write_planar_loops_csv(report: AnalysisReport, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "part_id",
                "part_name",
                "face_id",
                "loop_id",
                "shape_guess",
                "edge_count",
                "vertex_count",
                "area_2d",
                "perimeter",
                "width",
                "height",
                "bbox_center_x",
                "bbox_center_y",
                "vertices_2d",
            ],
        )
        writer.writeheader()
        for part in report.parts:
            for profile in part.planar_profiles:
                for loop in profile.loops:
                    writer.writerow(
                        {
                            "part_id": part.part_id,
                            "part_name": part.name,
                            "face_id": profile.face_id,
                            "loop_id": loop.loop_id,
                            "shape_guess": loop.shape_guess,
                            "edge_count": loop.edge_count,
                            "vertex_count": loop.vertex_count,
                            "area_2d": loop.area_2d,
                            "perimeter": loop.perimeter,
                            "width": loop.bbox_2d.get("width"),
                            "height": loop.bbox_2d.get("height"),
                            "bbox_center_x": loop.bbox_2d.get("center_x"),
                            "bbox_center_y": loop.bbox_2d.get("center_y"),
                            "vertices_2d": compact_points(loop.vertices_2d),
                        }
                    )


def compact_point(point: dict[str, float] | None) -> str:
    if not point:
        return ""
    return f"({point['x']}, {point['y']})"


def write_planar_edges_csv(report: AnalysisReport, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "part_id",
                "part_name",
                "face_id",
                "loop_id",
                "loop_shape_guess",
                "edge_index",
                "curve_type",
                "length",
                "angle_degrees",
                "radius",
                "diameter",
                "start_2d",
                "end_2d",
                "center_2d",
                "start_3d",
                "end_3d",
                "center_3d",
            ],
        )
        writer.writeheader()
        for part in report.parts:
            for profile in part.planar_profiles:
                for loop in profile.loops:
                    for edge in loop.edges:
                        writer.writerow(
                            {
                                "part_id": part.part_id,
                                "part_name": part.name,
                                "face_id": profile.face_id,
                                "loop_id": loop.loop_id,
                                "loop_shape_guess": loop.shape_guess,
                                "edge_index": edge.edge_index,
                                "curve_type": edge.curve_type,
                                "length": edge.length,
                                "angle_degrees": edge.angle_degrees,
                                "radius": edge.radius,
                                "diameter": edge.diameter,
                                "start_2d": compact_point(edge.start_2d),
                                "end_2d": compact_point(edge.end_2d),
                                "center_2d": compact_point(edge.center_2d),
                                "start_3d": flatten_vector(edge.start_3d),
                                "end_3d": flatten_vector(edge.end_3d),
                                "center_3d": flatten_vector(edge.center_3d),
                            }
                        )


def flatten_vector(value: dict[str, float] | None) -> str:
    if not value:
        return ""
    return f"({value.get('x')}, {value.get('y')}, {value.get('z')})"


def write_features_csv(report: AnalysisReport, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "part_id",
                "part_name",
                "feature_id",
                "source_face_id",
                "category",
                "confidence",
                "radius",
                "diameter",
                "height",
                "area",
                "orientation",
                "axis_origin",
                "axis_direction",
                "notes",
            ],
        )
        writer.writeheader()
        for part in report.parts:
            for feature in part.features:
                writer.writerow(
                    {
                        "part_id": feature.part_id,
                        "part_name": part.name,
                        "feature_id": feature.feature_id,
                        "source_face_id": feature.source_face_id,
                        "category": feature.category,
                        "confidence": feature.confidence,
                        "radius": feature.radius,
                        "diameter": feature.diameter,
                        "height": feature.height,
                        "area": feature.area,
                        "orientation": feature.orientation,
                        "axis_origin": flatten_vector(feature.axis_origin),
                        "axis_direction": flatten_vector(feature.axis_direction),
                        "notes": feature.notes,
                    }
                )


def html_table(headers: Iterable[str], rows: Iterable[Iterable[Any]]) -> str:
    header_html = "".join(f"<th>{html.escape(str(header))}</th>" for header in headers)
    row_html = []
    for row in rows:
        row_html.append(
            "<tr>"
            + "".join(f"<td>{html.escape('' if value is None else str(value))}</td>" for value in row)
            + "</tr>"
        )
    return f"<table><thead><tr>{header_html}</tr></thead><tbody>{''.join(row_html)}</tbody></table>"


def md_cell(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    return text.replace("|", "\\|").replace("\n", "<br>")


def md_table(headers: Iterable[str], rows: Iterable[Iterable[Any]]) -> str:
    header_list = [md_cell(header) for header in headers]
    lines = [
        "| " + " | ".join(header_list) + " |",
        "| " + " | ".join("---" for _ in header_list) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(md_cell(value) for value in row) + " |")
    return "\n".join(lines)


def vector_delta(
    value: dict[str, float] | None, reference: dict[str, float] | None
) -> dict[str, float] | None:
    if not value or not reference:
        return None
    return {
        "x": rounded(value["x"] - reference["x"]),
        "y": rounded(value["y"] - reference["y"]),
        "z": rounded(value["z"] - reference["z"]),
    }


def part_size_tuple(part: PartReport) -> tuple[float, float, float]:
    box = part.bounding_box
    return box.length_x, box.length_y, box.length_z


def role_guess(part: PartReport, reference_part: PartReport | None) -> str:
    dims = [dim for dim in part_size_tuple(part) if dim is not None]
    if reference_part and part.part_id == reference_part.part_id:
        return "largest body / likely base"
    if dims and min(dims) / max(dims) < 0.08:
        return "thin plate or pad"
    if part.features:
        return "feature body"
    return "separate solid"


def profile_shape_counts(part: PartReport) -> str:
    counts = Counter(
        loop.shape_guess for profile in part.planar_profiles for loop in profile.loops
    )
    return ", ".join(f"{shape}={count}" for shape, count in sorted(counts.items())) or "none"


def build123d_hint(part: PartReport) -> str:
    loop_shapes = {
        loop.shape_guess for profile in part.planar_profiles for loop in profile.loops
    }
    if "trapezium_candidate" in loop_shapes:
        return (
            "This is one shaped body with sloped/chamfered faces. Use trapezium loop vertices to "
            "understand the side faces, but do not create each face as a separate solid panel."
        )
    if any(shape.startswith("polygon_") for shape in loop_shapes):
        return (
            "This part has non-rectangular polygon profiles. Do not replace it with a simple Box; "
            "use the listed profile vertices, chamfers, fillets, or boolean cuts to reproduce the shape."
        )
    if loop_shapes and loop_shapes <= {"rectangle"}:
        return (
            "Box-like body or pad. Create one solid for this Part ID, then place it at the listed center."
        )
    if "circle" in loop_shapes or any(feature.category.endswith("candidate") for feature in part.features):
        return (
            "Use Cylinder/Circle operations for circular bodies or true hole/boss features. "
            "Do not turn a circular face into a cut unless it is marked as a hole candidate."
        )
    return "Use planar loop vertices and edge dimensions to recreate the body profile."


def is_critical_loop(loop: PlanarLoop) -> bool:
    return (
        loop.shape_guess not in {"rectangle", "circle"}
        or loop.edge_count > 4
        or any(edge.curve_type != "Line" for edge in loop.edges)
    )


def critical_profile_rows(report: AnalysisReport) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for part in report.parts:
        for profile in part.planar_profiles:
            for loop in profile.loops:
                if is_critical_loop(loop):
                    rows.append(
                        [
                            part.part_id,
                            profile.face_id,
                            loop.loop_id,
                            loop.shape_guess,
                            flatten_vector(profile.center_3d),
                            flatten_vector(profile.normal),
                            loop.bbox_2d.get("width"),
                            loop.bbox_2d.get("height"),
                            compact_points(loop.vertices_2d),
                            "; ".join(flatten_vector(point) for point in loop.vertices_3d),
                            "Must be modeled; do not simplify this part as a plain box.",
                        ]
                    )
    return rows


def loop_shape_counter(part: PartReport) -> Counter[str]:
    return Counter(
        loop.shape_guess for profile in part.planar_profiles for loop in profile.loops
    )


def all_profile_loops(part: PartReport) -> list[tuple[PlanarProfile, PlanarLoop]]:
    return [
        (profile, loop)
        for profile in part.planar_profiles
        for loop in profile.loops
    ]


def dominant_rectangle_profiles(
    part: PartReport, limit: int = 4
) -> list[tuple[PlanarProfile, PlanarLoop]]:
    rectangles = [
        (profile, loop)
        for profile, loop in all_profile_loops(part)
        if loop.shape_guess == "rectangle"
    ]
    rectangles.sort(key=lambda item: item[1].area_2d or 0, reverse=True)
    return rectangles[:limit]


def important_loop_refs(part: PartReport, limit: int = 8) -> str:
    loops = [
        (profile, loop)
        for profile, loop in all_profile_loops(part)
        if is_critical_loop(loop)
    ]
    loops.sort(key=lambda item: item[1].area_2d or 0, reverse=True)
    refs = [
        f"{profile.face_id}/{loop.loop_id} {loop.shape_guess} "
        f"w={loop.bbox_2d.get('width')} h={loop.bbox_2d.get('height')}"
        for profile, loop in loops[:limit]
    ]
    return "; ".join(refs) or "none"


def dominant_rectangle_refs(part: PartReport, limit: int = 4) -> str:
    refs = [
        f"{profile.face_id}/{loop.loop_id} center={flatten_vector(profile.center_3d)} "
        f"normal={flatten_vector(profile.normal)} w={loop.bbox_2d.get('width')} "
        f"h={loop.bbox_2d.get('height')}"
        for profile, loop in dominant_rectangle_profiles(part, limit=limit)
    ]
    return "; ".join(refs) or "none"


def feature_refs(part: PartReport) -> str:
    if not part.features:
        return "none"
    return "; ".join(
        f"{feature.feature_id} {feature.category} r={feature.radius} "
        f"d={feature.diameter} axis={flatten_vector(feature.axis_direction)}"
        for feature in part.features
    )


def modeling_strategy_for_part(part: PartReport, reference_part: PartReport | None) -> dict[str, str]:
    counts = loop_shape_counter(part)
    box = part.bounding_box
    dims = [box.length_x, box.length_y, box.length_z]
    min_dim = min(dims)
    max_dim = max(dims)
    has_trapezium = counts.get("trapezium_candidate", 0) > 0
    has_polygon = any(shape.startswith("polygon_") for shape in counts)
    has_rounds = any(
        "round" in feature.category or "fillet" in feature.category
        for feature in part.features
    )
    has_circle = counts.get("circle", 0) > 0 or any(
        "cylinder" in feature.category or "boss" in feature.category
        for feature in part.features
    )
    only_rectangles = bool(counts) and set(counts) <= {"rectangle"}
    is_thin = min_dim / max_dim < 0.08 if max_dim else False

    warnings: list[str] = []
    operations: list[str] = []
    avoid: list[str] = []

    if has_trapezium:
        operations.append(
            "Use loft/chamfer/wedge cuts from the trapezium_candidate faces; preserve listed 3D vertices."
        )
        avoid.append("Do not model this part as a plain rectangular Box.")
        warnings.append(
            "Trapezium faces mean the sides are sloped/chamfered and must be reproduced."
        )
    if has_polygon:
        operations.append(
            "Use polygon loop vertices or boolean cuts for non-rectangular side profiles."
        )
        avoid.append("Do not replace polygon_ profiles with rectangle faces.")
        warnings.append(
            "Polygon profiles indicate steps, notches, bevels, or shaped side caps."
        )
    if has_rounds:
        operations.append(
            "Add rounded/fillet/cylindrical cap geometry using the feature radii and axes."
        )
        avoid.append("Do not approximate rounded caps as square boxes if visual match matters.")
        warnings.append("Rounded feature candidates were detected.")
    if has_circle:
        operations.append(
            "Use Cylinder/Circle only for the listed circular body or boss; subtract only if marked as a hole."
        )
    if only_rectangles and not has_rounds:
        operations.append("A Box is acceptable for this body if its placement and dimensions match.")
    if is_thin:
        operations.append("This is thin relative to its largest dimension; likely a pad, plate, lead, or lens.")
    if reference_part and part.part_id == reference_part.part_id and not (has_trapezium or has_polygon):
        operations.append(
            "This is the largest body. Check whether the bbox includes protrusions before using it as the main block."
        )

    if not operations:
        operations.append("Recreate from dominant planar profiles and validate against the STEP.")
    if not avoid:
        avoid.append("Do not ignore face centers/normals when placing sketches.")

    return {
        "part": part.part_id,
        "name": part.name,
        "complexity_flags": "; ".join(warnings) or "box-like or simple analytic body",
        "recommended_operations": " ".join(operations),
        "do_not": " ".join(dict.fromkeys(avoid)),
        "key_dimensions": (
            f"bbox={box.length_x} x {box.length_y} x {box.length_z}; "
            f"center={flatten_vector(box.center)}"
        ),
        "dominant_rectangles": dominant_rectangle_refs(part),
        "critical_profiles": important_loop_refs(part),
        "feature_radii": feature_refs(part),
    }


def modeling_strategy_rows(
    parts: list[PartReport], reference_part: PartReport | None
) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for part in parts:
        strategy = modeling_strategy_for_part(part, reference_part)
        rows.append(
            [
                strategy["part"],
                strategy["name"],
                strategy["complexity_flags"],
                strategy["recommended_operations"],
                strategy["do_not"],
                strategy["key_dimensions"],
                strategy["critical_profiles"],
                strategy["feature_radii"],
            ]
        )
    return rows


def compact_json_data(report: AnalysisReport) -> dict[str, Any]:
    return {
        part.part_id: {
            "name": part.name,
            "modeling_strategy": modeling_strategy_for_part(part, None),
            "center": part.bounding_box.center,
            "size": {
                "x": part.bounding_box.length_x,
                "y": part.bounding_box.length_y,
                "z": part.bounding_box.length_z,
            },
            "bbox_min": {
                "x": part.bounding_box.min_x,
                "y": part.bounding_box.min_y,
                "z": part.bounding_box.min_z,
            },
            "bbox_max": {
                "x": part.bounding_box.max_x,
                "y": part.bounding_box.max_y,
                "z": part.bounding_box.max_z,
            },
            "planar_profiles": [
                {
                    "face_id": profile.face_id,
                    "center": profile.center_3d,
                    "normal": profile.normal,
                    "outer_shape": profile.outer_shape_guess,
                    "outer_width": profile.outer_width,
                    "outer_height": profile.outer_height,
                    "loops": [
                        {
                            "loop_id": loop.loop_id,
                            "shape": loop.shape_guess,
                            "vertices_2d": loop.vertices_2d,
                            "vertices_3d": loop.vertices_3d,
                        }
                        for loop in profile.loops
                    ],
                }
                for profile in part.planar_profiles
            ],
        }
        for part in report.parts
    }


def model_complexity_note(report: AnalysisReport) -> tuple[str, list[str]]:
    face_count = sum(part.topology_counts.get("faces", 0) for part in report.parts)
    feature_count = sum(len(part.features) for part in report.parts)
    curved_face_count = sum(
        1
        for part in report.parts
        for face in part.faces
        if face.surface_type not in {"Plane"}
    )
    freeform_like_count = sum(
        1
        for part in report.parts
        for face in part.faces
        if face.surface_type not in {"Plane", "Cylinder", "Cone", "Sphere", "Torus"}
    )

    warnings: list[str] = []
    if freeform_like_count:
        warnings.append(
            f"{freeform_like_count} non-analytic/freeform-like faces were found; these may need splines, lofts, sweeps, or approximation."
        )
    if curved_face_count > 12:
        warnings.append(
            f"{curved_face_count} curved faces were found; reconstructing every blend exactly may require multiple boolean/fillet steps."
        )
    if face_count > 80:
        warnings.append(
            f"{face_count} faces were found; ask the code generator to build and validate in stages, not in one giant script."
        )
    if feature_count == 0 and curved_face_count:
        warnings.append(
            "Curved faces exist but no confident feature candidates were inferred; inspect circular/curved profiles before treating them as cuts."
        )

    if face_count <= 35 and freeform_like_count == 0:
        level = "low to medium"
    elif face_count <= 90 and freeform_like_count <= 4:
        level = "medium"
    else:
        level = "high"

    if not warnings:
        warnings.append(
            "The model is mostly analytic geometry. A careful build123d reconstruction from this brief is realistic."
        )
    return level, warnings


def write_build123d_brief(report: AnalysisReport, path: Path) -> None:
    source_name = Path(report.source_file).name
    sorted_parts = sorted(
        report.parts, key=lambda part: part.mass_properties.volume or 0, reverse=True
    )
    reference_part = sorted_parts[0] if sorted_parts else None
    reference_center = reference_part.bounding_box.center if reference_part else None
    complexity_level, complexity_warnings = model_complexity_note(report)

    lines: list[str] = [
        f"# Build123d Reconstruction Brief: {source_name}",
        "",
        "This document is generated from exact STEP B-Rep geometry. It is meant to show the dimensions and arrangement needed to write a build123d script.",
        "",
        "## Coordinate System",
        "",
        f"- Source file: `{report.source_file}`",
        f"- Detected STEP units: {', '.join(report.detected_step_units) or 'not reported'}",
        "- Coordinates and dimensions are reported in the units imported by OpenCascade.",
        "- Part centers are bounding-box centers in the global STEP coordinate system.",
        "- Planar loop vertices are face-local 2D points; each profile also lists global 3D vertices for placement.",
        "",
        "## Reconstruction Rules",
        "",
        "- Create one build123d solid for each `Part ID` in the Arrangement Summary.",
        "- Do not create a separate solid for every `Face`, `Loop`, or planar profile. Those rows describe surfaces of an existing part.",
        "- Use `_parts` / Part Details for body size and placement. Use planar profiles only to understand shape, chamfers, trapezium sides, pockets, or face boundaries.",
        "- Treat separate Part IDs as additive bodies unless the feature table explicitly says `hole_candidate` or another subtractive feature.",
        "- Use `vertices_3d` when placement matters; `vertices_2d` are only local sketch coordinates on that one face.",
        "- STEP appearance/color is not a reliable modeling instruction. A colored circular face may be an appearance face, not a cut.",
        "",
        "## Reconstruction Notes",
        "",
        "Use these reconstruction notes when writing build123d code:",
        "",
        "```text",
        "Write build123d Python code from this reconstruction brief. Follow these rules strictly:",
        "1. Create one main solid/body only where the brief indicates one Part ID or one continuous original solid.",
        "2. Do not extrude every Face, Loop, or planar profile as a separate object. Faces/loops describe existing surfaces.",
        "3. Do not use the overall bounding box as the central body if protruding leads/tabs/terminals are present. Derive the body from dominant front/back/top/bottom profiles and add protrusions separately.",
        "4. If the Critical Shape Profiles table contains trapezium, polygon, triangle, or curved loops for a part, that part is not a plain Box. Model those profiles with chamfer/fillet/loft/boolean operations.",
        "5. Use global 3D vertices for placement-sensitive geometry. Use local 2D vertices only inside the matching face/sketch plane.",
        "6. For Polygon/Polyline points measured from the brief, preserve coordinates. In build123d, pass align=None to Polygon when using exact measured points.",
        "7. Treat feature candidates marked as holes/cuts as subtractive. Treat separate pads/leads/bosses as additive unless the brief explicitly says subtract.",
        "8. Build in stages: base body, protrusions/leads, cuts/notches, circular features, fillets/chamfers. Export STEP and validate against the original with symmetric boolean difference.",
        "9. If a surface is marked freeform/spline/unknown, do not invent a box. Use approximation only if acceptable and state the approximation.",
        "```",
        "",
        f"Complexity estimate for this model: **{complexity_level}**.",
        "",
        "Complexity notes:",
        "",
        *[f"- {warning}" for warning in complexity_warnings],
        "",
    ]

    critical_rows = critical_profile_rows(report)
    if critical_rows:
        lines.extend(
            [
                "## Critical Shape Profiles",
                "",
                "These profiles are the usual reason a generated build123d model looks too boxy. They must be represented in the code; do not flatten them into rectangular boxes.",
                "",
                md_table(
                    [
                        "Part",
                        "Face",
                        "Loop",
                        "Shape",
                        "Face Center",
                        "Normal",
                        "Width",
                        "Height",
                        "Vertices 2D",
                        "Vertices 3D",
                        "Action",
                    ],
                    critical_rows,
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Automatic Modeling Strategy",
            "",
            "This section is generated from the geometry patterns. It is the first place to look before writing build123d code.",
            "",
            md_table(
                [
                    "Part",
                    "Name",
                    "Detected Issue",
                    "Recommended Build123d Operations",
                    "Do Not",
                    "Key Dimensions",
                    "Critical Profiles",
                    "Feature Radii / Axes",
                ],
                modeling_strategy_rows(sorted_parts, reference_part),
            ),
            "",
        ]
    )

    if reference_part:
        lines.extend(
            [
                "## Arrangement Summary",
                "",
                f"Reference part for relative locations: `{reference_part.part_id}` `{reference_part.name}`.",
                "",
                md_table(
                    [
                        "Part",
                        "Name",
                        "Role Guess",
                        "Center XYZ",
                        f"Delta From {reference_part.part_id}",
                        "Size XYZ",
                        "Volume",
                        "Profile Shapes",
                    ],
                    [
                        [
                            part.part_id,
                            part.name,
                            role_guess(part, reference_part),
                            flatten_vector(part.bounding_box.center),
                            flatten_vector(vector_delta(part.bounding_box.center, reference_center)),
                            f"{part.bounding_box.length_x} x {part.bounding_box.length_y} x {part.bounding_box.length_z}",
                            part.mass_properties.volume,
                            profile_shape_counts(part),
                        ]
                        for part in sorted_parts
                    ],
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Suggested Build123d Data Block",
            "",
            "Use this as a starting constants block. It is not a final model by itself; it keeps the measured sizes, locations, and sketch loops together.",
            "",
            "```python",
            "from build123d import *",
            "",
            "STEP_PART_DATA = "
            + json.dumps(compact_json_data(report), indent=4).replace("\n", "\n"),
            "```",
            "",
            "## Part Details",
            "",
        ]
    )

    for part in sorted_parts:
        box = part.bounding_box
        strategy = modeling_strategy_for_part(part, reference_part)
        lines.extend(
            [
                f"### {part.part_id} - {part.name}",
                "",
                f"Build123d hint: {build123d_hint(part)}",
                "",
                "#### Modeling Strategy",
                "",
                md_table(
                    ["Item", "Instruction"],
                    [
                        ["Detected issue", strategy["complexity_flags"]],
                        ["Recommended operations", strategy["recommended_operations"]],
                        ["Do not", strategy["do_not"]],
                        ["Dominant rectangle profiles", strategy["dominant_rectangles"]],
                        ["Critical profiles", strategy["critical_profiles"]],
                        ["Feature radii / axes", strategy["feature_radii"]],
                    ],
                ),
                "",
                md_table(
                    ["Property", "Value"],
                    [
                        ["Valid B-Rep", part.is_valid],
                        ["Bounding box min", f"({box.min_x}, {box.min_y}, {box.min_z})"],
                        ["Bounding box max", f"({box.max_x}, {box.max_y}, {box.max_z})"],
                        ["Bounding box center", flatten_vector(box.center)],
                        ["Size X/Y/Z", f"{box.length_x} x {box.length_y} x {box.length_z}"],
                        ["Volume", part.mass_properties.volume],
                        ["Surface area", part.mass_properties.surface_area],
                        ["Center of mass", flatten_vector(part.mass_properties.center_of_mass)],
                        ["Topology", ", ".join(f"{k}={v}" for k, v in part.topology_counts.items())],
                    ],
                ),
                "",
                "#### Feature Candidates",
                "",
            ]
        )

        if part.features:
            lines.extend(
                [
                    md_table(
                        [
                            "ID",
                            "Face",
                            "Category",
                            "Confidence",
                            "Radius",
                            "Diameter",
                            "Height",
                            "Axis Origin",
                            "Axis Direction",
                        ],
                        [
                            [
                                feature.feature_id,
                                feature.source_face_id,
                                feature.category,
                                feature.confidence,
                                feature.radius,
                                feature.diameter,
                                feature.height,
                                flatten_vector(feature.axis_origin),
                                flatten_vector(feature.axis_direction),
                            ]
                            for feature in part.features
                        ],
                    ),
                    "",
                ]
            )
        else:
            lines.extend(["No cylindrical/conical/fillet feature candidates found.", ""])

        lines.extend(["#### Planar Profiles", ""])
        if part.planar_profiles:
            lines.extend(
                [
                    md_table(
                        [
                            "Face",
                            "Center XYZ",
                            "Normal",
                            "Outer Shape",
                            "Width",
                            "Height",
                            "Area",
                            "Loops",
                        ],
                        [
                            [
                                profile.face_id,
                                flatten_vector(profile.center_3d),
                                flatten_vector(profile.normal),
                                profile.outer_shape_guess,
                                profile.outer_width,
                                profile.outer_height,
                                profile.outer_area_2d,
                                profile.loop_count,
                            ]
                            for profile in part.planar_profiles
                        ],
                    ),
                    "",
                    "#### Sketch Loops",
                    "",
                    md_table(
                        [
                            "Face",
                            "Loop",
                            "Shape",
                            "Area",
                            "Perimeter",
                            "Width",
                            "Height",
                            "Vertices 2D",
                            "Vertices 3D",
                        ],
                        [
                            [
                                profile.face_id,
                                loop.loop_id,
                                loop.shape_guess,
                                loop.area_2d,
                                loop.perimeter,
                                loop.bbox_2d.get("width"),
                                loop.bbox_2d.get("height"),
                                compact_points(loop.vertices_2d),
                                "; ".join(flatten_vector(point) for point in loop.vertices_3d),
                            ]
                            for profile in part.planar_profiles
                            for loop in profile.loops
                        ],
                    ),
                    "",
                    "#### Sketch Edge Dimensions",
                    "",
                    md_table(
                        [
                            "Face",
                            "Loop",
                            "Edge",
                            "Curve",
                            "Length",
                            "Angle",
                            "Radius",
                            "Start 2D",
                            "End 2D",
                            "Start 3D",
                            "End 3D",
                        ],
                        [
                            [
                                profile.face_id,
                                loop.loop_id,
                                edge.edge_index,
                                edge.curve_type,
                                edge.length,
                                edge.angle_degrees,
                                edge.radius,
                                compact_point(edge.start_2d),
                                compact_point(edge.end_2d),
                                flatten_vector(edge.start_3d),
                                flatten_vector(edge.end_3d),
                            ]
                            for profile in part.planar_profiles
                            for loop in profile.loops
                            for edge in loop.edges
                        ],
                    ),
                    "",
                ]
            )
        else:
            lines.extend(["No planar profiles found for this part.", ""])

    lines.extend(
        [
            "## Notes",
            "",
            "- STEP usually does not preserve the original CAD feature tree, so profile and feature names are inferred from geometry.",
            "- For exact build123d reconstruction, use global part centers for arrangement and face-local loop vertices for sketch profiles.",
            "- Trapezium candidates are four-edge planar loops with one pair of opposite parallel edges.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def is_axis_z(axis: dict[str, float] | None) -> bool:
    if not axis:
        return False
    return abs(axis.get("z", 0.0)) > 0.99 and abs(axis.get("x", 0.0)) < 0.02 and abs(
        axis.get("y", 0.0)
    ) < 0.02


def part_is_simple_cylinder(part: PartReport) -> bool:
    box = part.bounding_box
    if not almost_equal(box.length_x, box.length_y, rel_tol=0.02):
        return False
    cylinder_features = [
        feature
        for feature in part.features
        if "cylinder" in feature.category or "boss" in feature.category
    ]
    if not cylinder_features:
        return False
    return any(is_axis_z(feature.axis_direction) for feature in cylinder_features)


def part_is_simple_box(part: PartReport) -> bool:
    counts = loop_shape_counter(part)
    if not counts or set(counts) != {"rectangle"}:
        return False
    if part.features:
        return False
    return True


def scaffold_code_for_part(part: PartReport) -> list[str]:
    box = part.bounding_box
    center = box.center
    var_name = part.part_id.lower()
    strategy = modeling_strategy_for_part(part, None)

    if part_is_simple_cylinder(part):
        radius = rounded(max(box.length_x, box.length_y) / 2)
        return [
            f"    # {part.part_id} {part.name}: simple vertical cylinder/pad",
            f"    {var_name} = Cylinder(radius={radius}, height={box.length_z}).located(",
            f"        Location(({center['x']}, {center['y']}, {center['z']}))",
            "    )",
            f"    bodies.append({var_name})",
            "",
        ]

    if part_is_simple_box(part):
        return [
            f"    # {part.part_id} {part.name}: box-like body",
            f"    {var_name} = Box({box.length_x}, {box.length_y}, {box.length_z}).located(",
            f"        Location(({center['x']}, {center['y']}, {center['z']}))",
            "    )",
            f"    bodies.append({var_name})",
            "",
        ]

    return [
        f"    # {part.part_id} {part.name}: intentionally not auto-boxed",
        f"    # Reason: {strategy['complexity_flags']}",
        f"    # Recommended: {strategy['recommended_operations']}",
        f"    # Critical profiles: {strategy['critical_profiles']}",
        "",
    ]


def write_build123d_model_script(
    report: AnalysisReport, path: Path, source_step_name: str, output_stem: str
) -> None:
    data_json = json.dumps(compact_json_data(report), indent=2)
    scaffold_lines: list[str] = []
    for part in sorted(report.parts, key=lambda item: item.mass_properties.volume or 0, reverse=True):
        scaffold_lines.extend(scaffold_code_for_part(part))
    if not any("bodies.append" in line for line in scaffold_lines):
        scaffold_lines.extend(
            [
                "    # No safely auto-reconstructable primitive bodies were detected.",
                "    # Keep MODEL_MODE = 'exact' for a faithful build123d model.",
                "",
            ]
        )

    script_lines = [
        "# Auto-generated by step_analyzer.py",
        "#",
        "# Default mode imports the local STEP copy exactly through build123d.",
        "# This avoids the common failure where complex geometry is simplified into boxes.",
        "# Change MODEL_MODE to 'scaffold' only when you intentionally want a partial",
        "# editable starting point made from safe primitive detections.",
        "",
        "from pathlib import Path",
        "import json",
        "",
        "from build123d import *",
        "",
        "MODEL_MODE = 'exact'  # 'exact' or 'scaffold'",
        f"SOURCE_STEP = Path(__file__).with_name({json.dumps(source_step_name)})",
        f"OUTPUT_STEP = Path(__file__).with_name({json.dumps(output_stem + '_from_build123d.step')})",
        "",
        "STEP_PART_DATA = json.loads(r'''",
        data_json,
        "''')",
        "",
        "",
        "def exact_reference_step():",
        "    if SOURCE_STEP.exists():",
        "        return SOURCE_STEP",
        "    if OUTPUT_STEP.exists():",
        "        return OUTPUT_STEP",
        "    raise FileNotFoundError(",
        "        f'Missing STEP reference beside this script. Expected either '",
        "        f'{SOURCE_STEP.name} or {OUTPUT_STEP.name}. Rerun the analyzer or copy the original STEP into this folder.'",
        "    )",
        "",
        "",
        "def total_volume(shape_or_shapes):",
        "    if hasattr(shape_or_shapes, 'volume'):",
        "        return shape_or_shapes.volume",
        "    return sum(getattr(shape, 'volume', 0.0) for shape in shape_or_shapes)",
        "",
        "",
        "def build_exact_model():",
        "    return import_step(str(exact_reference_step()))",
        "",
        "",
        "def build_parametric_scaffold():",
        "    bodies = []",
        "    # Complex parts are left as comments instead of being wrongly made as boxes.",
        *scaffold_lines,
        "    if not bodies:",
        "        raise RuntimeError('No safe primitive scaffold bodies were generated. Use MODEL_MODE = exact.')",
        "    model = bodies[0]",
        "    for body in bodies[1:]:",
        "        model += body",
        "    return model",
        "",
        "",
        "def build_model():",
        "    if MODEL_MODE == 'exact':",
        "        return build_exact_model()",
        "    if MODEL_MODE == 'scaffold':",
        "        return build_parametric_scaffold()",
        "    raise ValueError(\"MODEL_MODE must be 'exact' or 'scaffold'\")",
        "",
        "",
        "def validate(model):",
        "    reference = import_step(str(exact_reference_step()))",
        "    return total_volume(model - reference) + total_volume(reference - model)",
        "",
        "",
        "if __name__ == '__main__':",
        "    reference_step = exact_reference_step()",
        "    model = build_model()",
        "    export_step(model, str(OUTPUT_STEP))",
        "    print(f'Exported: {OUTPUT_STEP}')",
        "    if MODEL_MODE == 'exact':",
        "        print(f'Exact mode source: {reference_step.name}')",
        "    else:",
        "        try:",
        "            sym_diff = validate(model)",
        "            print(f'Symmetric difference vs {exact_reference_step().name}: {sym_diff:.9f}')",
        "        except Exception as exc:",
        "            print(f'Validation skipped: {exc}')",
        "",
    ]
    path.write_text("\n".join(script_lines), encoding="utf-8")


def write_exact_build123d_step(source_step: Path, output_step: Path) -> None:
    try:
        from build123d import export_step, import_step

        model = import_step(str(source_step))
        export_step(model, str(output_step))
    except Exception:
        # Fallback still preserves exact geometry if build123d import/export is unavailable.
        shutil.copy2(source_step, output_step)


def write_vscode_python_settings(folder: Path) -> Path:
    vscode_dir = folder / ".vscode"
    vscode_dir.mkdir(parents=True, exist_ok=True)
    settings_path = vscode_dir / "settings.json"

    settings: dict[str, Any] = {}
    if settings_path.exists():
        try:
            existing = json.loads(settings_path.read_text(encoding="utf-8-sig"))
            if isinstance(existing, dict):
                settings.update(existing)
        except json.JSONDecodeError:
            pass

    settings["python.defaultInterpreterPath"] = str(BUILD123D_PYTHON)
    settings["python.terminal.activateEnvironment"] = True
    settings_path.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    return settings_path


def package_build123d_outputs(
    paths: dict[str, Path], package_dir: Path, stem: str
) -> dict[str, Path]:
    target_dir = package_dir / safe_folder_name(stem)
    target_dir.mkdir(parents=True, exist_ok=True)

    script_target = target_dir / paths["build123d_script"].name
    source_target = target_dir / paths["source_step"].name
    step_target = target_dir / paths["build123d_step"].name
    shutil.copy2(paths["build123d_script"], script_target)
    shutil.copy2(paths["source_step"], source_target)
    shutil.copy2(paths["build123d_step"], step_target)
    vscode_settings = write_vscode_python_settings(target_dir)

    return {
        "package_dir": target_dir,
        "packaged_build123d_script": script_target,
        "packaged_source_step": source_target,
        "packaged_build123d_step": step_target,
        "vscode_settings": vscode_settings,
    }


def write_html(report: AnalysisReport, path: Path, max_rows: int = 300) -> None:
    source_name = Path(report.source_file).name
    part_rows = [
        [
            part.part_id,
            part.name,
            part.is_valid,
            flatten_vector(part.bounding_box.center),
            part.bounding_box.length_x,
            part.bounding_box.length_y,
            part.bounding_box.length_z,
            part.mass_properties.volume,
            part.mass_properties.surface_area,
            len(part.features),
        ]
        for part in report.parts
    ]

    sections = [
        f"<h1>STEP Analysis Report: {html.escape(source_name)}</h1>",
        "<h2>Model Summary</h2>",
        html_table(
            ["Source", "Parts", "Detected STEP Units", "System Length Unit"],
            [
                [
                    report.source_file,
                    report.part_count,
                    ", ".join(report.detected_step_units) or "Not reported",
                    report.import_system_length_unit,
                ]
            ],
        ),
        "<h2>Parts</h2>",
        html_table(
            [
                "Part",
                "Name",
                "Valid",
                "BBox Center",
                "X Length",
                "Y Length",
                "Z Length",
                "Volume",
                "Surface Area",
                "Feature Candidates",
            ],
            part_rows,
        ),
    ]

    if report.step_metadata.get("products"):
        sections.append("<h2>STEP Product Names</h2>")
        sections.append(
            "<ul>"
            + "".join(
                f"<li>{html.escape(product)}</li>"
                for product in report.step_metadata.get("products", [])
            )
            + "</ul>"
        )

    for part in report.parts:
        feature_rows = [
            [
                feature.feature_id,
                feature.source_face_id,
                feature.category,
                feature.confidence,
                feature.radius,
                feature.diameter,
                feature.height,
                feature.area,
                flatten_vector(feature.axis_direction),
                feature.notes,
            ]
            for feature in part.features[:max_rows]
        ]
        profile_rows = [
            [
                profile.face_id,
                profile.outer_shape_guess,
                profile.outer_width,
                profile.outer_height,
                profile.outer_area_2d,
                profile.area,
                flatten_vector(profile.center_3d),
                flatten_vector(profile.normal),
                profile.loop_count,
            ]
            for profile in part.planar_profiles[:max_rows]
        ]
        loop_rows = [
            [
                profile.face_id,
                loop.loop_id,
                loop.shape_guess,
                loop.edge_count,
                loop.vertex_count,
                loop.area_2d,
                loop.perimeter,
                loop.bbox_2d.get("width"),
                loop.bbox_2d.get("height"),
                compact_points(loop.vertices_2d),
            ]
            for profile in part.planar_profiles
            for loop in profile.loops
        ][:max_rows]
        profile_edge_rows = [
            [
                profile.face_id,
                loop.loop_id,
                edge.edge_index,
                edge.curve_type,
                edge.length,
                edge.angle_degrees,
                edge.radius,
                compact_point(edge.start_2d),
                compact_point(edge.end_2d),
            ]
            for profile in part.planar_profiles
            for loop in profile.loops
            for edge in loop.edges
        ][:max_rows]
        face_rows = [
            [
                face.face_id,
                face.surface_type,
                face.area,
                face.radius,
                face.diameter,
                face.height,
                face.u_span_degrees,
                face.orientation,
            ]
            for face in part.faces[:max_rows]
        ]
        edge_rows = [
            [
                edge.edge_id,
                edge.curve_type,
                edge.length,
                edge.radius,
                edge.diameter,
                edge.angle_degrees,
            ]
            for edge in part.edges[:max_rows]
        ]

        sections.append(f"<h2>{html.escape(part.part_id)} - {html.escape(part.name)}</h2>")
        sections.append("<h3>Feature Candidates</h3>")
        sections.append(
            html_table(
                [
                    "ID",
                    "Face",
                    "Category",
                    "Confidence",
                    "Radius",
                    "Diameter",
                    "Height",
                    "Area",
                    "Axis Dir",
                    "Notes",
                ],
                feature_rows,
            )
        )
        sections.append("<h3>Planar Face Profiles</h3>")
        sections.append(
            html_table(
                [
                    "Face",
                    "Outer Shape",
                    "Outer Width",
                    "Outer Height",
                    "Outer 2D Area",
                    "Face Area",
                    "Face Center",
                    "Normal",
                    "Loops",
                ],
                profile_rows,
            )
        )
        sections.append("<h3>Planar Loops / Sketch Boundaries</h3>")
        sections.append(
            html_table(
                [
                    "Face",
                    "Loop",
                    "Shape",
                    "Edges",
                    "Vertices",
                    "2D Area",
                    "Perimeter",
                    "Width",
                    "Height",
                    "2D Vertices",
                ],
                loop_rows,
            )
        )
        sections.append("<h3>Planar Edge Dimensions</h3>")
        sections.append(
            html_table(
                [
                    "Face",
                    "Loop",
                    "Edge",
                    "Curve",
                    "Length",
                    "Angle",
                    "Radius",
                    "Start 2D",
                    "End 2D",
                ],
                profile_edge_rows,
            )
        )
        sections.append("<h3>Face Inventory</h3>")
        sections.append(
            html_table(
                [
                    "Face",
                    "Surface",
                    "Area",
                    "Radius",
                    "Diameter",
                    "Height",
                    "Angular Span",
                    "Orientation",
                ],
                face_rows,
            )
        )
        sections.append("<h3>Edge Inventory</h3>")
        sections.append(
            html_table(
                ["Edge", "Curve", "Length", "Radius", "Diameter", "Angle"],
                edge_rows,
            )
        )
        if len(part.faces) > max_rows or len(part.edges) > max_rows or len(part.features) > max_rows:
            sections.append(
                f"<p>Only the first {max_rows} rows are shown in this HTML report. "
                "The JSON and CSV files contain the exported analysis data.</p>"
            )

    sections.append("<h2>Notes</h2>")
    sections.append("<ul>" + "".join(f"<li>{html.escape(note)}</li>" for note in report.notes) + "</ul>")

    stylesheet = """
    body { font-family: Segoe UI, Arial, sans-serif; margin: 28px; color: #202124; }
    h1 { font-size: 28px; margin-bottom: 8px; }
    h2 { margin-top: 28px; border-bottom: 1px solid #d7dce2; padding-bottom: 6px; }
    h3 { margin-top: 20px; }
    table { border-collapse: collapse; width: 100%; margin: 12px 0 20px; font-size: 13px; }
    th, td { border: 1px solid #d7dce2; padding: 7px 8px; text-align: left; vertical-align: top; }
    th { background: #eef2f6; font-weight: 600; }
    tr:nth-child(even) td { background: #fafbfc; }
    ul { line-height: 1.45; }
    """
    document = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>STEP Analysis - {html.escape(source_name)}</title>"
        f"<style>{stylesheet}</style></head><body>"
        + "".join(sections)
        + "</body></html>"
    )
    path.write_text(document, encoding="utf-8")


def safe_folder_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-")
    return cleaned or "step_report"


def write_reports(
    report: AnalysisReport,
    output_dir: Path,
    stem: str,
    max_rows: int,
    package_dir: Path | None = None,
) -> dict[str, Path]:
    safe_stem = safe_folder_name(stem)
    output_dir = output_dir / safe_stem
    output_dir.mkdir(parents=True, exist_ok=True)
    source_path = Path(report.source_file)
    source_copy = output_dir / f"{safe_stem}{source_path.suffix or '.step'}"
    if source_path.resolve() != source_copy.resolve():
        shutil.copy2(source_path, source_copy)
    paths = {
        "source_step": source_copy,
        "json": output_dir / f"{stem}_analysis.json",
        "parts_csv": output_dir / f"{stem}_parts.csv",
        "features_csv": output_dir / f"{stem}_features.csv",
        "planar_profiles_csv": output_dir / f"{stem}_planar_profiles.csv",
        "planar_loops_csv": output_dir / f"{stem}_planar_loops.csv",
        "planar_edges_csv": output_dir / f"{stem}_planar_edges.csv",
        "build123d_brief": output_dir / f"{stem}_build123d_brief.md",
        "build123d_script": output_dir / f"{stem}_build123d_model.py",
        "build123d_step": output_dir / f"{stem}_from_build123d.step",
        "html": output_dir / f"{stem}_report.html",
    }
    write_json(report, paths["json"])
    write_part_summary_csv(report, paths["parts_csv"])
    write_features_csv(report, paths["features_csv"])
    write_planar_profiles_csv(report, paths["planar_profiles_csv"])
    write_planar_loops_csv(report, paths["planar_loops_csv"])
    write_planar_edges_csv(report, paths["planar_edges_csv"])
    write_build123d_brief(report, paths["build123d_brief"])
    write_build123d_model_script(
        report,
        paths["build123d_script"],
        paths["source_step"].name,
        safe_stem,
    )
    write_exact_build123d_step(paths["source_step"], paths["build123d_step"])
    write_html(report, paths["html"], max_rows=max_rows)
    if package_dir is not None:
        paths.update(package_build123d_outputs(paths, package_dir, stem))
    return paths


def print_console_summary(report: AnalysisReport, outputs: dict[str, Path]) -> None:
    print(f"Analyzed: {report.source_file}")
    print(f"Parts found: {report.part_count}")
    print(f"Detected STEP units: {', '.join(report.detected_step_units) or 'not reported'}")
    for part in report.parts:
        box = part.bounding_box
        print(
            f"- {part.part_id} {part.name}: "
            f"{box.length_x} x {box.length_y} x {box.length_z}, "
            f"volume={part.mass_properties.volume}, "
            f"features={len(part.features)}, "
            f"planar_profiles={len(part.planar_profiles)}"
        )
        feature_counts = Counter(feature.category for feature in part.features)
        if feature_counts:
            print("  feature candidates: " + ", ".join(f"{k}={v}" for k, v in feature_counts.items()))
    print("Reports written:")
    for label, output_path in outputs.items():
        print(f"  {label}: {output_path.resolve()}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze STEP/STP CAD files and export dimensional reports."
    )
    parser.add_argument(
        "step_files",
        nargs="+",
        type=Path,
        help="Path(s) to .step or .stp file(s)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("reports"),
        help="Directory for JSON/CSV/HTML output files",
    )
    parser.add_argument(
        "--max-html-rows",
        type=int,
        default=300,
        help="Maximum face/edge/feature rows per section in the HTML report",
    )
    parser.add_argument(
        "--package-dir",
        type=Path,
        default=Path(r"D:\elctro mech assesment"),
        help="Directory where a clean build123d package folder is created",
    )
    parser.add_argument(
        "--no-package",
        action="store_true",
        help="Do not create the extra build123d .py/.step package folder",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    package_dir = None if args.no_package else args.package_dir
    failures = 0

    for index, step_file in enumerate(args.step_files, start=1):
        if len(args.step_files) > 1:
            print(f"\n=== [{index}/{len(args.step_files)}] {step_file} ===")

        if not step_file.exists():
            print(f"STEP file not found: {step_file}", file=sys.stderr)
            failures += 1
            continue
        if step_file.suffix.lower() not in {".step", ".stp"}:
            print(f"Expected a .step or .stp file, got: {step_file.suffix}", file=sys.stderr)
            failures += 1
            continue

        try:
            report = analyze_step(step_file)
            outputs = write_reports(
                report,
                args.output_dir,
                step_file.stem,
                args.max_html_rows,
                package_dir=package_dir,
            )
        except Exception as exc:
            print(f"Analysis failed for {step_file}: {exc}", file=sys.stderr)
            failures += 1
            continue

        print_console_summary(report, outputs)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
