#!/usr/bin/env python3
"""
metrics.py -- Geometric quality metrics for urban building geometry (STL).

Implements the metric definitions of Section 2.7 of the revised manuscript:

  Choi, Hong, Kim, and Jeong (2026), "Discretization-based Reconstruction of
  Watertight Building Geometry from Raw GIS Data for Urban CFD Simulation",
  Developments in the Built Environment (DIBE-D-26-01168R1).

Metrics
-------
  component count     connected components under manifold-edge adjacency:
                      two faces belong to the same component when they
                      share an edge that belongs to exactly two faces;
                      non-manifold edges (three or more faces) separate
                      the incident face groups, following the component
                      definition of Table 4 in the manuscript
  face count          total triangular faces (each face is assigned to
                      exactly one component)
  watertightness      components in which every edge is shared by exactly
                      two faces of that component (closed 2-manifolds)
  non-manifold edges  edges shared by more than two faces (whole-mesh level)
  open edges          edges belonging to a single face (whole-mesh level)
  degenerate faces    faces whose three vertices lie on an identical point
                      or line (area below the numerical tolerance)
  micro-faces         faces with an area smaller than --micro-area
                      (default 0.25 m^2); reported as the number and ratio
                      of components containing at least one micro-face and
                      as the ratio of micro-faces to the total face count

Components with fewer than --min-faces faces are reported separately as
residual fragments (small open shells), following the metric definitions of
the revised manuscript. All metrics are computed after merging coincident
vertices so that connectivity reflects true geometric adjacency. The
component decomposition is computed with an explicit union-find over the
manifold face-adjacency graph.

Usage
-----
  python metrics.py geometry.stl
  python metrics.py geometry.stl --micro-area 0.25 --min-faces 8 --json metrics.json
"""

import argparse
import json

import numpy as np
import trimesh


def face_areas(mesh):
    """Per-face areas of a triangle mesh (float64)."""
    tri = np.asarray(mesh.vertices, dtype=np.float64)[np.asarray(mesh.faces)]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    return np.linalg.norm(n, axis=1) / 2.0


def edge_face_map(faces):
    """Map each undirected edge (sorted vertex pair) to its incident faces."""
    edge_faces = {}
    for fi, (a, b, c) in enumerate(faces):
        for u, v in ((a, b), (b, c), (c, a)):
            e = (u, v) if u < v else (v, u)
            fl = edge_faces.get(e)
            if fl is None:
                edge_faces[e] = [fi]
            else:
                fl.append(fi)
    return edge_faces


def analyze(path, micro_area=0.25, min_faces=8, deg_tol=1e-12):
    """Compute all geometric quality metrics for an STL/OBJ/PLY file."""
    mesh = trimesh.load(path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh.dump())
    mesh.merge_vertices()

    faces = np.asarray(mesh.faces)
    n_faces = len(faces)
    if n_faces == 0:
        raise ValueError(f"No faces found in {path}")

    areas = face_areas(mesh)
    degenerate_mask = areas <= deg_tol
    micro_mask = areas < micro_area

    # ----- face-adjacency components (union-find over manifold edges) ------
    edge_faces = edge_face_map(faces)

    parent = list(range(n_faces))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for fl in edge_faces.values():
        if len(fl) == 2:  # manifold edge: connect its two faces
            r0, r1 = find(fl[0]), find(fl[1])
            if r0 != r1:
                parent[r1] = r0

    roots = {}
    labels = np.empty(n_faces, dtype=np.int64)
    for fi in range(n_faces):
        r = find(fi)
        if r not in roots:
            roots[r] = len(roots)
        labels[fi] = roots[r]
    n_comp = len(roots)

    # ----- per-component statistics ----------------------------------------
    comp_faces = np.bincount(labels, minlength=n_comp)
    comp_degenerate = np.bincount(labels, weights=degenerate_mask, minlength=n_comp)
    comp_micro = np.bincount(labels, weights=micro_mask, minlength=n_comp)

    # within-component edge multiplicity: an edge may be incident to faces
    # of several components around a non-manifold edge
    comp_edge_total = np.zeros(n_comp, dtype=np.int64)
    comp_edge_good = np.zeros(n_comp, dtype=np.int64)
    nm_edges = 0
    open_edges = 0
    for fl in edge_faces.values():
        per_comp = {}
        for fi in fl:
            c = labels[fi]
            per_comp[c] = per_comp.get(c, 0) + 1
        for c, k in per_comp.items():
            comp_edge_total[c] += 1
            if k == 2:
                comp_edge_good[c] += 1
        if len(fl) > 2:
            nm_edges += 1
        elif len(fl) == 1:
            open_edges += 1
    # a component is watertight iff every edge is shared by exactly two of
    # its own faces
    comp_watertight = (comp_edge_good == comp_edge_total) & (comp_faces > 0)

    comp_stats = [
        {
            "faces": int(comp_faces[c]),
            "watertight": bool(comp_watertight[c]),
            "degenerate_faces": int(comp_degenerate[c]),
            "micro_faces": int(comp_micro[c]),
        }
        for c in range(n_comp)
    ]
    bodies = [s for s in comp_stats if s["faces"] >= min_faces]
    fragments = [s for s in comp_stats if s["faces"] < min_faces]

    def summarize(group):
        n = len(group)
        f_total = sum(s["faces"] for s in group)
        wt = sum(1 for s in group if s["watertight"])
        deg_c = sum(1 for s in group if s["degenerate_faces"] > 0)
        mic_c = sum(1 for s in group if s["micro_faces"] > 0)
        mic_f = sum(s["micro_faces"] for s in group)
        return {
            "components": n,
            "faces": f_total,
            "watertight_components": wt,
            "watertight_ratio": round(wt / n, 4) if n else None,
            "components_with_degenerate_faces": deg_c,
            "components_with_micro_faces": mic_c,
            "micro_face_ratio_components": round(mic_c / n, 4) if n else None,
            "micro_faces": mic_f,
            "micro_face_ratio_faces": round(mic_f / f_total, 4) if f_total else None,
        }

    return {
        "file": path,
        "parameters": {
            "micro_area_threshold_m2": micro_area,
            "degenerate_area_tolerance_m2": deg_tol,
            "fragment_face_threshold": min_faces,
        },
        "total": {
            "components": n_comp,
            "faces": int(n_faces),
            "non_manifold_edges": nm_edges,
            "open_edges": open_edges,
            "degenerate_faces": int(degenerate_mask.sum()),
            "micro_faces": int(micro_mask.sum()),
            "micro_face_ratio_faces": round(float(micro_mask.sum()) / n_faces, 4),
        },
        "bodies": summarize(bodies),
        "residual_fragments": summarize(fragments),
    }


def print_report(r):
    """Print a human-readable summary of the analysis result."""
    t, b, f = r["total"], r["bodies"], r["residual_fragments"]
    line = "-" * 66
    print(f"\nGeometric quality metrics: {r['file']}")
    print(line)
    print(f"micro-face threshold : {r['parameters']['micro_area_threshold_m2']} m^2")
    print(f"fragment threshold   : components with < {r['parameters']['fragment_face_threshold']} faces")
    print(line)
    print(f"{'':36}{'all':>9}{'bodies':>10}{'fragments':>11}")
    def fm(v, w):
        return ("-" if v is None else str(v)).rjust(w)

    print(f"{'components':36}{t['components']:>9,}{b['components']:>10,}{f['components']:>11,}")
    print(f"{'faces':36}{t['faces']:>9,}{b['faces']:>10,}{f['faces']:>11,}")
    print(f"{'watertight components':36}{'':>9}{b['watertight_components']:>10,}{f['watertight_components']:>11,}")
    print(f"{'watertight ratio':36}{'':>9}{fm(b['watertight_ratio'],10)}{fm(f['watertight_ratio'],11)}")
    print(f"{'components w/ degenerate faces':36}{'':>9}{b['components_with_degenerate_faces']:>10,}{f['components_with_degenerate_faces']:>11,}")
    print(f"{'components w/ micro-faces':36}{'':>9}{b['components_with_micro_faces']:>10,}{f['components_with_micro_faces']:>11,}")
    print(f"{'micro-face ratio (components)':36}{'':>9}{fm(b['micro_face_ratio_components'],10)}{fm(f['micro_face_ratio_components'],11)}")
    print(f"{'micro-face ratio (faces)':36}{'':>9}{fm(b['micro_face_ratio_faces'],10)}{fm(f['micro_face_ratio_faces'],11)}")
    print(line)
    print(f"non-manifold edges (whole mesh) : {t['non_manifold_edges']:,}")
    print(f"open edges (whole mesh)         : {t['open_edges']:,}")
    print(f"degenerate faces (whole mesh)   : {t['degenerate_faces']:,}")
    print(f"micro-faces (whole mesh)        : {t['micro_faces']:,}")
    print(line)


def main():
    p = argparse.ArgumentParser(
        description="Geometric quality metrics for urban building geometry",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("input", help="Input mesh file (STL/OBJ/PLY)")
    p.add_argument("--micro-area", type=float, default=0.25,
                   help="Micro-face area threshold in m^2")
    p.add_argument("--min-faces", type=int, default=8,
                   help="Components with fewer faces are reported as residual fragments")
    p.add_argument("--deg-tol", type=float, default=1e-12,
                   help="Faces with area <= this tolerance count as degenerate (m^2)")
    p.add_argument("--json", default=None, help="Write the full result to a JSON file")
    a = p.parse_args()

    r = analyze(a.input, micro_area=a.micro_area, min_faces=a.min_faces,
                deg_tol=a.deg_tol)
    print_report(r)
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(r, fh, indent=2)
        print(f"Full result written to {a.json}")


if __name__ == "__main__":
    main()
