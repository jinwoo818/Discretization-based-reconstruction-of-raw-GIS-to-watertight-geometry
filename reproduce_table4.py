#!/usr/bin/env python3
"""
reproduce_table4.py -- Reproduce Table 4 of the revised manuscript
("Geometric quality comparison of baseline, reference, and modeled data")
from the benchmark geometries in data/benchmark/.

The script runs the released measurement script (metrics.py) on the three
benchmark geometries and compares every reported value against the
manuscript. A cell-by-cell PASS/FAIL summary is printed.

Usage
-----
  python reproduce_table4.py
  python reproduce_table4.py --data-dir data/benchmark
"""

import argparse
import os
import sys

import metrics

FILES = {
    "Baseline": "baseline_raw_gis.stl",
    "Reference": "reference_manual_cleanup.stl",
    "Modeled": "modeled_pipeline_output.stl",
}

# Values reported in Table 4 of the revised manuscript
EXPECTED = {
    "Baseline": dict(
        components=15659, faces=365668, watertightness=98.8, nm_edges=568,
        degenerate=293, degenerate_ratio=1.9,
        micro=1905, micro_ratio=12.2, micro_faces=8608, micro_face_ratio=2.4),
    "Reference": dict(
        components=15072, faces=284223, watertightness=96.9, nm_edges=527,
        degenerate=16, degenerate_ratio=0.1,
        micro=745, micro_ratio=4.9, micro_faces=2920, micro_face_ratio=1.0),
    "Modeled": dict(
        components=15056, faces=273432, watertightness=100.0, nm_edges=0,
        degenerate=0, degenerate_ratio=0.0,
        micro=188, micro_ratio=1.2, micro_faces=394, micro_face_ratio=0.1),
}


def table_row(result):
    """Compute the Table 4 values of one dataset from a metrics.analyze result."""
    t, b, f = result["total"], result["bodies"], result["residual_fragments"]
    comps = t["components"]
    faces = t["faces"]
    wt = b["watertight_components"] + f["watertight_components"]
    deg = b["components_with_degenerate_faces"] + f["components_with_degenerate_faces"]
    mic = b["components_with_micro_faces"] + f["components_with_micro_faces"]
    return {
        "components": comps,
        "faces": faces,
        "watertightness": round(wt / comps * 100, 1),
        "nm_edges": t["non_manifold_edges"],
        "degenerate": deg,
        "degenerate_ratio": round(deg / comps * 100, 1),
        "micro": mic,
        "micro_ratio": round(mic / comps * 100, 1),
        "micro_faces": t["micro_faces"],
        "micro_face_ratio": round(t["micro_faces"] / faces * 100, 1),
    }


ROWS = [
    ("Components", "components", "d"),
    ("Total faces", "faces", "d"),
    ("Watertightness (%)", "watertightness", ".1f"),
    ("Non-manifold edges", "nm_edges", "d"),
    ("Degenerate-face bldgs. (%)", "degenerate", "d"),
    ("  ratio", "degenerate_ratio", ".1f"),
    ("Micro-face bldgs. (%)", "micro", "d"),
    ("  ratio", "micro_ratio", ".1f"),
    ("Total micro-faces (%)", "micro_faces", "d"),
    ("  ratio", "micro_face_ratio", ".1f"),
]

INT_KEYS = {"components", "faces", "nm_edges", "degenerate", "micro", "micro_faces"}


def main():
    ap = argparse.ArgumentParser(
        description="Reproduce Table 4 (geometric quality comparison) from the "
                    "benchmark geometries")
    ap.add_argument("--data-dir", default="data/benchmark",
                    help="Directory containing the three benchmark STL files")
    ap.add_argument("--micro-area", type=float, default=0.25)
    ap.add_argument("--min-faces", type=int, default=8)
    args = ap.parse_args()

    results = {}
    for name, fname in FILES.items():
        path = os.path.join(args.data_dir, fname)
        if not os.path.exists(path):
            sys.exit(f"Benchmark file not found: {path}")
        results[name] = table_row(
            metrics.analyze(path, micro_area=args.micro_area,
                            min_faces=args.min_faces))

    names = list(FILES)
    print()
    print("Table 4. Geometric quality comparison of baseline, reference, "
          "and modeled data")
    print("-" * 78)
    print(f"{'Metric':32}" + "".join(f"{n:>15}" for n in names))
    print("-" * 78)
    all_ok = True
    for label, key, fmt in ROWS:
        line = f"{label:32}"
        for n in names:
            got = results[n][key]
            exp = EXPECTED[n][key]
            if key in INT_KEYS:
                ok = got == exp
            else:
                ok = abs(got - exp) < 0.05
            all_ok &= ok
            mark = "" if ok else "  (!)"
            line += f"{format(got, fmt):>13}{mark:2}"
        print(line)
    print("-" * 78)
    if all_ok:
        print("All values match Table 4 of the revised manuscript.")
    else:
        print("MISMATCH: at least one value differs from the manuscript "
              "(marked with (!)).")
        sys.exit(1)


if __name__ == "__main__":
    main()
