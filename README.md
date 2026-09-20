# Discretization-based Reconstruction of Raw GIS to Watertight Geometry

Source code and benchmark data for:

> Choi, J., Hong, T., Kim, H., and Jeong, K. (2026). *Discretization-based
> Reconstruction of Watertight Building Geometry from Raw GIS Data for Urban
> CFD Simulation*. Developments in the Built Environment (DIBE-D-26-01168R1,
> revised manuscript).

The pipeline reconstructs watertight LOD1 building geometry from heterogeneous
and defective GIS data. It accepts two input regimes (Section 2.2 of the
manuscript): **mesh-based** inputs (STL/OBJ/PLY) and **point-cloud** inputs
(LAS/LAZ/XYZ, e.g. classified airborne LiDAR). In both regimes it (i)
identifies building candidates (connected-component decomposition for meshes;
DBSCAN clustering for point clouds), (ii) discards the input connectivity
through surface sampling (meshes) or uses the clustered points directly
(point clouds), (iii) extracts building footprints as concave (alpha-shape)
hulls, (iv) simplifies the footprints with the Douglas–Peucker algorithm,
and (v) extrudes and regularizes the result into watertight closed manifolds
with integrated physics-aware filtering (minimum building height and
footprint area).

## Repository contents

| File | Description |
|---|---|
| `pipeline.py` | Complete reconstruction pipeline (v1.2, the exact version used for the revised manuscript) |
| `metrics.py` | Measurement script for the geometric quality metrics (Section 2.7 / Table 4) |
| `reproduce_table4.py` | One-command reproduction of Table 4 from the benchmark geometries |
| `benchmarks/` | Reproduction of the Section 4.3 comparative evaluation (TU Delft / Stanford public datasets) |
| `figures/render_process_figure.py` | Generates the process illustration figure (Section 2.5): reconstruction stages of a U-shaped building and the effect of the concave-hull ratio and DP tolerance |
| `configs/case_study_default.json` | Full configuration of the case study (mirrors built-in defaults / Table 2) |
| `example/input_defective_gis.stl` | Small synthetic defective input (gaps, non-manifold edges, open shells, debris) |
| `example/expected_output_LOD1.stl` | Expected pipeline output for the example input |
| `data/benchmark/baseline_raw_gis.stl` | Case-study baseline: raw extruded GIS geometry (15,659 components) |
| `data/benchmark/reference_manual_cleanup.stl` | Case-study reference: manually cleaned geometry (15,072 components) |
| `data/benchmark/modeled_pipeline_output.stl` | Case-study modeled: pipeline output geometry (15,056 components) |
| `results/benchmark_reference_values.md` | Reference values (Table 4), checksums, and the end-to-end processing log |
| `requirements.txt` | Python dependencies (pinned to the versions used for the manuscript) |
| `LICENSE` | MIT license (source code) |

## Requirements

- Python 3.9 or newer (the manuscript results were produced with Python 3.14)
- Dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

The manuscript results were produced with the following versions:
`numpy 2.5.3`, `trimesh 5.1.0` (the `seed` argument of
`trimesh.sample.sample_surface` is required), `shapely 2.1.2`,
`scikit-learn 1.9.1`, and `mapbox-earcut 2.1.0`.

## Running the pipeline

```bash
python pipeline.py input.stl -o output_LOD1.stl
```

All hyperparameters can be overridden from the command line. The defaults
correspond one-to-one to the values reported in Table 2 of the manuscript:

| CLI option | Default | Symbol in manuscript | Description |
|---|---|---|---|
| `--density` | 10.0 | ρ | Surface sampling density (pts/m²) |
| `--eps` | 2.5 | ε | DBSCAN neighborhood radius (m) |
| `--min_pts` | 30 | MinPts | DBSCAN minimum cluster size |
| `--alpha` | 0.2 | α | Concave-hull ratio |
| `--simplify` | 0.5 | ε_dp | Douglas–Peucker simplification tolerance (m) |
| `--min_height` | 2.0 | H_min | Minimum building height (m) |
| `--min_fp_area` | 8.0 | A_fp,min | Minimum footprint area (m²) |
| `--classification` | – | – | LAS class to keep for point-cloud inputs (e.g. 6 = building) |
| `--ground` | – | – | Ground point cloud for base-elevation estimation of point-cloud inputs |
| `--ground_buffer` | 5.0 | – | Buffer (m) around the cluster bounding box for the ground-point lookup |

Additional fixed defaults (documented in `configs/case_study_default.json`):
height percentiles Z_5 / Z_99 for building-height derivation,
`max_pts_per_building` = 2000, `min_pts_per_building` = 100,
`allow_holes` = False, `random_seed` = 0.

### Point-cloud input (v1.2)

```bash
python pipeline.py buildings.laz --classification 6 --ground ground.laz -o output_LOD1.stl
```

Point-cloud inputs (LAS/LAZ/XYZ) follow the point-cloud identification path
of Section 2.2: buildings are identified by DBSCAN clustering of the raw
points, and the clustered point set is used directly for footprint
extraction (Section 2.3). For classified airborne LiDAR, the building points
are dominated by roof returns, so the base elevation of each cluster is
estimated from the accompanying ground points (`--ground`): the median ground
elevation within the cluster bounding box expanded by `--ground_buffer`,
with the building height taken as the 99th-percentile roof elevation minus
this base elevation (Section 2.5). When no ground cloud is provided, the
percentile definition of Eq. (9) is used directly (appropriate for
mesh-sampled points, which include the facades).

### Geometry-consistency safeguards (v1.1)

STL files store vertex coordinates in single precision (float32). At
UTM-scale coordinates (e.g., ~5×10⁵ m) the coordinate resolution of a
float32 value is approximately 6 cm, so footprint vertices that are closer
than this resolution can collapse onto each other on export, producing
degenerate faces or coincident walls. The pipeline therefore applies two
deterministic safeguards, both with a 0.15 m threshold (above the
single-precision resolution):

- **Ring regularization**: after Douglas–Peucker simplification, footprint
  edges shorter than 0.15 m are removed, and so are vertices whose deviation
  from the line joining their neighbors is below 0.15 m (near-collinear
  vertices).
- **Coincident-wall resolution**: if a footprint shares a boundary vertex
  with an already reconstructed building, it is inset by 0.15 m (mitre
  join) so that adjacent buildings never share coincident wall faces.

After reconstruction, the pipeline verifies the final combined mesh at
single-precision resolution and reports the resulting non-manifold edge and
degenerate face counts (`[QC]` log line). The case-study output has zero of
both.

### Deterministic output

The random seed used for surface sampling is fixed (`random_seed` = 0 by
default), so identical input and configuration produce bit-identical output
across repeated executions.

### End-to-end example

A small synthetic defective input and its expected output are included in
`example/` (no licensing restrictions). Running

```bash
python pipeline.py example/input_defective_gis.stl -o example/output_LOD1.stl
```

produces a file identical to `example/expected_output_LOD1.stl`. The example
input contains 11 defective components (non-manifold edges from coincident
building faces, an open shell with a missing face, overlapping buildings, and
debris); the pipeline output contains 10 watertight LOD1 buildings with zero
non-manifold edges and zero open edges.

## Reproducing the case study

The modeled geometry of the case study is reproduced from the baseline (raw
GIS) geometry with the default configuration (no CLI overrides):

```bash
python pipeline.py data/benchmark/baseline_raw_gis.stl -o modeled_model.stl
```

This regenerates the released modeled geometry bit-for-bit (the run in
`results/benchmark_reference_values.md` was verified against
`data/benchmark/modeled_pipeline_output.stl` by checksum). The pipeline
reports the end-to-end accounting of the case study in its log: 15,251
building candidates were identified from the 15,659 input components, of
which 15,056 buildings (98.7%) were successfully reconstructed; 188
candidates were excluded by the physics-aware filtering (1 below the minimum
height, 187 below the minimum footprint area) and 7 failed reconstruction.
All 15,056 reconstructed buildings passed the watertightness verification.
A full case-study run completes within a few minutes on a modern desktop
CPU.

## Verifying the geometric quality metrics (Table 4)

`metrics.py` implements the metric definitions of Section 2.7 of the
manuscript (component counts, edge-based watertightness, non-manifold edges,
degenerate faces, and micro-face statistics):

```bash
python metrics.py data/benchmark/baseline_raw_gis.stl
python metrics.py data/benchmark/reference_manual_cleanup.stl
python metrics.py data/benchmark/modeled_pipeline_output.stl
```

To reproduce the entire quality-comparison table (Table 4 of the manuscript)
in one command, including an automatic check against the published values:

```bash
python reproduce_table4.py
```

Watertightness is evaluated per component as the closed 2-manifold condition
(every edge shared by exactly two faces). Components are decomposed under
manifold-edge adjacency (an edge shared by exactly two faces connects its
two faces; non-manifold edges separate the incident face groups). Components
with fewer than 8 faces are additionally reported as residual fragments
(small open shells); this is a diagnostic split, and the reported values
should be read together with the metric footnotes of Table 4. The
watertightness rate and the ratio metrics of Table 4 are computed over all
connected components, including residual fragments. Use `--json out.json`
to export the full statistics.

## Reproducing the Section 4.3 benchmark

`benchmarks/` reproduces the comparative evaluation on the public TU Delft
and Stanford campus datasets associated with City4CFD (Pađen et al. 2024),
including data-download instructions, preparation scripts, the four pipeline
outputs, and the figure script. See `benchmarks/README.md`.

## Hardware and timing

The processing time reported in the manuscript for the case study (190 s for
approximately 15,000 buildings) was measured on a workstation equipped with
an Intel Core i7-12700KF CPU (3.61 GHz) and 32 GB of memory. The reported
time is the total wall-clock time of a single pipeline execution, including
input loading and output export, measured with the Python `time` module.
The Section 4.3 benchmark runs were executed on a virtualized server CPU
(2 vCPUs, 2.9 GHz); see Section 4.3 of the manuscript and
`benchmarks/README.md` for the reference values and the hardware
qualifications.

## Versioning

- **v1.2** (revised manuscript): adds the point-cloud input path described
  in Sections 2.2–2.3 (LAS/LAZ/XYZ input, LAS classification filtering,
  DBSCAN identification, ground-point base-elevation estimation for airborne
  LiDAR). The mesh-based route is unchanged from v1.1 and reproduces the
  case-study output bit-identically (verified by checksum).
- **v1.1**: adds seeded sampling, ring regularization, coincident-wall
  resolution, and single-precision mesh-level verification, eliminating the
  residual non-manifold edges and degenerate faces of earlier versions.
- **v1.0**: initial release version.

The release tagged `v1.2` corresponds to the revised manuscript. All
results in the manuscript can be reproduced with this tagged version (the
case-study results of the original submission with `v1.1`).

## Data availability

- The three case-study geometries — **baseline** (raw extruded GIS),
  **reference** (manually cleaned), and **modeled** (pipeline output) — are
  included in `data/benchmark/` so that every value of Table 4 and the
  end-to-end pipeline run can be verified directly.
- The original GIS source layers are derived from national mapping data of
  the study area and are subject to data-provider licensing; they are
  available from the corresponding author upon reasonable request.
- The files in `example/` are synthetic and free of licensing restrictions.

## License

The source code is released under the MIT License (see `LICENSE`). The
geometry files in `data/` are provided solely for verification of the
manuscript results.
