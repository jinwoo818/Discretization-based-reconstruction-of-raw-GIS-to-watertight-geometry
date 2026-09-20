# Section 4.3 benchmark reproduction (TU Delft / Stanford)

Reproduces the comparative evaluation of Section 4.3 of the paper on the
public campus datasets associated with City4CFD (Pađen et al. 2024,
Build. Environ. 265:111978). All runs use the default configuration — no
parameter tuning, no manual repair, no post-hoc editing.

## 1. Get the data

The input point clouds are public and are **not** redistributed here
(sizes); download them into `benchmarks/data/`:

- `tud_buildings_pc.laz`, `tud_ground_pc.laz` — TU Delft campus, exactly as
  distributed with the City4CFD repository:
  https://github.com/tudelft3d/City4CFD (`examples/TUDCampus/point_cloud/`,
  files `sampled_buildings.laz` and `sampled_ground_1m.laz`)
- `stanford_buildings_pc.laz`, `stanford_ground_pc.laz` — generated from the
  USGS 3DEP LiDAR point cloud of Santa Clara County (2020 A20 collection)
  with `stanford_pc_prep.py` (edit the tile directory path at the top).
  Tiles: 07259800, 07259825, 07509800, 07509825 via the TNM Access API
  (https://tnmaccess.nationalmap.gov/). The script extracts the building
  (class 6) and ground (class 2) points within the 1.6 km² circular domain
  on the Stanford Science and Engineering Quad and reprojects to metric
  coordinates — the same separation of building and ground points that the
  City4CFD preparation workflow requires.

## 2. Run

```
# point-cloud input route (headline results of Section 4.3)
python ../pipeline.py data/tud_buildings_pc.laz      --ground data/tud_ground_pc.laz      -o outputs/tud_pc_modeled.stl
python ../pipeline.py data/stanford_buildings_pc.laz --ground data/stanford_ground_pc.laz -o outputs/stanford_pc_modeled.stl

# mesh input route (secondary; automatically extruded inputs, see paper)
python ../pipeline.py data/tud_raw_gis_extruded.stl      -o outputs/tud_modeled.stl
python ../pipeline.py data/stanford_raw_gis_extruded.stl -o outputs/stanford_modeled.stl

# geometric quality metrics (Section 2.7)
python ../metrics.py outputs/<each file>

# figure
python render_figure.py
```

## 3. Reference results

Reference results of the paper (virtualized server CPU, 2 vCPUs, 2.9 GHz):

| | TU Delft | Stanford |
|---|---|---|
| Input points (building / ground) | 5,988,641 / 1,000,000 | 6,847,242 / 14,064,668 |
| DBSCAN clusters | 224 | 184 |
| Reconstructed | 211 (100% watertight) | 181 (100% watertight) |
| Excluded by analysis filters (H < 2 m / A < 8 m²) | 13 (0 / 13) | 3 (2 / 1) |
| Failed reconstructions | 0 | 0 |
| Faces | 8,732 | 11,912 |
| Runtime (point route) | 26.7 s | 33.6 s |
| Runtime (mesh route) | 6.2 s | 2.3 s |

For comparison, City4CFD reports 136 s (TU Delft) and 385 s (Stanford) on a
24-core AMD Threadripper 3970X [Pađen et al. 2024]. The hardware, LoD
targets, and object definitions differ; see Section 4.3 of the paper for
the qualifications.

## Contents

- `stanford_pc_prep.py` — USGS 3DEP tiles -> building/ground point clouds
- `render_figure.py` — regenerates the Section 4.3 figure
- `outputs/` — the four pipeline outputs as shipped with the paper
