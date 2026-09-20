"""
Build raw GIS extrusion input for the TUD campus benchmark.

Input: City4CFD repository example dataset (examples/TUDCampus)
  - polygons/tudcampus.geojson      (BAG footprints, EPSG:28992)
  - point_cloud/sampled_buildings.laz
  - point_cloud/sampled_ground_1m.laz

Output: tud_raw_gis_extruded.stl  (buildings extruded from z=0,
height = P95(building points) - median(ground points), 3 m default
where no points — same convention as the case-study raw GIS baseline).

This is input PREPARATION (analogous to the pre-existing NGII extruded
Shapefile), not part of the pipeline run.
"""
import json, time
import numpy as np
import laspy
import shapely
from shapely.geometry import shape
from shapely.ops import unary_union
import trimesh

t0 = time.time()

BASE = 'city4cfd_repo/examples/TUDCampus'

# --- load point clouds ---
las = laspy.read(f'{BASE}/point_cloud/sampled_buildings.laz')
bpts = np.column_stack([np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)]).astype(np.float64)
las_g = laspy.read(f'{BASE}/point_cloud/sampled_ground_1m.laz')
gpts = np.column_stack([np.asarray(las_g.x), np.asarray(las_g.y), np.asarray(las_g.z)]).astype(np.float64)
print(f"building pts: {len(bpts):,}  ground pts: {len(gpts):,}  ({time.time()-t0:.0f}s)")

# --- load footprints ---
d = json.load(open(f'{BASE}/polygons/tudcampus.geojson'))
recs = [(ft['properties'].get('identificatie'), shape(ft['geometry']))
        for ft in d['features'] if ft.get('geometry')]

# restrict to point-cloud extent (buffered 5 m)
xmin, ymin = bpts[:, 0].min() - 5, bpts[:, 1].min() - 5
xmax, ymax = bpts[:, 0].max() + 5, bpts[:, 1].max() + 5
sel = []
for ident, p in recs:
    if p.is_empty:
        continue
    b = p.bounds
    if b[0] >= xmin and b[1] >= ymin and b[2] <= xmax and b[3] <= ymax:
        sel.append((ident, p))
    else:
        # keep if substantially inside (centroid within extent)
        if xmin <= p.centroid.x <= xmax and ymin <= p.centroid.y <= ymax:
            sel.append((ident, p))
print(f"selected footprints: {len(sel)}  ({time.time()-t0:.0f}s)")

# --- heights from point cloud ---
heights, n_default, n_pts = [], 0, []
gz_tree_pts = gpts
for ident, p in sel:
    minx, miny, maxx, maxy = p.bounds
    m = (bpts[:, 0] >= minx) & (bpts[:, 0] <= maxx) & (bpts[:, 1] >= miny) & (bpts[:, 1] <= maxy)
    cand = bpts[m]
    inside = shapely.contains_xy(p, cand[:, 0], cand[:, 1]) if len(cand) else np.zeros(0, bool)
    bz = cand[inside][:, 2]
    n_pts.append(int(inside.sum()))
    if len(bz) >= 5:
        # local ground level
        pb = p.buffer(10)
        gm = (gpts[:, 0] >= pb.bounds[0]) & (gpts[:, 0] <= pb.bounds[2]) & \
             (gpts[:, 1] >= pb.bounds[1]) & (gpts[:, 1] <= pb.bounds[3])
        gcand = gpts[gm]
        gi = shapely.contains_xy(pb, gcand[:, 0], gcand[:, 1]) if len(gcand) else np.zeros(0, bool)
        gz = np.median(gcand[gi][:, 2]) if gi.sum() >= 3 else 0.0
        h = float(np.percentile(bz, 95) - gz)
        if h < 1.0:  # degenerate/negative -> default
            h = 3.0
            n_default += 1
        heights.append(h)
    else:
        heights.append(3.0)
        n_default += 1

heights = np.array(heights)
n_pts = np.array(n_pts)
print(f"heights done: median {np.median(heights):.1f} m, p90 {np.percentile(heights,90):.1f}, "
      f"max {heights.max():.1f}; default(3m): {n_default}; "
      f"pts/building median {np.median(n_pts):.0f}  ({time.time()-t0:.0f}s)")

# --- extrude from z=0 ---
meshes, n_invalid_fixed = [], 0
for (ident, p), h in zip(sel, heights):
    if p.geom_type == 'MultiPolygon':
        parts = list(p.geoms)
    elif p.geom_type == 'Polygon':
        parts = [p]
    else:
        continue
    for part in parts:
        if not part.is_valid:
            part = part.buffer(0)
            n_invalid_fixed += 1
            if part.is_empty:
                continue
        try:
            m = trimesh.creation.extrude_polygon(part, float(h))
            meshes.append(m)
        except Exception as e:
            print(f"  extrude failed for {ident}: {e}")

raw = trimesh.util.concatenate(meshes)
raw.export('tud_raw_gis_extruded.stl')
print(f"extruded: {len(meshes)} parts, {len(raw.faces):,} faces, "
      f"{len(raw.split(only_watertight=False)):,} components; invalid fixed: {n_invalid_fixed}")
print(f"TOTAL input prep time: {time.time()-t0:.0f}s")
