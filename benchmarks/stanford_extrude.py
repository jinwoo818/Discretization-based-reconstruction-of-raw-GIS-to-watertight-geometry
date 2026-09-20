"""
Build raw GIS extrusion input for the Stanford benchmark.

Input (same public data sources as the City4CFD 2024 "Stanford case"):
  - OSM building footprints (Science and Engineering Quad area), WGS84
  - USGS 3DEP lidar, Santa Clara County 2020 A20 (4 LAZ tiles,
    NAD83(2011) / California zone 3 (ftUS) + NAVD88 height ftUS)

Domain: circle of 713 m radius (1.6 km^2) centered on the Stanford
Science and Engineering Quad (37.4274 N, -122.1731 W), matching the
published evaluation area size. All data reprojected to local metric
coordinates (UTM 10N) with z in meters.

Output: stanford_raw_gis_extruded.stl (extruded from z=0,
height = P95(points in footprint) - local ground, 3 m default).
"""
import json, time, glob
import numpy as np
import laspy
import shapely
from shapely.geometry import shape, Point, Polygon
from pyproj import Transformer
import trimesh

t0 = time.time()
LAT0, LON0, R = 37.4274, -122.1731, 713.0

# local metric frame = UTM 10N (EPSG:26910) meters
T_WGS84_UTM = Transformer.from_crs("EPSG:4326", "EPSG:26910", always_xy=True)
CX, CY = T_WGS84_UTM.transform(LON0, LAT0)  # domain center in UTM

t_utm2local = lambda x, y: (x - CX, y - CY)

# --- lidar: merge tiles, clip to circle ---
CA3_WKT = None
with laspy.open(sorted(glob.glob('stanford_laz/*.laz'))[0]) as r:
    CA3_CRS = r.header.parse_crs()
T_CA3_UTM = Transformer.from_crs(CA3_CRS, "EPSG:26910", always_xy=True)
FT = 0.3048006096012192

xs, ys, zs = [], [], []
for f in sorted(glob.glob('stanford_laz/*.laz')):
    las = laspy.read(f)
    x = np.asarray(las.x); y = np.asarray(las.y); z = np.asarray(las.z)
    xu, yu = T_CA3_UTM.transform(x, y)          # -> UTM meters
    xl = xu - CX; yl = yu - CY                   # -> local meters
    d2 = xl**2 + yl**2
    m = d2 <= (R + 50)**2
    xs.append(xl[m]); ys.append(yl[m]); zs.append(z[m] * FT)  # ft -> m
    print(f"{f.split('_')[-1]}: kept {m.sum():,} of {len(x):,}")
P = np.column_stack([np.concatenate(xs), np.concatenate(ys), np.concatenate(zs)])
dom_area = np.pi * R**2
print(f"total pts in domain: {len(P):,} | area {dom_area/1e6:.2f} km2 | "
      f"density {len(P)/dom_area:.2f} pts/m2  ({time.time()-t0:.0f}s)")

# --- OSM footprints ---
d = json.load(open('stanford_osm.json'))
polys = []
for e in d['elements']:
    if e['type'] == 'way' and e.get('tags', {}).get('building') and e.get('geometry'):
        polys.append([(g['lon'], g['lat']) for g in e['geometry']])
    elif e['type'] == 'relation':
        for mem in e.get('members', []):
            if mem.get('type') == 'way' and mem.get('role') == 'outer' and mem.get('geometry'):
                polys.append([(g['lon'], g['lat']) for g in mem['geometry']])
print("osm building rings:", len(polys))

shp = []
for coords in polys:
    utm = T_WGS84_UTM.transform([c[0] for c in coords], [c[1] for c in coords])
    ring = list(zip(utm[0], utm[1]))
    if len(ring) >= 4:
        try:
            p = Polygon(ring)
            if not p.is_valid:
                p = p.buffer(0)
            if not p.is_empty and p.area > 0.5:
                shp.append(p)
        except Exception:
            pass
print("valid polygons:", len(shp))

dom = Point(CX, CY).buffer(R)
sel = [p for p in shp if p.intersects(dom)]
# recenter to local coords
sel = [shapely.affinity.translate(p, xoff=-CX, yoff=-CY) for p in sel]
print("selected (intersect domain):", len(sel))

# --- heights ---
heights, defaults, npts = [], 0, []
for p in sel:
    minx, miny, maxx, maxy = p.bounds
    m = (P[:, 0] >= minx) & (P[:, 0] <= maxx) & (P[:, 1] >= miny) & (P[:, 1] <= maxy)
    cand = P[m]
    inside = shapely.contains_xy(p, cand[:, 0], cand[:, 1]) if len(cand) else np.zeros(0, bool)
    bz = cand[inside][:, 2]
    npts.append(int(inside.sum()))
    if len(bz) >= 5:
        pb = p.buffer(15)
        gm = (P[:, 0] >= pb.bounds[0]) & (P[:, 0] <= pb.bounds[2]) & \
             (P[:, 1] >= pb.bounds[1]) & (P[:, 1] <= pb.bounds[3])
        gcand = P[gm]
        out = ~shapely.contains_xy(p, gcand[:, 0], gcand[:, 1])
        gz = float(np.median(gcand[out][:, 2])) if out.sum() >= 3 else 0.0
        h = float(np.percentile(bz, 95) - gz)
        if h < 1.0:
            h = 3.0; defaults += 1
        heights.append(h)
    else:
        heights.append(3.0); defaults += 1

heights = np.array(heights); npts = np.array(npts)
print(f"heights: median {np.median(heights):.1f} m, p90 {np.percentile(heights,90):.1f}, "
      f"max {heights.max():.1f}; default: {defaults}; median pts/bldg {np.median(npts):.0f}  ({time.time()-t0:.0f}s)")

# --- extrude ---
meshes, fixed = [], 0
for p, h in zip(sel, heights):
    parts = list(p.geoms) if p.geom_type == 'MultiPolygon' else [p]
    for part in parts:
        if not part.is_valid:
            part = part.buffer(0); fixed += 1
            if part.is_empty: continue
        try:
            meshes.append(trimesh.creation.extrude_polygon(part, float(h)))
        except Exception as e:
            print("extrude fail:", e)

raw = trimesh.util.concatenate(meshes)
raw.export('stanford_raw_gis_extruded.stl')
print(f"extruded: {len(meshes)} parts, {len(raw.faces):,} faces, "
      f"{len(raw.split(only_watertight=False)):,} components; fixed {fixed}")
print(f"TOTAL input prep: {time.time()-t0:.0f}s")
