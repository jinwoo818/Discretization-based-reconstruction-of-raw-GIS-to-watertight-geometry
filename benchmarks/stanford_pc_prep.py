"""
Prepare the Stanford point-cloud benchmark input.

Same public data sources as the City4CFD 2024 "Stanford case" and the same
domain (circle of 713 m radius = 1.6 km^2 centered on the Science and
Engineering Quad). This script performs data acquisition/formatting only
(the same preparation City4CFD asks users to provide: separated building
points in a metric CRS):

  - USGS 3DEP lidar (Santa Clara County 2020 A20, 4 LAZ tiles,
    NAD83(2011) / California zone 3 ftUS)
  - keep LAS classification 6 (building) and 2 (ground) points inside
    the domain
  - reproject to local metric coordinates (UTM 10N, z in meters)

Output: stanford_buildings_pc.laz (building points, metric)
        stanford_ground_pc.laz   (ground points, metric)
"""
import glob, time
import numpy as np
import laspy
from pyproj import Transformer

t0 = time.time()
LAT0, LON0, R = 37.4274, -122.1731, 713.0
T_WGS84_UTM = Transformer.from_crs("EPSG:4326", "EPSG:26910", always_xy=True)
CX, CY = T_WGS84_UTM.transform(LON0, LAT0)

files = sorted(glob.glob('stanford_laz/*.laz'))
with laspy.open(files[0]) as r:
    CRS = r.header.parse_crs()
T = Transformer.from_crs(CRS, "EPSG:26910", always_xy=True)
FT = 0.3048006096012192

xs, ys, zs = [], [], []
gx, gy, gz = [], [], []
for f in files:
    las = laspy.read(f)
    cls = np.asarray(las.classification)
    for keep, acc in ((6, (xs, ys, zs)), (2, (gx, gy, gz))):
        x = np.asarray(las.x)[cls == keep]; y = np.asarray(las.y)[cls == keep]
        z = np.asarray(las.z)[cls == keep]
        xu, yu = T.transform(x, y)
        xl = xu - CX; yl = yu - CY
        m = xl**2 + yl**2 <= (R + 50)**2
        acc[0].append(xl[m]); acc[1].append(yl[m]); acc[2].append(z[m] * FT)
    print(f"{f.split('_')[-1]}: bld {len(xs[-1]):,} / grd {len(gx[-1]):,} "
          f"({time.time()-t0:.0f}s)")

for name, acc in (("stanford_buildings_pc.laz", (xs, ys, zs)),
                  ("stanford_ground_pc.laz", (gx, gy, gz))):
    X, Y, Z = map(np.concatenate, acc)
    hdr = laspy.LasHeader(point_format=0, version="1.2")
    hdr.offsets = [X.min(), Y.min(), Z.min()]
    hdr.scales = [0.001, 0.001, 0.001]
    las = laspy.LasData(hdr)
    las.x = X; las.y = Y; las.z = Z
    las.write(name)
    print(f"wrote {name}: {len(X):,} pts ({time.time()-t0:.0f}s)")
