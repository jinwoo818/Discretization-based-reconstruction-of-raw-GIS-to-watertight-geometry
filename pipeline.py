"""
Automated Geometry Pre-processing Pipeline for Urban CFD Simulation
====================================================================
Discretization-based Reconstruction: Raw GIS Geometry -> Watertight LOD1 Mesh

Pipeline Steps:
  Step 1. Building Identification (connectivity split / DBSCAN)
  Step 2. Per-building Surface Sampling & Normalization
  Step 3. Footprint Extraction via Concave Hull (alpha shape)
  Step 4. Douglas-Peucker Simplification, 2.5D Extrusion & Mesh Repair

Inputs: mesh-based GIS (STL/OBJ/PLY) or point-cloud data (LAS/LAZ/XYZ).
Point-cloud inputs are identified via DBSCAN (Section 2.2) and used
  directly, without separate surface sampling (Section 2.3).
For point-cloud inputs, an optional ground point cloud (--ground)
  provides the base elevation of each building; this is required for
  airborne LiDAR, where classified building points are dominated by
  roof returns and the lower percentile of the cluster corresponds to
  the eave level rather than the ground.

Dependencies: pip install numpy trimesh shapely scikit-learn mapbox-earcut
              (point-cloud input additionally requires: pip install laspy[lazrs])

Reference:
  Choi, J., Hong, T., Kim, H., and Jeong, K. (2026). Discretization-based
  Reconstruction of Watertight Building Geometry from Raw GIS Data for
  Urban CFD Simulation. Developments in the Built Environment
  (DIBE-D-26-01168R1).
"""

import numpy as np, trimesh, shapely, argparse, time, os, sys, logging
from collections import defaultdict
from shapely.geometry import MultiPoint, Polygon
from sklearn.cluster import DBSCAN

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    "sampling_density": 10.0, 
    "max_pts_per_building": 2000, 
    "min_pts_per_building": 100,
    "eps": 2.5, "min_pts": 30,
    "alpha_ratio": 0.2, "allow_holes": False,
    "simplify_tolerance": 0.5,
    "height_percentile_low": 5, 
    "height_percentile_high": 99,
    "min_building_height": 2.0,
    "min_footprint_area": 8.0,
    "random_seed": 0,
    # --- geometry-consistency safeguards (v1.1) ---
    # Both thresholds are set above the coordinate resolution of a
    # single-precision (float32) STL at UTM-scale coordinates (~6 cm at 5e5 m),
    # so that vertices can never collapse onto each other on export.
    "min_edge_length": 0.15,
    "coincident_inset": 0.15,
    # --- point-cloud input (Section 2.2) ---
    "ground_buffer": 5.0,
}

def _clean_polygon(poly, min_edge):
    """Regularize a footprint ring: remove edges shorter than min_edge and
    vertices whose deviation from the line joining their neighbors is below
    min_edge. Both conditions can collapse into degenerate faces after
    single-precision (STL) export. Returns None if the ring collapses."""
    def clean_ring(ring):
        pts = list(ring.coords)[:-1]
        out = []
        for p in pts:
            if out and (p[0]-out[-1][0])**2 + (p[1]-out[-1][1])**2 < min_edge**2:
                continue
            out.append(p)
        while len(out) > 1 and (out[0][0]-out[-1][0])**2 + (out[0][1]-out[-1][1])**2 < min_edge**2:
            out.pop()
        if len(out) < 3:
            return None
        changed = True
        while changed and len(out) > 3:
            changed = False
            for i in range(len(out)):
                a, b, c = out[i-1], out[i], out[(i+1) % len(out)]
                ac = ((c[0]-a[0])**2 + (c[1]-a[1])**2) ** 0.5
                if ac < 1e-12:
                    continue
                cross = abs((b[0]-a[0])*(c[1]-b[1]) - (b[1]-a[1])*(c[0]-b[0]))
                if cross / ac < min_edge:
                    out.pop(i); changed = True; break
        return out if len(out) >= 3 else None
    ring = clean_ring(poly.exterior)
    if ring is None:
        return None
    cleaned = Polygon(ring)
    return cleaned if cleaned.is_valid and cleaned.area > 0 else None

def _verify_mesh_integrity(mesh):
    """Mesh-level QC at single-precision (float32) resolution: counts
    non-manifold edges and degenerate faces exactly as they will appear in
    the written STL file."""
    v = np.asarray(mesh.vertices, dtype=np.float32).astype(np.float64)
    m = trimesh.Trimesh(vertices=v, faces=np.asarray(mesh.faces), process=False)
    m.merge_vertices()
    e2f = defaultdict(int)
    for f in m.faces:
        for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            e2f[tuple(sorted((int(a), int(b))))] += 1
    nm = sum(1 for c in e2f.values() if c > 2)
    deg = int(np.sum(m.area_faces <= 1e-10))
    return nm, deg

def _fix_sliver(fp, mesh, min_edge):
    """Perturb the footprint vertex responsible for a cap face that would be
    degenerate after single-precision (float32) export: such faces arise when
    a ring vertex lies (almost exactly) on the segment between two other ring
    vertices. The vertex is moved perpendicular to that segment by min_edge,
    on the side that keeps the polygon valid. Returns the fixed polygon or
    None if no safe fix exists."""
    v = np.asarray(mesh.vertices, dtype=np.float32).astype(np.float64)
    f = np.asarray(mesh.faces)
    tri = v[f]
    areas = 0.5 * np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    bad = np.where(areas <= 1e-10)[0]
    if len(bad) == 0:
        return None
    p = tri[bad[0]]
    # sliver vertex = the one closest to the line through the other two
    best = None
    for i in range(3):
        a, b = p[(i + 1) % 3], p[(i + 2) % 3]
        d = np.linalg.norm(np.cross(b - a, p[i] - a)) / (np.linalg.norm(b - a) + 1e-30)
        if best is None or d < best[0]:
            best = (d, p[i], a, b)
    _, B, A, C = best
    d = C[:2] - A[:2]
    L = float(np.linalg.norm(d))
    if L < 1e-12:
        return None
    n = np.array([-d[1], d[0]]) / L
    ring = list(fp.exterior.coords)[:-1]
    j = int(np.argmin([(pt[0] - B[0]) ** 2 + (pt[1] - B[1]) ** 2 for pt in ring]))
    if (ring[j][0] - B[0]) ** 2 + (ring[j][1] - B[1]) ** 2 > 1.0:
        return None  # sliver vertex not found on the ring
    a0 = fp.area
    for s in (1.0, -1.0):
        nb = (ring[j][0] + s * min_edge * n[0], ring[j][1] + s * min_edge * n[1])
        cand = Polygon(ring[:j] + [nb] + ring[j + 1:])
        if cand.is_valid and cand.area > 0 and abs(cand.area - a0) < 0.5 * a0:
            return cand
    return None

def _load_points(input_path, classification=None):
    """Load a point-cloud input (LAS/LAZ/XYZ/TXT). Returns an (N,3) float64
    array in metric coordinates, or None if the file is not point-based.
    If `classification` is given (LAS classification code, e.g. 6 = building),
    only points of that class are kept — the same preparation City4CFD-style
    workflows ask the user to provide as separated building points."""
    ext = os.path.splitext(input_path)[1].lower()
    if ext in (".las", ".laz"):
        import laspy
        las = laspy.read(input_path)
        pts = np.column_stack([np.asarray(las.x), np.asarray(las.y),
                               np.asarray(las.z)]).astype(np.float64)
        if classification is not None:
            keep = np.asarray(las.classification) == int(classification)
            pts = pts[keep]
        return pts
    if ext in (".xyz", ".txt"):
        return np.loadtxt(input_path, usecols=(0, 1, 2))
    return None

def _identify_points(pts, eps, min_pts, grid=1.0):
    """DBSCAN building identification for point-cloud inputs (Section 2.2).
    Points are binned into 1 m xy grid cells and DBSCAN is run on the unique
    cell centers with the per-cell point counts passed as sample_weight, so
    that the core-point condition keeps the point-count weighting of the
    MinPts criterion on the input points while reducing redundant LiDAR
    points for computational efficiency. Returns a list of (N_k,3) point
    arrays, one per identified building cluster."""
    keys = np.floor(pts[:, :2] / grid).astype(np.int64)
    _, inv, counts = np.unique(keys, axis=0, return_inverse=True,
                                return_counts=True)
    uniq = np.unique(keys, axis=0)
    centers = (uniq + 0.5) * grid
    log.info(f"  {len(pts):,} points -> {len(centers):,} grid cells")
    labels = DBSCAN(eps=eps, min_samples=min_pts).fit_predict(
        centers, sample_weight=counts.astype(np.float64))
    cell_labels = labels[inv]                      # back to raw points
    noise = int(np.sum(cell_labels == -1))
    ids = set(cell_labels.tolist()); ids.discard(-1)
    log.info(f"  {len(ids)} buildings, {noise:,} noise points")
    return [pts[cell_labels == l] for l in sorted(ids)]

def _ground_elevation(ground, xy, buf, min_pts=10):
    """Base elevation of a building cluster: median z of ground points
    within the xy bounding box of the cluster, expanded by `buf` meters
    (ground_buffer). Returns None when too few ground points are available
    (the caller then falls back to the lower percentile of the cluster
    points). `ground` must be pre-sorted by x (see _sort_by_x)."""
    x0, x1 = xy[:, 0].min() - buf, xy[:, 0].max() + buf
    lo = np.searchsorted(ground[:, 0], x0, side="left")
    hi = np.searchsorted(ground[:, 0], x1, side="right")
    cand = ground[lo:hi]
    m = (cand[:, 1] >= xy[:, 1].min() - buf) & (cand[:, 1] <= xy[:, 1].max() + buf)
    return float(np.median(cand[m, 2])) if int(m.sum()) >= min_pts else None

def _sort_by_x(pts):
    return pts[np.argsort(pts[:, 0], kind="stable")] if len(pts) else pts

def run_pipeline(input_path, output_path, config):
    t0 = time.time()
    log.info("=" * 60)
    log.info("  Urban CFD Pre-processing: GIS -> LOD1 Watertight Mesh")
    log.info("=" * 60)
    log.info(f"  Input:  {input_path}\n  Output: {output_path}\n")

    cloud_pts = _load_points(input_path, config.get("classification"))
    base_elev = None
    if cloud_pts is not None:
        log.info(f"[Load] point cloud: {len(cloud_pts):,} points")
        ground = _load_points(config["ground"]) if config.get("ground") else None
        if ground is not None:
            log.info(f"[Load] ground cloud: {len(ground):,} points")
            ground = _sort_by_x(ground)
        log.info(f"\n[Step 1] Point-cloud input - DBSCAN identification "
                 f"(eps={config['eps']}, MinPts={config['min_pts']})")
        clusters = _identify_points(cloud_pts, config["eps"], config["min_pts"])
        if ground is not None:
            base_elev = [_ground_elevation(ground, c[:, :2],
                                           config["ground_buffer"])
                         for c in clusters]
            n_fb = sum(b is None for b in base_elev)
            if n_fb:
                log.info(f"  ground-based base elevation unavailable for {n_fb} "
                         f"clusters (lower-percentile fallback)")
        building_meshes = [trimesh.Trimesh(vertices=c) for c in clusters]
    else:
        raw = trimesh.load(input_path, force="mesh")
        log.info(f"[Load] {len(raw.faces):,} faces, {len(raw.vertices):,} verts, wt={raw.is_watertight}")

        components = raw.split(only_watertight=False)
        if len(components) > 1:
            log.info(f"\n[Step 1] Connectivity split: {len(components)} components")
            building_meshes = [c for c in components if len(c.vertices) >= 4]
        else:
            log.info(f"\n[Step 1] Single body — DBSCAN path")
            n_s = min(int(raw.area * config["sampling_density"]), config["max_pts_per_building"] * 100)
            pts, _ = trimesh.sample.sample_surface(raw, max(n_s, 1000),
                                                    seed=config["random_seed"])
            center = np.mean(pts[:,:2], axis=0)
            pts[:,0] -= center[0]; pts[:,1] -= center[1]
            labels = DBSCAN(eps=config["eps"], min_samples=config["min_pts"]).fit_predict(pts)
            unique = set(labels); unique.discard(-1)
            log.info(f"  {len(unique)} buildings, {int(np.sum(labels==-1))} noise")
            building_meshes = []
            for l in unique:
                cp = pts[labels==l].copy(); cp[:,0]+=center[0]; cp[:,1]+=center[1]
                building_meshes.append(trimesh.Trimesh(vertices=cp))

    n_total = len(building_meshes)
    log.info(f"  Candidates: {n_total:,}")
    log.info(f"\n[Steps 2-4] density={config['sampling_density']}, alpha={config['alpha_ratio']}, "
             f"eps_dp={config['simplify_tolerance']}m, H_min={config['min_building_height']}m, "
             f"A_min={config['min_footprint_area']}m2\n")

    successful, filt_h, filt_a, failed = [], 0, 0, 0
    placed_vertices, resolved, sliver_fixes = set(), 0, 0

    def _vkeys(poly):
        # Quantize to single-precision coordinates: two vertices collapse onto
        # each other in an STL file exactly when their float32 values match.
        return {(float(np.float32(x)), float(np.float32(y)))
                for x, y in poly.exterior.coords}

    for i, comp in enumerate(building_meshes):
        try:
            verts = np.array(comp.vertices)
            zh = np.percentile(verts[:,2], config["height_percentile_high"])
            if base_elev is not None and base_elev[i] is not None:
                # point-cloud input with ground cloud: base from ground points
                zl = base_elev[i]
            else:
                zl = np.percentile(verts[:,2], config["height_percentile_low"])
            h = zh - zl
            if h < config["min_building_height"]: filt_h += 1; continue

            mp_c = MultiPoint(verts[:,:2])
            cv = mp_c.convex_hull
            if cv.geom_type != 'Polygon' or cv.area < config["min_footprint_area"]:
                filt_a += 1; continue

            if hasattr(comp,'area') and comp.area > 0 and len(comp.faces) > 0:
                n = min(max(int(comp.area*config["sampling_density"]), config["min_pts_per_building"]),
                        config["max_pts_per_building"])
                sampled, _ = trimesh.sample.sample_surface(comp, n,
                                                            seed=config["random_seed"])
            else:
                # point-cloud component (Section 2.3): points are used directly;
                # for hull extraction only, subsample to the same per-building
                # cap as the mesh path (height percentiles use all points)
                if len(verts) > config["max_pts_per_building"]:
                    rng = np.random.default_rng(config["random_seed"] + i)
                    idx = rng.choice(len(verts), config["max_pts_per_building"],
                                     replace=False)
                    sampled = verts[idx]
                else:
                    sampled = verts

            pts_2d = sampled[:,:2]
            mp = MultiPoint(pts_2d)
            hull = mp.convex_hull if len(pts_2d)<6 else shapely.concave_hull(
                mp, ratio=config["alpha_ratio"], allow_holes=config["allow_holes"])
            if hull.is_empty or hull.geom_type in ('Point','LineString'): failed+=1; continue
            if hull.geom_type == 'MultiPolygon': hull = max(hull.geoms, key=lambda p: p.area)

            fp = hull.simplify(config["simplify_tolerance"], preserve_topology=True)
            if fp.is_empty: fp = hull

            # v1.1: ring regularization (sub-precision edges, collinear vertices)
            cleaned = _clean_polygon(fp, config["min_edge_length"])
            if cleaned is not None:
                fp = cleaned

            # v1.1: resolve coincident walls with already-placed buildings
            if _vkeys(fp) & placed_vertices:
                for attempt in (1, 2):
                    cand = fp.buffer(-attempt * config["coincident_inset"], join_style=2)
                    if cand.geom_type != 'Polygon' or cand.is_empty or not cand.is_valid:
                        break
                    cand = _clean_polygon(cand, config["min_edge_length"])
                    if cand is None:
                        break
                    fp = cand; resolved += 1
                    if not (_vkeys(fp) & placed_vertices):
                        break

            # v1.1: extrude; if a cap face would degenerate at single-precision
            # resolution, perturb the responsible footprint vertex and retry
            m = None
            for _attempt in range(4):
                m = trimesh.creation.extrude_polygon(fp, height=h)
                m.apply_translation([0,0,zl]); m.merge_vertices()
                trimesh.repair.fix_normals(m); trimesh.repair.fix_inversion(m)

                # Remove degenerate faces
                if len(m.faces) > 0:
                    valid = m.area_faces > 1e-10
                    if not np.all(valid):
                        m.update_faces(valid); m.remove_unreferenced_vertices()

                if not m.is_watertight:
                    break
                if _verify_mesh_integrity(m)[1] == 0:
                    break
                fp2 = _fix_sliver(fp, m, config["min_edge_length"])
                if fp2 is None:
                    break
                fp = fp2; sliver_fixes += 1

            placed_vertices |= _vkeys(fp)

            if m is not None and m.is_watertight: successful.append(m)
            else: failed += 1
        except Exception as e:
            failed += 1; log.debug(f"  Building {i+1} failed: {e}")
        finally:
            if (i+1) % 2000 == 0 or (i+1) == n_total:
                nf = filt_h + filt_a
                log.info(f"  {i+1:,}/{n_total:,}: {len(successful):,} OK, "
                         f"{nf} filtered (h:{filt_h}, fp:{filt_a}), {failed} failed")

    if not successful: log.error("No valid buildings."); return None

    city = trimesh.util.concatenate(successful)

    # v1.1: mesh-level QC at single-precision (export) resolution
    nm_edges, deg_faces = _verify_mesh_integrity(city)
    log.info(f"\n[QC] float32 mesh-level check: non-manifold edges={nm_edges}, "
             f"degenerate faces={deg_faces}, coincident-wall insets={resolved}, "
             f"sliver-vertex fixes={sliver_fixes}")
    if nm_edges or deg_faces:
        log.warning("[QC] residual mesh-level defects detected - inspect output")

    city.export(output_path)
    elapsed = time.time() - t0
    log.info(f"\n{'='*60}")
    log.info(f"  Completed in {elapsed:.1f}s")
    log.info(f"  Buildings: {len(successful):,}/{n_total:,} ({len(successful)/n_total*100:.1f}%)")
    log.info(f"  Faces: {len(city.faces):,} | Output: {output_path}")
    log.info(f"{'='*60}")
    return city

def main():
    p = argparse.ArgumentParser(description="Urban CFD LOD1 Pipeline",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("input", help="Input mesh (STL/OBJ/PLY) or point cloud (LAS/LAZ/XYZ)")
    p.add_argument("-o","--output", default=None, help="Output STL path")
    p.add_argument("--classification", type=int, default=None,
                   help="LAS classification code to keep (e.g. 6 = building); "
                        "point-cloud inputs only")
    p.add_argument("--ground", default=None,
                   help="Ground point cloud (LAS/LAZ/XYZ) for base-elevation "
                        "estimation of point-cloud inputs (airborne LiDAR, "
                        "roof-dominated building points)")
    p.add_argument("--ground_buffer", type=float,
                   default=DEFAULT_CONFIG["ground_buffer"],
                   help="Buffer (m) around the cluster bounding box for "
                        "ground-point base-elevation lookup")
    p.add_argument("--density", type=float, default=DEFAULT_CONFIG["sampling_density"])
    p.add_argument("--eps", type=float, default=DEFAULT_CONFIG["eps"])
    p.add_argument("--min_pts", type=int, default=DEFAULT_CONFIG["min_pts"])
    p.add_argument("--alpha", type=float, default=DEFAULT_CONFIG["alpha_ratio"])
    p.add_argument("--simplify", type=float, default=DEFAULT_CONFIG["simplify_tolerance"])
    p.add_argument("--min_height", type=float, default=DEFAULT_CONFIG["min_building_height"])
    p.add_argument("--min_fp_area", type=float, default=DEFAULT_CONFIG["min_footprint_area"])
    p.add_argument("-v","--verbose", action="store_true")
    a = p.parse_args()
    if a.verbose: logging.getLogger().setLevel(logging.DEBUG)
    if not os.path.exists(a.input): log.error(f"Not found: {a.input}"); sys.exit(1)
    cfg = DEFAULT_CONFIG.copy()
    cfg.update({"sampling_density":a.density,"eps":a.eps,"min_pts":a.min_pts,
                "alpha_ratio":a.alpha,"simplify_tolerance":a.simplify,
                "min_building_height":a.min_height,"min_footprint_area":a.min_fp_area,
                "classification":a.classification,"ground":a.ground,
                "ground_buffer":a.ground_buffer})
    out = a.output or os.path.splitext(a.input)[0] + "_LOD1.stl"
    run_pipeline(a.input, out, cfg)

if __name__ == "__main__":
    main()
