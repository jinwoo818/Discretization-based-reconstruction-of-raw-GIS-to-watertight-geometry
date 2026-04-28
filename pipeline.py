"""
Automated Geometry Pre-processing Pipeline for Urban CFD Simulation
====================================================================
Discretization-based Reconstruction: Raw GIS Geometry -> Watertight LOD1 Mesh

Pipeline Steps:
  Step 1. Building Identification (connectivity split / DBSCAN)
  Step 2. Per-building Surface Sampling & Normalization
  Step 3. Footprint Extraction via Concave Hull (alpha shape)
  Step 4. Douglas-Peucker Simplification, 2.5D Extrusion & Mesh Repair

Dependencies: pip install numpy trimesh shapely scikit-learn mapbox-earcut

Reference:
  Choi, J., Hong, T. (2026). Automated Geometry Pre-processing Pipeline
  for Urban-scale CFD Simulation. [Journal TBD].
"""

import numpy as np, trimesh, shapely, argparse, time, os, sys, logging
from shapely.geometry import MultiPoint
from sklearn.cluster import DBSCAN

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    "sampling_density": 10.0, "max_pts_per_building": 2000, "min_pts_per_building": 100,
    "eps": 2.5, "min_pts": 30,
    "alpha_ratio": 0.2, "allow_holes": False,
    "simplify_tolerance": 0.6,
    "height_percentile_low": 5, "height_percentile_high": 99,
    "min_building_height": 2.0,
    "min_footprint_area": 8.0,
}

def run_pipeline(input_path, output_path, config):
    t0 = time.time()
    log.info("=" * 60)
    log.info("  Urban CFD Pre-processing: GIS -> LOD1 Watertight Mesh")
    log.info("=" * 60)
    log.info(f"  Input:  {input_path}\n  Output: {output_path}\n")

    raw = trimesh.load(input_path, force="mesh")
    log.info(f"[Load] {len(raw.faces):,} faces, {len(raw.vertices):,} verts, wt={raw.is_watertight}")

    components = raw.split(only_watertight=False)
    if len(components) > 1:
        log.info(f"\n[Step 1] Connectivity split: {len(components)} components")
        building_meshes = [c for c in components if len(c.vertices) >= 4]
    else:
        log.info(f"\n[Step 1] Single body — DBSCAN path")
        n_s = min(int(raw.area * config["sampling_density"]), config["max_pts_per_building"] * 100)
        pts, _ = trimesh.sample.sample_surface(raw, max(n_s, 1000))
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

    for i, comp in enumerate(building_meshes):
        try:
            verts = np.array(comp.vertices)
            zl = np.percentile(verts[:,2], config["height_percentile_low"])
            zh = np.percentile(verts[:,2], config["height_percentile_high"])
            h = zh - zl
            if h < config["min_building_height"]: filt_h += 1; continue

            mp_c = MultiPoint(verts[:,:2])
            cv = mp_c.convex_hull
            if cv.geom_type != 'Polygon' or cv.area < config["min_footprint_area"]:
                filt_a += 1; continue

            if hasattr(comp,'area') and comp.area > 0 and len(comp.faces) > 0:
                n = min(max(int(comp.area*config["sampling_density"]), config["min_pts_per_building"]),
                        config["max_pts_per_building"])
                sampled, _ = trimesh.sample.sample_surface(comp, n)
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

            m = trimesh.creation.extrude_polygon(fp, height=h)
            m.apply_translation([0,0,zl]); m.merge_vertices()
            trimesh.repair.fix_normals(m); trimesh.repair.fix_inversion(m)

            # Remove degenerate faces
            if len(m.faces) > 0:
                valid = m.area_faces > 1e-10
                if not np.all(valid):
                    m.update_faces(valid); m.remove_unreferenced_vertices()

            if m.is_watertight: successful.append(m)
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
    p.add_argument("input", help="Input mesh file (STL/OBJ/PLY)")
    p.add_argument("-o","--output", default=None, help="Output STL path")
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
                "min_building_height":a.min_height,"min_footprint_area":a.min_fp_area})
    out = a.output or os.path.splitext(a.input)[0] + "_LOD1.stl"
    run_pipeline(a.input, out, cfg)

if __name__ == "__main__":
    main()
