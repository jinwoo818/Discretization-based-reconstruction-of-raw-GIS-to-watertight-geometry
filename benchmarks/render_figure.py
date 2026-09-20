"""Render the Section 4.3 benchmark figure: (input cloud vs output mesh) x (TUD, Stanford).

Requires the input point clouds (see README.md for download instructions) in
benchmarks/data/ and the outputs in benchmarks/outputs/ (or regenerate them
with the commands in README.md).

Usage:  python render_figure.py
Output: fig_pc_benchmark.png
"""
import os
import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image, ImageChops
import io

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')
OUT = os.path.join(HERE, 'outputs')


def load_faces(path):
    m = trimesh.load(path, force='mesh')
    m.merge_vertices()
    return m.vertices, m.faces


def load_pts(path):
    import laspy
    las = laspy.read(path)
    return np.column_stack([np.asarray(las.x), np.asarray(las.y),
                            np.asarray(las.z)]).astype(np.float64)


def _tight_crop(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.02,
                transparent=False)
    plt.close(fig)
    buf.seek(0)
    im = Image.open(buf).convert('RGB')
    bg = Image.new('RGB', im.size, (255, 255, 255))
    bbox = ImageChops.difference(im, bg).getbbox()
    return np.array(im.crop(bbox)) if bbox else np.array(im)


def render_mesh(verts, faces, color, elev=32, azim=-60, zscale=0.42, dpi=220):
    fig = plt.figure(figsize=(6, 6), dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')
    tri = verts[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)
    light = np.array([0.4, 0.3, 0.85]); light /= np.linalg.norm(light)
    shade = np.clip(n @ light, 0.25, 1.0)
    base = np.array(matplotlib.colors.to_rgb(color))
    cols = np.clip(shade[:, None] * base[None, :], 0, 1)
    ax.add_collection3d(Poly3DCollection(tri, facecolors=cols, edgecolors='none'))
    mins, maxs = verts.min(0), verts.max(0)
    c = (mins + maxs) / 2
    r = max(maxs[0] - mins[0], maxs[1] - mins[1]) / 2 * 1.02
    zr = maxs[2] - mins[2]
    ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(mins[2] - zr * 0.05, maxs[2] + zr * 0.10)
    ax.set_box_aspect((1, 1, zscale))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    return _tight_crop(fig)


def render_cloud(pts, elev=32, azim=-60, zscale=0.42, dpi=220, max_pts=350_000):
    fig = plt.figure(figsize=(6, 6), dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')
    if len(pts) > max_pts:
        idx = np.random.default_rng(0).choice(len(pts), max_pts, replace=False)
        pts = pts[idx]
    zmin, zmax = pts[:, 2].min(), pts[:, 2].max()
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=0.35, c=pts[:, 2],
               cmap='viridis', vmin=zmin, vmax=zmax, linewidths=0, depthshade=False)
    c = (pts.min(0) + pts.max(0)) / 2
    r = max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1])) / 2 * 1.02
    zr = zmax - zmin
    ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(zmin - zr * 0.05, zmax + zr * 0.10)
    ax.set_box_aspect((1, 1, zscale))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    return _tight_crop(fig)


def main():
    tud_pts = load_pts(os.path.join(DATA, 'tud_buildings_pc.laz'))
    st_pts = load_pts(os.path.join(DATA, 'stanford_buildings_pc.laz'))
    tv, tf = load_faces(os.path.join(OUT, 'tud_pc_modeled.stl'))
    sv, sf = load_faces(os.path.join(OUT, 'stanford_pc_modeled.stl'))

    panels = [
        ('(a) TU Delft building point cloud\n(5.99M points, as distributed)',
         render_cloud(tud_pts)),
        ('(b) Pipeline output\n(211 buildings, 100% watertight)',
         render_mesh(tv, tf, '#2f6db3')),
        ('(c) Stanford building point cloud\n(USGS 3DEP, class 6, 6.85M points)',
         render_cloud(st_pts)),
        ('(d) Pipeline output\n(181 buildings, 100% watertight)',
         render_mesh(sv, sf, '#2f6db3')),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10.4, 11.0), dpi=220)
    for ax, (title, img) in zip(axes.flat, panels):
        ax.imshow(img); ax.set_axis_off()
        ax.set_title(title, fontsize=11, pad=6)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.01,
                        wspace=0.03, hspace=0.04)
    out = os.path.join(HERE, 'fig_pc_benchmark.png')
    fig.savefig(out, bbox_inches='tight', pad_inches=0.05, facecolor='white')
    plt.close(fig)
    print('saved', out)


if __name__ == '__main__':
    main()
