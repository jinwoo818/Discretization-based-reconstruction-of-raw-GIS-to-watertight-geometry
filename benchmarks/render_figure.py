"""Render the Section 4.3 benchmark figure: (input cloud vs output mesh) x (TUD, Stanford).

Each row is rendered with identical axis limits, view angle, and aspect, so the
input cloud and the output mesh read as the same area. Input clouds are drawn
as uniform single-color points; output meshes as dark gray solids. Panel titles
use a Times-compatible serif font (Times New Roman where available, otherwise
the metrically compatible Liberation Serif).

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

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman', 'Liberation Serif',
                                     'Nimbus Roman', 'DejaVu Serif']

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')
OUT = os.path.join(HERE, 'outputs')

ELEV, AZIM, ZSCALE = 32, -60, 0.42
CLOUD_COLOR = '#4f6d8f'   # uniform, no colormap
MESH_COLOR = '#3a3f44'    # dark gray


def load_faces(path):
    m = trimesh.load(path, force='mesh')
    m.merge_vertices()
    return m.vertices, m.faces


def load_pts(path):
    import laspy
    las = laspy.read(path)
    return np.column_stack([np.asarray(las.x), np.asarray(las.y),
                            np.asarray(las.z)]).astype(np.float64)


def _frame(ax, center, r, zmin, zmax):
    ax.set_xlim(center[0] - r, center[0] + r)
    ax.set_ylim(center[1] - r, center[1] + r)
    ax.set_zlim(zmin - (zmax - zmin) * 0.05, zmax + (zmax - zmin) * 0.10)
    ax.set_box_aspect((1, 1, ZSCALE))
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_axis_off()


def _finish(fig, pad_frac=0.10):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.02,
                transparent=False)
    plt.close(fig)
    buf.seek(0)
    im = Image.open(buf).convert('RGB')
    bg = Image.new('RGB', im.size, (255, 255, 255))
    bbox = ImageChops.difference(im, bg).getbbox()
    if bbox:
        im = im.crop(bbox)
    # symmetric white margin so each panel breathes
    pad = int(pad_frac * max(im.size))
    out = Image.new('RGB', (im.width + 2 * pad, im.height + 2 * pad),
                    (255, 255, 255))
    out.paste(im, (pad, pad))
    return np.array(out)


def render_cloud_shared(pts, center, r, zmin, zmax, max_pts=350_000, dpi=220):
    fig = plt.figure(figsize=(6, 6), dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')
    if len(pts) > max_pts:
        idx = np.random.default_rng(0).choice(len(pts), max_pts, replace=False)
        pts = pts[idx]
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=0.45, c=CLOUD_COLOR,
               linewidths=0, depthshade=False)
    _frame(ax, center, r, zmin, zmax)
    return _finish(fig)


def render_mesh_shared(verts, faces, center, r, zmin, zmax, dpi=220):
    fig = plt.figure(figsize=(6, 6), dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')
    tri = verts[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)
    light = np.array([0.4, 0.3, 0.85]); light /= np.linalg.norm(light)
    shade = np.clip(n @ light, 0.25, 1.0)
    base = np.array(matplotlib.colors.to_rgb(MESH_COLOR))
    cols = np.clip(shade[:, None] * base[None, :], 0, 1)
    ax.add_collection3d(Poly3DCollection(tri, facecolors=cols, edgecolors='none'))
    _frame(ax, center, r, zmin, zmax)
    return _finish(fig)


def shared_extent(cloud, verts):
    """Union of cloud and mesh bounds -> common center, radius, z-range."""
    allmin = np.minimum(cloud.min(0), verts.min(0))
    allmax = np.maximum(cloud.max(0), verts.max(0))
    c = (allmin + allmax) / 2
    r = max(allmax[0] - allmin[0], allmax[1] - allmin[1]) / 2 * 1.02
    return c[:2], r, allmin[2], allmax[2]


def main():
    tud_pts = load_pts(os.path.join(DATA, 'tud_buildings_pc.laz'))
    st_pts = load_pts(os.path.join(DATA, 'stanford_buildings_pc.laz'))
    tv, tf = load_faces(os.path.join(OUT, 'tud_pc_modeled.stl'))
    sv, sf = load_faces(os.path.join(OUT, 'stanford_pc_modeled.stl'))

    # each row shares one camera/extent so input and output align
    tc, tr, tz0, tz1 = shared_extent(tud_pts, tv)
    sc, sr, sz0, sz1 = shared_extent(st_pts, sv)
    panels = [
        ('(a) TU Delft building point cloud\n(5.99M points, as distributed)',
         render_cloud_shared(tud_pts, tc, tr, tz0, tz1)),
        ('(b) Pipeline output\n(211 buildings, 100% watertight)',
         render_mesh_shared(tv, tf, tc, tr, tz0, tz1)),
        ('(c) Stanford building point cloud\n(USGS 3DEP, class 6, 6.85M points)',
         render_cloud_shared(st_pts, sc, sr, sz0, sz1)),
        ('(d) Pipeline output\n(181 buildings, 100% watertight)',
         render_mesh_shared(sv, sf, sc, sr, sz0, sz1)),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10.4, 11.0), dpi=220)
    for ax, (title, img) in zip(axes.flat, panels):
        ax.imshow(img); ax.set_axis_off()
        ax.set_title(title, fontsize=12, pad=8)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.01,
                        wspace=0.04, hspace=0.10)
    # quadrant divider lines
    fig.canvas.draw()
    pos = [ax.get_position() for ax in axes.flat]
    x_div = (pos[0].x1 + pos[1].x0) / 2
    y_div = (pos[0].y0 + pos[2].y1) / 2
    ylo = min(p.y0 for p in pos); yhi = max(p.y1 for p in pos)
    xlo = min(p.x0 for p in pos); xhi = max(p.x1 for p in pos)
    fig.add_artist(plt.Line2D([x_div, x_div], [ylo, yhi],
                              transform=fig.transFigure, color='0.55', lw=1.0))
    fig.add_artist(plt.Line2D([xlo, xhi], [y_div, y_div],
                              transform=fig.transFigure, color='0.55', lw=1.0))
    out = os.path.join(HERE, 'fig_pc_benchmark.png')
    fig.savefig(out, bbox_inches='tight', pad_inches=0.05, facecolor='white')
    plt.close(fig)
    print('saved', out)


if __name__ == '__main__':
    main()
