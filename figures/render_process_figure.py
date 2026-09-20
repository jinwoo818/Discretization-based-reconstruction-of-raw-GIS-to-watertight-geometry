"""Generate the process illustration figure of the manuscript (Section 2.5).

Row 1: reconstruction stages of one U-shaped building -- (a) sampled point set,
(b) xy projection, (c) alpha-shape, (d) DP simplification, (e) regularization,
(f) extruded watertight LOD1 block.
Row 2: parameter effects -- (g-i) concave-hull ratio alpha, (j-l) DP tolerance.

The building is a synthetic U-shaped footprint with boundary irregularities.
All stages are produced by the actual pipeline logic (pipeline.py, default
configuration); the resulting extruded block is verified to be watertight.

Usage:  python figures/render_process_figure.py
Output: figures/fig_reconstruction_stages.png
"""
import numpy as np
import trimesh
import shapely
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image
import io
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import _clean_polygon
from shapely.geometry import Polygon, MultiPoint

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman', 'Liberation Serif',
                                     'Nimbus Roman', 'DejaVu Serif']

ELEV, AZIM, ZSCALE = 32, -60, 0.42
SLATE = '#4f6d8f'
DARK = '#3a3f44'
PT_GRAY = '#b8bcc2'
XLIM, YLIM = (-1.5, 27.5), (-1.5, 19.5)

# ---------------------------------------------------------------- geometry
outer = Polygon([(0, 0), (10, 0), (10, 0.6), (18, 0.6), (18, 0), (26, 0),
                 (26, 18), (0, 18)])
U = outer.difference(Polygon([(6, 7), (20, 7), (20, 18), (6, 18)]))
H_BUILD = 15.0
mesh = trimesh.creation.extrude_polygon(U, H_BUILD)

# pipeline stages (mirror run_pipeline, single component, default config)
verts = np.array(mesh.vertices)
h = (np.percentile(verts[:, 2], 99) - np.percentile(verts[:, 2], 5))
n = min(max(int(mesh.area * 10.0), 100), 2000)
sampled, _ = trimesh.sample.sample_surface(mesh, n, seed=0)
mp = MultiPoint(sampled[:, :2])

hull_a005 = shapely.concave_hull(mp, ratio=0.05, allow_holes=False)
hull_a02 = shapely.concave_hull(mp, ratio=0.2, allow_holes=False)
hull_a10 = shapely.concave_hull(mp, ratio=1.0, allow_holes=False)

fp_e01 = _clean_polygon(hull_a02.simplify(0.1, preserve_topology=True), 0.15)
fp_e05 = _clean_polygon(hull_a02.simplify(0.5, preserve_topology=True), 0.15)
fp_e20 = _clean_polygon(hull_a02.simplify(2.0, preserve_topology=True), 0.15)

final = trimesh.creation.extrude_polygon(fp_e05, height=h)
final.merge_vertices()
trimesh.repair.fix_normals(final); trimesh.repair.fix_inversion(final)
assert final.is_watertight

# ---------------------------------------------------------------- 2D panels
def panel_2d(poly=None, pts=None, mark_vertices=True, dpi=220):
    fig = plt.figure(figsize=(3.3, 2.55), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    if pts is not None:
        ax.scatter(pts[:, 0], pts[:, 1], s=1.4, c=PT_GRAY, linewidths=0,
                   zorder=1)
    if poly is not None:
        ax.fill(*poly.exterior.xy, color=SLATE, alpha=0.10, zorder=2)
        ax.plot(*poly.exterior.xy, color=SLATE, lw=1.6, zorder=3)
        if mark_vertices:
            xs, ys = poly.exterior.xy
            ax.plot(xs, ys, 'o', ms=3.0, color='#31506f', zorder=4)
    ax.set_xlim(*XLIM); ax.set_ylim(*YLIM)
    ax.set_aspect('equal')
    ax.set_axis_off()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', pad_inches=0, transparent=False)
    plt.close(fig)
    buf.seek(0)
    return np.array(Image.open(buf).convert('RGB'))

# ---------------------------------------------------------------- 3D panels
def _finish(fig, pad_frac=0.10):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.02,
                transparent=False)
    plt.close(fig)
    buf.seek(0)
    im = Image.open(buf).convert('RGB')
    from PIL import ImageChops
    bg = Image.new('RGB', im.size, (255, 255, 255))
    bbox = ImageChops.difference(im, bg).getbbox()
    if bbox:
        im = im.crop(bbox)
    pad = int(pad_frac * max(im.size))
    out = Image.new('RGB', (im.width + 2 * pad, im.height + 2 * pad),
                    (255, 255, 255))
    out.paste(im, (pad, pad))
    return np.array(out)

def _frame(ax):
    ax.set_xlim(-1.5, 27.5); ax.set_ylim(-1.5, 19.5)
    ax.set_zlim(-1.2, 16.8)
    ax.set_box_aspect((1, 1, 0.42))
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_axis_off()

def panel_3d_points(pts, s=3.0):
    fig = plt.figure(figsize=(4.2, 4.2), dpi=220)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=s, c=SLATE,
               linewidths=0, depthshade=False)
    _frame(ax)
    return _finish(fig)

def panel_3d_mesh(m):
    fig = plt.figure(figsize=(4.2, 4.2), dpi=220)
    ax = fig.add_subplot(111, projection='3d')
    v, f = np.asarray(m.vertices), np.asarray(m.faces)
    tri = v[f]
    nn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nn = nn / (np.linalg.norm(nn, axis=1, keepdims=True) + 1e-12)
    light = np.array([0.4, 0.3, 0.85]); light /= np.linalg.norm(light)
    shade = np.clip(nn @ light, 0.25, 1.0)
    base = np.array(matplotlib.colors.to_rgb(DARK))
    cols = np.clip(shade[:, None] * base[None, :], 0, 1)
    ax.add_collection3d(Poly3DCollection(tri, facecolors=cols,
                                         edgecolors='none'))
    _frame(ax)
    return _finish(fig)

# ---------------------------------------------------------------- render
a = panel_3d_points(sampled)
b = panel_2d(pts=sampled[:, :2])
c = panel_2d(poly=hull_a02, pts=sampled[:, :2], mark_vertices=False)
d = panel_2d(poly=hull_a02.simplify(0.5, preserve_topology=True),
             mark_vertices=False)
e = panel_2d(poly=fp_e05)
f = panel_3d_mesh(final)
print('row 1 done')

g = panel_2d(poly=hull_a005, pts=sampled[:, :2], mark_vertices=False)
h_ = panel_2d(poly=hull_a02, pts=sampled[:, :2], mark_vertices=False)
i = panel_2d(poly=hull_a10, pts=sampled[:, :2], mark_vertices=False)
j = panel_2d(poly=fp_e01)
k = panel_2d(poly=fp_e05)
l = panel_2d(poly=fp_e20)
print('row 2 done')

# ---------------------------------------------------------------- compose
row1 = [
    ('(a) Sampled point set', a),
    ('(b) xy projection', b),
    ('(c) \u03b1-shape (\u03b1 = 0.2)', c),
    ('(d) DP simplification\n(\u03b5 = 0.5 m)', d),
    ('(e) Regularization', e),
    ('(f) Extrusion (LOD1)', f),
]
row2 = [
    ('(g) \u03b1 = 0.05', g),
    ('(h) \u03b1 = 0.2 (default)', h_),
    ('(i) \u03b1 = 1.0', i),
    ('(j) \u03b5 = 0.1 m', j),
    ('(k) \u03b5 = 0.5 m (default)', k),
    ('(l) \u03b5 = 2.0 m', l),
]

fig = plt.figure(figsize=(14.0, 8.6), dpi=220)
gs = fig.add_gridspec(2, 6, height_ratios=[1.22, 1.0],
                      left=0.015, right=0.985, top=0.88, bottom=0.02,
                      wspace=0.05, hspace=0.24)
axes = [[fig.add_subplot(gs[r, c_]) for c_ in range(6)] for r in range(2)]
for r, row in enumerate((row1, row2)):
    for ax, (title, img) in zip(axes[r], row):
        ax.imshow(img)
        ax.set_axis_off()
        ax.set_title(title, fontsize=10.5, pad=6)

# group headers
fig.text(0.5, 0.965, 'Reconstruction stages (default parameters)',
         fontsize=12.5, ha='center')

# dividers
p = [[ax.get_position() for ax in row] for row in axes]
y_div = (p[0][0].y0 + p[1][0].y1) / 2
x_div = (p[1][2].x1 + p[1][3].x0) / 2
ylo = p[1][0].y0; yhi = p[0][0].y1
xlo = p[0][0].x0; xhi = p[0][5].x1
fig.add_artist(plt.Line2D([xlo, xhi], [y_div, y_div],
                          transform=fig.transFigure, color='0.55', lw=1.0))
fig.add_artist(plt.Line2D([x_div, x_div], [ylo, y_div],
                          transform=fig.transFigure, color='0.55', lw=1.0))

# parameter-group headers sit in the gap between the two rows, above the
# divider, directly over the group they label
hdr_y = p[0][0].y0 - 0.30 * (p[0][0].y0 - p[1][0].y1)
fig.text(0.5 * (p[1][0].x0 + p[1][2].x1), hdr_y,
         'Effect of \u03b1 (concave-hull ratio, \u03b1-shape)',
         fontsize=12.5, ha='center')
fig.text(0.5 * (p[1][3].x0 + p[1][5].x1), hdr_y,
         'Effect of \u03b5 (DP tolerance, final footprint)',
         fontsize=12.5, ha='center')

fig.savefig(Path(__file__).resolve().parent / 'fig_reconstruction_stages.png',
            bbox_inches='tight',
            pad_inches=0.06, facecolor='white')
plt.close(fig)
print('saved figures/fig_reconstruction_stages.png')
