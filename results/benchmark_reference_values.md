# Benchmark reference values

Reference values for the three case-study geometries in `data/benchmark/`,
computed with `metrics.py` (micro-face threshold 0.25 m², fragment
threshold 8 faces). These values correspond to Table 4 of the revised
manuscript and are checked automatically by `reproduce_table4.py`.

## Checksums

| File | MD5 |
|---|---|
| `data/benchmark/baseline_raw_gis.stl` | `033927353773e6fa1d715bf9882be4e1` |
| `data/benchmark/reference_manual_cleanup.stl` | `b287d44c647089fd8763f7def6afbf5e` |
| `data/benchmark/modeled_pipeline_output.stl` | `6d552d7fb341794e1002d1c8f2b64209` |

## Table 4 — geometric quality comparison

| Metric | Baseline | Reference | Modeled |
|---|---|---|---|
| Components | 15,659 | 15,072 | 15,056 |
| Total faces | 365,668 | 284,223 | 273,432 |
| Watertightness rate | 98.8% | 96.9% | 100.0% |
| Non-manifold edges | 568 | 527 | 0 |
| Degenerate-face bldgs. | 293 (1.9%) | 16 (0.1%) | 0 (0.0%) |
| Micro-face bldgs. | 1,905 (12.2%) | 745 (4.9%) | 188 (1.2%) |
| Total micro-faces | 8,608 (2.4%) | 2,920 (1.0%) | 394 (0.1%) |

Additional whole-mesh diagnostics (not part of Table 4):

| Diagnostic | Baseline | Reference | Modeled |
|---|---|---|---|
| Open edges | 0 | 519 | 0 |
| Degenerate faces | 586 | 32 | 0 |

## End-to-end pipeline run (modeled geometry)

Command: `python pipeline.py data/benchmark/baseline_raw_gis.stl -o modeled_model.stl`

```
Buildings: 15,056/15,251 (98.7%)
  excluded: 188 filtered (h:1, fp:187), 7 failed reconstruction
Faces: 273,432
Output checksum (MD5): 6d552d7fb341794e1002d1c8f2b64209
  (bit-identical to data/benchmark/modeled_pipeline_output.stl)
```

The 15,251 building candidates are identified from the 15,659 connected
components of the baseline geometry; all 15,056 reconstructed buildings
passed the watertightness verification (100% watertight, 0 non-manifold
edges, 0 degenerate faces, 0 residual fragments).

## Environment

Python 3.14.3, numpy 2.5.3, trimesh 5.1.0, shapely 2.1.2,
scikit-learn 1.9.1, mapbox-earcut 2.1.0 (see `requirements.txt`).
