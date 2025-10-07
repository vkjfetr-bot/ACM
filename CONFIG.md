# ACM Next — Config Overview

This document describes dials for the next-gen ACM components. The current drop focuses on the reporting path; core modeling dials will be wired in subsequent milestones.

Sections:
- features: `enable_wavelet|entropy|trend|hurst` (default: off)
- model: `h1_type = {AR, VAR, AE}` (default: VAR)
- regimes: `algo = {kmeans, gmm, hdbscan}`
- context: `mask_transients = true|false`, `key_tags = []`
- dcff: `enable = false`
- loss: `{type: huber|mse, delta: 1.0}`
- threshold: `{type: ndt|pot, q: 0.95}`
- sampling.attention: `{enable: false, heads: 2, k: 16, downsample_ratio: 4}`

Notes:
- Determinism: set `seed` in CLI; plotting and PCA are seeded.
- Large matrices are clipped for visualization robustness and file size.

