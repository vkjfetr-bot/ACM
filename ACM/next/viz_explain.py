"""
viz_explain.py

Explainability and latent views for the next-gen ACM report.
All functions are optional and tolerate missing inputs.
"""

from __future__ import annotations

import os
import json
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .viz_core import ensure_dir, save_fig, sanitize_array, set_seed


def read_contrib_jsonl(path: str) -> Dict[str, List[Tuple[str, float]]]:
    out: Dict[str, List[Tuple[str, float]]] = {}
    if not path or not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                eid = str(obj.get("window_id") or obj.get("event_id") or len(out))
                contrib = obj.get("contrib") or obj.get("contributors") or []
                pairs = [(str(c.get("tag")), float(c.get("score", 0.0))) for c in contrib]
                out[eid] = pairs
            except Exception:
                continue
    return out


def plot_contrib_bars(event_id: str, contrib: List[Tuple[str, float]], out_path: str, top_k: int = 10) -> str:
    pairs = contrib or []
    pairs = sorted(pairs, key=lambda x: x[1], reverse=True)[:top_k]
    if not pairs:
        fig, ax = plt.subplots(figsize=(6, 3), constrained_layout=True)
        ax.axis("off")
        ax.text(0.5, 0.5, f"No contributions for {event_id}", ha="center", va="center")
        return save_fig(fig, out_path)
    tags, scores = zip(*pairs)
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.barh(tags, scores, color="#86efac")
    ax.invert_yaxis()
    ax.set_title(f"Top contributors • event {event_id}")
    ax.set_xlabel("score")
    return save_fig(fig, out_path)


def plot_attention_maps(attn_npz: str, temporal_out: str, spatial_out: str) -> Tuple[str, str]:
    if not attn_npz or not os.path.exists(attn_npz):
        # placeholders
        for out in (temporal_out, spatial_out):
            fig, ax = plt.subplots(figsize=(6, 3), constrained_layout=True)
            ax.axis("off")
            ax.text(0.5, 0.5, "No attention inputs", ha="center", va="center")
            save_fig(fig, out)
        return temporal_out, spatial_out

    with np.load(attn_npz) as npz:
        # expected optional keys: temporal [T], spatial [T x tags]
        temporal = npz.get("temporal")
        spatial = npz.get("spatial")

    if temporal is None or np.size(temporal) == 0:
        t_mat = np.zeros((1, 10), dtype=float)
    else:
        t = sanitize_array(np.asarray(temporal))
        t_mat = t[None, :] if t.ndim == 1 else t

    fig, ax = plt.subplots(figsize=(8, 2.6), constrained_layout=True)
    im = ax.imshow(t_mat, aspect="auto", cmap="viridis")
    ax.set_title("Temporal attention")
    save_fig(fig, temporal_out)

    if spatial is None or np.size(spatial) == 0:
        s_mat = np.zeros((10, 10), dtype=float)
    else:
        s = sanitize_array(np.asarray(spatial))
        s_mat = s

    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    im = ax.imshow(s_mat, aspect="auto", cmap="magma")
    ax.set_title("Spatial/tag attention")
    save_fig(fig, spatial_out)

    return temporal_out, spatial_out


def plot_latent_space(emb: np.ndarray, out_path: str, seed: int = 42) -> str:
    set_seed(seed)
    if emb is None or np.size(emb) == 0:
        fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
        ax.axis("off")
        ax.text(0.5, 0.5, "No latent embeddings", ha="center", va="center")
        return save_fig(fig, out_path)

    X = sanitize_array(np.asarray(emb))
    n, d = X.shape if X.ndim == 2 else (X.shape[0], 1)
    # Use PCA for dependency-light 2D
    Xc = X - X.mean(axis=0, keepdims=True)
    cov = np.cov(Xc, rowvar=False) if d > 1 else np.array([[1.0]])
    vals, vecs = np.linalg.eigh(cov)
    idx = np.argsort(vals)[::-1][:2]
    P = vecs[:, idx]
    Z = (Xc @ P) if d > 1 else np.column_stack([np.arange(n), Xc.reshape(-1)])

    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    ax.scatter(Z[:, 0], Z[:, 1], s=6, c="#60a5fa", alpha=0.7)
    ax.set_title("Latent space (PCA)")
    ax.grid(True, alpha=0.2)
    return save_fig(fig, out_path)

