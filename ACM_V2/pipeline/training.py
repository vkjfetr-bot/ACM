"""Training pipeline orchestrator."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.preprocessing import StandardScaler

from .artifacts import ArtifactManager
from .config import PipelineConfig
from .data import prepare_window
from .features import FeatureConfig, build_feature_matrix
from .regime import (
    RegimeModel,
    cluster_segments,
    expand_labels_to_samples,
    fit_pca,
    segment_latent,
    smooth_labels,
)


@dataclass
class TrainingResult:
    run_id: str
    features_path: Path
    regimes_path: Path
    model_paths: Dict[str, Path]
    metadata: Dict[str, object]


def run_training(
    csv_source: Path,
    *,
    equip: str,
    config: PipelineConfig,
    artifact_manager: ArtifactManager,
    t0: Optional[datetime] = None,
    t1: Optional[datetime] = None,
) -> TrainingResult:
    art_paths = artifact_manager.start_run()

    sampling = config.get("sampling.period", "10s")
    feature_cfg = FeatureConfig(
        window_seconds=int(pd.to_timedelta(config.get("features.window", "60s")).total_seconds()),
        step_seconds=int(pd.to_timedelta(config.get("features.step", "10s")).total_seconds()),
        spectral_bands=[tuple(band) for band in config.get("features.spectral_bands", [[0.0, 0.1]])],
    )

    clamp_sigma = 6.0
    data_window = prepare_window(
        csv_source,
        equip=equip,
        resample_rule=sampling,
        clamp_sigma=clamp_sigma,
        t0=t0,
        t1=t1,
    )

    artifact_manager.write_table(data_window.raw, "raw.parquet")
    artifact_manager.write_table(data_window.clean, "clean.parquet")
    artifact_manager.write_json(data_window.dq.to_dict(orient="records"), "dq.json")

    period_seconds = int(pd.to_timedelta(sampling).total_seconds())
    feature_df = build_feature_matrix(
        data_window.clean,
        period_seconds=period_seconds,
        config=feature_cfg,
        key_tags=config.get("report.key_tags", []),
    )
    features_path = artifact_manager.write_table(feature_df, "features.parquet")

    scaler = StandardScaler()
    feature_matrix = scaler.fit_transform(feature_df.fillna(method="ffill").fillna(0.0).values)

    pca, latent = fit_pca(feature_matrix, config.get("pca.variance", 0.95))

    min_duration_s = config.get("segmentation.min_duration_s", 60)
    min_size = max(int(min_duration_s / feature_cfg.step_seconds), 5)
    segments = segment_latent(latent, min_size=min_size)

    cluster_algo = config.get("clustering.algo", "kmeans_auto")
    k_range = tuple(config.get("clustering.k_range", [2, 6]))
    min_cluster_minutes = config.get("clustering.min_cluster_minutes", 3)
    min_cluster_samples = max(int((min_cluster_minutes * 60) / feature_cfg.step_seconds), 2)
    labels, cluster_model, cluster_type = cluster_segments(
        latent,
        segments,
        algo=cluster_algo,
        k_range=k_range,
        min_cluster_size=min_cluster_samples,
    )

    sample_labels = expand_labels_to_samples(segments, labels, len(feature_df))
    min_state_seconds = config.get("hmm.min_state_seconds", 60)
    min_state_samples = max(int(min_state_seconds / feature_cfg.step_seconds), 1)
    smoothed_labels = smooth_labels(sample_labels, min_state_samples)

    regimes_df = pd.DataFrame({"Ts": feature_df.index, "Regime": smoothed_labels})
    regimes_path = artifact_manager.write_json(regimes_df.to_dict(orient="records"), "regimes.json")

    metadata = {
        "k": int(len(np.unique(smoothed_labels))),
        "segments": len(segments),
        "cluster_type": cluster_type,
        "period_seconds": period_seconds,
        "feature_window_seconds": feature_cfg.window_seconds,
        "feature_step_seconds": feature_cfg.step_seconds,
        "reference_hist": {int(k): float(v) for k, v in regimes_df["Regime"].value_counts(normalize=True).items()},
    }
    regime_model = RegimeModel(pca=pca, cluster=cluster_model, cluster_type=cluster_type, metadata=metadata)

    scaler_path = artifact_manager.latest_model_path("scaler.joblib")
    dump(scaler, scaler_path)
    pca_path = artifact_manager.latest_model_path("pca.joblib")
    dump(pca, pca_path)
    cluster_path = artifact_manager.latest_model_path("cluster.joblib")
    dump(cluster_model, cluster_path)
    meta_path = artifact_manager.latest_model_path("regime_meta.json")
    meta_path.write_text(pd.Series(metadata).to_json(), encoding="utf-8")
    # TODO: Upsert regime model to SQL (usp_WriteRegimeModels).

    artifact_manager.mark_success({"stage": "train", "k": metadata["k"]})
    artifact_manager.emit_run_summary(
        {
            "run_id": art_paths.run_id,
            "ts_utc": datetime.utcnow().isoformat() + "Z",
            "equip": equip,
            "cmd": "train",
            "rows_in": len(data_window.clean),
            "tags": len(data_window.tags),
            "feat_rows": len(feature_df),
            "regimes": metadata["k"],
            "events": 0,
            "data_span_min": (feature_df.index[-1] - feature_df.index[0]).total_seconds() / 60.0 if not feature_df.empty else 0.0,
            "phase": "train",
            "k_selected": metadata["k"],
            "theta_p95": 0.0,
            "drift_flag": 0,
            "guardrail_state": "ok",
            "theta_step_pct": 0.0,
            "latency_s": 0.0,
            "artifacts_age_min": 0.0,
            "status": "ok",
            "err_msg": "",
        }
    )

    return TrainingResult(
        run_id=art_paths.run_id,
        features_path=features_path,
        regimes_path=regimes_path,
        model_paths={
            "scaler": scaler_path,
            "pca": pca_path,
            "cluster": cluster_path,
            "meta": meta_path,
        },
        metadata=metadata,
    )
