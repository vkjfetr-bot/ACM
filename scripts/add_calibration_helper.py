#!/usr/bin/env python3
"""Add CalibrationResult dataclass and _calibrate_scores helper function."""

# Read the file
with open("core/acm_main.py", "r", encoding="utf-8") as f:
    content = f.read()

# Find the location to insert (before ScoreResult dataclass)
insert_marker = "@dataclass\nclass ScoreResult:"

# New code to insert
new_code = '''@dataclass
class CalibrationResult:
    """Result of score calibration."""
    frame: pd.DataFrame  # Frame with calibrated z-scores
    train_frame: pd.DataFrame  # Train frame with raw and calibrated scores
    calibrators: List[Tuple[str, Any]]  # List of (name, calibrator) tuples
    spe_p95_train: float
    t2_p95_train: float
    pca_train_spe_z: np.ndarray
    pca_train_t2_z: np.ndarray


def _calibrate_scores(
    frame: pd.DataFrame,
    train: pd.DataFrame,
    train_regime_labels: Optional[np.ndarray],
    score_regime_labels: Optional[np.ndarray],
    regime_quality_ok: bool,
    ar1_detector: Any,
    pca_detector: Any,
    iforest_detector: Any,
    gmm_detector: Any,
    omr_detector: Optional[Any],
    omr_enabled: bool,
    pca_train_spe: Optional[np.ndarray],
    pca_train_t2: Optional[np.ndarray],
    cfg: Dict[str, Any],
    equip: str,
    output_manager: Any,
    T: Timer,
) -> CalibrationResult:
    """Calibrate detector raw scores to z-scores using training data.
    
    Fits calibrators on TRAIN data, transforms SCORE data.
    Generates per-regime threshold tables if quality is acceptable.
    
    Args:
        frame: Frame with raw scores from detectors.
        train: Training features.
        train_regime_labels: Regime labels for training data (if quality ok).
        score_regime_labels: Regime labels for score data (if quality ok).
        regime_quality_ok: Whether regime quality is acceptable for per-regime thresholds.
        ar1_detector: Fitted AR1 detector.
        pca_detector: Fitted PCA detector.
        iforest_detector: Fitted IForest detector.
        gmm_detector: Fitted GMM detector.
        omr_detector: Fitted OMR detector (optional).
        omr_enabled: Whether OMR is enabled.
        pca_train_spe: Cached PCA SPE scores on training data.
        pca_train_t2: Cached PCA T2 scores on training data.
        cfg: Configuration dictionary.
        equip: Equipment name.
        output_manager: OutputManager for writing tables.
        T: Timer for profiling.
    
    Returns:
        CalibrationResult with calibrated frame and training metrics.
    """
    with T.section("calibrate"):
        cal_q = float((cfg or {}).get("thresholds", {}).get("q", 0.98))
        self_tune_cfg = (cfg or {}).get("thresholds", {}).get("self_tune", {})
        use_per_regime = (cfg.get("fusion", {}) or {}).get("per_regime", False)
        quality_ok = bool(use_per_regime and regime_quality_ok and train_regime_labels is not None and score_regime_labels is not None)
        
        # Score TRAIN data with all fitted detectors
        Console.info("Scoring TRAIN data for calibration baseline...", component="CAL")
        train_frame = pd.DataFrame(index=train.index)
        train_frame["ar1_raw"] = ar1_detector.score(train)
        
        # CRITICAL FIX: Reuse cached PCA train scores to avoid recomputation
        if pca_train_spe is not None and pca_train_t2 is not None:
            Console.info("Using cached PCA train scores (optimization)", component="CAL")
            train_frame["pca_spe"], train_frame["pca_t2"] = pca_train_spe, pca_train_t2
        else:
            Console.warn("Cache miss - recomputing PCA train scores", component="CAL",
                         equip=equip)
            train_frame["pca_spe"], train_frame["pca_t2"] = pca_detector.score(train)
        train_frame["iforest_raw"] = iforest_detector.score(train)
        train_frame["gmm_raw"] = gmm_detector.score(train)
        if omr_enabled and omr_detector:
            train_frame["omr_raw"] = omr_detector.score(train, return_contributions=False)
        
        # Compute adaptive z-clip
        default_clip = float(self_tune_cfg.get("clip_z", 8.0))
        temp_cfg = dict(self_tune_cfg)
        train_z_p99: Dict[str, float] = {}
        
        det_list = [("ar1", "ar1_raw"), ("pca_spe", "pca_spe"), ("pca_t2", "pca_t2"),
                   ("iforest", "iforest_raw"), ("gmm", "gmm_raw")]
        if omr_enabled:
            det_list.append(("omr", "omr_raw"))
        
        for det_name, raw_col in det_list:
            if raw_col in train_frame.columns:
                temp_cal = fuse.ScoreCalibrator(q=cal_q, self_tune_cfg=temp_cfg).fit(
                    train_frame[raw_col].to_numpy(copy=False), regime_labels=None
                )
                temp_z = temp_cal.transform(train_frame[raw_col].to_numpy(copy=False), regime_labels=None)
                finite_z = temp_z[np.isfinite(temp_z)]
                if len(finite_z) > 10:
                    p99 = float(np.percentile(finite_z, 99))
                    if 0 < p99 < 100:
                        train_z_p99[det_name] = p99
        
        # Set adaptive clip
        if train_z_p99:
            max_train_p99 = max(train_z_p99.values())
            adaptive_clip = max(default_clip, min(max_train_p99 * 1.5, 50.0))
            self_tune_cfg["clip_z"] = adaptive_clip
            Console.info(f"Adaptive clip_z={adaptive_clip:.2f} (TRAIN P99 max={max_train_p99:.2f})", component="CAL")
        
        fit_regimes = train_regime_labels if quality_ok else None
        transform_regimes = score_regime_labels if quality_ok else None

        # Surface per-regime calibration activity
        frame["per_regime_active"] = 1 if quality_ok else 0
        
        # Calibrate each detector
        cal_ar = fuse.ScoreCalibrator(q=cal_q, self_tune_cfg=self_tune_cfg, name="ar1_z").fit(
            train_frame["ar1_raw"].to_numpy(copy=False), regime_labels=fit_regimes
        )
        frame["ar1_z"] = cal_ar.transform(frame["ar1_raw"].to_numpy(copy=False), regime_labels=transform_regimes)
        
        cal_pca_spe = fuse.ScoreCalibrator(q=cal_q, self_tune_cfg=self_tune_cfg, name="pca_spe_z").fit(
            train_frame["pca_spe"].to_numpy(copy=False), regime_labels=fit_regimes
        )
        frame["pca_spe_z"] = cal_pca_spe.transform(frame["pca_spe"].to_numpy(copy=False), regime_labels=transform_regimes)
        
        cal_pca_t2 = fuse.ScoreCalibrator(q=cal_q, self_tune_cfg=self_tune_cfg, name="pca_t2_z").fit(
            train_frame["pca_t2"].to_numpy(copy=False), regime_labels=fit_regimes
        )
        frame["pca_t2_z"] = cal_pca_t2.transform(frame["pca_t2"].to_numpy(copy=False), regime_labels=transform_regimes)
        
        cal_if = fuse.ScoreCalibrator(q=cal_q, self_tune_cfg=self_tune_cfg, name="iforest_z").fit(
            train_frame["iforest_raw"].to_numpy(copy=False), regime_labels=fit_regimes
        )
        frame["iforest_z"] = cal_if.transform(frame["iforest_raw"].to_numpy(copy=False), regime_labels=transform_regimes)
        
        cal_gmm = fuse.ScoreCalibrator(q=cal_q, self_tune_cfg=self_tune_cfg, name="gmm_z").fit(
            train_frame["gmm_raw"].to_numpy(copy=False), regime_labels=fit_regimes
        )
        frame["gmm_z"] = cal_gmm.transform(frame["gmm_raw"].to_numpy(copy=False), regime_labels=transform_regimes)
        
        # OMR calibration
        cal_omr = None
        if omr_enabled and "omr_raw" in train_frame.columns and "omr_raw" in frame.columns:
            cal_omr = fuse.ScoreCalibrator(q=cal_q, self_tune_cfg=self_tune_cfg, name="omr_z").fit(
                train_frame["omr_raw"].to_numpy(copy=False), regime_labels=fit_regimes
            )
            frame["omr_z"] = cal_omr.transform(frame["omr_raw"].to_numpy(copy=False), regime_labels=transform_regimes)
        
        # River Half-Space Trees (if enabled)
        cal_river = None
        if "river_hst_raw" in frame.columns:
            cal_river = fuse.ScoreCalibrator(q=cal_q, self_tune_cfg=self_tune_cfg).fit(
                frame["river_hst_raw"].to_numpy(copy=False), regime_labels=transform_regimes
            )
            frame["river_hst_z"] = cal_river.transform(
                frame["river_hst_raw"].to_numpy(copy=False), regime_labels=transform_regimes
            )

        # Compute TRAIN z-scores for PCA metrics
        pca_train_spe_z = cal_pca_spe.transform(
            train_frame["pca_spe"].to_numpy(dtype=np.float32), regime_labels=fit_regimes
        )
        pca_train_t2_z = cal_pca_t2.transform(
            train_frame["pca_t2"].to_numpy(dtype=np.float32), regime_labels=fit_regimes
        )
        spe_p95_train = float(np.nanpercentile(pca_train_spe_z, 95))
        t2_p95_train = float(np.nanpercentile(pca_train_t2_z, 95))
        
        # Build calibrators list
        calibrators: List[Tuple[str, Any]] = [
            ("ar1_z", cal_ar),
            ("pca_spe_z", cal_pca_spe),
            ("pca_t2_z", cal_pca_t2),
            ("iforest_z", cal_if),
            ("gmm_z", cal_gmm),
        ]
        if cal_omr is not None:
            calibrators.append(("omr_z", cal_omr))
        if cal_river is not None:
            calibrators.append(("river_hst_z", cal_river))

        # Generate per-regime threshold table
        if quality_ok and use_per_regime:
            Console.info("Generating per-regime threshold table...", component="CAL")
            per_regime_rows = []
            for detector_name, calibrator in calibrators:
                for regime_id in sorted(calibrator.regime_thresh_.keys()):
                    med_r, scale_r = calibrator.regime_params_[regime_id]
                    thresh_z = calibrator.regime_thresh_[regime_id]
                    per_regime_rows.append({
                        "detector": detector_name,
                        "regime": int(regime_id),
                        "median": float(med_r),
                        "scale": float(scale_r),
                        "z_threshold": float(thresh_z),
                        "global_median": float(calibrator.med),
                        "global_scale": float(calibrator.scale),
                    })
            
            if per_regime_rows:
                per_regime_df = pd.DataFrame(per_regime_rows)
                output_manager.write_dataframe(per_regime_df, "per_regime_thresholds")
                Console.info(f"Wrote per-regime thresholds: {len(per_regime_rows)} regime-detector pairs", component="CAL")

        # Write thresholds table
        threshold_rows: List[Dict[str, Any]] = []
        for detector_name, calibrator in calibrators:
            threshold_rows.append({
                "detector": detector_name,
                "regime": "GLOBAL",
                "median": float(calibrator.med),
                "scale": float(calibrator.scale),
                "z_threshold": float(calibrator.q_z),
                "raw_threshold": float(calibrator.q_thresh),
            })
            for regime_id, regime_thresh in calibrator.regime_thresh_.items():
                med_r, scale_r = calibrator.regime_params_.get(regime_id, (calibrator.med, calibrator.scale))
                threshold_rows.append({
                    "detector": detector_name,
                    "regime": int(regime_id),
                    "median": float(med_r),
                    "scale": float(scale_r),
                    "z_threshold": float(regime_thresh),
                    "raw_threshold": float(med_r + regime_thresh * max(scale_r, 1e-9)),
                })

        if threshold_rows:
            thresholds_df = pd.DataFrame(threshold_rows)
            output_manager.write_dataframe(thresholds_df, "acm_thresholds")
            Console.info(f"Wrote thresholds table with {len(threshold_rows)} rows -> acm_thresholds", component="CAL")

    return CalibrationResult(
        frame=frame,
        train_frame=train_frame,
        calibrators=calibrators,
        spe_p95_train=spe_p95_train,
        t2_p95_train=t2_p95_train,
        pca_train_spe_z=pca_train_spe_z,
        pca_train_t2_z=pca_train_t2_z,
    )


'''

# Check if already added
if "class CalibrationResult:" in content:
    print("CalibrationResult already exists, skipping")
else:
    # Insert before ScoreResult
    if insert_marker in content:
        content = content.replace(insert_marker, new_code + insert_marker)
        
        with open("core/acm_main.py", "w", encoding="utf-8") as f:
            f.write(content)
        
        print("SUCCESS: Added CalibrationResult and _calibrate_scores helper")
    else:
        print("ERROR: Could not find insertion marker")
