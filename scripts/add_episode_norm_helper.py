#!/usr/bin/env python3
"""Add EpisodeNormResult dataclass and _normalize_episodes helper function."""

# Read the file
with open("core/acm_main.py", "r", encoding="utf-8") as f:
    content = f.read()

# Find the location to insert (before DriftResult dataclass)
insert_marker = "@dataclass\nclass DriftResult:"

# New code to insert
new_code = '''@dataclass
class EpisodeNormResult:
    """Result of episode normalization."""
    episodes: pd.DataFrame  # Normalized episodes with all columns
    frame: pd.DataFrame  # Frame (potentially sorted/deduped)


def _normalize_episodes(
    episodes: Optional[pd.DataFrame],
    frame: pd.DataFrame,
    equip: str,
) -> EpisodeNormResult:
    """Normalize episodes schema for report/export.
    
    Ensures episodes have required columns (episode_id, severity, regime, 
    start_ts, end_ts) and maps indices to timestamps.
    
    Args:
        episodes: Raw episodes DataFrame (may be None or empty).
        frame: Frame with fused scores and regime labels.
        equip: Equipment name for logging.
    
    Returns:
        EpisodeNormResult with normalized episodes and potentially sorted frame.
    """
    # Defensive copy: ensure episodes is a DataFrame before .copy()
    episodes = (episodes if isinstance(episodes, pd.DataFrame) else pd.DataFrame()).copy()
    
    # Ensure required columns exist
    if "episode_id" not in episodes.columns:
        episodes.insert(0, "episode_id", np.arange(1, len(episodes) + 1, dtype=int))
    if "severity" not in episodes.columns:
        episodes["severity"] = "info"
    if "regime" not in episodes.columns:
        episodes["regime"] = ""
    if "start_ts" not in episodes.columns:
        episodes["start_ts"] = pd.NaT
    if "end_ts" not in episodes.columns:
        episodes["end_ts"] = pd.NaT
    
    start_idx_series = episodes.get("start")
    end_idx_series = episodes.get("end")
    
    # Ensure frame is sorted before any indexing operations
    if not frame.index.is_monotonic_increasing:
        Console.warn("Sorting frame index for timestamp mapping", component="EPISODE", equip=equip)
        frame = frame.sort_index()
    idx_array = frame.index.to_numpy()

    # Prefer nearest mapping; preserve NaT (avoid clip-to-zero artefacts)
    if start_idx_series is None:
        # CRITICAL FIX: Deduplicate frame index before episode mapping to prevent aggregation errors
        if not frame.index.is_unique:
            Console.warning(
                f"Deduplicating {len(frame)} - {frame.index.nunique()} = {len(frame) - frame.index.nunique()} duplicate timestamps",
                component="EPISODES"
            )
            frame = frame.groupby(frame.index).first()
            idx_array = frame.index.to_numpy()  # Update after deduplication

        start_positions = _nearest_indexer(frame.index, episodes["start_ts"], label="EPISODE.start")
        start_idx_series = pd.Series(start_positions, index=episodes.index, dtype="int64")
    
    if end_idx_series is None:
        end_positions = _nearest_indexer(frame.index, episodes["end_ts"], label="EPISODE.end")
        end_idx_series = pd.Series(end_positions, index=episodes.index, dtype="int64")
    
    start_idx_series = start_idx_series.fillna(-1).astype(int)
    end_idx_series = end_idx_series.fillna(-1).astype(int)
    
    if len(idx_array):
        start_idx = start_idx_series.clip(-1, len(idx_array) - 1).to_numpy()
        end_idx = end_idx_series.clip(-1, len(idx_array) - 1).to_numpy()
        s_idx_safe = np.where(start_idx >= 0, start_idx, 0)
        e_idx_safe = np.where(end_idx >= 0, end_idx, 0)
        # Create datetime arrays, use pd.NaT for invalid indices
        start_times = idx_array[s_idx_safe]
        end_times = idx_array[e_idx_safe]
        episodes["start_ts"] = pd.Series(start_times, index=episodes.index, dtype='datetime64[ns]')
        episodes["end_ts"] = pd.Series(end_times, index=episodes.index, dtype='datetime64[ns]')
        # Set NaT for invalid indices
        episodes.loc[start_idx < 0, "start_ts"] = pd.NaT
        episodes.loc[end_idx < 0, "end_ts"] = pd.NaT
    else:
        start_idx = np.zeros(len(episodes), dtype=int)
        end_idx = np.zeros(len(episodes), dtype=int)
        episodes["start_ts"] = pd.NaT
        episodes["end_ts"] = pd.NaT

    # Map regime labels to episodes
    label_series = frame.get("regime_label")
    state_series = frame.get("regime_state")
    if label_series is not None:
        label_array = label_series.to_numpy()
        state_array = state_series.to_numpy() if state_series is not None else None
        regime_vals: List[Any] = []
        regime_states: List[str] = []
        for s_idx, e_idx in zip(start_idx, end_idx):
            if len(label_array) == 0:
                regime_vals.append(-1)
                regime_states.append("unknown")
                continue
            s_clamped = int(np.clip(s_idx, 0, len(label_array) - 1))
            e_clamped = int(np.clip(e_idx, 0, len(label_array) - 1))
            if e_clamped < s_clamped:
                e_clamped = s_clamped
            slice_labels = label_array[s_clamped:e_clamped + 1]
            if slice_labels.size:
                counts = np.bincount(slice_labels.astype(int))
                majority_label = int(np.argmax(counts))
            else:
                majority_label = -1
            regime_vals.append(majority_label)
            if state_array is not None and slice_labels.size:
                slice_states = state_array[s_clamped:e_clamped + 1]
                values, counts_arr = np.unique(slice_states, return_counts=True)
                majority_state = str(values[np.argmax(counts_arr)])
            else:
                majority_state = "unknown"
            regime_states.append(majority_state)
        episodes["regime"] = regime_vals
        episodes["regime_state"] = regime_states
    else:
        episodes["regime_state"] = "unknown"

    # Map severity from regime state
    severity_map = {"critical": "critical", "suspect": "warning", "warning": "warning"}
    severity_override = episodes["regime_state"].map(lambda s: severity_map.get(str(s)))
    episodes["severity"] = severity_override.fillna(episodes["severity"])

    # Compute duration columns
    start_ts = pd.to_datetime(episodes["start_ts"], errors="coerce")
    end_ts = pd.to_datetime(episodes["end_ts"], errors="coerce")
    episodes["duration_s"] = (end_ts - start_ts).dt.total_seconds()
    
    # Convenience: duration in hours for operator tables
    try:
        episodes["duration_hours"] = episodes["duration_s"].astype(float) / 3600.0
    except Exception:
        episodes["duration_hours"] = np.where(
            pd.notna(episodes.get("duration_s")),
            episodes.get("duration_s").astype(float) / 3600.0,
            0.0
        )
    
    # Sort and clean up
    episodes = episodes.sort_values(["start_ts", "end_ts", "episode_id"]).reset_index(drop=True)
    episodes["regime"] = episodes["regime"].astype(str)
    
    return EpisodeNormResult(episodes=episodes, frame=frame)


'''

# Check if already added
if "class EpisodeNormResult:" in content:
    print("EpisodeNormResult already exists, skipping")
else:
    # Insert before DriftResult
    if insert_marker in content:
        content = content.replace(insert_marker, new_code + insert_marker)
        
        with open("core/acm_main.py", "w", encoding="utf-8") as f:
            f.write(content)
        
        print("SUCCESS: Added EpisodeNormResult and _normalize_episodes helper")
    else:
        print("ERROR: Could not find insertion marker")
