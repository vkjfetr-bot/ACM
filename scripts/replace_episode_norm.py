#!/usr/bin/env python3
"""Replace the episode normalization section in main() with _normalize_episodes() call."""

with open("core/acm_main.py", "r", encoding="utf-8") as f:
    content = f.read()

# The old code block to replace
old_code = '''        # --- Normalize episodes schema for report/export ------------
        # Defensive copy: ensure episodes is a DataFrame before .copy()
        episodes = (episodes if isinstance(episodes, pd.DataFrame) else pd.DataFrame()).copy()
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
            Console.warn("Sorting frame index for timestamp mapping", component="EPISODE",
                         equip=equip)
            frame = frame.sort_index()
        idx_array = frame.index.to_numpy()

        # Prefer nearest mapping; preserve NaT (avoid clip-to-zero artefacts)
        if start_idx_series is None:
            # CRITICAL FIX: Deduplicate frame index before episode mapping to prevent aggregation errors
            if not frame.index.is_unique:
                Console.warning(f"Deduplicating {len(frame)} - {frame.index.nunique()} = {len(frame) - frame.index.nunique()} duplicate timestamps", component="EPISODES")
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
                    values, counts = np.unique(slice_states, return_counts=True)
                    majority_state = str(values[np.argmax(counts)])
                else:
                    majority_state = "unknown"
                regime_states.append(majority_state)
            episodes["regime"] = regime_vals
            episodes["regime_state"] = regime_states
        else:
            episodes["regime_state"] = "unknown"

        severity_map = {"critical": "critical", "suspect": "warning", "warning": "warning"}
        severity_override = episodes["regime_state"].map(lambda s: severity_map.get(str(s)))
        episodes["severity"] = severity_override.fillna(episodes["severity"])

        # Ensure both timestamps are parsed before subtraction
        start_ts = pd.to_datetime(episodes["start_ts"], errors="coerce")
        end_ts = pd.to_datetime(episodes["end_ts"], errors="coerce")
        episodes["duration_s"] = (end_ts - start_ts).dt.total_seconds()
        # Convenience: duration in hours for operator tables
        try:
            episodes["duration_hours"] = episodes["duration_s"].astype(float) / 3600.0
        except Exception:
            episodes["duration_hours"] = np.where(pd.notna(episodes.get("duration_s")), episodes.get("duration_s").astype(float) / 3600.0, 0.0)
        episodes = episodes.sort_values(["start_ts", "end_ts", "episode_id"]).reset_index(drop=True)
        episodes["regime"] = episodes["regime"].astype(str)'''

# New code with helper function call
new_code = '''        # --- Normalize episodes schema for report/export ------------
        episode_norm = _normalize_episodes(episodes=episodes, frame=frame, equip=equip)
        episodes = episode_norm.episodes
        frame = episode_norm.frame'''

if old_code in content:
    content = content.replace(old_code, new_code)
    
    with open("core/acm_main.py", "w", encoding="utf-8") as f:
        f.write(content)
    
    old_lines = len(old_code.split('\n'))
    new_lines = len(new_code.split('\n'))
    print(f"SUCCESS: Replaced episode normalization section ({old_lines} lines -> {new_lines} lines)")
    print(f"         Removed {old_lines - new_lines} lines from main()")
else:
    print("ERROR: Could not find old_code block to replace")
    # Debug: check if parts exist
    if "# --- Normalize episodes schema for report/export" in content:
        print("Found section header, trying to find why match failed...")
        # Check for encoding/whitespace issues
        idx = content.find("# --- Normalize episodes schema for report/export")
        print(f"Section found at char {idx}")
    else:
        print("Section header not found")
