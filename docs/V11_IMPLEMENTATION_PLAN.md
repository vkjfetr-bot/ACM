# V11 PROPER IMPLEMENTATION PLAN

**Created**: 2025-12-29
**Branch**: `feature/v11-refactor`
**Status**: IN PROGRESS

---

## What Went Wrong

The "V11 refactor" to date:
- Created 23 modules that were never called from acm_main.py
- Deleted those 23 modules as "dead code"
- Added DataContract validation (fail-fast only)
- Added observability stack
- Cleaned up dead code
- Optimized calibration

What V11 was supposed to do (but did not):
- ONLINE/OFFLINE pipeline separation
- UNKNOWN regime handling
- Confidence model for all outputs
- RUL reliability gate
- Maturity state lifecycle
- Regime stability (no rediscovery every batch)

The tracker claimed 60/60 complete by counting module creation as "done" even though nothing was wired up.

---

## V10 Flaws That V11 Must Fix

| V10 Flaw | V11 Solution |
|----------|--------------|
| Regime discovery runs in online mode - regimes created/modified during every batch | Strict ONLINE/OFFLINE split - online never creates regimes |
| Forced regime assignment - every point forced to nearest regime | UNKNOWN/EMERGING regimes - RegimeLabel=-1 allowed |
| Circular dependency - PCA/detector outputs fed into regime clustering | Clean regime inputs - only operational sensor data |
| No train/score separation - batches could influence their own scores | Strict fit_baseline()/score() API - baseline frozen |
| Point anomalies trigger alerts - noisy, operator fatigue | Episodes as only alerting primitive |
| RUL always returns numbers - even with 5 data points | RUL reliability gate - explicit NOT_RELIABLE |
| Confidence not tracked - outputs presented as certain | Unified confidence model - every output has confidence |
| Cold start hidden - system pretends it knows things | Cold start as first-class state - explicit NULL version |
| Drift is dashboard-only - detected but not actionable | Drift as control signal - triggers offline replay |
| Health is instantaneous score - can jump abruptly | Health as time-evolving state with hysteresis |

---

## The 20 Non-Negotiable Rules

From Major Refactor Plan - these MUST be enforced:

1. Online code must never create/modify regimes
2. Offline code must never overwrite historical truth
3. No forced assignment when confidence is low
4. Cold start must be explicit and visible
5. Regime features must be operational only
6. Detector outputs must never define regimes
7. Point anomalies must never trigger alerts
8. Episodes are the only alerting primitive
9. Health must be time-based and confidence-aware
10. RUL must be gated or suppressed
11. Drift must influence control flow
12. All model changes must be versioned
13. Promotion requires objective evidence
14. UNKNOWN is a valid system output
15. SQL is the single source of truth
16. Replays must be reproducible
17. Confidence must always be exposed
18. Dashboards must reflect uncertainty
19. No module may silently reinterpret historical data
20. If unsure, the system must say "not reliable"

---

## Architecture

```
                           SINGLE ENTRY POINT
  
   python -m core.acm --equip FD_FAN
          |
          v
   +------------------+
   |  ACMController   | <-- New orchestrator class
   |  (core/acm.py)   |
   +--------+---------+
            |
            +---------------+------------------+
            |               |                  |
            v               v                  v
   +----------------+ +----------------+ +----------------+
   | OFFLINE Thread | | ONLINE Thread  | | Observability  |
   | (every 24h)    | | (every 30min)  | | (continuous)   |
   +----------------+ +----------------+ +----------------+
```

### ONLINE vs OFFLINE

| Aspect | ONLINE Pipeline | OFFLINE Pipeline |
|--------|----------------|------------------|
| Purpose | Current operational belief | Is belief still best explanation? |
| Frequency | Every 5-30 minutes | Every 1-7 days |
| Data Window | Last N minutes (small) | Historical span (large) |
| Regime Discovery | FORBIDDEN - uses existing | ALLOWED - creates new versions |
| Model Fitting | FORBIDDEN - loads cache | ALLOWED - refits all |
| Threshold Recalc | Uses active version | May create new version |
| Coldstart | Requires existing models | Handles coldstart |
| Latency Target | <30 seconds | <10 minutes |
| Failure Mode | Skip if no model, trigger OFFLINE | Create model |

---

## Implementation Phases

### Phase 0: Foundation

| Task | Description | Files |
|------|-------------|-------|
| 0.1 | Create core/acm.py - single entry point | NEW |
| 0.2 | Create core/pipeline_context.py - extract dataclasses | NEW |
| 0.3 | Add --mode online/offline/auto CLI arg | acm_main.py |
| 0.4 | Gate regime discovery with if mode == OFFLINE | acm_main.py |
| 0.5 | Gate model fitting with if mode == OFFLINE | acm_main.py |

Exit Criteria: --mode online with no model fails fast. --mode offline trains. --mode online then works.

---

### Phase 1: OFFLINE Pipeline

| Task | Description | Files |
|------|-------------|-------|
| 1.1 | Create core/offline_pipeline.py - extract main() phases | NEW |
| 1.2 | Create core/detector_manager.py - detector fit/score | NEW |
| 1.3 | Create core/data_pipeline.py - data quality/features | NEW |
| 1.4 | Implement ACM_ActiveModels state machine | NEW |
| 1.5 | Implement auto-promotion when quality passes | MODIFY |
| 1.6 | Wire OfflinePipeline into acm.py | MODIFY |

Exit Criteria: python -m core.acm --equip FD_FAN --mode offline runs discovery, creates model, promotes.

---

### Phase 2: ONLINE Pipeline

| Task | Description | Files |
|------|-------------|-------|
| 2.1 | Create core/online_pipeline.py - scoring-only | NEW |
| 2.2 | Load frozen model from ModelRegistry | MODIFY |
| 2.3 | Regime assignment (predict only, no fit) | MODIFY |
| 2.4 | Add UNKNOWN regime (label=-1) when confidence < threshold | MODIFY |
| 2.5 | Wire OnlinePipeline into acm.py | MODIFY |
| 2.6 | ONLINE fails triggers OFFLINE automatically | MODIFY |

Exit Criteria: python -m core.acm --equip FD_FAN --mode online uses existing model, scores.

---

### Phase 3: Confidence and Reliability

| Task | Description | Files |
|------|-------------|-------|
| 3.1 | Create core/confidence.py - unified confidence model | NEW |
| 3.2 | Add confidence column to ACM_HealthTimeline | MODIFY |
| 3.3 | Add confidence column to ACM_RUL | MODIFY |
| 3.4 | Add confidence column to ACM_Episodes | MODIFY |
| 3.5 | Implement RUL reliability gate (CONVERGED + min data) | MODIFY |
| 3.6 | Output RUL_Status=NOT_RELIABLE when gate fails | MODIFY |

Exit Criteria: All outputs have confidence 0-1. RUL shows NOT_RELIABLE when prerequisites fail.

---

### Phase 4: Regime Stability

| Task | Description | Files |
|------|-------------|-------|
| 4.1 | Create core/regime_lifecycle.py - version management | NEW |
| 4.2 | OFFLINE creates new RegimeVersion, never overwrites | MODIFY |
| 4.3 | Add AssignmentConfidence to ACM_RegimeTimeline | MODIFY |
| 4.4 | ONLINE uses frozen regime model only | MODIFY |
| 4.5 | Add promotion criteria (silhouette>0.15, stability>0.8, 7+ days) | MODIFY |

Exit Criteria: Regimes versioned, ONLINE never modifies, promotions logged.

Regime Lifecycle:
```
OFFLINE run 1  --> Creates RegimeVersion=1 (state=LEARNING)
OFFLINE run 2  --> Evaluates V1, still learning
OFFLINE run 7  --> V1 passes criteria --> Promote to CONVERGED
ONLINE runs    --> Use V1 for assignment, never modify
OFFLINE run 30 --> Creates V2 candidate (if drift detected)
OFFLINE run 35 --> V2 passes criteria --> Promote V2, deprecate V1
```

---

### Phase 5: Single Entry Point

| Task | Description | Files |
|------|-------------|-------|
| 5.1 | Implement ACMController.start() in core/acm.py | MODIFY |
| 5.2 | Auto mode: check model exists, route to ONLINE or OFFLINE | MODIFY |
| 5.3 | Update sql_batch_runner.py to use new entry point | MODIFY |
| 5.4 | Add scheduling logic (ONLINE 30min, OFFLINE 24h) | MODIFY |
| 5.5 | Create scripts/acm_launcher.py for production | NEW |

Exit Criteria: Single command python -m core.acm --equip FD_FAN handles everything.

---

### Phase 6: Observability Dashboard

| Task | Description | Files |
|------|-------------|-------|
| 6.1 | Create acm_v11_operations.json dashboard | NEW |
| 6.2 | Panel: Pipeline mode (ONLINE/OFFLINE) per equipment | |
| 6.3 | Panel: Model maturity state per equipment | |
| 6.4 | Panel: Confidence distributions over time | |
| 6.5 | Panel: Phase timings | |
| 6.6 | Panel: ONLINE latency percentiles | |
| 6.7 | Add @Span.trace() decorators to all new modules | MODIFY |

Exit Criteria: Dashboard shows operations, confidence, maturity, latency.

---

### Phase 7: Cleanup

| Task | Description | Files |
|------|-------------|-------|
| 7.1 | Deprecate acm_main.py (thin wrapper) | MODIFY |
| 7.2 | Update copilot-instructions.md with V11 architecture | MODIFY |
| 7.3 | Update version to 11.1.0 | MODIFY |
| 7.4 | Tag release | GIT |
| 7.5 | Update tracker with actual status | MODIFY |

---

## File Structure After Refactor

```
core/
    acm.py                  # NEW - Single entry point (50 lines)
    pipeline_context.py     # NEW - Dataclasses (150 lines)
    offline_pipeline.py     # NEW - OFFLINE batch pipeline (800 lines)
    online_pipeline.py      # NEW - ONLINE scoring pipeline (300 lines)
    detector_manager.py     # NEW - Detector fit/score (400 lines)
    data_pipeline.py        # NEW - Data quality/features (500 lines)
    health_pipeline.py      # NEW - Episode/health builders (400 lines)
    regime_lifecycle.py     # NEW - Maturity state machine (300 lines)
    confidence.py           # NEW - Unified confidence (200 lines)
    acm_main.py             # DEPRECATED - Thin wrapper (100 lines)
    
    # Existing (unchanged)
    observability.py        # Traces/metrics/logs
    output_manager.py       # SQL/file writes
    fuse.py                 # Fusion
    regimes.py              # Regime clustering
    forecast_engine.py      # RUL/forecasting
```

---

## Commit Strategy

| Commit | Content |
|--------|---------|
| v11: Add --mode CLI argument and gate regime discovery | Phase 0 |
| v11: Extract offline pipeline from acm_main | Phase 1 |
| v11: Implement online pipeline with model loading | Phase 2 |
| v11: Add confidence model and RUL reliability gate | Phase 3 |
| v11: Implement regime versioning and maturity lifecycle | Phase 4 |
| v11: Create single entry point and launcher | Phase 5 |
| v11: Add operations dashboard | Phase 6 |
| v11.1.0: Complete V11 proper implementation | Phase 7 + Tag |

---

## Development Rules

- No emojis or unicode characters
- No lazy wrappers
- No over-engineering - POC first, edge cases as discovered
- Rewrite properly, no shortcuts
- Optimize memory and CPU
- Follow source control practices (commit, branch, tag)
- Maintain versioning
- Maintain this task list
- Delete single-use scripts when done
- No unnecessary documentation
- Follow V11 philosophy always

---

## Progress Tracking

### Phase 0: Foundation
- [ ] 0.1 Create core/acm.py
- [ ] 0.2 Create core/pipeline_context.py
- [ ] 0.3 Add --mode CLI arg
- [ ] 0.4 Gate regime discovery
- [ ] 0.5 Gate model fitting

### Phase 1: OFFLINE Pipeline
- [ ] 1.1 Create core/offline_pipeline.py
- [ ] 1.2 Create core/detector_manager.py
- [ ] 1.3 Create core/data_pipeline.py
- [ ] 1.4 Implement ACM_ActiveModels state machine
- [ ] 1.5 Implement auto-promotion
- [ ] 1.6 Wire OfflinePipeline

### Phase 2: ONLINE Pipeline
- [ ] 2.1 Create core/online_pipeline.py
- [ ] 2.2 Load frozen model
- [ ] 2.3 Regime assignment predict-only
- [ ] 2.4 Add UNKNOWN regime
- [ ] 2.5 Wire OnlinePipeline
- [ ] 2.6 ONLINE fail triggers OFFLINE

### Phase 3: Confidence and Reliability
- [ ] 3.1 Create core/confidence.py
- [ ] 3.2 Confidence in ACM_HealthTimeline
- [ ] 3.3 Confidence in ACM_RUL
- [ ] 3.4 Confidence in ACM_Episodes
- [ ] 3.5 RUL reliability gate
- [ ] 3.6 RUL_Status=NOT_RELIABLE

### Phase 4: Regime Stability
- [ ] 4.1 Create core/regime_lifecycle.py
- [ ] 4.2 OFFLINE creates new version
- [ ] 4.3 AssignmentConfidence in timeline
- [ ] 4.4 ONLINE uses frozen model
- [ ] 4.5 Promotion criteria

### Phase 5: Single Entry Point
- [ ] 5.1 ACMController.start()
- [ ] 5.2 Auto mode routing
- [ ] 5.3 Update sql_batch_runner.py
- [ ] 5.4 Scheduling logic
- [ ] 5.5 scripts/acm_launcher.py

### Phase 6: Observability Dashboard
- [ ] 6.1 Create dashboard JSON
- [ ] 6.2-6.6 Dashboard panels
- [ ] 6.7 Add @Span.trace() decorators

### Phase 7: Cleanup
- [ ] 7.1 Deprecate acm_main.py
- [ ] 7.2 Update copilot-instructions.md
- [ ] 7.3 Version 11.1.0
- [ ] 7.4 Tag release
- [ ] 7.5 Update tracker
