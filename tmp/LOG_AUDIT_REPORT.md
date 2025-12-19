# ACM Log Message Audit Report

**Generated**: Automated analysis of Console logging calls
**Scope**: Core modules and key scripts

## Executive Summary

- **Total log calls analyzed**: 941
- **Modules scanned**: 30
- **Average quality score**: 75.0/100
- **Calls with component tag**: 798/941
- **Calls with context data**: 86/941

## Log Level Distribution

- **info**: 426 calls (45.3%)
- **warn**: 390 calls (41.4%)
- **error**: 81 calls (8.6%)
- **debug**: 13 calls (1.4%)
- **status**: 12 calls (1.3%)
- **ok**: 8 calls (0.9%)
- **header**: 5 calls (0.5%)
- **warning**: 3 calls (0.3%)
- **section**: 3 calls (0.3%)

## Module-by-Module Analysis

### acm_main.py

- **Call count**: 302
- **Quality score**: 77.7/100
- **With component**: 286/302
- **With context**: 3/302

**Issues found**:
  - No context data for error/warning: 132 occurrences
  - Message too vague/short: 16 occurrences
  - Missing component tag: 15 occurrences

**Low-quality examples**:
  - Line 666: `???` (score: 30)
    Issues: Message too vague/short, Missing component tag
  - Line 709: `???` (score: 30)
    Issues: Message too vague/short, Missing component tag
  - Line 1221: `???` (score: 30)
    Issues: Message too vague/short, Missing component tag

### output_manager.py

- **Call count**: 115
- **Quality score**: 74.4/100
- **With component**: 108/115
- **With context**: 1/115

**Issues found**:
  - No context data for error/warning: 77 occurrences
  - Message too vague/short: 21 occurrences
  - Missing component tag: 4 occurrences

**Low-quality examples**:
  - Line 963: `???` (score: 30)
    Issues: Message too vague/short, Missing component tag, No context data for error/warning

### sql_batch_runner.py

- **Call count**: 105
- **Quality score**: 78.9/100
- **With component**: 2/105
- **With context**: 79/105

**Issues found**:
  - Missing component tag: 52 occurrences
  - Message too vague/short: 21 occurrences
  - No context data for error/warning: 4 occurrences

**Low-quality examples**:
  - Line 99: `???` (score: 30)
    Issues: Message too vague/short, Missing component tag, No context data for error/warning
  - Line 103: `???` (score: 30)
    Issues: Message too vague/short, Missing component tag
  - Line 109: `???` (score: 30)
    Issues: Message too vague/short, Missing component tag

### model_persistence.py

- **Call count**: 62
- **Quality score**: 82.3/100
- **With component**: 62/62
- **With context**: 0/62

**Issues found**:
  - No context data for error/warning: 38 occurrences
  - Message too vague/short: 1 occurrences

### regimes.py

- **Call count**: 52
- **Quality score**: 67.7/100
- **With component**: 39/52
- **With context**: 1/52

**Issues found**:
  - No context data for error/warning: 32 occurrences
  - Message too vague/short: 13 occurrences
  - Missing component tag: 13 occurrences

**Low-quality examples**:
  - Line 447: `???` (score: 30)
    Issues: Message too vague/short, Missing component tag, No context data for error/warning
  - Line 514: `???` (score: 30)
    Issues: Message too vague/short, Missing component tag, No context data for error/warning
  - Line 563: `???` (score: 30)
    Issues: Message too vague/short, Missing component tag, No context data for error/warning

### forecast_engine.py

- **Call count**: 34
- **Quality score**: 65.4/100
- **With component**: 0/34
- **With context**: 0/34

**Issues found**:
  - No context data for error/warning: 22 occurrences
  - Message too vague/short: 9 occurrences
  - Missing component tag: 9 occurrences

**Low-quality examples**:
  - Line 356: `???` (score: 30)
    Issues: Message too vague/short, Missing component tag, No context data for error/warning
  - Line 361: `???` (score: 30)
    Issues: Message too vague/short, Missing component tag
  - Line 387: `???` (score: 30)
    Issues: Message too vague/short, Missing component tag

### smart_coldstart.py

- **Call count**: 29
- **Quality score**: 81.0/100
- **With component**: 29/29
- **With context**: 0/29

**Issues found**:
  - No context data for error/warning: 14 occurrences

### fuse.py

- **Call count**: 25
- **Quality score**: 76.6/100
- **With component**: 17/25
- **With context**: 0/25

**Issues found**:
  - No context data for error/warning: 14 occurrences
  - Missing component tag: 3 occurrences
  - Message too vague/short: 1 occurrences

**Low-quality examples**:
  - Line 410: `???` (score: 30)
    Issues: Message too vague/short, Missing component tag

### correlation.py

- **Call count**: 22
- **Quality score**: 72.7/100
- **With component**: 20/22
- **With context**: 0/22

**Issues found**:
  - No context data for error/warning: 3 occurrences
  - Message too vague/short: 2 occurrences
  - Missing component tag: 2 occurrences

**Low-quality examples**:
  - Line 174: `???` (score: 30)
    Issues: Message too vague/short, Missing component tag
  - Line 276: `???` (score: 30)
    Issues: Message too vague/short, Missing component tag, No context data for error/warning

### observability.py

- **Call count**: 19
- **Quality score**: 59.7/100
- **With component**: 4/19
- **With context**: 1/19

**Issues found**:
  - Missing component tag: 9 occurrences
  - Message too vague/short: 7 occurrences
  - No context data for error/warning: 6 occurrences

**Low-quality examples**:
  - Line 591: `=` (score: 35)
    Issues: Message too vague/short, Missing component tag
  - Line 609: `???` (score: 30)
    Issues: Message too vague/short, Missing component tag
  - Line 610: `???` (score: 30)
    Issues: Message too vague/short, Missing component tag

### run_metadata_writer.py

- **Call count**: 16
- **Quality score**: 71.2/100
- **With component**: 13/16
- **With context**: 0/16

**Issues found**:
  - No context data for error/warning: 10 occurrences
  - Message too vague/short: 3 occurrences
  - Missing component tag: 3 occurrences

**Low-quality examples**:
  - Line 284: `???` (score: 30)
    Issues: Message too vague/short, Missing component tag, No context data for error/warning
  - Line 299: `???` (score: 30)
    Issues: Message too vague/short, Missing component tag
  - Line 306: `???` (score: 30)
    Issues: Message too vague/short, Missing component tag, No context data for error/warning

### episode_culprits_writer.py

- **Call count**: 14
- **Quality score**: 77.9/100
- **With component**: 14/14
- **With context**: 0/14

**Issues found**:
  - No context data for error/warning: 8 occurrences

### config_history_writer.py

- **Call count**: 12
- **Quality score**: 82.5/100
- **With component**: 12/12
- **With context**: 0/12

**Issues found**:
  - No context data for error/warning: 7 occurrences

### multivariate_forecast.py

- **Call count**: 12
- **Quality score**: 79.6/100
- **With component**: 0/12
- **With context**: 0/12

**Issues found**:
  - No context data for error/warning: 8 occurrences

### sql_performance.py

- **Call count**: 12
- **Quality score**: 73.3/100
- **With component**: 12/12
- **With context**: 0/12

**Issues found**:
  - No context data for error/warning: 3 occurrences
  - Missing units for measurement: 1 occurrences

### state_manager.py

- **Call count**: 12
- **Quality score**: 69.6/100
- **With component**: 0/12
- **With context**: 0/12

**Issues found**:
  - No context data for error/warning: 8 occurrences
  - Message too vague/short: 3 occurrences
  - Missing component tag: 3 occurrences

**Low-quality examples**:
  - Line 139: `???` (score: 30)
    Issues: Message too vague/short, Missing component tag
  - Line 212: `???` (score: 30)
    Issues: Message too vague/short, Missing component tag, No context data for error/warning
  - Line 227: `???` (score: 30)
    Issues: Message too vague/short, Missing component tag, No context data for error/warning

### export_comprehensive_schema.py

- **Call count**: 12
- **Quality score**: 58.3/100
- **With component**: 0/12
- **With context**: 0/12

**Issues found**:
  - Missing component tag: 11 occurrences
  - No context data for error/warning: 3 occurrences
  - Message too vague/short: 1 occurrences

**Low-quality examples**:
  - Line 420: `???` (score: 30)
    Issues: Message too vague/short, Missing component tag, No context data for error/warning

### health_tracker.py

- **Call count**: 11
- **Quality score**: 50.0/100
- **With component**: 0/11
- **With context**: 0/11

**Issues found**:
  - Missing component tag: 7 occurrences
  - No context data for error/warning: 7 occurrences
  - Message too vague/short: 6 occurrences

**Low-quality examples**:
  - Line 214: `???` (score: 30)
    Issues: Message too vague/short, Missing component tag, No context data for error/warning
  - Line 222: `???` (score: 30)
    Issues: Message too vague/short, Missing component tag
  - Line 241: `???` (score: 30)
    Issues: Message too vague/short, Missing component tag, No context data for error/warning

### simplified_output_manager.py

- **Call count**: 11
- **Quality score**: 77.3/100
- **With component**: 9/11
- **With context**: 0/11

**Issues found**:
  - No context data for error/warning: 6 occurrences

### omr.py

- **Call count**: 10
- **Quality score**: 69.0/100
- **With component**: 8/10
- **With context**: 0/10

**Issues found**:
  - No context data for error/warning: 5 occurrences
  - Message too vague/short: 2 occurrences
  - Missing component tag: 2 occurrences

**Low-quality examples**:
  - Line 406: `???` (score: 30)
    Issues: Message too vague/short, Missing component tag
  - Line 415: `???` (score: 30)
    Issues: Message too vague/short, Missing component tag, No context data for error/warning

## Common Issues Across Codebase

- **No context data for error/warning**: 432 occurrences (45.9% of calls)
- **Missing component tag**: 143 occurrences (15.2% of calls)
- **Message too vague/short**: 117 occurrences (12.4% of calls)
- **Missing units for measurement**: 1 occurrences (0.1% of calls)

## Best Practices Analysis

### Component Tagging

- **Explicit component=** parameter: 663 (70.5%)
- **Inline [TAG] format**: 135 (14.3%)
- **No tagging**: 143 (15.2%)

### Context Data Usage

- **Calls with context data**: 86/941 (9.1%)
- **Errors/warnings with context**: 39/474 (8.2% if error_warnings else 0)

## Recommendations

### High Priority

1. **Add context to errors**: 435 error/warning calls lack contextual data. Include relevant variables (table names, IDs, counts) as kwargs.

2. **Improve message clarity**: 117 calls have very short/vague messages. Add more descriptive text explaining what happened.

### Medium Priority

1. **Standardize component naming**: Use consistent component names (DATA, MODEL, SQL, FUSE, etc.)
2. **Add units to measurements**: Include units (s, ms, rows, MB) in messages about durations/sizes
3. **Use Console.status() for decorative output**: Move separator lines and banners to Console.status() to avoid Loki pollution

## Compliance with Documentation

### LOGGING_GUIDE.md Alignment

- ✓ Using Console.info/warn/error/ok methods correctly
- ✓ Using component parameter for filtering (where present)
- ⚠ Context metadata usage (9% coverage)

### OBSERVABILITY.md Alignment

- ✓ Console methods route to Loki properly
- ✓ Component tags enable Loki label filtering
- ✓ Appropriate use of console-only methods (status/header/section)

---

**Total issues identified**: 693
**Overall assessment**: GOOD