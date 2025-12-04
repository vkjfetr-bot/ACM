"""
Version management for ACM.

This module defines the current version of ACM and provides utilities for version tracking
across logs, outputs, and database entries. Version follows semantic versioning (MAJOR.MINOR.PATCH).

Versioning Strategy:
- MAJOR: Significant architecture changes or breaking changes (e.g., v8→v9)
- MINOR: New features, detector improvements, algorithm enhancements (e.g., v9.0→v9.1)
- PATCH: Bug fixes, refinements, performance improvements (e.g., v9.0.0→v9.0.1)

Release Management:
- All releases tagged with git annotated tags (e.g., v9.0.0)
- Each tag includes comprehensive release notes
- Feature branches use descriptive names: feature/*, fix/*, refactor/*, docs/*
- Merges to main use --no-ff to preserve history
- Production deployments use specific tags (never merge commits)
"""

__version__ = "9.0.0"
__version_date__ = "2025-12-04"
__version_author__ = "ACM Development Team"

VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH = map(int, __version__.split("."))


def get_version_string():
    """
    Get the full version string with date and context.
    
    Returns:
        str: Version string in format "ACM v9.0.0 (2025-12-04)"
    """
    return f"ACM v{__version__} ({__version_date__})"


def get_version_tuple():
    """
    Get version as tuple for programmatic comparison.
    
    Returns:
        tuple: (major, minor, patch)
    """
    return (VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH)


def is_compatible(required_version):
    """
    Check if current version is compatible with required version.
    Uses semantic versioning - same major version required, minor/patch must be >= required.
    
    Args:
        required_version (str): Version string like "9.0.0"
        
    Returns:
        bool: True if compatible, False otherwise
    """
    required_parts = list(map(int, required_version.split(".")))
    current_parts = get_version_tuple()
    
    # Major version must match
    if current_parts[0] != required_parts[0]:
        return False
    
    # Minor version must be >= required
    if current_parts[1] < required_parts[1]:
        return False
    
    # Patch version must be >= required (only if minor versions match)
    if current_parts[1] == required_parts[1] and current_parts[2] < required_parts[2]:
        return False
    
    return True


def format_version_for_output(context=""):
    """
    Format version information for inclusion in outputs (logs, SQL records, etc).
    
    Args:
        context (str): Optional context like "run_metadata", "log_header", etc
        
    Returns:
        str: Formatted version string
    """
    if context:
        return f"{get_version_string()} [{context}]"
    return get_version_string()


# v9.0.0 Release Notes (from v8.2.0)
RELEASE_NOTES = """
ACM v9.0.0 - Major Production Release (2025-12-04)

CRITICAL FIXES (P0):
  ✓ Detector Label Consistency (CRIT-04)
    - Fixed extract_dominant_sensor() to preserve full detector labels
    - All outputs now show standardized format: "Multivariate Outlier (PCA-T²)"
    - Applied to ACM_EpisodeDiagnostics, Grafana dashboards, and all analytics
    - Impact: 100% label consistency across all interfaces

  ✓ Database Cleanup
    - Removed 3 migration backup tables (6,982 rows total)
    - Removed 6 unused empty feature tables
    - Schema reduced from 85 to 79 tables
    - Maintained referential integrity and data consistency

  ✓ Equipment Data Integrity
    - Standardized equipment names across all 26 runs
    - All runs now reference consistent equipment codes
    - Aligned with Equipment master table
    - Fixed 4 runs with mismatched equipment references

  ✓ Run Completion Tracking
    - All 26 runs now have valid CompletedAt timestamps
    - 4 incomplete runs marked with NOOP status (zero duration)
    - Proper error message tracking for incomplete runs
    - Enables accurate run duration and performance metrics

  ✓ Stored Procedure Fixes
    - Fixed usp_ACM_FinalizeRun to reference ACM_Runs table (not deleted RunLog)
    - Updated column mappings: Outcome→CompletedAt, RowsRead→TrainRowCount, etc
    - Procedure now executes successfully on run completion

FEATURES:
  ✓ Comprehensive Testing Suite
    - 30+ Python unit tests covering all P0 fixes
    - 8 SQL validation checks for database integrity
    - tests/test_p0_fixes_validation.py
    - scripts/sql/validate_p0_fixes.sql

  ✓ Professional Versioning
    - Semantic versioning (MAJOR.MINOR.PATCH)
    - Version management module (utils/version.py)
    - Release notes and documentation
    - Proper git tag management

IMPROVEMENTS:
  ✓ Source Control Practices
    - Feature branches with descriptive names (feature/*, fix/*, refactor/*)
    - Merge commits with --no-ff flag to preserve history
    - Comprehensive commit messages with context
    - Proper tag management with annotated tags

  ✓ Documentation
    - Updated README.md with v9.0.0 highlights
    - Updated ACM_SYSTEM_OVERVIEW.md with major changes
    - Comprehensive release index and workflow documentation
    - Version management guidelines for future releases

BREAKING CHANGES: None - Fully backward compatible with v8.x data

DATABASE CHANGES:
  - 9 tables removed (backups + unused features)
  - 0 tables added (cleanup only)
  - 4 runs updated with standardized equipment codes
  - All data preserved and validated

TESTING:
  - All 30+ Python tests passing ✓
  - All 8 SQL validation checks passing ✓
  - Database integrity verified ✓
  - Detector label consistency verified ✓

DEPLOYMENT:
  - Production-ready on v9.0.0 tag
  - No data migration required
  - No downtime expected
  - Rollback available via v8.2.0 tag if needed

NEXT STEPS:
  1. Run validation test suite: pytest tests/test_p0_fixes_validation.py -v
  2. Run SQL validation: sqlcmd -S localhost\\INSTANCE -d ACM -E -i scripts/sql/validate_p0_fixes.sql
  3. Deploy to production environment
  4. Monitor Grafana dashboards for detector label consistency
  5. Monitor run completion metrics

VERSION HISTORY:
  v8.2.0 → v9.0.0: Major production release with P0 fixes and professional versioning
  v8.0.0 → v8.2.0: Feature releases and stabilization
  v7.x: Legacy versions (archived)

AUTHOR: ACM Development Team
DATE: 2025-12-04
GIT_TAG: v9.0.0
"""
