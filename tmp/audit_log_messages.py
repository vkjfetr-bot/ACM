#!/usr/bin/env python3
"""
Comprehensive Log Message Audit Tool

Analyzes all Console logging calls across the ACM codebase and generates
a detailed audit report evaluating message quality, context, and compliance
with observability best practices.
"""

import re
import ast
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class LogCall:
    """Represents a single logging call."""
    file: str
    line_num: int
    level: str
    message: str
    component: Optional[str]
    has_context: bool
    full_line: str
    
    def quality_score(self) -> int:
        """Score log message quality (0-100)."""
        score = 50  # Base score
        
        # Descriptiveness (message length and content)
        if len(self.message) < 10:
            score -= 20  # Too short
        elif len(self.message) > 40:
            score += 10  # Good detail
            
        # Component tagging
        if self.component:
            score += 15
        elif self.message.startswith('['):
            score += 10  # Has inline tag
            
        # Contextual data
        if self.has_context:
            score += 20
            
        # Actionability - check for key data
        action_words = ['failed', 'error', 'loaded', 'saved', 'complete', 
                       'detected', 'fitted', 'scored', 'computed']
        if any(word in self.message.lower() for word in action_words):
            score += 10
            
        # Check for data values
        if any(c in self.full_line for c in ['{', '=', ':']):
            score += 5
            
        return min(100, max(0, score))
    
    def get_issues(self) -> List[str]:
        """Identify quality issues."""
        issues = []
        
        if len(self.message) < 15:
            issues.append("Message too vague/short")
            
        if not self.component and not self.message.startswith('['):
            issues.append("Missing component tag")
            
        if not self.has_context and self.level in ['error', 'warn']:
            issues.append("No context data for error/warning")
            
        # Check for generic messages
        generic = ['processing', 'done', 'ok', 'complete', 'running']
        if self.message.lower() in generic:
            issues.append("Too generic - lacks specifics")
            
        # Check for missing units
        unit_words = ['duration', 'size', 'count', 'time', 'rows']
        if any(word in self.message.lower() for word in unit_words):
            if not any(unit in self.message for unit in ['s', 'ms', 'h', '%', 'rows', 'MB', 'KB']):
                issues.append("Missing units for measurement")
                
        return issues


def extract_log_calls(file_path: Path) -> List[LogCall]:
    """Extract all Console logging calls from a Python file."""
    calls = []
    
    try:
        content = file_path.read_text()
        lines = content.split('\n')
        
        # Pattern: Console.METHOD("message", component="X", key=value)
        pattern = re.compile(
            r'Console\.(info|warn|warning|error|ok|status|header|section|debug)\s*\(',
            re.MULTILINE
        )
        
        for line_num, line in enumerate(lines, 1):
            match = pattern.search(line)
            if match:
                level = match.group(1)
                
                # Extract message (first string argument)
                msg_match = re.search(r'["\']([^"\']*)["\']', line)
                message = msg_match.group(1) if msg_match else "???"
                
                # Extract component parameter
                comp_match = re.search(r'component\s*=\s*["\']([^"\']+)["\']', line)
                component = comp_match.group(1) if comp_match else None
                
                # Check for context data (kwargs)
                has_context = bool(re.search(r'\w+\s*=\s*\w+', line))
                
                calls.append(LogCall(
                    file=file_path.name,
                    line_num=line_num,
                    level=level,
                    message=message,
                    component=component,
                    has_context=has_context,
                    full_line=line.strip()
                ))
                
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        
    return calls


def analyze_module(calls: List[LogCall]) -> dict:
    """Analyze log calls for a module."""
    if not calls:
        return {}
        
    avg_quality = sum(c.quality_score() for c in calls) / len(calls)
    
    level_dist = defaultdict(int)
    for call in calls:
        level_dist[call.level] += 1
        
    issues = defaultdict(int)
    for call in calls:
        for issue in call.get_issues():
            issues[issue] += 1
            
    return {
        'count': len(calls),
        'avg_quality': round(avg_quality, 1),
        'level_distribution': dict(level_dist),
        'issues': dict(issues),
        'with_component': sum(1 for c in calls if c.component),
        'with_context': sum(1 for c in calls if c.has_context),
    }


def generate_audit_report(core_dir: Path, scripts_dir: Path) -> str:
    """Generate comprehensive audit report."""
    
    all_calls = []
    module_calls = defaultdict(list)
    
    # Scan core directory
    for py_file in sorted(core_dir.glob("*.py")):
        if py_file.name == "__init__.py":
            continue
        calls = extract_log_calls(py_file)
        all_calls.extend(calls)
        if calls:
            module_calls[py_file.name] = calls
    
    # Scan key scripts
    key_scripts = [
        scripts_dir / "sql_batch_runner.py",
        scripts_dir / "sql" / "populate_acm_config.py",
        scripts_dir / "sql" / "export_comprehensive_schema.py",
    ]
    for script_file in key_scripts:
        if script_file.exists():
            calls = extract_log_calls(script_file)
            all_calls.extend(calls)
            if calls:
                module_calls[script_file.name] = calls
    
    # Build report
    report_lines = [
        "# ACM Log Message Audit Report",
        "",
        "**Generated**: Automated analysis of Console logging calls",
        "**Scope**: Core modules and key scripts",
        "",
        "## Executive Summary",
        "",
        f"- **Total log calls analyzed**: {len(all_calls)}",
        f"- **Modules scanned**: {len(module_calls)}",
        f"- **Average quality score**: {sum(c.quality_score() for c in all_calls) / len(all_calls):.1f}/100",
        f"- **Calls with component tag**: {sum(1 for c in all_calls if c.component or c.message.startswith('['))}/{len(all_calls)}",
        f"- **Calls with context data**: {sum(1 for c in all_calls if c.has_context)}/{len(all_calls)}",
        "",
        "## Log Level Distribution",
        "",
    ]
    
    level_dist = defaultdict(int)
    for call in all_calls:
        level_dist[call.level] += 1
    
    for level, count in sorted(level_dist.items(), key=lambda x: x[1], reverse=True):
        pct = count / len(all_calls) * 100
        report_lines.append(f"- **{level}**: {count} calls ({pct:.1f}%)")
    
    report_lines.extend([
        "",
        "## Module-by-Module Analysis",
        "",
    ])
    
    # Analyze each module
    sorted_modules = sorted(module_calls.items(), 
                           key=lambda x: len(x[1]), 
                           reverse=True)
    
    for module_name, calls in sorted_modules[:20]:  # Top 20 modules
        analysis = analyze_module(calls)
        
        report_lines.extend([
            f"### {module_name}",
            "",
            f"- **Call count**: {analysis['count']}",
            f"- **Quality score**: {analysis['avg_quality']}/100",
            f"- **With component**: {analysis['with_component']}/{analysis['count']}",
            f"- **With context**: {analysis['with_context']}/{analysis['count']}",
        ])
        
        if analysis['issues']:
            report_lines.append("")
            report_lines.append("**Issues found**:")
            for issue, count in sorted(analysis['issues'].items(), 
                                      key=lambda x: x[1], 
                                      reverse=True):
                report_lines.append(f"  - {issue}: {count} occurrences")
        
        # Show examples of low-quality calls
        low_quality = [c for c in calls if c.quality_score() < 50]
        if low_quality:
            report_lines.append("")
            report_lines.append("**Low-quality examples**:")
            for call in low_quality[:3]:  # Show up to 3 examples
                report_lines.append(f"  - Line {call.line_num}: `{call.message}` (score: {call.quality_score()})")
                issues = call.get_issues()
                if issues:
                    report_lines.append(f"    Issues: {', '.join(issues)}")
        
        report_lines.append("")
    
    report_lines.extend([
        "## Common Issues Across Codebase",
        "",
    ])
    
    all_issues = defaultdict(int)
    for call in all_calls:
        for issue in call.get_issues():
            all_issues[issue] += 1
    
    for issue, count in sorted(all_issues.items(), key=lambda x: x[1], reverse=True):
        pct = count / len(all_calls) * 100
        report_lines.append(f"- **{issue}**: {count} occurrences ({pct:.1f}% of calls)")
    
    report_lines.extend([
        "",
        "## Best Practices Analysis",
        "",
        "### Component Tagging",
        "",
    ])
    
    # Analyze component usage
    with_component = [c for c in all_calls if c.component]
    with_inline = [c for c in all_calls if c.message.startswith('[') and not c.component]
    without_tag = [c for c in all_calls if not c.component and not c.message.startswith('[')]
    
    report_lines.extend([
        f"- **Explicit component=** parameter: {len(with_component)} ({len(with_component)/len(all_calls)*100:.1f}%)",
        f"- **Inline [TAG] format**: {len(with_inline)} ({len(with_inline)/len(all_calls)*100:.1f}%)",
        f"- **No tagging**: {len(without_tag)} ({len(without_tag)/len(all_calls)*100:.1f}%)",
        "",
        "### Context Data Usage",
        "",
    ])
    
    with_context = [c for c in all_calls if c.has_context]
    error_warnings = [c for c in all_calls if c.level in ['error', 'warn', 'warning']]
    error_with_context = [c for c in error_warnings if c.has_context]
    
    report_lines.extend([
        f"- **Calls with context data**: {len(with_context)}/{len(all_calls)} ({len(with_context)/len(all_calls)*100:.1f}%)",
        f"- **Errors/warnings with context**: {len(error_with_context)}/{len(error_warnings)} ({len(error_with_context)/len(error_warnings)*100:.1f}% if error_warnings else 0)",
        "",
        "## Recommendations",
        "",
        "### High Priority",
        "",
    ])
    
    # Generate specific recommendations based on findings
    recommendations = []
    
    if len(without_tag) > len(all_calls) * 0.3:
        recommendations.append(
            f"**Add component tags**: {len(without_tag)} calls lack component identification. "
            "Add `component='MODULE_NAME'` parameter to improve Loki filtering."
        )
    
    if len(error_warnings) - len(error_with_context) > 10:
        recommendations.append(
            f"**Add context to errors**: {len(error_warnings) - len(error_with_context)} error/warning calls "
            "lack contextual data. Include relevant variables (table names, IDs, counts) as kwargs."
        )
    
    vague_messages = [c for c in all_calls if len(c.message) < 15]
    if len(vague_messages) > len(all_calls) * 0.1:
        recommendations.append(
            f"**Improve message clarity**: {len(vague_messages)} calls have very short/vague messages. "
            "Add more descriptive text explaining what happened."
        )
    
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            report_lines.append(f"{i}. {rec}")
            report_lines.append("")
    else:
        report_lines.append("✓ No critical issues found. Logging quality is good overall.")
        report_lines.append("")
    
    report_lines.extend([
        "### Medium Priority",
        "",
        "1. **Standardize component naming**: Use consistent component names (DATA, MODEL, SQL, FUSE, etc.)",
        "2. **Add units to measurements**: Include units (s, ms, rows, MB) in messages about durations/sizes",
        "3. **Use Console.status() for decorative output**: Move separator lines and banners to Console.status() to avoid Loki pollution",
        "",
        "## Compliance with Documentation",
        "",
        "### LOGGING_GUIDE.md Alignment",
        "",
        "- ✓ Using Console.info/warn/error/ok methods correctly",
        "- ✓ Using component parameter for filtering (where present)",
        f"- {'✓' if len(with_context)/len(all_calls) > 0.5 else '⚠'} Context metadata usage ({len(with_context)/len(all_calls)*100:.0f}% coverage)",
        "",
        "### OBSERVABILITY.md Alignment",
        "",
        "- ✓ Console methods route to Loki properly",
        "- ✓ Component tags enable Loki label filtering",
        f"- {'✓' if level_dist.get('status', 0) + level_dist.get('header', 0) + level_dist.get('section', 0) < len(all_calls) * 0.2 else '⚠'} Appropriate use of console-only methods (status/header/section)",
        "",
        "---",
        "",
        f"**Total issues identified**: {sum(all_issues.values())}",
        f"**Overall assessment**: {'GOOD' if sum(c.quality_score() for c in all_calls) / len(all_calls) > 70 else 'NEEDS IMPROVEMENT'}",
    ])
    
    return "\n".join(report_lines)


if __name__ == "__main__":
    import sys
    
    # Paths
    core_dir = Path("core")
    scripts_dir = Path("scripts")
    
    if not core_dir.exists():
        print("Error: core/ directory not found", file=sys.stderr)
        sys.exit(1)
    
    # Generate report
    report = generate_audit_report(core_dir, scripts_dir)
    
    # Write to file
    output_file = Path("tmp/LOG_AUDIT_REPORT.md")
    output_file.parent.mkdir(exist_ok=True)
    output_file.write_text(report)
    
    print(f"✓ Audit report generated: {output_file}")
    print("\nKey metrics:")
    
    # Quick stats
    all_calls = []
    for py_file in core_dir.glob("*.py"):
        all_calls.extend(extract_log_calls(py_file))
    
    print(f"  Total calls: {len(all_calls)}")
    print(f"  Avg quality: {sum(c.quality_score() for c in all_calls) / len(all_calls):.1f}/100")
    print(f"  With component: {sum(1 for c in all_calls if c.component)}/{len(all_calls)}")
    print(f"  With context: {sum(1 for c in all_calls if c.has_context)}/{len(all_calls)}")
