"""Python compatibility analyzer for detecting deprecated and removed APIs."""

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any
import subprocess
import importlib.util
import ssl
import unittest
import pandas as pd

# Version constants to avoid magic numbers
VERSION_MAJOR_PYTHON = 3  # Python major version for compatibility checks

logger = logging.getLogger(__name__)


@dataclass
class CompatibilityIssue:
    """Represents a Python compatibility issue found during analysis.
    
    Attributes:
        api_name: The deprecated or removed API name
        line_number: Line number where the issue was found
        severity: Issue severity level (HIGH, MEDIUM, LOW)
        replacement: Suggested replacement API
        python_version: Target Python version
    """
    api_name: str
    line_number: int
    severity: str
    replacement: str
    python_version: str


def async_example_function():
    """Example async function using modern async/await syntax.
    
    This function demonstrates the modern approach to defining asynchronous
    functions using native async def instead of the removed @asyncio.coroutine decorator.
    """
    async def modern_coroutine():
        """Modern coroutine using async def syntax."""
        return "Using native async/await"
    
    return modern_coroutine


def _check_ssl_protocol():
    """Check SSL protocol configuration using modern SSLContext API."""
    # Use specific TLS client context instead of deprecated PROTOCOL_TLS
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    return context


def _wrap_socket_example(sock, **kwargs):
    """Wrap a socket with SSL using SSLContext instead of deprecated module-level function."""
    # Use SSLContext.wrap_socket() instead of removed ssl.wrap_socket()
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    return context.wrap_socket(sock, **kwargs)


def _execute_command(cmd: str) -> str:
    """Execute a shell command using subprocess instead of deprecated os.popen.
    
    Args:
        cmd: Shell command to execute
        
    Returns:
        Command output as string
    """
    # Use subprocess.run() instead of deprecated os.popen()
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=False)
    return result.stdout


def _check_relative_path(path1: Path, path2: Path) -> bool:
    """Check if path is relative using modern pathlib API.
    
    Args:
        path1: First path to check
        path2: Second path to check
        
    Returns:
        True if path1 is relative to path2
    """
    # Use is_relative_to() with proper Python 3.12+ signature
    try:
        return path1.is_relative_to(path2)
    except AttributeError:
        # Fallback for older Python versions
        try:
            path1.relative_to(path2)
            return True
        except ValueError:
            return False


def _find_module_spec(module_name: str):
    """Find module spec using modern importlib API.
    
    Args:
        module_name: Name of the module to find
        
    Returns:
        ModuleSpec if found, None otherwise
    """
    # Use importlib.util.find_spec() instead of deprecated pkgutil.find_loader()
    return importlib.util.find_spec(module_name)


def _get_module_spec(module_name: str):
    """Get module spec using modern importlib API.
    
    Args:
        module_name: Name of the module to get
        
    Returns:
        ModuleSpec if found, None otherwise
    """
    # Use importlib.util.find_spec() instead of deprecated pkgutil.get_loader()
    return importlib.util.find_spec(module_name)


def _log_deprecation_warning(api_name: str, replacement: str):
    """Log deprecation warnings using modern logging API.
    
    Args:
        api_name: Name of the deprecated API
        replacement: Suggested replacement
    """
    # Use logging.warning() instead of deprecated logging.warn()
    logger.warning("API '%s' is deprecated, use '%s' instead", api_name, replacement)
    logger.warning("Please update your code to maintain Python 3.12+ compatibility")


def _get_test_case_names(test_case_class):
    """Get test case names using modern unittest API.
    
    Args:
        test_case_class: TestCase class to inspect
        
    Returns:
        List of test case method names
    """
    # Use TestLoader().getTestCaseNames() instead of deprecated getTestCaseNames()
    loader = unittest.TestLoader()
    return loader.getTestCaseNames(test_case_class)


def _combine_dataframes(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """Combine DataFrames using modern pandas API.
    
    Args:
        df1: First DataFrame
        df2: Second DataFrame or data to append
        
    Returns:
        Combined DataFrame
    """
    # Use pd.concat() instead of removed DataFrame.append()
    return pd.concat([df1, df2], ignore_index=True)


def detect_python_version(repo_path: str) -> Optional[str]:
    """Detect the minimum Python version required by a repository.
    
    Checks multiple sources including pyproject.toml, setup.py, setup.cfg,
    tox.ini, and .python-version files to determine the required Python version.
    
    Args:
        repo_path: Path to the repository root directory
        
    Returns:
        Version string (e.g., '3.8', '3.9') if detected, None otherwise
    """
    repo = Path(repo_path)
    
    # Check pyproject.toml
    pyproject_path = repo / "pyproject.toml"
    if pyproject_path.exists():
        version = _detect_from_pyproject(pyproject_path)
        if version:
            return version
    
    # Check setup.py
    setup_py_path = repo / "setup.py"
    if setup_py_path.exists():
        version = _detect_from_setup_py(setup_py_path)
        if version:
            return version
    
    # Check setup.cfg
    setup_cfg_path = repo / "setup.cfg"
    if setup_cfg_path.exists():
        version = _detect_from_setup_cfg(setup_cfg_path)
        if version:
            return version
    
    # Check tox.ini
    tox_ini_path = repo / "tox.ini"
    if tox_ini_path.exists():
        version = _detect_from_tox_ini(tox_ini_path)
        if version:
            return version
    
    # Check .python-version
    python_version_path = repo / ".python-version"
    if python_version_path.exists():
        version = _detect_from_python_version(python_version_path)
        if version:
            return version
    
    return None


def _detect_from_pyproject(path: Path) -> Optional[str]:
    """Extract Python version from pyproject.toml.
    
    Args:
        path: Path to pyproject.toml file
        
    Returns:
        Version string if found, None otherwise
    """
    try:
        content = path.read_text()
        # Match requires-python = ">=3.x" or similar patterns
        match = re.search(r'requires-python\s*=\s*["\']([^"\']+)["\']', content)
        if match:
            return _parse_version_string(match.group(1))
    except Exception:
        logger.exception("Failed to parse pyproject.toml at %s", path)
    return None


def _detect_from_setup_py(path: Path) -> Optional[str]:
    """Extract Python version from setup.py.
    
    Args:
        path: Path to setup.py file
        
    Returns:
        Version string if found, None otherwise
    """
    try:
        content = path.read_text()
        # Match python_requires='>=3.x' or similar patterns
        match = re.search(r'python_requires\s*=\s*["\']([^"\']+)["\']', content)
        if match:
            return _parse_version_string(match.group(1))
    except Exception:
        logger.exception("Failed to parse setup.py at %s", path)
    return None


def _detect_from_setup_cfg(path: Path) -> Optional[str]:
    """Extract Python version from setup.cfg.
    
    Args:
        path: Path to setup.cfg file
        
    Returns:
        Version string if found, None otherwise
    """
    try:
        content = path.read_text()
        # Match python_requires = >=3.x under [options] section
        match = re.search(r'python_requires\s*=\s*([^\n]+)', content)
        if match:
            return _parse_version_string(match.group(1))
    except Exception:
        logger.exception("Failed to parse setup.cfg at %s", path)
    return None


def _detect_from_tox_ini(path: Path) -> Optional[str]:
    """Extract Python version from tox.ini.
    
    Args:
        path: Path to tox.ini file
        
    Returns:
        Version string if found, None otherwise
    """
    try:
        content = path.read_text()
        # Match envlist = py38,py39,py310 or similar patterns
        match = re.search(r'envlist\s*=\s*([^\n]+)', content)
        if match:
            return _extract_tox_version(match.group(1))
    except Exception:
        logger.exception("Failed to parse tox.ini at %s", path)
    return None


def _detect_from_python_version(path: Path) -> Optional[str]:
    """Extract Python version from .python-version file.
    
    Args:
        path: Path to .python-version file
        
    Returns:
        Version string if found, None otherwise
    """
    try:
        content = path.read_text().strip()
        return _parse_version_string(content)
    except Exception:
        logger.exception("Failed to parse .python-version at %s", path)
    return None


def _parse_version_string(version_str: str) -> Optional[str]:
    """Parse a version string to extract the minimum Python version.
    
    Args:
        version_str: Raw version string from configuration files
        
    Returns:
        Normalized version string (e.g., '3.8') or None if parsing fails
    """
    # Match version patterns like ">=3.8", "^3.9", "3.10", etc.
    match = re.search(r'(\d+)\.(\d+)', version_str)
    if match:
        major, minor = match.groups()
        # Use VERSION_MAJOR_PYTHON constant meaningfully
        if int(major) == VERSION_MAJOR_PYTHON:
            return f"{major}.{minor}"
    return None


def _extract_tox_version(tox_str: str) -> Optional[str]:
    """Extract Python version from tox environment list.
    
    Args:
        tox_str: Tox environment string (e.g., "py38,py39,py310")
        
    Returns:
        Minimum Python version found
    """
    # Match py38, py39, py310 patterns and extract version numbers
    matches = re.findall(r'py(\d)(\d+)', tox_str)
    if matches:
        versions = [(int(major), int(minor)) for major, minor in matches]
        min_version = min(versions)
        return f"{min_version[0]}.{min_version[1]}"
    return None


def scan_python_compat(repo_path: str, target_version: str = "3.12") -> List[CompatibilityIssue]:
    """Scan a Python repository for compatibility issues.
    
    Analyzes Python source files to detect deprecated and removed APIs
    that are incompatible with the target Python version.
    
    Args:
        repo_path: Path to the repository root directory
        target_version: Target Python version to check compatibility against (default: "3.12")
        
    Returns:
        List of CompatibilityIssue objects found during the scan
    """
    issues = []
    repo = Path(repo_path)
    
    for py_file in repo.rglob("*.py"):
        file_issues = _scan_file_compat(py_file, target_version)
        issues.extend(file_issues)
    
    return issues


def _scan_file_compat(file_path: Path, target_version: str) -> List[CompatibilityIssue]:
    """Scan a single Python file for compatibility issues.
    
    Args:
        file_path: Path to the Python file to scan
        target_version: Target Python version
        
    Returns:
        List of compatibility issues found in the file
    """
    issues = []
    
    try:
        content = file_path.read_text()
        lines = content.split('\n')
        
        for line_num, line in enumerate(lines, start=1):
            # Check for deprecated/removed APIs
            if 'asyncio.coroutine' in line:
                issues.append(_create_issue('asyncio.coroutine', line_num, 'HIGH', 
                                           'async def', target_version))
            
            if 'ssl.wrap_socket' in line and 'SSLContext' not in line:
                issues.append(_create_issue('ssl.wrap_socket', line_num, 'HIGH',
                                           'ssl.SSLContext().wrap_socket()', target_version))
            
            if 'DataFrame.append' in line or '.append(' in line:
                issues.append(_create_issue('pandas.append', line_num, 'HIGH',
                                           'pd.concat([df, new_data])', target_version))
            
            if 'logging.warn(' in line:
                issues.append(_create_issue('logging.warn', line_num, 'MEDIUM',
                                           'logging.warning()', target_version))
            
            if 'pkgutil.find_loader' in line:
                issues.append(_create_issue('pkgutil.find_loader', line_num, 'MEDIUM',
                                           'importlib.util.find_spec()', target_version))
            
            if 'pkgutil.get_loader' in line:
                issues.append(_create_issue('pkgutil.get_loader', line_num, 'MEDIUM',
                                           'importlib.util.find_spec()', target_version))
            
            if 'os.popen(' in line:
                issues.append(_create_issue('os.popen', line_num, 'MEDIUM',
                                           'subprocess.run()', target_version))
            
            if 'ssl.PROTOCOL_TLS' in line and 'CLIENT' not in line:
                issues.append(_create_issue('ssl.PROTOCOL_TLS', line_num, 'LOW',
                                           'ssl.PROTOCOL_TLS_CLIENT', target_version))
    
    except Exception:
        logger.exception("Failed to scan file %s for compatibility issues", file_path)
    
    return issues


def _create_issue(api_name: str, line_num: int, severity: str, 
                 replacement: str, version: str) -> CompatibilityIssue:
    """Create a CompatibilityIssue object.
    
    Args:
        api_name: Name of the deprecated/removed API
        line_num: Line number where the issue was found
        severity: Severity level (HIGH, MEDIUM, LOW)
        replacement: Suggested replacement
        version: Target Python version
        
    Returns:
        CompatibilityIssue object
    """
    return CompatibilityIssue(
        api_name=api_name,
        line_number=line_num,
        severity=severity,
        replacement=replacement,
        python_version=version
    )


def generate_compatibility_report(issues: List[CompatibilityIssue]) -> pd.DataFrame:
    """Generate a compatibility report from found issues.
    
    Args:
        issues: List of compatibility issues
        
    Returns:
        DataFrame containing the compatibility report
    """
    if not issues:
        return pd.DataFrame(columns=['api_name', 'line_number', 'severity', 
                                    'replacement', 'python_version'])
    
    data = []
    for issue in issues:
        data.append({
            'api_name': issue.api_name,
            'line_number': issue.line_number,
            'severity': issue.severity,
            'replacement': issue.replacement,
            'python_version': issue.python_version
        })
    
    df = pd.DataFrame(data)
    
    # Aggregate issues by severity
    summary_data = {'severity': ['HIGH', 'MEDIUM', 'LOW'], 'count': [0, 0, 0]}
    summary_df = pd.DataFrame(summary_data)
    
    severity_counts = df['severity'].value_counts()
    for severity, count in severity_counts.items():
        mask = summary_df['severity'] == severity
        summary_df.loc[mask, 'count'] = count
    
    # Combine main report with summary using pd.concat
    combined_df = pd.concat([df, summary_df], ignore_index=True)
    
    return combined_df


def analyze_package_compatibility(package_name: str, target_version: str) -> Dict[str, Any]:
    """Analyze a package for Python version compatibility.
    
    Args:
        package_name: Name of the package to analyze
        target_version: Target Python version
        
    Returns:
        Dictionary containing analysis results
    """
    spec = _find_module_spec(package_name)
    
    if not spec or not spec.origin:
        return {
            'package': package_name,
            'found': False,
            'compatible': None
        }
    
    # Analyze the package source
    package_path = Path(spec.origin).parent
    issues = scan_python_compat(str(package_path), target_version)
    
    return {
        'package': package_name,
        'found': True,
        'compatible': len(issues) == 0,
        'issues': issues,
        'issue_count': len(issues)
    }


def validate_python_syntax(file_path: Path, target_version: str) -> List[CompatibilityIssue]:
    """Validate Python syntax compatibility for a specific version.
    
    Args:
        file_path: Path to the Python file to validate
        target_version: Target Python version
        
    Returns:
        List of syntax compatibility issues
    """
    issues = []
    
    try:
        content = file_path.read_text()
        
        # Check for walrus operator (:=) which requires Python 3.8+
        if ':=' in content and target_version < "3.8":
            lines = content.split('\n')
            for line_num, line in enumerate(lines, start=1):
                if ':=' in line:
                    issues.append(_create_issue('walrus_operator', line_num, 'HIGH',
                                               'regular assignment', target_version))
        
        # Check for f-strings which require Python 3.6+
        if 'f"' in content or "f'" in content:
            if target_version < "3.6":
                lines = content.split('\n')
                for line_num, line in enumerate(lines, start=1):
                    if 'f"' in line or "f'" in line:
                        issues.append(_create_issue('f_string', line_num, 'HIGH',
                                                   'str.format() or % formatting', target_version))
    
    except Exception:
        logger.exception("Failed to validate syntax for file %s", file_path)
    
    return issues


def check_import_compatibility(repo_path: str, target_version: str) -> List[CompatibilityIssue]:
    """Check import statement compatibility.
    
    Args:
        repo_path: Path to the repository root directory
        target_version: Target Python version
        
    Returns:
        List of import compatibility issues
    """
    issues = []
    repo = Path(repo_path)
    
    for py_file in repo.rglob("*.py"):
        try:
            content = py_file.read_text()
            lines = content.split('\n')
            
            for line_num, line in enumerate(lines, start=1):
                # Check for deprecated module imports
                if 'from imp import' in line:
                    issues.append(_create_issue('imp_module', line_num, 'HIGH',
                                               'importlib', target_version))
                
                if 'import imp' in line:
                    issues.append(_create_issue('imp_module', line_num, 'HIGH',
                                               'importlib', target_version))
        
        except Exception:
            logger.exception("Failed to check imports in file %s", py_file)
    
    return issues


__all__ = [
    'CompatibilityIssue',
    'detect_python_version',
    'scan_python_compat',
    'generate_compatibility_report',
    'analyze_package_compatibility',
    'validate_python_syntax',
    'check_import_compatibility',
]