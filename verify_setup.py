#!/usr/bin/env python3

"""
MetaEvolve Setup Verification Script
==================================

This script verifies that all dependencies are installed and optionally runs
a minimal test to ensure everything is working correctly.
"""

import importlib
import os
from pathlib import Path
import re
import subprocess
import sys


def check_symbol(success: bool, message: str) -> None:
    """Print a status message with colored symbols."""
    symbol = "✅" if success else "❌"
    print(f"{symbol} {message}")
    return success


def parse_dependencies_from_pyproject() -> list[str]:
    """Parse dependencies from pyproject.toml file."""
    pyproject_path = Path("pyproject.toml")

    if not pyproject_path.exists():
        print("❌ pyproject.toml not found")
        return []

    try:
        content = pyproject_path.read_text()

        # Find the dependencies section
        in_dependencies = False
        dependencies = []

        for line in content.split("\n"):
            line = line.strip()

            # Start of dependencies section
            if line == "dependencies = [":
                in_dependencies = True
                continue

            # End of dependencies section
            if in_dependencies and line == "]":
                break

            # Parse dependency line - handle both "package", and "package" (last item)
            if in_dependencies and line.startswith('"'):
                # Extract package name from "package>=version", or "package>=version" format
                dep_line = line.strip('",').strip('"')
                # Get just the package name (before any version specifiers)
                package_name = re.split(r"[>=<!=]", dep_line)[0].strip()
                # Handle extras like "fakeredis[lua]" -> "fakeredis"
                package_name = package_name.split("[")[0]
                dependencies.append(package_name)

        return dependencies

    except Exception as e:
        print(f"❌ Error parsing pyproject.toml: {e}")
        return []


def check_python_packages() -> bool:
    """Check if required Python packages are installed."""
    print("🔍 Checking Python packages...")

    required_packages = parse_dependencies_from_pyproject()

    if not required_packages:
        print("❌ Could not parse dependencies from pyproject.toml")
        return False

    print(f"  Found {len(required_packages)} dependencies in pyproject.toml")

    all_good = True
    for package in required_packages:
        try:
            importlib.import_module(package)
            check_symbol(True, f"Package '{package}' is installed")
        except ImportError:
            check_symbol(False, f"Package '{package}' is missing")
            all_good = False

    return all_good


def check_redis_connection() -> bool:
    """Check if Redis is running and accessible."""
    print("\n🔍 Checking Redis connection...")

    try:
        result = subprocess.run(
            ["redis-cli", "ping"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and "PONG" in result.stdout:
            check_symbol(True, "Redis is running and accessible")
            return True
        else:
            check_symbol(False, "Redis is not responding correctly")
            return False
    except subprocess.TimeoutExpired:
        check_symbol(False, "Redis connection timed out")
        return False
    except FileNotFoundError:
        check_symbol(False, "Redis CLI not found - Redis may not be installed")
        print("  Install Redis: brew install redis")
        return False
    except Exception as e:
        check_symbol(False, f"Redis check failed: {e}")
        return False


def check_api_key() -> bool:
    """Check if OpenRouter API key is set."""
    print("\n🔍 Checking API key...")

    api_key = os.getenv("OPENROUTER_API_KEY")
    if api_key:
        check_symbol(True, "OPENROUTER_API_KEY environment variable is set")
        return True
    else:
        check_symbol(
            False, "OPENROUTER_API_KEY environment variable is not set"
        )
        print("  Get your API key from: https://openrouter.ai/keys")
        print("  Set it with: export OPENROUTER_API_KEY=your_api_key_here")
        return False


def check_problem_directory() -> bool:
    """Check if the hexagon pack problem directory exists."""
    print("\n🔍 Checking problem directory...")

    problem_dir = Path("problems/hexagon_pack")
    if problem_dir.exists():
        check_symbol(True, f"Problem directory '{problem_dir}' exists")

        # Check required files
        required_files = [
            "task_description.txt",
            "task_hints.txt",
            "validate.py",
            "mutation_system_prompt.txt",
            "mutation_user_prompt.txt",
        ]

        all_files_present = True
        for file in required_files:
            if (problem_dir / file).exists():
                check_symbol(True, f"  Required file '{file}' found")
            else:
                check_symbol(False, f"  Required file '{file}' missing")
                all_files_present = False

        # Check initial programs
        initial_programs = problem_dir / "initial_programs"
        if initial_programs.exists():
            py_files = list(initial_programs.glob("*.py"))
            if py_files:
                check_symbol(True, f"  Found {len(py_files)} initial programs")
            else:
                check_symbol(False, "  No initial programs found")
                all_files_present = False
        else:
            check_symbol(False, "  Initial programs directory missing")
            all_files_present = False

        return all_files_present
    else:
        check_symbol(False, f"Problem directory '{problem_dir}' not found")
        return False


def main():
    """Main verification routine."""
    print("🚀 MetaEvolve Setup Verification")
    print("================================")

    # Check virtual environment
    if os.getenv("VIRTUAL_ENV"):
        check_symbol(
            True, f"Virtual environment active: {os.getenv('VIRTUAL_ENV')}"
        )
    else:
        check_symbol(False, "No virtual environment detected")
        print("  Activate with: source metaevolve/bin/activate")

    # Run all checks
    checks = [
        check_python_packages(),
        check_redis_connection(),
        check_api_key(),
        check_problem_directory(),
    ]

    all_checks_passed = all(checks)

    print(f"\n📊 Summary: {sum(checks)}/{len(checks)} checks passed")

    if all_checks_passed:
        print("\n🎉 All checks passed! MetaEvolve is ready to run.")
        print("\nTo start evolution:")
        print(
            "  python restart_llm_evolution_improved.py --problem-dir problems/hexagon_pack --min-fitness -7 --max-fitness -3.9"
        )
    else:
        print(
            "\n⚠️  Some checks failed. Please fix the issues above before running MetaEvolve."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
