#!/usr/bin/env python3
"""
test_setup.py — Quick check that your Python environment is working in VS Code.
Just run this file and check the output.
"""

import sys
import platform

print("=" * 45)
print("  ✅  Python is working!")
print("=" * 45)
print(f"  Python version : {sys.version.split()[0]}")
print(f"  Platform       : {platform.system()} {platform.release()}")
print(f"  Interpreter    : {sys.executable}")
print("=" * 45)

# Check common scientific packages
packages = ["numpy", "matplotlib", "torch"]
print("\n  Checking packages:")
for pkg in packages:
    try:
        mod = __import__(pkg)
        version = getattr(mod, "__version__", "installed")
        print(f"  ✅  {pkg:12s} {version}")
    except ImportError:
        print(f"  ❌  {pkg:12s} not found")

print("\n  All done — VS Code is set up correctly! 🎉")