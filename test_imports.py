#!/usr/bin/env python3
"""Test imports to diagnose the issue."""

import os
import sys

print("=== Current Environment ===")
print(f"Working directory: {os.getcwd()}")
print(f"Script location: {os.path.abspath(__file__)}")
print(f"Python path (first 5):")
for i, p in enumerate(sys.path[:5]):
    print(f"  {i}: {p}")

# Check what directories exist
print("\n=== Checking for required directories ===")
dirs_to_check = ['core', 'compression', 'evaluation', 'experiments', 'configs']
for d in dirs_to_check:
    full_path = os.path.abspath(d)
    exists = os.path.exists(d) and os.path.isdir(d)
    print(f"{d}: {'EXISTS' if exists else 'NOT FOUND'} at {full_path}")

# Try to import
print("\n=== Testing imports ===")
try:
    import core
    print("✓ import core - SUCCESS")
    print(f"  core location: {core.__file__}")
except ImportError as e:
    print(f"✗ import core - FAILED: {e}")

# Try adding current directory
sys.path.insert(0, os.getcwd())
print(f"\nAdded {os.getcwd()} to path")

try:
    import core
    print("✓ import core - SUCCESS after adding cwd")
except ImportError as e:
    print(f"✗ import core - STILL FAILED: {e}")

# List what's actually in the current directory
print("\n=== Current directory contents ===")
for item in sorted(os.listdir('.')):
    if os.path.isdir(item):
        print(f"[DIR]  {item}/")
    else:
        print(f"[FILE] {item}")