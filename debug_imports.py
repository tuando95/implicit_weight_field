#!/usr/bin/env python3
"""Debug import issues."""

import os
import sys

print("=== Import Debug ===")
print(f"Current working directory: {os.getcwd()}")
print(f"Script location: {os.path.abspath(__file__)}")
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print("\n=== Python Path ===")
for i, p in enumerate(sys.path[:5]):
    print(f"{i}: {p}")

print("\n=== Directory Structure ===")
# Check current directory
for item in os.listdir('.'):
    if os.path.isdir(item) and not item.startswith('.'):
        print(f"DIR: {item}/")
        # Check if it has __init__.py
        init_path = os.path.join(item, '__init__.py')
        if os.path.exists(init_path):
            print(f"  -> has __init__.py")
        else:
            print(f"  -> NO __init__.py!")

print("\n=== Testing Imports ===")
# Try different import methods
try:
    import experiments
    print("✓ import experiments - SUCCESS")
except ImportError as e:
    print(f"✗ import experiments - FAILED: {e}")

try:
    from experiments import models
    print("✓ from experiments import models - SUCCESS")
except ImportError as e:
    print(f"✗ from experiments import models - FAILED: {e}")

try:
    import experiments.models
    print("✓ import experiments.models - SUCCESS")
except ImportError as e:
    print(f"✗ import experiments.models - FAILED: {e}")

# Check if we can find the experiments directory
exp_dir = None
for path in sys.path:
    potential = os.path.join(path, 'experiments')
    if os.path.exists(potential) and os.path.isdir(potential):
        exp_dir = potential
        print(f"\n=== Found experiments at: {exp_dir} ===")
        break

if exp_dir:
    print("Contents:")
    for item in os.listdir(exp_dir):
        print(f"  {item}")