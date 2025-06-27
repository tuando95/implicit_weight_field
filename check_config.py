#!/usr/bin/env python3
"""Check config directory."""

import os
from pathlib import Path

print(f"Current directory: {os.getcwd()}")
print(f"Looking for configs in: {os.path.abspath('./configs')}")

if os.path.exists('./configs'):
    print("✓ configs directory exists")
    print("Contents:")
    for item in os.listdir('./configs'):
        print(f"  - {item}")
else:
    print("✗ configs directory NOT FOUND")
    
# Check for default.yaml
if os.path.exists('./configs/default.yaml'):
    print("\n✓ default.yaml exists")
else:
    print("\n✗ default.yaml NOT FOUND")