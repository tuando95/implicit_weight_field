"""Experiments module for Implicit Neural Weight Fields."""

import os
import sys

# Ensure project root is in path for all submodules
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from .models import *
from .ablation import *
from .scaling import *
from .robustness import *