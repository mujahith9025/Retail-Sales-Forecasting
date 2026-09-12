"""
Retail Sales Forecasting Package
"""

import sys
import os
from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _PKG_DIR.parent

for _p in [str(_ROOT_DIR), str(_PKG_DIR), os.getcwd()]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
