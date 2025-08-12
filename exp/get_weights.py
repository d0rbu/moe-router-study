"""Utilities for locating experiment weight files.

This module provides the path where experiment weight tensors (e.g., router.pt,
layer projection weights) are stored. Keeping it centralized avoids import
issues in visualization scripts and helps static type checkers.
"""

from __future__ import annotations

import os

# Default directory for weight files. Users can override this via the
# MOE_ROUTER_WEIGHT_DIR environment variable if desired.
WEIGHT_DIR = os.environ.get("MOE_ROUTER_WEIGHT_DIR", "data")

