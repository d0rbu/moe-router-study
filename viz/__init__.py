"""Visualization utilities for MoE router study."""

import os

from exp import BASE_OUTPUT_DIR

# Base figure directory
BASE_FIGURE_DIR = "fig"


def get_figure_dir(experiment_name: str | None = None) -> str:
    """
    Get the figure directory for an experiment.

    Args:
        experiment_name: Optional name of the experiment. If provided, figures will be
                         saved to a subdirectory with this name.

    Returns:
        Path to the figure directory.
    """
    if experiment_name is None:
        return BASE_FIGURE_DIR

    return os.path.join(BASE_FIGURE_DIR, experiment_name)
