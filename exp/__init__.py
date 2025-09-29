"""Experiments for MoE router study."""

import os

OUTPUT_DIR = "out"
ACTIVATION_DIRNAME = "activations"
ROUTER_LOGITS_DIRNAME = "router_logits"
WEIGHT_DIR = os.path.join(OUTPUT_DIR, "weights")
MODEL_DIRNAME = "models"
DATASET_DIRNAME = "datasets"
