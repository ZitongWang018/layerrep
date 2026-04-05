"""Paths and defaults for ETD experiments."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
DEFAULT_MODEL_PATH = "/root/autodl-tmp/model_qwen"

# Angle collection: small subset per dataset (offline-friendly)
DEFAULT_ANGLE_MAX_SAMPLES = 128

# Evaluation: full test set can be slow; 0 = all
DEFAULT_EVAL_LIMIT = 0
