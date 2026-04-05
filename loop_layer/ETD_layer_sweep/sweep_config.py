"""Sweep experiment defaults."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
FIGURES = ROOT / "figures"

DEFAULT_MODEL_PATH = "/root/autodl-tmp/model_qwen"

# T block: inclusive layer indices (0-based) for thinking layers
T_START_MIN = 5
T_START_MAX = 15
T_END_MIN = 20
T_END_MAX = 28

NUM_LAYERS = 36

# Same evaluation sets for every sweep cell (fixed order / count)
DEFAULT_BOOLQ_SPLIT = "validation"
DEFAULT_ARC_SPLIT = "test"
DEFAULT_BOOLQ_LIMIT = 500
DEFAULT_ARC_LIMIT = 500

ETD_ROOT = ROOT.parent / "ETD"
