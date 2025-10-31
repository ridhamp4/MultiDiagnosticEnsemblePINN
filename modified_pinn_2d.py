# Compatibility shim for existing training scripts
# Exposes EnsembleAllenCahnPINN2D used by train_and_test_2d.py
from multidiagnostic_2d import EnsembleAllenCahnPINN2D

__all__ = ["EnsembleAllenCahnPINN2D"]
