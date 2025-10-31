# Compatibility shim for existing training scripts
# Exposes EnsembleAllenCahnPINN used by train_and_test_1d.py
from multidiagnostic_1d import EnsembleAllenCahnPINN

__all__ = ["EnsembleAllenCahnPINN"]
