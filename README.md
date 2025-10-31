# MultiDiagnostic Ensemble PINN

This repository contains an implementation of a "multidiagnostic" ensemble-enhanced
Physics-Informed Neural Network (PINN) specialized for the Allen–Cahn equation (1D and 2D).
The implementation combines three main components:

- A standard PINN architecture (fully-connected feed-forward networks using tanh activations).
- An adaptive loss-weighter that adjusts PDE/BC/IC loss terms based on high-frequency content
  in the PDE residual (spectral analysis).
- An ensemble of novel analyzers (topological, manifold complexity, information-theoretic,
  and geometric) that produce an additional auxiliary loss used to guide training.

## Files of interest

- `multidiagnostic_1d.py` — 1D Allen–Cahn PINN implementation including ensemble analyzers,
  adaptive loss weighting, supervised-term handling and logging.
- `multidiagnostic_2d.py` — 2D Allen–Cahn PINN equivalent (x,y,t input).
- `ensemble.py` — analyzers and utilities: `SpectralAnalyzer`, `AdaptiveLossWeighter`,
  `EnsembleNovelAnalyzer`, and several fallback analyzers (topological, manifold,
  information-theoretic, geometric). This module is robust to optional dependencies.
- `train_and_test_1d.py`, `train_and_test_2d.py` — training/demo scripts that expect
  importable `modified_pinn.py` / `modified_pinn_2d.py` modules.
- `modified_pinn.py`, `modified_pinn_2d.py` — lightweight shims that expose the
  classes from the `multidiagnostic_*.py` files for compatibility with existing scripts.

## Design contract (brief)

Inputs:

- data_dict: a dictionary used during training with keys: `collocation`, `boundary_points`,
  `boundary_values`, `initial_points`, `initial_values`. Optionally `collocation_values`,
  `supervised_points`, `supervised_values`.
- model hyperparameters: `epsilon`, `alpha`, MLP `layers`.

Outputs:

- A PyTorch module instance that exposes `forward(x)` and `compute_ensemble_adaptive_loss(data_dict, epoch)`.
- Training history saved in `model.history` (loss breakdowns, weights, spectral metrics,
  analyzer outputs).

## Error modes / expectations

- The analyzers in `ensemble.py` use optional libraries (gudhi, umap, sklearn, scipy). They
  fall back gracefully to simple statistics if those packages are missing. The code is robust
  to missing `collocation_values` or `supervised_*` entries in the dataset.
- The training scripts (`train_and_test_*.py`) provide a fallback data generator if `data/*.npz`
  is missing, but reproducing full experiments requires prepared datasets (`data/train.npz` and `data/test.npz`).

## How the ensemble guides training

- SpectralAnalyzer computes band-wise energy of the PDE residual (FFT-based). If high-frequency
  energy is significant, `AdaptiveLossWeighter` increases the PDE loss weight to force the PINN
  to focus on higher-frequency components.
- `EnsembleNovelAnalyzer` computes several non-differentiable metrics (topological persistence,
  manifold complexity, information metrics, geometric flow) and converts them to scalar "boosts".
  These boosts are combined with a small learnable mixture (softmax over analyzer logits) to form
  a differentiable auxiliary loss term the optimizer can influence.

## Running the demo training (1D)

From the repository root run:

```bash
python3 train_and_test_1d.py
```

This script loads `data/train.npz` and `data/test.npz` if present. If not, it attempts to
generate a small fallback dataset (which may be slower and is intended for quick demos).

Outputs from training:

- The model state dict is saved to `models/ensemble_pinn.pth`.
- The training loop prints per-epoch loss information and stores detailed history in `model.history`.

## Notes and next steps

- The repository aims to be robust and informative; the analyzer modules include many fallbacks
  so experiments can run without heavy optional dependencies. If you want the most accurate
  topological or manifold measures, install the optional packages (`pip install gudhi umap-learn scikit-learn scipy`).
- Future improvements could include: automated hyperparameter tuning, a unit-test harness for
  analyzer outputs, and optional callbacks to visualize spectral snapshots during training.

