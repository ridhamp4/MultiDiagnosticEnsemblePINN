import torch
import torch.nn as nn
import numpy as np
from scipy import fft
import matplotlib.pyplot as plt

# from dataget import BurgersDataGenerator, save_dataset, load_dataset
import os
# from vis_eval import evaluate_model, plot_solution_comparison, plot_training_history


class SpectralAnalyzer:
    """Enhanced spectral analysis for PINNs"""

    def __init__(self, domain_bounds, n_bands=12):
        self.domain_bounds = domain_bounds
        self.n_bands = n_bands
        self.frequency_bands = self._initialize_frequency_bands()

    def _initialize_frequency_bands(self):
        domain_size = self.domain_bounds[0][1] - self.domain_bounds[0][0]
        max_freq = 1.0 / (2.0 * (domain_size / 50))
        low_freqs = np.logspace(-1, np.log10(max_freq), self.n_bands)
        high_freqs = np.logspace(-1, np.log10(max_freq), self.n_bands + 1)[1:]
        bands = [(low, high) for low, high in zip(low_freqs, high_freqs)]
        return bands

    def compute_spectral_errors(self, residual, coordinates):
        x = coordinates[:, 0].detach().cpu().numpy()
        res_vals = residual.detach().cpu().numpy().ravel()

        sort_idx = np.argsort(x)
        x_sorted = x[sort_idx]
        res_sorted = res_vals[sort_idx]

        x_unique, unique_idx = np.unique(x_sorted, return_index=True)
        res_unique = res_sorted[unique_idx]

        x_unique = np.asarray(x_unique).ravel()
        res_unique = np.asarray(res_unique).ravel()

        if x_unique.size < 2:
            return {f'band_{i}': 0.0 for i in range(self.n_bands)}

        n_points = min(256, len(x_unique))
        x_uniform = np.linspace(float(x_unique.min()), float(x_unique.max()), n_points)
        res_uniform = np.interp(x_uniform, x_unique, res_unique)

        fft_vals = fft.fft(res_uniform)
        freqs = fft.fftfreq(len(res_uniform), x_uniform[1] - x_uniform[0])

        spectral_energy = {}
        total_energy = np.sum(np.abs(fft_vals)**2) + 1e-12
        for i, (low, high) in enumerate(self.frequency_bands):
            mask = (np.abs(freqs) >= low) & (np.abs(freqs) < high)
            band_energy = np.sum(np.abs(fft_vals[mask])**2)
            spectral_energy[f'band_{i}'] = float(band_energy / total_energy)

        return spectral_energy

    def compute_spectral_energies_absolute(self, residual, coordinates):
        """Return absolute band energies (not normalized) and the total energy.

        Returns:
          (band_dict, total_energy)
        where band_dict maps 'band_i' -> absolute energy (float).
        """
        x = coordinates[:, 0].detach().cpu().numpy()
        res_vals = residual.detach().cpu().numpy().ravel()

        sort_idx = np.argsort(x)
        x_sorted = x[sort_idx]
        res_sorted = res_vals[sort_idx]

        x_unique, unique_idx = np.unique(x_sorted, return_index=True)
        res_unique = res_sorted[unique_idx]

        x_unique = np.asarray(x_unique).ravel()
        res_unique = np.asarray(res_unique).ravel()

        if x_unique.size < 2:
            return ({f'band_{i}': 0.0 for i in range(self.n_bands)}, 0.0)

        n_points = min(256, len(x_unique))
        x_uniform = np.linspace(float(x_unique.min()), float(x_unique.max()), n_points)
        res_uniform = np.interp(x_uniform, x_unique, res_unique)

        fft_vals = fft.fft(res_uniform)
        freqs = fft.fftfreq(len(res_uniform), x_uniform[1] - x_uniform[0])

        spectral_energy = {}
        total_energy = float(np.sum(np.abs(fft_vals)**2) + 1e-12)
        for i, (low, high) in enumerate(self.frequency_bands):
            mask = (np.abs(freqs) >= low) & (np.abs(freqs) < high)
            band_energy = float(np.sum(np.abs(fft_vals[mask])**2))
            spectral_energy[f'band_{i}'] = band_energy

        return spectral_energy, total_energy


########## Novel analyzers from README.md (with safe fallbacks) ##########


class TopologicalFeatureAnalyzer:
    """Use persistent-homology-like features to detect structure in residuals.

    Falls back to simple statistics when gudhi is not available.
    """
    def __init__(self, max_dimension=1, persistence_threshold=0.1):
        try:
            import gudhi as gd  # optional
            self.gd = gd
        except Exception:
            self.gd = None
        self.max_dimension = max_dimension
        self.persistence_threshold = persistence_threshold
        self.feature_history = []

    def compute_persistence(self, residual, coordinates):
        if self.gd is None:
            # fallback: return simple stats
            res = residual.detach().cpu().numpy().ravel()
            features = {
                'topological_complexity': float(np.var(res)),
                'feature_persistence': float(np.mean(np.abs(res))),
                'cycle_strength': 0.0,
                'connectivity_pattern': 0.0
            }
            return features

        X = coordinates.detach().cpu().numpy()
        y = residual.detach().cpu().numpy().reshape(-1, 1)
        point_cloud = np.hstack([X, y])
        try:
            rips = self.gd.RipsComplex(points=point_cloud, max_edge_length=2.0)
            st = rips.create_simplex_tree(max_dimension=self.max_dimension)
            persistence = st.persistence()
        except Exception:
            return self.compute_persistence(residual, coordinates)

        # crude analysis of persistence
        features = {
            'topological_complexity': 0.0,
            'feature_persistence': 0.0,
            'cycle_strength': 0.0,
            'connectivity_pattern': 0.0
        }
        for dim, pair in persistence:
            birth, death = pair
            pval = death - birth
            features['feature_persistence'] += pval
            if dim == 1:
                features['cycle_strength'] += pval
                features['topological_complexity'] += 1.0
            elif dim == 0:
                features['connectivity_pattern'] += pval

        total_pairs = max(1, len(persistence))
        for k in features:
            features[k] = float(features[k] / total_pairs)

        return features

    def compute_adaptive_boost(self, features, epoch):
        complexity_boost = 1.0 + 2.0 * features.get('topological_complexity', 0.0)
        cycle_boost = 1.0 + features.get('cycle_strength', 0.0)
        topological_importance = min(3.0, complexity_boost * cycle_boost)
        self.feature_history.append({'epoch': epoch, 'boost_factor': topological_importance})
        return topological_importance


class ManifoldComplexityAnalyzer:
    """Estimate manifold/cluster complexity with fallbacks if sklearn/umap missing.
    """
    def __init__(self, n_components=2):
        try:
            import umap
            from sklearn.neighbors import NearestNeighbors
            self.umap = umap
            self.nn = NearestNeighbors
        except Exception:
            self.umap = None
            self.nn = None
        self.n_components = n_components
        self.complexity_history = []

    def compute_manifold_complexity(self, residual, coordinates):
        X = coordinates.detach().cpu().numpy()
        y = residual.detach().cpu().numpy().reshape(-1, 1)
        features = np.hstack([X, y])
        # fallback measures
        metrics = {
            'intrinsic_dimensionality': 1.0,
            'manifold_curvature': 0.0,
            'cluster_complexity': 0.0,
            'nonlinearity': 0.0
        }
        if self.umap is None or self.nn is None:
            # try simple PCA variance ratio as intrinsic dim proxy
            try:
                uu, s, vv = np.linalg.svd(features - features.mean(0), full_matrices=False)
                explained = s**2 / np.sum(s**2)
                metrics['intrinsic_dimensionality'] = float(np.sum(explained > 0.01))
                metrics['manifold_curvature'] = float(np.var(features)) * 1e-3
                metrics['cluster_complexity'] = float(len(np.unique(y)) / max(1, len(y)))
            except Exception:
                pass
            return metrics

        try:
            reducer = self.umap.UMAP(n_components=self.n_components, n_neighbors=15, min_dist=0.1)
            emb = reducer.fit_transform(features)
            # curvature proxy
            from sklearn.metrics import pairwise_distances
            od = pairwise_distances(features)
            ed = pairwise_distances(emb)
            curvature = np.mean(np.abs(od - ed) / (od + 1e-8))
            metrics['manifold_curvature'] = float(curvature)
            # intrinsic dim via SVD
            uu, s, vv = np.linalg.svd(features - features.mean(0), full_matrices=False)
            explained = s**2 / np.sum(s**2)
            metrics['intrinsic_dimensionality'] = float(np.sum(explained > 0.01))
            # clustering complexity
            from sklearn.cluster import DBSCAN
            lab = DBSCAN(eps=0.5, min_samples=10).fit_predict(features)
            n_clusters = len(set(lab)) - (1 if -1 in lab else 0)
            metrics['cluster_complexity'] = float((n_clusters + list(lab).count(-1) / max(1, len(lab))) / 10.0)
        except Exception:
            pass
        return metrics

    def compute_complexity_boost(self, metrics, epoch):
        dim_boost = 1.0 + 0.5 * metrics.get('intrinsic_dimensionality', 1.0)
        curvature_boost = 1.0 + metrics.get('manifold_curvature', 0.0)
        cluster_boost = 1.0 + 2.0 * metrics.get('cluster_complexity', 0.0)
        nonlinear_boost = 1.0 + metrics.get('nonlinearity', 0.0)
        combined_boost = min(3.0, dim_boost * curvature_boost * cluster_boost * nonlinear_boost)
        self.complexity_history.append({'epoch': epoch, 'boost': combined_boost})
        return combined_boost


class InformationTheoreticAnalyzer:
    """Information-theoretic metrics with fallbacks."""
    def __init__(self, n_bins=50):
        self.n_bins = n_bins
        self.info_history = []

    def compute_information_metrics(self, residual, coordinates):
        try:
            from scipy.stats import entropy
            from scipy.stats import gaussian_kde
        except Exception:
            entropy = None
            gaussian_kde = None

        res_vals = residual.detach().cpu().numpy().flatten()
        x_vals = coordinates[:, 0].detach().cpu().numpy()
        t_vals = coordinates[:, 1].detach().cpu().numpy()
        metrics = {'entropy': 0.0, 'mi_spatial': 0.0, 'mi_temporal': 0.0, 'predictability': 0.0, 'information_density': 0.0}
        try:
            if entropy is not None:
                hist, _ = np.histogram(res_vals, bins=self.n_bins, density=True)
                hist = hist[hist > 0]
                metrics['entropy'] = float(entropy(hist))

            # mutual information coarse estimate
            def mi(a, b, bins=20):
                H, xedges, yedges = np.histogram2d(a, b, bins=bins)
                pxy = H / np.sum(H)
                px = np.sum(pxy, axis=1)
                py = np.sum(pxy, axis=0)
                mi_v = 0.0
                for i in range(pxy.shape[0]):
                    for j in range(pxy.shape[1]):
                        if pxy[i, j] > 0:
                            mi_v += pxy[i, j] * np.log(pxy[i, j] / (px[i] * py[j] + 1e-10))
                return mi_v

            metrics['mi_spatial'] = float(mi(x_vals, res_vals))
            metrics['mi_temporal'] = float(mi(t_vals, res_vals))

            # predictability via autocorrelation
            ac = np.correlate(res_vals, res_vals, mode='full')
            ac = ac[len(ac)//2:]
            if ac[0] != 0:
                ac = ac / (ac[0] + 1e-12)
            metrics['predictability'] = float(np.mean(np.abs(ac[1:min(10, len(ac))])))

            # information density via KDE concentration
            if gaussian_kde is not None:
                kde = gaussian_kde(res_vals)
                dens = kde(res_vals)
                sd = np.sort(dens)[::-1]
                cum = np.cumsum(sd)
                half = np.argmax(cum > 0.5 * cum[-1])
                metrics['information_density'] = float(half / max(1, len(res_vals)))
        except Exception:
            pass
        return metrics

    def compute_information_boost(self, metrics, epoch):
        entropy_boost = 1.0 + 0.3 * metrics.get('entropy', 0.0)
        spatial_boost = 1.0 + metrics.get('mi_spatial', 0.0)
        temporal_boost = 1.0 + metrics.get('mi_temporal', 0.0)
        unpredictability_boost = 2.0 - metrics.get('predictability', 0.0)
        concentration_boost = 1.0 + (1.0 - metrics.get('information_density', 0.0))
        combined_boost = min(3.0, entropy_boost * spatial_boost * temporal_boost * unpredictability_boost * concentration_boost)
        self.info_history.append({'epoch': epoch, 'boost': combined_boost})
        return combined_boost


class GeometricFlowAnalyzer:
    """Geometric flow / differential geometry inspired measures with fallbacks."""
    def __init__(self, neighborhood_size=20):
        self.neighborhood_size = neighborhood_size
        self.flow_history = []

    def compute_geometric_metrics(self, residual, coordinates):
        res_vals = residual.detach().cpu().numpy().flatten()
        x_vals = coordinates[:, 0].detach().cpu().numpy()
        t_vals = coordinates[:, 1].detach().cpu().numpy()
        metrics = {'gradient_magnitude': 0.0, 'divergence': 0.0, 'vorticity': 0.0, 'topological_defects': 0.0}
        try:
            from scipy.interpolate import griddata
            from numpy import gradient
            xi = np.linspace(x_vals.min(), x_vals.max(), 50)
            ti = np.linspace(t_vals.min(), t_vals.max(), 50)
            X, T = np.meshgrid(xi, ti)
            Z = griddata((x_vals, t_vals), res_vals, (X, T), method='linear')
            dx, dt = gradient(Z)
            grad_mag = np.mean(np.sqrt(np.nan_to_num(dx)**2 + np.nan_to_num(dt)**2))
            metrics['gradient_magnitude'] = float(grad_mag)
            # laplacian
            d2x, _ = gradient(dx)
            _, d2t = gradient(dt)
            lap = d2x + d2t
            metrics['divergence'] = float(np.nanmean(np.abs(lap)))
            metrics['vorticity'] = float(np.nanmean(np.abs(np.nan_to_num(dx) - np.nan_to_num(dt))))
            from scipy.ndimage import maximum_filter, minimum_filter
            local_max = maximum_filter(Z, size=3) == Z
            local_min = minimum_filter(Z, size=3) == Z
            n_defects = np.sum(np.nan_to_num(local_max)) + np.sum(np.nan_to_num(local_min))
            metrics['topological_defects'] = float(n_defects / (Z.shape[0] * Z.shape[1]))
        except Exception:
            pass
        return metrics

    def compute_geometric_boost(self, metrics, epoch):
        gradient_boost = 1.0 + metrics.get('gradient_magnitude', 0.0)
        divergence_boost = 1.0 + metrics.get('divergence', 0.0)
        vorticity_boost = 1.0 + metrics.get('vorticity', 0.0)
        defect_boost = 1.0 + 5.0 * metrics.get('topological_defects', 0.0)
        combined_boost = min(3.0, gradient_boost * divergence_boost * vorticity_boost * defect_boost)
        self.flow_history.append({'epoch': epoch, 'boost': combined_boost})
        return combined_boost


class EnsembleNovelAnalyzer(nn.Module):
    """Combine multiple novel analysis methods with learned weighting.

    Lightweight nn.Module wrapper to keep ensemble weights as learnable
    parameters (optional). Uses safe fallbacks when analyzers are missing.
    """
    def __init__(self):
        super().__init__()
        # instantiate analyzers (they are pure python, no params)
        self.topological = TopologicalFeatureAnalyzer()
        self.manifold = ManifoldComplexityAnalyzer()
        self.information = InformationTheoreticAnalyzer()
        self.geometric = GeometricFlowAnalyzer()

        # include spectral analyzer so the ensemble can learn to weight FFT opinions
        try:
            self.spectral = SpectralAnalyzer(domain_bounds=[(-1, 1)], n_bands=12)
        except Exception:
            self.spectral = None

        # learnable logits for softmax weighting; keep a learnable temperature
        # so the model can control sharpness of the mixing.
        self.ensemble_weights = nn.ParameterDict({
            'topological': nn.Parameter(torch.tensor(0.25)),
            'manifold': nn.Parameter(torch.tensor(0.25)),
            'information': nn.Parameter(torch.tensor(0.25)),
            'geometric': nn.Parameter(torch.tensor(0.25)),
            'spectral': nn.Parameter(torch.tensor(0.25))
        })
        # softmax temperature (learnable)
        self.register_parameter('temperature', nn.Parameter(torch.tensor(1.0)))

        self.ensemble_history = []

    def compute_ensemble_boost(self, residual, coordinates, epoch):
        """Return a differentiable torch scalar boost and a dict of analyzer boosts.

        The analyzers themselves produce non-differentiable float boosts (numpy-based).
        We convert those floats to a torch vector and compute a softmax-weighted
        combination with the learnable ensemble logits. The returned boost tensor
        is suitable for use as an auxiliary loss multiplier (it participates in
        autograd for the ensemble parameters).
        Returns: (boost_tensor, boosts_dict, weight_floats)
        """
        boosts = {}
        metrics = {}
        try:
            feats = self.topological.compute_persistence(residual, coordinates)
            boosts['topological'] = self.topological.compute_adaptive_boost(feats, epoch)
            metrics['topological'] = feats
        except Exception:
            boosts['topological'] = 1.0

        try:
            m = self.manifold.compute_manifold_complexity(residual, coordinates)
            boosts['manifold'] = self.manifold.compute_complexity_boost(m, epoch)
            metrics['manifold'] = m
        except Exception:
            boosts['manifold'] = 1.0

        try:
            im = self.information.compute_information_metrics(residual, coordinates)
            boosts['information'] = self.information.compute_information_boost(im, epoch)
            metrics['information'] = im
        except Exception:
            boosts['information'] = 1.0

        try:
            gm = self.geometric.compute_geometric_metrics(residual, coordinates)
            boosts['geometric'] = self.geometric.compute_geometric_boost(gm, epoch)
            metrics['geometric'] = gm
        except Exception:
            boosts['geometric'] = 1.0

        # spectral opinion (FFT-based) -> convert high-frequency ratio into a boost
        try:
            if self.spectral is not None:
                band_dict, total_energy = self.spectral.compute_spectral_energies_absolute(residual, coordinates)
                if total_energy > 0:
                    n_bands = len(band_dict)
                    # use upper quarter as high-frequency bands
                    start = max(0, n_bands - max(1, n_bands // 4))
                    high_energy = sum(band_dict.get(f'band_{i}', 0.0) for i in range(start, n_bands))
                    high_ratio = high_energy / (total_energy + 1e-12)
                else:
                    high_ratio = 0.0
                spectral_boost = min(3.0, 1.0 + 2.0 * float(high_ratio))
                boosts['spectral'] = spectral_boost
                metrics['spectral'] = {'high_freq_ratio': float(high_ratio), 'total_energy': float(total_energy)}
            else:
                boosts['spectral'] = 1.0
        except Exception:
            boosts['spectral'] = 1.0

        # Build tensors for boosts and logits in a deterministic key order
        keys = list(self.ensemble_weights.keys())
        device = next(self.ensemble_weights[keys[0]].device for _ in [0]) if len(keys) > 0 else torch.device('cpu')

        boost_vals = []
        for k in keys:
            boost_vals.append(float(boosts.get(k, 1.0)))

        # convert boosts to a torch tensor on same device as params (no grad on boosts)
        boosts_tensor = torch.tensor(boost_vals, dtype=torch.float32, device=device)

        # assemble logits tensor from ParameterDict in same order
        logits = torch.stack([self.ensemble_weights[k] for k in keys])

        # compute softmax weights with temperature
        temp = torch.clamp(self.temperature, min=1e-6)
        try:
            weight_tensor = torch.softmax(logits / temp, dim=0)
        except Exception:
            # fallback in case of numerical issues
            weight_tensor = torch.softmax(logits, dim=0)

        # final differentiable scalar boost
        boost_tensor = torch.dot(weight_tensor, boosts_tensor)

        # expose human-friendly floats for logging
        weight_vals = {k: float(w) for k, w in zip(keys, weight_tensor.detach().cpu().numpy().tolist())}

        self.ensemble_history.append({'epoch': epoch, 'boosts': boosts, 'weights': weight_vals, 'total_boost': float(boost_tensor.detach().cpu().item())})
        return boost_tensor, boosts, weight_vals

##########################################################################


class AdaptiveLossWeighter:
    """Incremental adaptive loss weighter with configurable behavior and logging."""

    def __init__(self, initial_weights, adaptation_rate=0.05, min_weight=0.1, max_weight=3.0, hf_threshold=0.01, time_scale=2000):
        self.weights = {k: float(v) for k, v in initial_weights.items()}
        self.base_weights = {k: float(v) for k, v in initial_weights.items()}
        self.adaptation_rate = adaptation_rate
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.hf_threshold = hf_threshold
        self.time_scale = time_scale
        self.error_history = []

    def update_weights(self, spectral_errors, current_losses, epoch=0):
        total_energy = sum(spectral_errors.values()) if spectral_errors else 0.0
        n_bands = len(spectral_errors) if spectral_errors else 0
        if n_bands == 0:
            return

        high_start = n_bands // 2
        high_energy = sum(spectral_errors.get(f'band_{i}', 0.0) for i in range(high_start, n_bands))
        high_ratio = high_energy / (total_energy + 1e-12)

        old_weights = self.weights.copy()

        time_factor = 1.0 + min(epoch / max(1, self.time_scale), 1.0)

        # incorporate the current PDE loss into the PDE weight: we want to
        # reduce the PDE loss toward zero, so include the PDE loss value
        # (from current_losses) into the weight baseline before other tweaks.
        pde_loss_val = float(current_losses.get('pde', 0.0))
        # pde_val is the base weight magnitude: pde_loss + time factor
        # pde_val = pde_loss_val + time_factor
        pde_val = pde_loss_val

        # apply an HF-driven small boost if the high-frequency ratio is above threshold
        if high_ratio > self.hf_threshold:
            hf_boost = high_ratio * 2.0 * self.adaptation_rate * (1.0 + time_factor - 1.0)
            pde_val = min(pde_val + hf_boost, self.max_weight)

        bc_val = max(self.base_weights.get('bc', 1.0) / time_factor, self.min_weight)
        ic_val = max(self.base_weights.get('ic', 1.0) / time_factor, self.min_weight)

        self.weights['pde'] = float(pde_val)
        self.weights['bc'] = float(bc_val)
        self.weights['ic'] = float(ic_val)
        # self._normalize_weights()

        changed = any(abs(self.weights[k] - old_weights.get(k, 0.0)) > 1e-9 for k in self.weights)
        self.error_history.append({
            'epoch': epoch,
            'high_freq_ratio': high_ratio,
            'weights': self.weights.copy(),
            'losses': current_losses.copy(),
            'changed': changed
        })

        # if changed:
            # print(f"[AdaptiveLossWeighter] epoch={epoch} hf_ratio={high_ratio:.4f} weights={self.weights}")

    def _normalize_weights(self):
        total = sum(self.weights.values())
        if total > 0:
            for key in self.weights:
                # normalize to sum to 3 (previously sum to 1)
                self.weights[key] = self.weights[key] / total * 3.0


class AdaptiveSpectralPINN(nn.Module):
    """Adaptive PINN with improved update rules and more frequent adaptation"""

    def __init__(self, layers=[2, 50, 50, 50, 1], nu=0.01, domain_bounds=[(-1, 1)]):
        super(AdaptiveSpectralPINN, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers)-1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))

        self.nu = nu
        self.spectral_analyzer = SpectralAnalyzer(domain_bounds, n_bands=12)
        self.loss_weighter = AdaptiveLossWeighter(
            initial_weights={'pde': 1.0, 'bc': 1.0, 'ic': 1.0},
            adaptation_rate=0.05,
            max_weight=2.5,
            hf_threshold=0.01,
            time_scale=2000
        )

        # ensemble analyzer (novel analyzers) — kept optional and resilient
        try:
            self.ensemble_analyzer = EnsembleNovelAnalyzer()
        except Exception:
            self.ensemble_analyzer = None

        # coefficient for auxiliary ensemble-driven loss (user-configurable)
        self.ensemble_loss_coeff = 0.1

        self.history = {
            'total_loss': [], 'pde_loss': [], 'bc_loss': [], 'ic_loss': [],
            'weights': [], 'spectral_errors': [], 'high_freq_ratios': [], 'ensemble_boosts': [],
            'ensemble_losses': [], 'ensemble_weight_vals': []
        }

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.tanh(layer(x))
        x = self.layers[-1](x)
        return x

    def compute_pde_residual(self, x, u):
        u.requires_grad_(True)
        grad_u = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
        u_x = grad_u[:, 0:1]
        u_t = grad_u[:, 1:2]

        u_x.requires_grad_(True)
        grad_u_x = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        u_xx = grad_u_x[:, 0:1]

        residual = u_t + u * u_x - self.nu * u_xx
        return residual

    def compute_adaptive_loss(self, data_dict, epoch):
        x_colloc = data_dict['collocation']
        u_colloc = self.forward(x_colloc)
        residual = self.compute_pde_residual(x_colloc, u_colloc)
        pde_loss = torch.mean(residual**2)

        x_bc = data_dict['boundary_points']
        u_bc_pred = self.forward(x_bc)
        u_bc_true = data_dict['boundary_values'].unsqueeze(1)
        bc_loss = torch.mean((u_bc_pred - u_bc_true)**2)

        x_ic = data_dict['initial_points']
        u_ic_pred = self.forward(x_ic)
        u_ic_true = data_dict['initial_values'].unsqueeze(1)
        ic_loss = torch.mean((u_ic_pred - u_ic_true)**2)

        if epoch % 20 == 0:
            with torch.no_grad():
                spectral_errors = self.spectral_analyzer.compute_spectral_errors(residual, x_colloc)
                current_losses = {'pde': pde_loss.item(), 'bc': bc_loss.item(), 'ic': ic_loss.item()}

                self.history['spectral_errors'].append(spectral_errors)
                total_energy = sum(spectral_errors.values()) if spectral_errors else 0.0
                high_freq_energy = sum([spectral_errors.get(f'band_{i}', 0.0) for i in range(8, 12)])
                high_freq_ratio = high_freq_energy / (total_energy + 1e-8)
                self.history['high_freq_ratios'].append(high_freq_ratio)

                pre_weights = self.loss_weighter.weights.copy()
                # print(f"[AdaptiveSpectralPINN] epoch={epoch} total_energy={total_energy:.6e} hf_ratio={high_freq_ratio:.6f} pre_weights={pre_weights}")

                self.loss_weighter.update_weights(spectral_errors, current_losses, epoch)

                # compute ensemble boost (novel analyzers) and log
                ensemble_boost = None
                ensemble_weight_vals = {}
                try:
                    if self.ensemble_analyzer is not None:
                        # returns (tensor_boost, boosts_dict, weight_vals)
                        ensemble_boost, ensemble_metrics, ensemble_weight_vals = self.ensemble_analyzer.compute_ensemble_boost(residual, x_colloc, epoch)
                    else:
                        ensemble_boost = torch.tensor(1.0, dtype=torch.float32, device=residual.device)
                except Exception:
                    ensemble_boost = torch.tensor(1.0, dtype=torch.float32, device=residual.device)

                # log a float copy and weight values
                try:
                    self.history['ensemble_boosts'].append(float(ensemble_boost.detach().cpu().item()))
                except Exception:
                    self.history['ensemble_boosts'].append(1.0)
                self.history['ensemble_weight_vals'].append(ensemble_weight_vals)

                # we'll compute ensemble auxiliary loss below (outside no_grad)

                post_weights = self.loss_weighter.weights.copy()
                # if post_weights != pre_weights:
                    # print(f"[AdaptiveSpectralPINN] epoch={epoch} POST-UPDATE weights={post_weights}")
        else:
            spectral_errors = self.history['spectral_errors'][-1] if self.history['spectral_errors'] else {}
            # if no new ensemble boost this epoch, reuse last one if present (create tensor)
            if self.history['ensemble_boosts']:
                last_boost = float(self.history['ensemble_boosts'][-1])
                ensemble_boost = torch.tensor(last_boost, dtype=torch.float32, device=residual.device)
            else:
                ensemble_boost = torch.tensor(1.0, dtype=torch.float32, device=residual.device)

        weights = self.loss_weighter.weights

        # Keep ensemble boost as a separate auxiliary loss so it can be used for
        # general-purpose objectives (not only HF error minimization). We compute
        # an auxiliary loss by multiplying the (differentiable) ensemble boost
        # tensor with the current PDE loss. Users can tune ensemble_loss_coeff
        # to control its influence.
        try:
            # ensure ensemble_boost is a torch tensor (it may already be one)
            if not isinstance(ensemble_boost, torch.Tensor):
                ensemble_boost = torch.tensor(float(ensemble_boost), device=pde_loss.device)
        except Exception:
            ensemble_boost = torch.tensor(1.0, dtype=torch.float32, device=pde_loss.device)

        ensemble_aux_loss = ensemble_boost * pde_loss

        # primary weighted losses remain governed by the loss_weighter (unchanged)
        total_loss = (weights['pde'] * pde_loss + weights['bc'] * bc_loss + weights['ic'] * ic_loss)

        # add ensemble auxiliary loss scaled by a user-configurable coefficient
        total_loss = total_loss + float(self.ensemble_loss_coeff) * ensemble_aux_loss

        # log ensemble loss
        try:
            self.history['ensemble_losses'].append(float((float(self.ensemble_loss_coeff) * ensemble_aux_loss).detach().cpu().item()))
        except Exception:
            self.history['ensemble_losses'].append(0.0)

        self.history['total_loss'].append(total_loss.item())
        self.history['pde_loss'].append(pde_loss.item())
        self.history['bc_loss'].append(bc_loss.item())
        self.history['ic_loss'].append(ic_loss.item())
        # store weights (effective_pde is the current PDE weight from the weighter)
        wcopy = weights.copy()
        wcopy['effective_pde'] = float(weights.get('pde', 1.0))
        try:
            wcopy['ensemble_boost'] = float(ensemble_boost.detach().cpu().item())
        except Exception:
            wcopy['ensemble_boost'] = float(ensemble_boost)
        self.history['weights'].append(wcopy)

        return total_loss


class FixedAdaptivePINN(nn.Module):
    """Fixed version with incremental weight updates (debugging)"""
    def __init__(self, layers=[2, 50, 50, 50, 1], nu=0.01, domain_bounds=[(-1, 1)]):
        super(FixedAdaptivePINN, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers)-1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))

        self.nu = nu
        self.spectral_analyzer = SpectralAnalyzer(domain_bounds, n_bands=12)

        self.weights = {'pde': 1.0, 'bc': 1.0, 'ic': 1.0}
        self.base_weights = {'pde': 1.0, 'bc': 1.0, 'ic': 1.0}
        self.adaptation_count = 0

        self.history = {
            'total_loss': [], 'pde_loss': [], 'bc_loss': [], 'ic_loss': [],
            'weights': [], 'high_freq_ratios': [], 'adaptation_events': []
        }

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.tanh(layer(x))
        x = self.layers[-1](x)
        return x

    def compute_pde_residual(self, x, u):
        u.requires_grad_(True)
        grad_u = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
        u_x = grad_u[:, 0:1]
        u_t = grad_u[:, 1:2]

        u_x.requires_grad_(True)
        grad_u_x = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        u_xx = grad_u_x[:, 0:1]

        residual = u_t + u * u_x - self.nu * u_xx
        return residual

    def update_weights_incremental(self, high_freq_ratio, epoch):
        # print(f"  [FixedAdaptivePINN] Current weights: {self.weights}")
        # print(f"  [FixedAdaptivePINN] High-freq ratio: {high_freq_ratio:.6f}")

        old_weights = self.weights.copy()

        time_factor = 1.0 + min(epoch / 2000, 1.0)
        self.weights['pde'] = self.base_weights['pde'] * time_factor

        if high_freq_ratio > 0.01:
            hf_boost = high_freq_ratio * 2.0
            self.weights['pde'] += hf_boost
            # print(f"  🔥 High-freq boost: +{hf_boost:.3f}")

        self.weights['bc'] = max(self.base_weights['bc'] / time_factor, 0.2)
        self.weights['ic'] = max(self.base_weights['ic'] / time_factor, 0.2)

        total = sum(self.weights.values())
        for key in self.weights:
            # normalize to sum to 3
            self.weights[key] = self.weights[key] / (total + 1e-12) * 3.0

        weight_changed = any(abs(self.weights[key] - old_weights[key]) > 1e-6 for key in self.weights)
        if weight_changed:
            self.adaptation_count += 1
            # print(f"  ✅ WEIGHTS UPDATED: {old_weights} -> {self.weights}")
        else:
            pass
            # print(f"  ⚠️  Weights unchanged")

    def compute_adaptive_loss(self, data_dict, epoch):
        x_colloc = data_dict['collocation']
        u_colloc = self.forward(x_colloc)
        residual = self.compute_pde_residual(x_colloc, u_colloc)
        pde_loss = torch.mean(residual**2)

        x_bc = data_dict['boundary_points']
        u_bc_pred = self.forward(x_bc)
        u_bc_true = data_dict['boundary_values'].unsqueeze(1)
        bc_loss = torch.mean((u_bc_pred - u_bc_true)**2)

        x_ic = data_dict['initial_points']
        u_ic_pred = self.forward(x_ic)
        u_ic_true = data_dict['initial_values'].unsqueeze(1)
        ic_loss = torch.mean((u_ic_pred - u_ic_true)**2)

        if epoch % 20 == 0:
            with torch.no_grad():
                spectral_errors = self.spectral_analyzer.compute_spectral_errors(residual, x_colloc)
                total_energy = sum(spectral_errors.values()) if spectral_errors else 0.0
                high_freq_energy = sum([spectral_errors.get(f'band_{i}', 0.0) for i in range(len(spectral_errors)//2, len(spectral_errors))])
                high_freq_ratio = high_freq_energy / (total_energy + 1e-8)
                print(f"[FixedAdaptivePINN] epoch={epoch} total_energy={total_energy:.6e} hf_ratio={high_freq_ratio:.6f}")
                self.update_weights_incremental(high_freq_ratio, epoch)
                self.history['high_freq_ratios'].append(high_freq_ratio)
                self.history['adaptation_events'].append(self.adaptation_count)

        total_loss = (self.weights['pde'] * pde_loss + self.weights['bc'] * bc_loss + self.weights['ic'] * ic_loss)

        self.history['total_loss'].append(total_loss.item())
        self.history['pde_loss'].append(pde_loss.item())
        self.history['bc_loss'].append(bc_loss.item())
        self.history['ic_loss'].append(ic_loss.item())
        self.history['weights'].append(self.weights.copy())

        return total_loss


def train_adaptive_pinn(data_dict, n_epochs=5000, lr=0.001, test_data=None, eval_every=1):
    model = AdaptiveSpectralPINN(domain_bounds=[(-1, 1)])
    # use separate param groups: ensemble params get smaller LR and light weight decay
    ensemble_params = []
    try:
        if model.ensemble_analyzer is not None:
            ensemble_params = list(model.ensemble_analyzer.parameters())
    except Exception:
        ensemble_params = []

    if len(ensemble_params) > 0:
        # base params = all params not in ensemble
        ensemble_set = set([id(p) for p in ensemble_params])
        base_params = [p for p in model.parameters() if id(p) not in ensemble_set]
        optimizer = torch.optim.Adam([
            {'params': base_params},
            {'params': ensemble_params, 'lr': lr * 0.1, 'weight_decay': 1e-4}
        ], lr=lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # prepare history keys for per-epoch evaluation if not present
    if 'mse_epoch' not in model.history:
        model.history['mse_epoch'] = []
    if 'l2_epoch' not in model.history:
        model.history['l2_epoch'] = []

    print("Training Adaptive Spectral PINN...")
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        total_loss = model.compute_adaptive_loss(data_dict, epoch)
        total_loss.backward()

        # gradient clipping to stabilize late-stage training
        try:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        except Exception:
            pass

        optimizer.step()

        # optional per-epoch test evaluation
        if test_data is not None and (eval_every is not None) and (epoch % max(1, eval_every) == 0):
            try:
                # use local evaluate_model (from vis_eval import evaluate_model at top of file)
                _, mse_val, l2_val = evaluate_model(model, test_data)
                model.history['mse_epoch'].append(float(mse_val))
                model.history['l2_epoch'].append(float(l2_val))
            except Exception:
                model.history['mse_epoch'].append(float('nan'))
                model.history['l2_epoch'].append(float('nan'))

        if epoch % 200 == 0:
            current_weights = model.loss_weighter.weights
            current_losses = {'total': total_loss.item(), 'pde': model.history['pde_loss'][-1], 'bc': model.history['bc_loss'][-1], 'ic': model.history['ic_loss'][-1]}
            high_freq_ratio = model.history['high_freq_ratios'][-1] if model.history['high_freq_ratios'] else 0.0
            print(f"Epoch {epoch}: Total Loss = {total_loss.item():.6f}")
            print(f"  Weights - PDE: {current_weights['pde']:.3f}, BC: {current_weights['bc']:.3f}, IC: {current_weights['ic']:.3f}")
            print(f"  High-freq ratio: {high_freq_ratio:.3f}")

    return model, model.history


def train_fixed_pinn(data_dict, n_epochs=400, lr=0.001):
    model = FixedAdaptivePINN(domain_bounds=[(-1, 1)])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print("="*60)
    print("FIXED INCREMENTAL ADAPTATION (debug)")
    print("="*60)

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        total_loss = model.compute_adaptive_loss(data_dict, epoch)
        total_loss.backward()
        optimizer.step()

        if epoch % 200 == 0 and epoch > 0:
            print(f"\nEpoch {epoch}: Total Loss = {total_loss.item():.6f}")
            print(f"Current weights: {model.weights}")
            print(f"Total adaptation events: {model.adaptation_count}")

    return model


def main():
    print("\n" + "="*50)
    print("STARTING ADAPTIVE FREQUENCY APPROACH")
    print("="*50)

    data_path = os.path.join('data', 'burgers_dataset.npz')

    if os.path.exists(data_path):
        print(f"Loading dataset from {data_path}...")
        train_data, test_data = load_dataset(data_path)
    else:
        print("Generating training and test data with exact Cole-Hopf solution...")
        data_gen = BurgersDataGenerator(nu=0.01, L=1.0, T=1.0)

        train_data = data_gen.generate_training_data(
            n_collocation=1000, 
            n_boundary=100, 
            n_initial=100
        )

        test_data = data_gen.generate_test_data(n_x=100, n_t=50)

        print(f"Saving generated dataset to {data_path}...")
        save_dataset(data_path, train_data, test_data)

    adaptive_model = train_adaptive_pinn(train_data, n_epochs=5000, lr=0.001)

    print("\nEvaluating adaptive model...")
    adaptive_predictions, adaptive_mse, adaptive_rel_l2 = evaluate_model(adaptive_model, test_data)

    print("\n" + "="*50)
    print("COMPARISON RESULTS")
    print("="*50)
    print(f"Baseline PINN:")
    print(f"  Test MSE: {0.457547:.6f}")
    print(f"  Relative L2 Error: {1.134239:.6f}")
    print(f"Adaptive Spectral PINN:")
    print(f"  Test MSE: {adaptive_mse:.6f}")
    print(f"  Relative L2 Error: {adaptive_rel_l2:.6f}")
    print(f"Improvement: {((1.134239 - adaptive_rel_l2) / 1.134239 * 100):.1f}%")

    plot_training_history(adaptive_model.history, "Adaptive Spectral PINN Training History")
    plot_solution_comparison(adaptive_model, test_data, "Adaptive Spectral PINN Solution")

    plt.figure(figsize=(10, 4))
    weights_history = adaptive_model.history['weights']
    epochs = range(len(weights_history))
    pde_weights = [w['pde'] for w in weights_history]
    bc_weights = [w['bc'] for w in weights_history] 
    ic_weights = [w['ic'] for w in weights_history]

    plt.plot(epochs, pde_weights, label='PDE Weight', linewidth=2)
    plt.plot(epochs, bc_weights, label='BC Weight', linewidth=2)
    plt.plot(epochs, ic_weights, label='IC Weight', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Weight')
    plt.title('Adaptive Loss Weight Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == '__main__':
    main()
