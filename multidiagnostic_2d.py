import torch.nn as nn
from ensemble import AdaptiveLossWeighter, SpectralAnalyzer, EnsembleNovelAnalyzer
import torch

class EnsembleAllenCahnPINN2D(nn.Module):
    """2D Allen-Cahn PINN with your complete ensemble analysis framework"""
    
    def __init__(self, layers=[3, 64, 64, 64, 1], epsilon=0.001, alpha=5.0, domain_bounds=[(-1, 1), (-1, 1)]):
        super(EnsembleAllenCahnPINN2D, self).__init__()
        
        # Base network - input dimension changed to 3 (x, y, t)
        self.layers = nn.ModuleList()
        for i in range(len(layers)-1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
            
        self.epsilon = epsilon
        self.alpha = alpha
        
        # YOUR ENSEMBLE COMPONENTS (from your original code)
        self.spectral_analyzer = SpectralAnalyzer(domain_bounds, n_bands=12)
        
        # Loss weighter (adaptive weighting) - include supervised data term ('sup')
        self.loss_weighter = AdaptiveLossWeighter(
            initial_weights={'pde': 1.0, 'bc': 1.0, 'ic': 1.0, 'sup': 2.0},
            adaptation_rate=0.05,
            max_weight=2.5, 
            hf_threshold=0.01,
            time_scale=2000
        )
        
        # NOVEL ANALYZERS (your key innovation)
        try:
            self.ensemble_analyzer = EnsembleNovelAnalyzer()
        except:
            self.ensemble_analyzer = None
            
        self.ensemble_loss_coeff = 0.1  # Weight for ensemble-driven loss
        
        # Training history
        self.history = {
            'total_loss': [], 'pde_loss': [], 'bc_loss': [], 'ic_loss': [],
            'weights': [], 'spectral_errors': [], 'high_freq_ratios': [],
            'ensemble_boosts': [], 'ensemble_losses': [], 'ensemble_weight_vals': [],
            'topological_features': [], 'geometric_metrics': [], 'information_metrics': [],
            'sup_loss': [],
            'sup_n': []
        }
        # internal flag to avoid spamming warning every epoch
        self._warned_no_sup = False
    
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.tanh(layer(x))
        x = self.layers[-1](x)
        return x
    
    def compute_allen_cahn_residual(self, x, u):
        """2D Allen-Cahn residual: u_t - ε(∇²u) + α(u³ - u) = 0"""
        u.requires_grad_(True)
        
        # Compute first derivatives
        grad_u = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), 
                                   create_graph=True, retain_graph=True)[0]
        u_t = grad_u[:, 2:3]  # t is the third coordinate (x, y, t)
        u_x = grad_u[:, 0:1]
        u_y = grad_u[:, 1:2]
        
        # Compute second derivatives for Laplacian
        u_x.requires_grad_(True)
        grad_u_x = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), 
                                     create_graph=True, retain_graph=True)[0]
        u_xx = grad_u_x[:, 0:1]
        
        u_y.requires_grad_(True)
        grad_u_y = torch.autograd.grad(u_y, x, grad_outputs=torch.ones_like(u_y), 
                                     create_graph=True)[0]
        u_yy = grad_u_y[:, 1:2]
        
        # 2D Laplacian
        laplacian_u = u_xx + u_yy
        
        # 2D Allen-Cahn residual
        residual = u_t - self.epsilon * laplacian_u + self.alpha * (u**3 - u)
        return residual
    
    def compute_ensemble_adaptive_loss(self, data_dict, epoch):
        """
        Compute loss with FULL ensemble adaptation
        This is where your novel analyzers actively guide the training
        """
        
        # Get predictions and residuals
        x_colloc = data_dict['collocation']
        u_colloc = self.forward(x_colloc)
        residual = self.compute_allen_cahn_residual(x_colloc, u_colloc)
        
        # Basic loss components
        pde_loss = torch.mean(residual**2)
        
        # Boundary loss (all boundaries)
        x_bc = data_dict['boundary_points']
        u_bc_pred = self.forward(x_bc)
        u_bc_true = data_dict['boundary_values'].unsqueeze(1)
        bc_loss = torch.mean((u_bc_pred - u_bc_true)**2)
        
        # Initial condition loss
        x_ic = data_dict['initial_points']
        u_ic_pred = self.forward(x_ic)
        u_ic_true = data_dict['initial_values'].unsqueeze(1)
        ic_loss = torch.mean((u_ic_pred - u_ic_true)**2)
        
        # ENSEMBLE ADAPTATION (every 20 epochs)
        if epoch % 20 == 0:
            with torch.no_grad():
                # 1. SPECTRAL ANALYSIS
                spectral_errors = self.spectral_analyzer.compute_spectral_errors(residual, x_colloc)
                current_losses = {'pde': pde_loss.item(), 'bc': bc_loss.item(), 'ic': ic_loss.item()}
                # include supervised loss in the current_losses dict if available (filled below)
                
                self.history['spectral_errors'].append(spectral_errors)
                total_energy = sum(spectral_errors.values()) if spectral_errors else 0.0
                high_freq_energy = sum([spectral_errors.get(f'band_{i}', 0.0) for i in range(8, 12)])
                high_freq_ratio = high_freq_energy / (total_energy + 1e-8)
                self.history['high_freq_ratios'].append(high_freq_ratio)
                
                # Update adaptive weights based on spectral analysis
                self.loss_weighter.update_weights(spectral_errors, current_losses, epoch)
                
                # 2. ENSEMBLE ANALYSIS (YOUR KEY INNOVATION)
                ensemble_weight_vals = {}
                ensemble_metrics = {}
                ensemble_term = torch.tensor(0.0, device=residual.device)

                if self.ensemble_analyzer is not None:
                    # compute_ensemble_boost historically returned (boost, metrics, weight_vals)
                    _, ensemble_metrics, ensemble_weight_vals = \
                        self.ensemble_analyzer.compute_ensemble_boost(residual, x_colloc, epoch)

                    # Log individual analyzer metrics for analysis (keep same fields where present)
                    if 'topological' in ensemble_metrics:
                        self.history['topological_features'].append(ensemble_metrics['topological'])
                    if 'geometric' in ensemble_metrics:
                        self.history['geometric_metrics'].append(ensemble_metrics['geometric'])
                    if 'information' in ensemble_metrics:
                        # we record information metric but explicitly do not include it in the ensemble term
                        self.history['information_metrics'].append(ensemble_metrics['information'])

                    # Build ensemble_term = mean of all metrics excluding 'information'
                    metric_vals = []
                    for k, v in ensemble_metrics.items():
                        if k == 'information':
                            continue
                        try:
                            metric_vals.append(float(v))
                        except Exception:
                            # ignore non-scalar metric entries
                            pass
                    if metric_vals:
                        ensemble_term = torch.tensor(sum(metric_vals) / len(metric_vals), device=residual.device)

                # record ensemble_term and weight vals for later inspection
                self.history.setdefault('ensemble_terms', []).append(float(ensemble_term.item()))
                self.history['ensemble_weight_vals'].append(ensemble_weight_vals)
        
        # Get current adaptive weights
        weights = self.loss_weighter.weights

        # Supervised data-driven L2 term (optional)
        # If explicit supervised points aren't provided, use all labeled training
        # points we do have (boundary + initial + any explicit supervised).
        sup_pts = data_dict.get('supervised_points', None)
        sup_vals = data_dict.get('supervised_values', None)

        # If user didn't provide supervised_points, build from BC and IC labels
        built_from_parts = False
        if sup_pts is None or sup_vals is None:
            parts_pts = []
            parts_vals = []
            if 'boundary_points' in data_dict and 'boundary_values' in data_dict:
                parts_pts.append(data_dict['boundary_points'])
                parts_vals.append(data_dict['boundary_values'])
            if 'initial_points' in data_dict and 'initial_values' in data_dict:
                parts_pts.append(data_dict['initial_points'])
                parts_vals.append(data_dict['initial_values'])
            # include collocation points if true values are provided (user requested)
            if 'collocation' in data_dict and ('collocation_values' in data_dict or 'collocation_u' in data_dict):
                parts_pts.append(data_dict['collocation'])
                # prefer 'collocation_values' name, otherwise fallback to 'collocation_u'
                if 'collocation_values' in data_dict:
                    parts_vals.append(data_dict['collocation_values'])
                else:
                    parts_vals.append(data_dict['collocation_u'])
            # also include any explicitly provided supervised arrays if present
            if 'supervised_points' in data_dict and 'supervised_values' in data_dict:
                parts_pts.append(data_dict['supervised_points'])
                parts_vals.append(data_dict['supervised_values'])

            if parts_pts and parts_vals:
                # concatenate parts along first dimension
                try:
                    # coerce to tensors and align device/dtype later in conversion block
                    sup_pts = torch.cat([p if torch.is_tensor(p) else torch.tensor(p) for p in parts_pts], dim=0)
                    sup_vals = torch.cat([v if torch.is_tensor(v) else torch.tensor(v) for v in parts_vals], dim=0)
                    built_from_parts = True
                except Exception:
                    # fallback: leave sup_pts/sup_vals as None to trigger safe path below
                    sup_pts = data_dict.get('supervised_points', None)
                    sup_vals = data_dict.get('supervised_values', None)

        sup_loss = torch.tensor(0.0, device=residual.device)
        sup_count = 0
        if sup_pts is not None and sup_vals is not None:
            # Coerce supervised inputs to tensors on the correct device/dtype
            try:
                if not torch.is_tensor(sup_pts):
                    sup_pts = torch.tensor(sup_pts, dtype=residual.dtype, device=residual.device)
                else:
                    sup_pts = sup_pts.to(residual.device, dtype=residual.dtype)

                if not torch.is_tensor(sup_vals):
                    sup_vals = torch.tensor(sup_vals, dtype=residual.dtype, device=residual.device)
                else:
                    sup_vals = sup_vals.to(residual.device, dtype=residual.dtype)

                # Ensure shape (N,1) for values
                if sup_vals.dim() == 1:
                    sup_vals = sup_vals.unsqueeze(1)

                # If pts are 1D (unlikely), ensure proper shape
                if sup_pts.dim() == 1:
                    sup_pts = sup_pts.unsqueeze(1)

                # Compute supervised MSE
                u_sup_pred = self.forward(sup_pts)
                sup_loss = torch.mean((u_sup_pred - sup_vals)**2)
                sup_count = int(sup_pts.shape[0]) if hasattr(sup_pts, 'shape') else 0
            except Exception as e:
                # Keep sup_loss as zero but emit a warning so it's visible in logs
                sup_loss = torch.tensor(0.0, device=residual.device)
                print(f"Warning: failed to compute supervised loss: {e}")

        # ENSEMBLE-DRIVEN LOSS COMPONENT (now a separate term, not a boost)
        # Use the last recorded ensemble_term if available, otherwise zero.
        if 'ensemble_terms' in self.history and self.history['ensemble_terms']:
            ensemble_term_val = torch.tensor(self.history['ensemble_terms'][-1], device=residual.device)
        else:
            ensemble_term_val = torch.tensor(0.0, device=residual.device)

        # TOTAL LOSS with ensemble guidance as a separate loss term and supervised term
        total_loss = (weights.get('pde', 1.0) * pde_loss + 
                     weights.get('bc', 1.0) * bc_loss + 
                     weights.get('ic', 1.0) * ic_loss +
                     weights.get('sup', 1.0) * sup_loss +
                     self.ensemble_loss_coeff * ensemble_term_val)
        
        # Logging
        self.history['total_loss'].append(total_loss.item())
        self.history['pde_loss'].append(pde_loss.item())
        self.history['bc_loss'].append(bc_loss.item())
        self.history['ic_loss'].append(ic_loss.item())
        # log supervised loss
        try:
            self.history['sup_loss'].append(float(sup_loss.item()))
        except Exception:
            self.history['sup_loss'].append(0.0)
        # log supervised sample count
        try:
            self.history['sup_n'].append(int(sup_count))
        except Exception:
            self.history['sup_n'].append(0)
        
        wcopy = weights.copy()
        # record the latest ensemble term value (additive term), not a multiplicative boost
        try:
            wcopy['ensemble_term'] = float(ensemble_term_val.item())
        except Exception:
            wcopy['ensemble_term'] = 0.0
        # include supervised weight in logged copy if present
        if 'sup' in weights:
            wcopy['sup'] = float(weights['sup'])
        self.history['weights'].append(wcopy)
        self.history['ensemble_losses'].append(
            float((self.ensemble_loss_coeff * ensemble_term_val).item())
        )
        # Print loss and supervised MSE for visibility during training
        try:
            # include supervised sample count for clarity
            print(f"Epoch {epoch}: loss={total_loss.item():.6e} mse={float(sup_loss.item()):.6e} sup_n={int(sup_count)}")
        except Exception:
            # if sup_loss or epoch aren't available for some reason, fall back to printing total loss
            print(f"Epoch {epoch}: loss={total_loss.item():.6e}")
        # Warn once if supervised count is zero so user notices missing labels
        if sup_count == 0 and not getattr(self, '_warned_no_sup', False):
            print("Warning: supervised sample count is 0 — no supervised points found in data_dict. "
                  "If you expect a data-driven MSE term, add 'supervised_points' and 'supervised_values' to data_dict, "
                  "or ensure boundary/initial labels are present so they can be used automatically.")
            self._warned_no_sup = True

        return total_loss