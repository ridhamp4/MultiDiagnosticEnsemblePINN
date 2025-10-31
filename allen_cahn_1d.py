import torch
import numpy as np
from scipy import fft
import matplotlib.pyplot as plt

class AllenCahnDataGenerator:
    """Generate exact Allen-Cahn solutions using spectral method"""
    
    def __init__(self, epsilon=0.001, alpha=5.0, L=1.0, T=1.0):
        self.epsilon = epsilon
        self.alpha = alpha
        self.L = L
        self.T = T
    
    def spectral_solve_allen_cahn(self, u0, dt, n_steps, n_x):
        """Solve Allen-Cahn equation using pseudo-spectral method with ETD-RK4"""
        
        # Spatial grid and wave numbers
        x = np.linspace(-self.L, self.L, n_x, endpoint=False)
        k = 2 * np.pi * fft.fftfreq(n_x, d=(2*self.L/n_x))
        
        # Precompute linear operator in Fourier space
        L_operator = -self.epsilon * k**2 - self.alpha
        
        # Exponential time-stepping coefficients for ETD-RK4
        E = np.exp(dt * L_operator)
        E2 = np.exp(dt * L_operator / 2)
        
        M1 = (np.exp(dt * L_operator / 2) - 1) / (dt * L_operator / 2)
        M2 = (np.exp(dt * L_operator) - 1) / (dt * L_operator)
        
        # Initialize solution
        u = u0.copy()
        u_history = [u0.copy()]
        
        # Time stepping
        for i in range(n_steps):
            u_hat = fft.fft(u)
            
            # ETD-RK4 scheme
            # Stage 1
            N1 = -self.alpha * fft.fft(u**3)
            a_hat = E2 * u_hat + dt * M1 * N1
            
            # Stage 2
            a = np.real(fft.ifft(a_hat))
            N2 = -self.alpha * fft.fft(a**3)
            b_hat = E2 * u_hat + dt * M1 * N2
            
            # Stage 3
            b = np.real(fft.ifft(b_hat))
            N3 = -self.alpha * fft.fft(b**3)
            c_hat = E2 * a_hat + dt * M1 * (2 * N3 - N1)
            
            # Stage 4
            c = np.real(fft.ifft(c_hat))
            N4 = -self.alpha * fft.fft(c**3)
            
            # Final update
            u_hat_new = (E * u_hat + dt * (
                (N1 * (-4 - dt * L_operator + E * (4 - 3 * dt * L_operator + (dt * L_operator)**2))) / (2 * (dt * L_operator)**2) +
                (N2 + N3) * (2 + dt * L_operator + E * (-2 + dt * L_operator)) / ((dt * L_operator)**2) +
                N4 * (-4 - 3 * dt * L_operator - (dt * L_operator)**2 + E * (4 - dt * L_operator)) / (2 * (dt * L_operator)**2)
            ))
            
            u = np.real(fft.ifft(u_hat_new))
            u_history.append(u.copy())
            
        return np.array(u_history), x
    
    def generate_exact_solution(self, n_x=256, n_t=100, initial_condition='cosine'):
        """Generate exact solution using spectral method"""
        
        # Temporal grid
        t = np.linspace(0, self.T, n_t)
        dt = t[1] - t[0]
        
        # Spatial grid
        x = np.linspace(-self.L, self.L, n_x, endpoint=False)
        
        # Initial condition
        if initial_condition == 'cosine':
            u0 = x**2 * np.cos(np.pi * x)  # Smooth initial condition
        elif initial_condition == 'random_phases':
            # Random phases between -1 and 1 (more challenging)
            u0 = np.random.choice([-0.9, 0.9], size=n_x)
            # Add some smoothing
            from scipy.ndimage import gaussian_filter1d
            u0 = gaussian_filter1d(u0.astype(float), sigma=2)
        elif initial_condition == 'tanh_profile':
            # Sharp interface initial condition
            u0 = np.tanh(x / np.sqrt(2 * self.epsilon))
        else:
            u0 = x**2 * np.cos(np.pi * x)  # Default
        
        # Solve using spectral method
        print("Solving Allen-Cahn equation with spectral method...")
        u_history, x = self.spectral_solve_allen_cahn(u0, dt, n_t-1, n_x)
        
        # Create meshgrid for output
        T_grid, X_grid = np.meshgrid(t, x, indexing='ij')
        U_exact = u_history
        
        return X_grid, T_grid, U_exact
    
    def generate_training_data(self, n_collocation=2000, n_boundary=200, n_initial=200, 
                             initial_condition='cosine'):
        """Generate training data for Allen-Cahn equation"""
        
        # Generate exact solution for interpolation
        X_grid, T_grid, U_exact = self.generate_exact_solution(n_x=256, n_t=100, 
                                                              initial_condition=initial_condition)
        
        # Collocation points (interior)
        x_colloc = torch.FloatTensor(n_collocation, 1).uniform_(-self.L, self.L)
        t_colloc = torch.FloatTensor(n_collocation, 1).uniform_(0, self.T)
        x_train = torch.cat([x_colloc, t_colloc], dim=1)
        
        # Get exact values at collocation points using interpolation
        from scipy.interpolate import RegularGridInterpolator
        x_points = np.linspace(-self.L, self.L, 256, endpoint=False)
        t_points = np.linspace(0, self.T, 100)
        
        # Create interpolator (note: U_exact has shape (n_t, n_x))
        interpolator = RegularGridInterpolator((t_points, x_points), U_exact, 
                                             method='cubic', bounds_error=False, 
                                             fill_value=None)
        
        # Convert collocation points to numpy for interpolation
        x_colloc_np = x_train.detach().numpy()
        collocation_values_np = interpolator(x_colloc_np[:, [1, 0]])  # Swap columns for (t,x) ordering
        
        collocation_values = torch.FloatTensor(collocation_values_np)
        
        # Boundary points (spatial boundaries - periodic BC)
        x_boundary = torch.cat([
            -self.L * torch.ones(n_boundary//2, 1),  # x = -L
            self.L * torch.ones(n_boundary//2, 1)    # x = L
        ])
        t_boundary = torch.FloatTensor(n_boundary, 1).uniform_(0, self.T)
        x_bc = torch.cat([x_boundary, t_boundary], dim=1)
        
        # For periodic BC, values at x=-L and x=L should be equal
        # Get exact boundary values using interpolation
        x_bc_np = x_bc.detach().numpy()
        boundary_values_np = interpolator(x_bc_np[:, [1, 0]])
        boundary_values = torch.FloatTensor(boundary_values_np)
        
        # Initial condition points (t=0)
        x_initial = torch.FloatTensor(n_initial, 1).uniform_(-self.L, self.L)
        t_initial = torch.zeros(n_initial, 1)
        x_ic = torch.cat([x_initial, t_initial], dim=1)
        
        # Get exact initial condition
        x_ic_np = x_ic.detach().numpy()
        initial_values_np = interpolator(x_ic_np[:, [1, 0]])
        initial_values = torch.FloatTensor(initial_values_np)
        
        return {
            'collocation': x_train,
            'collocation_values': collocation_values,
            'boundary_points': x_bc,
            'boundary_values': boundary_values,
            'initial_points': x_ic,
            'initial_values': initial_values
        }
    
    def generate_test_data(self, n_x=256, n_t=100, initial_condition='cosine'):
        """Generate high-resolution test data using exact spectral solution"""
        
        print("Generating exact test solution...")
        X, T, U_exact = self.generate_exact_solution(n_x, n_t, initial_condition)
        
        # Flatten for PINN format
        x_test = torch.FloatTensor(np.column_stack([X.ravel(), T.ravel()]))
        u_test = torch.FloatTensor(U_exact.ravel())
        
        return x_test, u_test, X, T, U_exact

    def plot_exact_solution(self, n_x=256, n_t=100, initial_condition='cosine'):
        """Plot the exact Allen-Cahn solution"""
        
        X, T, U_exact = self.generate_exact_solution(n_x, n_t, initial_condition)
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(131)
        plt.contourf(T, X, U_exact, levels=50, cmap='RdBu_r')
        plt.colorbar()
        plt.xlabel('Time (t)')
        plt.ylabel('Space (x)')
        plt.title('Allen-Cahn Exact Solution')
        
        plt.subplot(132)
        plt.plot(X[0], U_exact[0], 'b-', linewidth=2, label='t=0')
        plt.plot(X[n_t//2], U_exact[n_t//2], 'r-', linewidth=2, label=f't={self.T/2:.2f}')
        plt.plot(X[-1], U_exact[-1], 'g-', linewidth=2, label=f't={self.T:.2f}')
        plt.xlabel('Space (x)')
        plt.ylabel('u(x,t)')
        plt.legend()
        plt.title('Solution Profiles')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(133)
        plt.imshow(U_exact, aspect='auto', cmap='RdBu_r', 
                  extent=[-self.L, self.L, self.T, 0])
        plt.colorbar()
        plt.xlabel('Space (x)')
        plt.ylabel('Time (t)')
        plt.title('Space-Time Plot')
        
        plt.tight_layout()
        plt.show()
        
        return X, T, U_exact