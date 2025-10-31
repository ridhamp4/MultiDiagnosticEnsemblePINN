import torch
import numpy as np
from scipy import fft
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class AllenCahnDataGenerator2D:
    """Generate exact 2D Allen-Cahn solutions using spectral method"""
    
    def __init__(self, epsilon=0.001, alpha=5.0, L=1.0, T=1.0):
        self.epsilon = epsilon
        self.alpha = alpha
        self.L = L
        self.T = T
    
    def spectral_solve_allen_cahn_2d(self, u0, dt, n_steps, n_x, n_y):
        """Solve 2D Allen-Cahn equation using pseudo-spectral method with ETD-RK4"""
        
        # Spatial grid and wave numbers
        x = np.linspace(-self.L, self.L, n_x, endpoint=False)
        y = np.linspace(-self.L, self.L, n_y, endpoint=False)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        kx = 2 * np.pi * fft.fftfreq(n_x, d=(2*self.L/n_x))
        ky = 2 * np.pi * fft.fftfreq(n_y, d=(2*self.L/n_y))
        KX, KY = np.meshgrid(kx, ky, indexing='ij')
        
        # Precompute linear operator in Fourier space
        K2 = KX**2 + KY**2
        L_operator = -self.epsilon * K2 - self.alpha
        
        # Exponential time-stepping coefficients for ETD-RK4
        E = np.exp(dt * L_operator)
        E2 = np.exp(dt * L_operator / 2)
        
        M1 = (np.exp(dt * L_operator / 2) - 1) / (dt * L_operator / 2)
        M2 = (np.exp(dt * L_operator) - 1) / (dt * L_operator)
        
        # Handle division by zero (where L_operator = 0)
        M1 = np.where(L_operator == 0, 1.0, M1)
        M2 = np.where(L_operator == 0, 1.0, M2)
        
        # Initialize solution
        u = u0.copy()
        u_history = [u0.copy()]
        
        # Time stepping
        for i in range(n_steps):
            u_hat = fft.fft2(u)
            
            # ETD-RK4 scheme
            # Stage 1
            N1 = -self.alpha * fft.fft2(u**3)
            a_hat = E2 * u_hat + dt * M1 * N1
            
            # Stage 2
            a = np.real(fft.ifft2(a_hat))
            N2 = -self.alpha * fft.fft2(a**3)
            b_hat = E2 * u_hat + dt * M1 * N2
            
            # Stage 3
            b = np.real(fft.ifft2(b_hat))
            N3 = -self.alpha * fft.fft2(b**3)
            c_hat = E2 * a_hat + dt * M1 * (2 * N3 - N1)
            
            # Stage 4
            c = np.real(fft.ifft2(c_hat))
            N4 = -self.alpha * fft.fft2(c**3)
            
            # Final update (simplified ETD-RK4 for stability)
            u_hat_new = E * u_hat + dt * (
                E2 * M1 * N1 + 
                2 * E2 * M1 * (N2 + N3) + 
                M1 * N4
            ) / 6
            
            u = np.real(fft.ifft2(u_hat_new))
            u_history.append(u.copy())
            
        return np.array(u_history), X, Y
    
    def generate_exact_solution(self, n_x=128, n_y=128, n_t=50, initial_condition='cosine'):
        """Generate exact 2D solution using spectral method"""
        
        # Temporal grid
        t = np.linspace(0, self.T, n_t)
        dt = t[1] - t[0]
        
        # Spatial grid
        x = np.linspace(-self.L, self.L, n_x, endpoint=False)
        y = np.linspace(-self.L, self.L, n_y, endpoint=False)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Initial condition
        if initial_condition == 'cosine':
            u0 = np.cos(np.pi * X) * np.cos(np.pi * Y)
        elif initial_condition == 'random_phases':
            # Random phases between -1 and 1
            u0 = np.random.choice([-0.9, 0.9], size=(n_x, n_y))
            # Add some smoothing
            from scipy.ndimage import gaussian_filter
            u0 = gaussian_filter(u0.astype(float), sigma=2)
        elif initial_condition == 'tanh_profile':
            # Circular interface
            r = np.sqrt(X**2 + Y**2)
            u0 = np.tanh((r - 0.5) / np.sqrt(2 * self.epsilon))
        elif initial_condition == 'bubble':
            # Multiple bubbles
            u0 = -np.ones((n_x, n_y))
            centers = [(-0.5, -0.5), (0.5, 0.5), (-0.5, 0.5), (0.5, -0.5)]
            for cx, cy in centers:
                r = np.sqrt((X - cx)**2 + (Y - cy)**2)
                bubble = np.tanh((0.2 - r) / np.sqrt(2 * self.epsilon))
                u0 = np.maximum(u0, bubble)
        else:
            u0 = np.cos(np.pi * X) * np.cos(np.pi * Y)  # Default
        
        # Solve using spectral method
        print("Solving 2D Allen-Cahn equation with spectral method...")
        u_history, X, Y = self.spectral_solve_allen_cahn_2d(u0, dt, n_t-1, n_x, n_y)
        
        # Create meshgrid for output
        T_grid = np.zeros((n_t, n_x, n_y))
        for i in range(n_t):
            T_grid[i] = t[i]
        
        return X, Y, T_grid, u_history
    
    def generate_training_data(self, n_collocation=5000, n_boundary=500, n_initial=500, 
                             initial_condition='cosine'):
        """Generate training data for 2D Allen-Cahn equation"""
        
        # Generate exact solution for interpolation
        X, Y, T, U_exact = self.generate_exact_solution(n_x=64, n_y=64, n_t=30, 
                                                       initial_condition=initial_condition)
        
        # Collocation points (interior)
        x_colloc = torch.FloatTensor(n_collocation, 1).uniform_(-self.L, self.L)
        y_colloc = torch.FloatTensor(n_collocation, 1).uniform_(-self.L, self.L)
        t_colloc = torch.FloatTensor(n_collocation, 1).uniform_(0, self.T)
        x_train = torch.cat([x_colloc, y_colloc, t_colloc], dim=1)
        
        # Get exact values at collocation points using interpolation
        from scipy.interpolate import RegularGridInterpolator
        x_points = np.linspace(-self.L, self.L, 64, endpoint=False)
        y_points = np.linspace(-self.L, self.L, 64, endpoint=False)
        t_points = np.linspace(0, self.T, 30)
        
        # Create interpolator (U_exact has shape (n_t, n_x, n_y))
        interpolator = RegularGridInterpolator((t_points, x_points, y_points), U_exact, 
                                             method='linear', bounds_error=False, 
                                             fill_value=None)
        
        # Convert collocation points to numpy for interpolation
        x_colloc_np = x_train.detach().numpy()
        collocation_values_np = interpolator(x_colloc_np[:, [2, 0, 1]])  # Order: (t, x, y)
        
        collocation_values = torch.FloatTensor(collocation_values_np)
        
        # Boundary points (spatial boundaries - periodic BC)
        n_boundary_each = n_boundary // 4
        
        # x = -L boundary
        x_b1 = -self.L * torch.ones(n_boundary_each, 1)
        y_b1 = torch.FloatTensor(n_boundary_each, 1).uniform_(-self.L, self.L)
        t_b1 = torch.FloatTensor(n_boundary_each, 1).uniform_(0, self.T)
        
        # x = L boundary  
        x_b2 = self.L * torch.ones(n_boundary_each, 1)
        y_b2 = torch.FloatTensor(n_boundary_each, 1).uniform_(-self.L, self.L)
        t_b2 = torch.FloatTensor(n_boundary_each, 1).uniform_(0, self.T)
        
        # y = -L boundary
        x_b3 = torch.FloatTensor(n_boundary_each, 1).uniform_(-self.L, self.L)
        y_b3 = -self.L * torch.ones(n_boundary_each, 1)
        t_b3 = torch.FloatTensor(n_boundary_each, 1).uniform_(0, self.T)
        
        # y = L boundary
        x_b4 = torch.FloatTensor(n_boundary_each, 1).uniform_(-self.L, self.L)
        y_b4 = self.L * torch.ones(n_boundary_each, 1)
        t_b4 = torch.FloatTensor(n_boundary_each, 1).uniform_(0, self.T)
        
        x_bc = torch.cat([
            torch.cat([x_b1, y_b1, t_b1], dim=1),
            torch.cat([x_b2, y_b2, t_b2], dim=1),
            torch.cat([x_b3, y_b3, t_b3], dim=1),
            torch.cat([x_b4, y_b4, t_b4], dim=1)
        ])
        
        # Get exact boundary values using interpolation
        x_bc_np = x_bc.detach().numpy()
        boundary_values_np = interpolator(x_bc_np[:, [2, 0, 1]])
        boundary_values = torch.FloatTensor(boundary_values_np)
        
        # Initial condition points (t=0)
        x_initial = torch.FloatTensor(n_initial, 1).uniform_(-self.L, self.L)
        y_initial = torch.FloatTensor(n_initial, 1).uniform_(-self.L, self.L)
        t_initial = torch.zeros(n_initial, 1)
        x_ic = torch.cat([x_initial, y_initial, t_initial], dim=1)
        
        # Get exact initial condition
        x_ic_np = x_ic.detach().numpy()
        initial_values_np = interpolator(x_ic_np[:, [2, 0, 1]])
        initial_values = torch.FloatTensor(initial_values_np)
        
        return {
            'collocation': x_train,
            'collocation_values': collocation_values,
            'boundary_points': x_bc,
            'boundary_values': boundary_values,
            'initial_points': x_ic,
            'initial_values': initial_values
        }
    
    def generate_test_data(self, n_x=128, n_y=128, n_t=50, initial_condition='cosine'):
        """Generate high-resolution test data using exact spectral solution"""
        
        print("Generating exact 2D test solution...")
        X, Y, T, U_exact = self.generate_exact_solution(n_x, n_y, n_t, initial_condition)
        
        # Flatten for PINN format
        # U_exact has shape (n_t, n_x, n_y). We need a full space-time grid of points
        # with shape (n_t * n_x * n_y, 3) where columns are [x, y, t].
        # X, Y are (n_x, n_y); T is (n_t, n_x, n_y).
        # Build full grids for X and Y repeated over time and then ravel consistently.
        n_t = U_exact.shape[0]
        # Expand X and Y to shape (n_t, n_x, n_y)
        X_full = np.tile(X[None, :, :], (n_t, 1, 1))
        Y_full = np.tile(Y[None, :, :], (n_t, 1, 1))
        # T is already (n_t, n_x, n_y)
        x_test_arr = np.column_stack([X_full.ravel(), Y_full.ravel(), T.ravel()])
        x_test = torch.FloatTensor(x_test_arr)
        u_test = torch.FloatTensor(U_exact.ravel())
        
        return x_test, u_test, X, Y, T, U_exact

    def plot_exact_solution(self, n_x=128, n_y=128, n_t=50, initial_condition='cosine'):
        """Plot the exact 2D Allen-Cahn solution"""
        
        X, Y, T, U_exact = self.generate_exact_solution(n_x, n_y, n_t, initial_condition)
        
        # Select time slices to plot
        time_indices = [0, n_t//3, 2*n_t//3, n_t-1]
        
        fig = plt.figure(figsize=(16, 12))
        
        # 2D contour plots at different times
        for i, idx in enumerate(time_indices):
            plt.subplot(2, 4, i+1)
            contour = plt.contourf(X, Y, U_exact[idx], levels=50, cmap='RdBu_r')
            plt.colorbar(contour)
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title(f'Solution at t={T[idx,0,0]:.2f}')
            plt.axis('equal')
        
        # 3D surface plot at final time
        ax = fig.add_subplot(2, 4, 5, projection='3d')
        surf = ax.plot_surface(X, Y, U_exact[-1], cmap='RdBu_r', 
                              linewidth=0, antialiased=True)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('u(x,y,t)')
        ax.set_title(f'Final solution at t={self.T}')
        
        # Evolution of center point
        plt.subplot(2, 4, 6)
        center_idx_x = n_x // 2
        center_idx_y = n_y // 2
        center_evolution = U_exact[:, center_idx_x, center_idx_y]
        plt.plot(np.linspace(0, self.T, n_t), center_evolution, 'b-', linewidth=2)
        plt.xlabel('Time')
        plt.ylabel('u(0,0,t)')
        plt.title('Evolution at center point')
        plt.grid(True, alpha=0.3)
        
        # Energy evolution
        plt.subplot(2, 4, 7)
        energy = np.array([np.mean(0.25 * (u**2 - 1)**2 + 0.5 * self.epsilon * 
                                ((np.gradient(u, axis=0)[1]**2 + np.gradient(u, axis=1)[1]**2)))
                        for u in U_exact])
        plt.plot(np.linspace(0, self.T, n_t), energy, 'r-', linewidth=2)
        plt.xlabel('Time')
        plt.ylabel('Energy')
        plt.title('Energy evolution')
        plt.grid(True, alpha=0.3)
        
        # Final state cross-section
        plt.subplot(2, 4, 8)
        plt.plot(X[:, n_y//2], U_exact[-1, :, n_y//2], 'g-', linewidth=2, label='y=0 cross-section')
        plt.plot(Y[n_x//2, :], U_exact[-1, n_x//2, :], 'm-', linewidth=2, label='x=0 cross-section')
        plt.xlabel('Position')
        plt.ylabel('u(x,y)')
        plt.title('Final state cross-sections')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return X, Y, T, U_exact

# Example usage
if __name__ == "__main__":
    # Create 2D Allen-Cahn generator
    generator = AllenCahnDataGenerator2D(epsilon=0.01, alpha=1.0, L=1.0, T=0.5)
    
    # Generate and plot exact solution
    X, Y, T, U_exact = generator.plot_exact_solution(initial_condition='bubble')
    
    # Generate training data
    training_data = generator.generate_training_data(initial_condition='cosine')
    
    print("Training data shapes:")
    for key, value in training_data.items():
        print(f"{key}: {value.shape}")
    
    # Generate test data
    x_test, u_test, X_test, Y_test, T_test, U_test = generator.generate_test_data()
    print(f"Test data shapes: x_test {x_test.shape}, u_test {u_test.shape}")