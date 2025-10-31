import os
import numpy as np
import torch

from modified_pinn_2d import EnsembleAllenCahnPINN2D


def load_or_generate_data(data_dir):
    train_path = os.path.join(data_dir, 'train.npz')
    test_path = os.path.join(data_dir, 'test.npz')

    if os.path.exists(train_path) and os.path.exists(test_path):
        print(f"Loading data from {data_dir}/*.npz")
        train = np.load(train_path)
        test = np.load(test_path)

        collocation = train['collocation']
        # optional supervised values at collocation points
        collocation_values = train['collocation_values'] if 'collocation_values' in train.files else None
        boundary_points = train['boundary_points']
        boundary_values = train['boundary_values']
        initial_points = train['initial_points']
        initial_values = train['initial_values']
        print("Training data keys:", train.keys())

        x_test = test['x_test']
        u_test = test['u_test']
        X = test['X']
        Y = test['Y']
        T = test['T']
        U_exact = test['U_exact']
    else:
        # Fallback: generate small dataset (this may be slower)...
        print("Data files not found. Generating fallback data (this may be slower)...")
        from allen_cahn_2d import AllenCahnDataGenerator2D
        gen = AllenCahnDataGenerator2D(epsilon=0.01, alpha=1.0, L=1.0, T=0.5)
        d = gen.generate_training_data(n_collocation=5000, n_boundary=500, n_initial=500)
        collocation = d['collocation'].numpy()
        # generator may or may not provide collocation target values
        collocation_values = d.get('collocation_values', None)
        if collocation_values is not None:
            collocation_values = collocation_values.numpy()
        boundary_points = d['boundary_points'].numpy()
        boundary_values = d['boundary_values'].numpy()
        initial_points = d['initial_points'].numpy()
        initial_values = d['initial_values'].numpy()
        print("Generated data keys:", d.keys())
        
        x_test, u_test, X, Y, T, U_exact = gen.generate_test_data(n_x=128, n_y=128, n_t=50)
        x_test = x_test.numpy()
        u_test = u_test.numpy()
        X = X.numpy()
        Y = Y.numpy()
        T = T.numpy()
        U_exact = U_exact.numpy()

    return (collocation, collocation_values, boundary_points, boundary_values, 
            initial_points, initial_values, x_test, u_test, X, Y, T, U_exact)


def to_tensor(x, device, dtype=torch.float32):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device=device, dtype=dtype)
    if torch.is_tensor(x):
        return x.to(device=device, dtype=dtype)
    return torch.tensor(x, device=device, dtype=dtype)


def build_data_dict(collocation, collocation_values, boundary_points, boundary_values, 
                   initial_points, initial_values, device):
    coll = to_tensor(collocation, device)
    # collocation coordinates must require gradients for PDE residual computation
    try:
        coll.requires_grad_(True)
    except Exception:
        pass
    bc_pts = to_tensor(boundary_points, device)
    bc_vals = to_tensor(boundary_values, device)
    ic_pts = to_tensor(initial_points, device)
    ic_vals = to_tensor(initial_values, device)

    # optional collocation target values
    coll_vals = None
    if collocation_values is not None:
        coll_vals = to_tensor(collocation_values, device)

    # Ensure shapes: inputs should be (N,3) for (x,y,t), values (N,)
    if coll.dim() == 1:
        coll = coll.unsqueeze(1)
    if bc_pts.dim() == 1:
        bc_pts = bc_pts.unsqueeze(1)
    if ic_pts.dim() == 1:
        ic_pts = ic_pts.unsqueeze(1)

    out = {
        'collocation': coll,
        'boundary_points': bc_pts,
        'boundary_values': bc_vals,
        'initial_points': ic_pts,
        'initial_values': ic_vals
    }
    if coll_vals is not None:
        out['collocation_values'] = coll_vals
    return out


def main():
    root = os.path.dirname(__file__)
    data_dir = os.path.join(root, 'data_2d')
    models_dir = os.path.join(root, 'models_2d')
    os.makedirs(models_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    (collocation, collocation_values, boundary_points, boundary_values, 
     initial_points, initial_values, x_test, u_test, X, Y, T, U_exact) = load_or_generate_data(data_dir)

    data_dict = build_data_dict(collocation, collocation_values, boundary_points, 
                               boundary_values, initial_points, initial_values, device)
    
    print("Data shapes:")
    print(f"  collocation: {data_dict['collocation'].shape}")
    print(f"  boundary_points: {data_dict['boundary_points'].shape}")
    print(f"  initial_points: {data_dict['initial_points'].shape}")
    if 'collocation_values' in data_dict:
        print(f"  collocation_values: {data_dict['collocation_values'].shape}")

    # Convert test data
    x_test_t = to_tensor(x_test, device)
    u_test_t = to_tensor(u_test, device)

    # Initialize 2D model
    model = EnsembleAllenCahnPINN2D(
        epsilon=0.01, 
        alpha=1.0, 
        domain_bounds=[(-1, 1), (-1, 1)]  # 2D spatial domain
    )
    model.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training loop (short demo by default)
    n_epochs = 5000
    print(f"Starting training for {n_epochs} epochs (demo run)...")

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        loss = model.compute_ensemble_adaptive_loss(data_dict, epoch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if epoch % 100 == 0 or epoch == n_epochs - 1:
            print(f"Epoch {epoch:4d} - loss: {loss.item():.6e}")

    # Save model
    model_path = os.path.join(models_dir, 'ensemble_pinn_2d.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to {model_path}")

    # Evaluate on test set (MSE)
    model.eval()
    with torch.no_grad():
        preds = model.forward(x_test_t)
        mse = torch.mean((preds.view(-1) - u_test_t.view(-1))**2).item()
    print(f"Test MSE: {mse:.6e}")

    # Additional evaluation: plot some results
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        # Select a time slice for visualization
        time_idx = -1  # Final time
        # X and Y are 2D arrays with shape (n_x, n_y)
        spatial_slice = (X, Y)

        # Get exact solution at final time
        U_exact_final = U_exact[time_idx].reshape(X.shape)

        # Get predictions at final time. Build test points with same ordering as x_test (time-major)
        test_points_final = np.column_stack([
            X.ravel(), Y.ravel(), np.full(X.size, T[time_idx, 0, 0])
        ])
        test_points_final_t = to_tensor(test_points_final, device)
        with torch.no_grad():
            U_pred_final = model.forward(test_points_final_t).cpu().numpy().reshape(X.shape[0], Y.shape[1])
        
        # Plot comparison (increase height so colorbars and titles don't overlap)
        fig = plt.figure(figsize=(15, 7))
        
        # Exact solution
        ax1 = fig.add_subplot(131, projection='3d')
        surf1 = ax1.plot_surface(X, Y, U_exact_final, cmap='RdBu_r', alpha=0.8)
        # colorbar for exact solution
        fig.colorbar(surf1, ax=ax1, shrink=0.6, pad=0.04, aspect=20)
        ax1.set_title('Exact Solution')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_zlabel('u(x,y)')
        
        # Predicted solution
        ax2 = fig.add_subplot(132, projection='3d')
        surf2 = ax2.plot_surface(X, Y, U_pred_final, cmap='RdBu_r', alpha=0.8)
        # colorbar for prediction
        fig.colorbar(surf2, ax=ax2, shrink=0.6, pad=0.04, aspect=20)
        ax2.set_title('Ensemble Method Prediction')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_zlabel('u(x,y)')
        
        # Error
        ax3 = fig.add_subplot(133, projection='3d')
        error = np.abs(U_pred_final - U_exact_final)
        surf3 = ax3.plot_surface(X, Y, error, cmap='hot', alpha=0.8)
        # colorbar for error
        fig.colorbar(surf3, ax=ax3, shrink=0.6, pad=0.04, aspect=20)
        ax3.set_title('Absolute Error')
        ax3.set_xlabel('x')
        ax3.set_ylabel('y')
        ax3.set_zlabel('Error')
        
        # tighten layout and leave a bit of top margin for titles/colorbars
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        plt.savefig(os.path.join(models_dir, '2d_allen_cahn_comparison.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
    except ImportError:
        print("Matplotlib not available for plotting results")


if __name__ == '__main__':
    main()