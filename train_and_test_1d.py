import os
import numpy as np
import torch

from modified_pinn import EnsembleAllenCahnPINN



import os
import numpy as np
import torch

from modified_pinn import EnsembleAllenCahnPINN


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
        print(train.keys(), "jay ganesh")

        x_test = test['x_test']
        u_test = test['u_test']
    else:
        # Fallback: generate small dataset (this may be slower)...
        print("Data files not found. Generating fallback data (this may be slower)...")
        from allen_cahn import AllenCahnDataGenerator
        gen = AllenCahnDataGenerator()
        d = gen.generate_training_data(n_collocation=2000, n_boundary=200, n_initial=200)
        collocation = d['collocation'].numpy()
        # generator may or may not provide collocation target values
        collocation_values = d.get('collocation_values', None)
        boundary_points = d['boundary_points'].numpy()
        boundary_values = d['boundary_values'].numpy()
        initial_points = d['initial_points'].numpy()
        initial_values = d['initial_values'].numpy()
        print(d.keys(), "jay ganesh")
        x_test, u_test, _, _, _ = gen.generate_test_data(n_x=256, n_t=100)
        x_test = x_test.numpy()
        u_test = u_test.numpy()
        print(collocation_values)
    return (collocation, collocation_values, boundary_points, boundary_values, initial_points, initial_values,
            x_test, u_test)


def to_tensor(x, device, dtype=torch.float32):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device=device, dtype=dtype)
    if torch.is_tensor(x):
        return x.to(device=device, dtype=dtype)
    return torch.tensor(x, device=device, dtype=dtype)


def build_data_dict(collocation, collocation_values, boundary_points, boundary_values, initial_points, initial_values, device):
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

    # Ensure shapes: inputs should be (N,2), values (N,)
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
    data_dir = os.path.join(root, 'data')
    models_dir = os.path.join(root, 'models')
    os.makedirs(models_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    collocation, collocation_values, boundary_points, boundary_values, initial_points, initial_values, x_test, u_test = \
        load_or_generate_data(data_dir)

    data_dict = build_data_dict(collocation, collocation_values, boundary_points, boundary_values, initial_points, initial_values, device)
    print(data_dict.keys())
    print(collocation_values)
    # Convert test data
    x_test_t = to_tensor(x_test, device)
    u_test_t = to_tensor(u_test, device)

    # Initialize model
    model = EnsembleAllenCahnPINN(epsilon=0.001, alpha=5.0, domain_bounds=[(-1, 1)])
    model.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training loop (short demo by default)
    n_epochs = 800
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
    model_path = os.path.join(models_dir, 'ensemble_pinn.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to {model_path}")

    # Evaluate on test set (MSE)
    model.eval()
    with torch.no_grad():
        preds = model.forward(x_test_t)
        mse = torch.mean((preds.view(-1) - u_test_t.view(-1))**2).item()
    print(f"Test MSE: {mse:.6e}")


if __name__ == '__main__':
    main()
