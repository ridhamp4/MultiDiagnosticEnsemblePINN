import os
import numpy as np
import torch
from allen_cahn import AllenCahnDataGenerator


def tensor_to_np(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.array(x)


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(out_dir, exist_ok=True)

    gen = AllenCahnDataGenerator()

    print("Generating training data...")
    train = gen.generate_training_data(n_collocation=2000, n_boundary=200, n_initial=200, initial_condition='cosine')

    # Convert torch tensors to numpy arrays where needed
    collocation = tensor_to_np(train['collocation'])
    collocation_values = tensor_to_np(train['collocation_values'])  # NEW: true values at collocation points
    boundary_points = tensor_to_np(train['boundary_points'])
    boundary_values = tensor_to_np(train['boundary_values'])
    initial_points = tensor_to_np(train['initial_points'])
    initial_values = tensor_to_np(train['initial_values'])

    train_path = os.path.join(out_dir, 'train.npz')
    np.savez_compressed(train_path,
                        collocation=collocation,
                        collocation_values=collocation_values,  # NEW: save collocation values
                        boundary_points=boundary_points,
                        boundary_values=boundary_values,
                        initial_points=initial_points,
                        initial_values=initial_values)

    print(f"Saved training data -> {train_path}")

    print("Generating test data (high-res exact solution)...")
    x_test, u_test, X, T, U_exact = gen.generate_test_data(n_x=256, n_t=100, initial_condition='cosine')

    # convert
    x_test_np = tensor_to_np(x_test)
    u_test_np = tensor_to_np(u_test)
    X_np = np.array(X)
    T_np = np.array(T)
    U_exact_np = np.array(U_exact)

    test_path = os.path.join(out_dir, 'test.npz')
    np.savez_compressed(test_path,
                        x_test=x_test_np,
                        u_test=u_test_np,
                        X=X_np,
                        T=T_np,
                        U_exact=U_exact_np)

    print(f"Saved test data -> {test_path}")

    # Print shapes for quick verification
    print("Training shapes:")
    print("  collocation:", collocation.shape)
    print("  collocation_values:", collocation_values.shape)  # NEW
    print("  boundary_points:", boundary_points.shape)
    print("  boundary_values:", boundary_values.shape)
    print("  initial_points:", initial_points.shape)
    print("  initial_values:", initial_values.shape)

    print("Test shapes:")
    print("  x_test:", x_test_np.shape)
    print("  u_test:", u_test_np.shape)
    print("  X, T, U_exact:", X_np.shape, T_np.shape, U_exact_np.shape)


if __name__ == '__main__':
    main()