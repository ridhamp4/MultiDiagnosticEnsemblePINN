import os
import numpy as np
import torch
from allen_cahn_2d import AllenCahnDataGenerator2D


def tensor_to_np(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.array(x)


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "data_2d")
    os.makedirs(out_dir, exist_ok=True)

    gen = AllenCahnDataGenerator2D(epsilon=0.01, alpha=1.0, L=1.0, T=0.5)

    print("Generating 2D training data...")
    train = gen.generate_training_data(
        n_collocation=5000, 
        n_boundary=500, 
        n_initial=500, 
        initial_condition='cosine'
    )

    # Convert torch tensors to numpy arrays where needed
    collocation = tensor_to_np(train['collocation'])
    collocation_values = tensor_to_np(train['collocation_values'])
    boundary_points = tensor_to_np(train['boundary_points'])
    boundary_values = tensor_to_np(train['boundary_values'])
    initial_points = tensor_to_np(train['initial_points'])
    initial_values = tensor_to_np(train['initial_values'])

    train_path = os.path.join(out_dir, 'train.npz')
    np.savez_compressed(train_path,
                        collocation=collocation,
                        collocation_values=collocation_values,
                        boundary_points=boundary_points,
                        boundary_values=boundary_values,
                        initial_points=initial_points,
                        initial_values=initial_values)

    print(f"Saved 2D training data -> {train_path}")

    print("Generating 2D test data (high-res exact solution)...")
    x_test, u_test, X, Y, T, U_exact = gen.generate_test_data(
        n_x=128, 
        n_y=128, 
        n_t=50, 
        initial_condition='cosine'
    )

    # convert
    x_test_np = tensor_to_np(x_test)
    u_test_np = tensor_to_np(u_test)
    X_np = np.array(X)
    Y_np = np.array(Y)
    T_np = np.array(T)
    U_exact_np = np.array(U_exact)

    test_path = os.path.join(out_dir, 'test.npz')
    np.savez_compressed(test_path,
                        x_test=x_test_np,
                        u_test=u_test_np,
                        X=X_np,
                        Y=Y_np,
                        T=T_np,
                        U_exact=U_exact_np)

    print(f"Saved 2D test data -> {test_path}")

    # Print shapes for quick verification
    print("Training shapes:")
    print("  collocation:", collocation.shape, "-> (x, y, t) points")
    print("  collocation_values:", collocation_values.shape, "-> u values")
    print("  boundary_points:", boundary_points.shape, "-> boundary (x, y, t) points")
    print("  boundary_values:", boundary_values.shape, "-> boundary u values")
    print("  initial_points:", initial_points.shape, "-> initial (x, y, t=0) points")
    print("  initial_values:", initial_values.shape, "-> initial u values")

    print("\nTest shapes:")
    print("  x_test:", x_test_np.shape, "-> flattened (x, y, t) points")
    print("  u_test:", u_test_np.shape, "-> flattened u values")
    print("  X, Y, T grids:", X_np.shape, Y_np.shape, T_np.shape)
    print("  U_exact:", U_exact_np.shape, "-> (time, x, y)")

    # Generate additional datasets with different initial conditions
    print("\nGenerating additional datasets with different initial conditions...")
    
    # Dataset with bubble initial condition
    print("Generating bubble initial condition dataset...")
    train_bubble = gen.generate_training_data(
        n_collocation=5000, 
        n_boundary=500, 
        n_initial=500, 
        initial_condition='bubble'
    )
    
    # Convert and save bubble dataset
    train_bubble_np = {k: tensor_to_np(v) for k, v in train_bubble.items()}
    train_bubble_path = os.path.join(out_dir, 'train_bubble.npz')
    np.savez_compressed(train_bubble_path, **train_bubble_np)
    print(f"Saved bubble training data -> {train_bubble_path}")

    # Test data with bubble initial condition
    x_test_bubble, u_test_bubble, X_b, Y_b, T_b, U_exact_b = gen.generate_test_data(
        n_x=128, n_y=128, n_t=50, initial_condition='bubble'
    )
    
    test_bubble_path = os.path.join(out_dir, 'test_bubble.npz')
    np.savez_compressed(test_bubble_path,
                        x_test=tensor_to_np(x_test_bubble),
                        u_test=tensor_to_np(u_test_bubble),
                        X=np.array(X_b),
                        Y=np.array(Y_b), 
                        T=np.array(T_b),
                        U_exact=np.array(U_exact_b))
    print(f"Saved bubble test data -> {test_bubble_path}")

    # Dataset with random phases initial condition
    print("Generating random phases initial condition dataset...")
    train_random = gen.generate_training_data(
        n_collocation=5000, 
        n_boundary=500, 
        n_initial=500, 
        initial_condition='random_phases'
    )
    
    # Convert and save random dataset
    train_random_np = {k: tensor_to_np(v) for k, v in train_random.items()}
    train_random_path = os.path.join(out_dir, 'train_random.npz')
    np.savez_compressed(train_random_path, **train_random_np)
    print(f"Saved random phases training data -> {train_random_path}")

    # Test data with random phases initial condition
    x_test_random, u_test_random, X_r, Y_r, T_r, U_exact_r = gen.generate_test_data(
        n_x=128, n_y=128, n_t=50, initial_condition='random_phases'
    )
    
    test_random_path = os.path.join(out_dir, 'test_random.npz')
    np.savez_compressed(test_random_path,
                        x_test=tensor_to_np(x_test_random),
                        u_test=tensor_to_np(u_test_random),
                        X=np.array(X_r),
                        Y=np.array(Y_r),
                        T=np.array(T_r),
                        U_exact=np.array(U_exact_r))
    print(f"Saved random phases test data -> {test_random_path}")

    print(f"\nAll datasets saved to: {out_dir}")
    print("\nDataset summary:")
    print("  train.npz, test.npz - Cosine initial condition")
    print("  train_bubble.npz, test_bubble.npz - Bubble initial condition") 
    print("  train_random.npz, test_random.npz - Random phases initial condition")


if __name__ == '__main__':
    main()