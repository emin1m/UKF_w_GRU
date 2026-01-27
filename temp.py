"""
Learned measurement model for UKF (no GRU):
- Train an MLP h_theta(x) to approximate the nonlinear measurement function z = h(x) + v
- Create a clean train/val/test split from simulated data
- Train and validate the model
- Visualize training curves and prediction quality

Next step (after this): plug h_theta into your UKF by replacing h(x) with h_theta(x).
"""

import numpy as np
from math import sqrt
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# 1) Original sensor geometry + physical measurement function
# ============================================================

P0 = 1e4
n = 2

num_sensors_x = 5
num_sensors_y = 5
num_sensors = num_sensors_x * num_sensors_y
rng_values = np.linspace(-20, 20, num_sensors_x)
sensor_coords = np.array([[x, y] for x in rng_values for y in rng_values])

def h_physics(x_t: np.ndarray) -> np.ndarray:
    """
    Physics measurement model (ground truth generator).
    x_t: [x, y, vx, vy] (vx, vy are ignored by physics here)
    Returns: z (25,)
    """
    z_t = np.zeros(len(sensor_coords), dtype=np.float64)
    for i, (x_i, y_i) in enumerate(sensor_coords):
        d_i = sqrt((x_i - x_t[0])**2 + (y_i - x_t[1])**2)
        z_t[i] = sqrt(P0 / (1.0 + d_i**n))
    return z_t


# ============================================================
# 2) Dataset generation (train/val/test)
# ============================================================

def generate_dataset(
    N: int,
    R_std: float = 1.0,
    xy_range: float = 25.0,
    v_range: float = 3.0,
    seed: int = 0
):
    """
    Generate supervised pairs (x -> z):
      x = [x, y, vx, vy]
      z = h_physics(x) + Gaussian noise

    We sample position uniformly in [-xy_range, xy_range]^2
    and velocity uniformly in [-v_range, v_range]^2.

    Returns:
      X: (N, 4) float32
      Z: (N, 25) float32
    """
    rng = np.random.default_rng(seed)

    x_pos = rng.uniform(-xy_range, xy_range, size=(N, 2))
    v = rng.uniform(-v_range, v_range, size=(N, 2))
    X = np.concatenate([x_pos, v], axis=1).astype(np.float32)

    Z = np.zeros((N, num_sensors), dtype=np.float32)
    for i in range(N):
        z_clean = h_physics(X[i].astype(np.float64))
        z_noisy = z_clean + rng.normal(0.0, R_std, size=(num_sensors,))
        Z[i] = z_noisy.astype(np.float32)

    return X, Z


def train_val_test_split(X, Z, train=0.7, val=0.15, test=0.15, seed=0):
    assert abs(train + val + test - 1.0) < 1e-9
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))

    n_train = int(train * len(X))
    n_val = int(val * len(X))
    n_test = len(X) - n_train - n_val

    i_train = idx[:n_train]
    i_val = idx[n_train:n_train + n_val]
    i_test = idx[n_train + n_val:]

    return (X[i_train], Z[i_train]), (X[i_val], Z[i_val]), (X[i_test], Z[i_test])


class SupervisedXZDataset(Dataset):
    def __init__(self, X, Z):
        self.X = torch.from_numpy(X)
        self.Z = torch.from_numpy(Z)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i], self.Z[i]


# ============================================================
# 3) Model: simple MLP for h_theta(x) -> z
#    (This is the most suitable baseline for your case.)
# ============================================================

class MeasurementMLP(nn.Module):
    """
    Predicts 25D measurement vector from 4D state.
    Notes:
      - This learns h_theta(x) directly.
      - For your physical model, only x,y matter; vx,vy can still be input.
    """
    def __init__(self, in_dim=4, out_dim=25, hidden=(128, 128, 128), dropout=0.0):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(inplace=True)]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
            d = h
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ============================================================
# 4) Training utilities
# ============================================================

def run_epoch(model, loader, optimizer=None, device="cpu"):
    """
    If optimizer is provided -> train mode.
    Otherwise -> eval mode.
    Returns average MSE loss.
    """
    is_train = optimizer is not None
    model.train(is_train)

    mse = nn.MSELoss(reduction="mean")
    total_loss = 0.0
    n_batches = 0

    for Xb, Zb in loader:
        Xb = Xb.to(device)
        Zb = Zb.to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        pred = model(Xb)
        loss = mse(pred, Zb)

        if is_train:
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item())
        n_batches += 1

    return total_loss / max(1, n_batches)


@torch.no_grad()
def evaluate_quality(model, X_test, device="cpu", n_examples=5):
    """
    Quick qualitative check:
    - prints per-example MSE
    - returns a few predictions for plotting
    """
    model.eval()
    X = torch.from_numpy(X_test).to(device)
    Zhat = model(X).cpu().numpy()
    return Zhat


# ============================================================
# 5) Main: generate data -> train -> validate -> test -> plots
# ============================================================

def main():
    # ----------------------------
    # Data config
    # ----------------------------
    N = 60000            # total samples
    R_std = 1.0          # measurement noise std used in labels
    xy_range = 25.0      # cover beyond sensor grid to make model robust
    v_range = 3.0
    seed = 42

    # ----------------------------
    # Train config
    # ----------------------------
    batch_size = 512
    epochs = 40
    lr = 1e-3
    weight_decay = 1e-5

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ----------------------------
    # Generate dataset
    # ----------------------------
    X, Z = generate_dataset(N=N, R_std=R_std, xy_range=xy_range, v_range=v_range, seed=seed)
    (Xtr, Ztr), (Xva, Zva), (Xte, Zte) = train_val_test_split(X, Z, seed=seed)

    train_ds = SupervisedXZDataset(Xtr, Ztr)
    val_ds   = SupervisedXZDataset(Xva, Zva)
    test_ds  = SupervisedXZDataset(Xte, Zte)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    # ----------------------------
    # Build model
    # ----------------------------
    model = MeasurementMLP(in_dim=4, out_dim=num_sensors, hidden=(256, 256, 256), dropout=0.0).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # ----------------------------
    # Train loop
    # ----------------------------
    train_losses = []
    val_losses = []

    best_val = float("inf")
    best_state = None

    for ep in range(1, epochs + 1):
        tr_loss = run_epoch(model, train_loader, optimizer=optimizer, device=device)
        va_loss = run_epoch(model, val_loader, optimizer=None, device=device)

        train_losses.append(tr_loss)
        val_losses.append(va_loss)

        if va_loss < best_val:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if ep == 1 or ep % 5 == 0:
            print(f"Epoch {ep:3d}/{epochs} | train MSE: {tr_loss:.6f} | val MSE: {va_loss:.6f}")

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    # ----------------------------
    # Final test evaluation
    # ----------------------------
    te_loss = run_epoch(model, test_loader, optimizer=None, device=device)
    print(f"Best val MSE: {best_val:.6f}")
    print(f"Test MSE    : {te_loss:.6f}")

    # ----------------------------
    # Plots: training curves
    # ----------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train MSE")
    plt.plot(val_losses, label="Val MSE")
    plt.title("Training Curves (MLP learns measurement model)")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ----------------------------
    # Plots: prediction quality on a few samples
    # ----------------------------
    # Choose a few random test samples and compare predicted vs measured vs clean
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Xte), size=4, replace=False)

    X_examples = Xte[idx]
    Z_meas = Zte[idx]  # noisy labels used in training

    with torch.no_grad():
        Z_pred = model(torch.from_numpy(X_examples).to(device)).cpu().numpy()

    # Also compute clean physics output for reference
    Z_clean = np.stack([h_physics(x.astype(np.float64)).astype(np.float32) for x in X_examples], axis=0)

    for k in range(len(idx)):
        plt.figure(figsize=(10, 4))
        plt.plot(Z_clean[k], label="Clean physics h(x)")
        plt.plot(Z_meas[k], label="Noisy measurement (label)", alpha=0.7)
        plt.plot(Z_pred[k], label="NN prediction h_theta(x)", linestyle="--")
        plt.title(f"Example {k} | state = {X_examples[k]}")
        plt.xlabel("Sensor index")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    # ----------------------------
    # Save model (optional)
    # ----------------------------
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {
                "P0": P0, "n": n,
                "num_sensors": num_sensors,
                "xy_range": xy_range, "v_range": v_range,
                "R_std": R_std
            }
        },
        "measurement_mlp.pt"
    )
    print("Saved: measurement_mlp.pt")

if __name__ == "__main__":
    main()
