from __future__ import annotations

from math import sqrt
from pathlib import Path
from typing import Callable, Tuple

import numpy as np
import matplotlib.pyplot as plt

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None


# ============================================================
# 0) Measurement model constants (PDF Eq.(4),(5))
#    IMPORTANT: avoid name collision with covariance P0
# ============================================================

P_TX = 1e4  # PDF uses P0 as target power (Eq.(4)); rename to avoid shadowing.
# In the PDF: a_{n,k} = sqrt(P0 / (1 + d_{n,k}^2))  -> exponent is fixed to 2. :contentReference[oaicite:6]{index=6}

num_sensors_x = 5
num_sensors_y = 5
num_sensors = num_sensors_x * num_sensors_y

# ROI is 40x40 in the paper (b=40), sensors shown from -20..20 grid (Fig.1).
rng_values = np.linspace(-20, 20, num_sensors_x)
sensor_coords = np.array([[x, y] for x in rng_values for y in rng_values], dtype=np.float64)


def h_physics(x_t: np.ndarray) -> np.ndarray:
    """
    Physical measurement model aligned with the PDF Eq.(4):
        a_n(x) = sqrt(P_TX / (1 + d^2))
    State x_t = [x, y, vx, vy]; vx/vy are ignored by h(·).
    Returns z shape: (N,)
    """
    x_t = np.asarray(x_t, dtype=np.float64).reshape(4)
    z_t = np.zeros(len(sensor_coords), dtype=np.float64)
    x, y = x_t[0], x_t[1]

    for i, (x_i, y_i) in enumerate(sensor_coords):
        dx = x_i - x
        dy = y_i - y
        d2 = dx * dx + dy * dy  # distance squared
        z_t[i] = sqrt(P_TX / (1.0 + d2))
    return z_t


# ============================================================
# 1) Optional learned measurement model loader (unchanged logic)
# ============================================================

if nn is not None:
    class MeasurementMLP(nn.Module):
        def __init__(self, in_dim: int = 4, out_dim: int = 25, hidden=(128, 128, 128), dropout: float = 0.0):
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
else:
    class MeasurementMLP:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for MeasurementMLP.")


def _extract_linear_shapes(state_dict: dict) -> list[tuple[int, int]]:
    shapes = []
    for key, value in state_dict.items():
        if key.endswith(".weight"):
            parts = key.split(".")
            if len(parts) >= 3 and parts[0] == "net" and parts[1].isdigit():
                layer_idx = int(parts[1])
                out_dim, in_dim = value.shape
                shapes.append((layer_idx, in_dim, out_dim))
    shapes.sort(key=lambda t: t[0])
    return [(in_dim, out_dim) for _, in_dim, out_dim in shapes]


def load_h_theta_from_checkpoint(ckpt_path: str = "measurement_mlp.pt", device: str = "cpu") -> Callable[[np.ndarray], np.ndarray]:
    """
    Loads h_theta(x) from checkpoint.
    Returns callable hx(x)->z with z shape (N,).
    """
    if torch is None:
        raise ImportError("PyTorch not installed. Cannot load measurement_mlp.pt.")

    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint["state_dict"]

    shapes = _extract_linear_shapes(state_dict)
    if len(shapes) < 2:
        raise ValueError("Unexpected checkpoint format for MeasurementMLP.")

    hidden = tuple(out_dim for _, out_dim in shapes[:-1])
    in_dim = shapes[0][0]
    out_dim = shapes[-1][1]

    model = MeasurementMLP(in_dim=in_dim, out_dim=out_dim, hidden=hidden, dropout=0.0)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    @torch.no_grad()
    def hx_nn(x_t: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(np.asarray(x_t, dtype=np.float32).reshape(1, -1)).to(device)
        z = model(x).cpu().numpy().reshape(-1).astype(np.float64)
        return z

    return hx_nn


# ============================================================
# 2) System model: matches the PDF Section II, Eq.(1)-(3)
# ============================================================

def F_constant_velocity(dt: float) -> np.ndarray:
    """
    PDF Eq.(2) uses Δ (time step) in the (x,vx) and (y,vy) coupling.
    Here we use dt as Δ:
        [1 0 dt 0
         0 1 0  dt
         0 0 1  0
         0 0 0  1]
    """
    return np.array(
        [
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def Q_white_accel_pdf(dt: float, tau: float) -> np.ndarray:
    """
    Process noise covariance matches PDF Eq.(3):
        Q = τ * [[dt^3/3, 0,      dt^2/2, 0     ],
                 [0,      dt^3/3, 0,      dt^2/2],
                 [dt^2/2, 0,      dt,     0     ],
                 [0,      dt^2/2, 0,      dt    ]]
    """
    dt2 = dt * dt
    dt3 = dt2 * dt
    Q = np.array(
        [
            [dt3 / 3.0, 0.0,      dt2 / 2.0, 0.0],
            [0.0,       dt3 / 3.0, 0.0,      dt2 / 2.0],
            [dt2 / 2.0, 0.0,      dt,        0.0],
            [0.0,       dt2 / 2.0, 0.0,      dt],
        ],
        dtype=np.float64,
    )
    return tau * Q


def fx_constant_velocity(x: np.ndarray, dt: float) -> np.ndarray:
    """
    x_{k} = F x_{k-1} + v_k (PDF Eq.(1))
    """
    F = F_constant_velocity(dt)
    return F @ np.asarray(x, dtype=np.float64).reshape(4)


def simulate_sequence(
    T: int,
    dt: float,
    x0_true: np.ndarray,
    tau: float = 1e-2,
    sigma: float = 1.0,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulates trajectory and noisy raw measurements consistent with PDF Eq.(1)-(5).

    - Process: x_k = F x_{k-1} + v_k,  v_k ~ N(0, Q)  (Eq.(1)-(3))
    - Measurement: z_k = h(x_k) + n_k, n_k ~ N(0, R), R = sigma^2 I  (Eq.(5))
    """
    rng = np.random.default_rng(seed)
    x_true = np.asarray(x0_true, dtype=np.float64).reshape(4).copy()

    Q = Q_white_accel_pdf(dt, tau)
    R_std = sigma

    xs = np.zeros((T, 4), dtype=np.float64)
    zs = np.zeros((T, num_sensors), dtype=np.float64)

    for k in range(T):
        x_true = fx_constant_velocity(x_true, dt)
        x_true = x_true + rng.multivariate_normal(mean=np.zeros(4), cov=Q)

        z_k = h_physics(x_true) + rng.normal(0.0, R_std, size=num_sensors)

        xs[k] = x_true
        zs[k] = z_k

    return xs, zs


# ============================================================
# 3) UKF core (your implementation + minor robustness)
# ============================================================

def _force_symmetry(M: np.ndarray) -> np.ndarray:
    return 0.5 * (M + M.T)


def _safe_cholesky(P: np.ndarray, max_tries: int = 8, eps: float = 1e-10) -> np.ndarray:
    P = _force_symmetry(P)
    eye = np.eye(P.shape[0], dtype=P.dtype)
    jitter = 0.0
    for _ in range(max_tries):
        try:
            return np.linalg.cholesky(P + jitter * eye)
        except np.linalg.LinAlgError:
            jitter = eps if jitter == 0.0 else jitter * 10.0
    raise np.linalg.LinAlgError("Cholesky failed: covariance is not positive definite.")


def _safe_solve_spd(A: np.ndarray, B: np.ndarray, max_tries: int = 8, eps: float = 1e-10) -> np.ndarray:
    A = _force_symmetry(A)
    eye = np.eye(A.shape[0], dtype=A.dtype)
    jitter = 0.0
    for _ in range(max_tries):
        try:
            return np.linalg.solve(A + jitter * eye, B)
        except np.linalg.LinAlgError:
            jitter = eps if jitter == 0.0 else jitter * 10.0
    raise np.linalg.LinAlgError("Solve failed: innovation covariance is singular.")


class MerweSigmaPoints:
    def __init__(self, n_dim: int, alpha: float = 0.3, beta: float = 2.0, kappa: float = 0.0):
        """
        alpha default increased from 1e-3 -> 0.3 for better numerical behavior in many practical settings.
        If you want, you can set it back when calling UKF.
        """
        self.n_dim = n_dim
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa

        self.lambda_ = (alpha ** 2) * (n_dim + kappa) - n_dim
        self.c = n_dim + self.lambda_

        self.Wm = np.full(2 * n_dim + 1, 0.5 / self.c, dtype=np.float64)
        self.Wc = np.full(2 * n_dim + 1, 0.5 / self.c, dtype=np.float64)
        self.Wm[0] = self.lambda_ / self.c
        self.Wc[0] = self.lambda_ / self.c + (1.0 - alpha ** 2 + beta)

    def sigma_points(self, x: np.ndarray, P: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).reshape(self.n_dim)
        P = np.asarray(P, dtype=np.float64).reshape(self.n_dim, self.n_dim)

        U = _safe_cholesky(self.c * P)
        sigmas = np.zeros((2 * self.n_dim + 1, self.n_dim), dtype=np.float64)
        sigmas[0] = x

        for i in range(self.n_dim):
            sigmas[i + 1] = x + U[:, i]
            sigmas[self.n_dim + i + 1] = x - U[:, i]
        return sigmas


def unscented_transform(sigmas: np.ndarray, Wm: np.ndarray, Wc: np.ndarray, noise_cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.dot(Wm, sigmas)
    d = sigmas.shape[1]
    cov = np.zeros((d, d), dtype=np.float64)

    for i in range(sigmas.shape[0]):
        delta = (sigmas[i] - mean).reshape(-1, 1)
        cov += Wc[i] * (delta @ delta.T)

    cov += noise_cov
    cov = _force_symmetry(cov)
    return mean, cov


class UKF:
    def __init__(
        self,
        dim_x: int,
        dim_z: int,
        fx: Callable[[np.ndarray, float], np.ndarray],
        hx: Callable[[np.ndarray], np.ndarray],
        dt: float,
        Q: np.ndarray,
        R: np.ndarray,
        x0: np.ndarray,
        P_init: np.ndarray,
        alpha: float = 0.01,
        beta: float = 2.0,
        kappa: float = 0.0,
    ):
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.fx = fx
        self.hx = hx
        self.dt = dt

        self.Q = np.asarray(Q, dtype=np.float64).reshape(dim_x, dim_x)
        self.R = np.asarray(R, dtype=np.float64).reshape(dim_z, dim_z)
        self.x = np.asarray(x0, dtype=np.float64).reshape(dim_x)
        self.P = np.asarray(P_init, dtype=np.float64).reshape(dim_x, dim_x)

        self.points = MerweSigmaPoints(dim_x, alpha=alpha, beta=beta, kappa=kappa)
        self.Wm = self.points.Wm
        self.Wc = self.points.Wc

        self.sigmas_f = None
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()
        self.y = np.zeros(dim_z, dtype=np.float64)
        self.S = np.zeros((dim_z, dim_z), dtype=np.float64)
        self.K = np.zeros((dim_x, dim_z), dtype=np.float64)

    def predict(self):
        sigmas = self.points.sigma_points(self.x, self.P)
        sigmas_f = np.zeros_like(sigmas)

        for i in range(sigmas.shape[0]):
            sigmas_f[i] = np.asarray(self.fx(sigmas[i], self.dt), dtype=np.float64).reshape(self.dim_x)

        self.sigmas_f = sigmas_f
        self.x, self.P = unscented_transform(sigmas_f, self.Wm, self.Wc, self.Q)
        self.P = _force_symmetry(self.P)

        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

    def update(self, z: np.ndarray):
        if self.sigmas_f is None:
            raise RuntimeError("predict() must be called before update().")

        z = np.asarray(z, dtype=np.float64).reshape(self.dim_z)
        n_sigmas = self.sigmas_f.shape[0]

        sigmas_h = np.zeros((n_sigmas, self.dim_z), dtype=np.float64)
        for i in range(n_sigmas):
            sigmas_h[i] = np.asarray(self.hx(self.sigmas_f[i]), dtype=np.float64).reshape(self.dim_z)

        z_pred, S = unscented_transform(sigmas_h, self.Wm, self.Wc, self.R)

        Pxz = np.zeros((self.dim_x, self.dim_z), dtype=np.float64)
        for i in range(n_sigmas):
            dx = (self.sigmas_f[i] - self.x).reshape(-1, 1)
            dz = (sigmas_h[i] - z_pred).reshape(-1, 1)
            Pxz += self.Wc[i] * (dx @ dz.T)

        S = _force_symmetry(S)
        K = _safe_solve_spd(S, Pxz.T).T  # K = Pxz * inv(S)

        self.y = z - z_pred
        self.K = K
        self.S = S

        self.x = self.x + K @ self.y
        self.P = self.P - K @ S @ K.T
        self.P = _force_symmetry(self.P)

        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

    def step(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.predict()
        self.update(z)
        return self.x.copy(), self.P.copy()


# ============================================================
# 4) Example run
# ============================================================

if __name__ == "__main__":
    # PDF uses Δ=0.25s in experiments, but model is general. :contentReference[oaicite:7]{index=7}
    dt = 0.25
    T = 100

    # PDF uses τ=1e-2 and P0=1e4 in simulation section (matches defaults here conceptually).
    tau = 1e-2
    sigma = 1.0

    x0_true = np.array([-8.0, 10.0, 1.2, -0.8], dtype=np.float64)
    x0_est = np.array([-7.0, 9.0, 0.0, 0.0], dtype=np.float64)

    xs_true, zs = simulate_sequence(T=T, dt=dt, x0_true=x0_true, tau=tau, sigma=sigma, seed=7)

    Q = Q_white_accel_pdf(dt, tau)
    R = np.eye(num_sensors, dtype=np.float64) * (sigma ** 2)

    # Initial covariance (your choice). PDF example for x0 distribution uses diag[2,2,0.01,0.01] in sims,
    # but that was for generating trajectories; filter init can match if you want.
    P_init = np.diag([2.0, 2.0, 1.0, 1.0]).astype(np.float64)

    # Default: physical measurement model (PDF Eq.(4)-(5))
    hx_used = h_physics

    # Optional: use learned h_theta(x) from measurement_mlp.pt if available
    ckpt = Path("measurement_mlp.pt")
    if ckpt.exists() and torch is not None:
        try:
            hx_used = load_h_theta_from_checkpoint(str(ckpt), device="cpu")
            print("Using learned measurement model: h_theta(x) from measurement_mlp.pt")
        except Exception as exc:
            print(f"Could not load measurement_mlp.pt ({exc}). Falling back to h_physics.")
    else:
        print("measurement_mlp.pt not found or torch missing. Using h_physics.")

    ukf = UKF(
        dim_x=4,
        dim_z=num_sensors,
        fx=fx_constant_velocity,
        hx=hx_used,
        dt=dt,
        Q=Q,
        R=R,
        x0=x0_est,
        P_init=P_init,
        alpha=0.3,  # you can set 1e-3 if you insist, but 0.3 is usually saner numerically
        beta=2.0,
        kappa=0.0,
    )

    xs_est = np.zeros_like(xs_true)
    for k in range(T):
        x_est, _ = ukf.step(zs[k])
        xs_est[k] = x_est

    pos_rmse = np.sqrt(np.mean((xs_est[:, :2] - xs_true[:, :2]) ** 2))
    vel_rmse = np.sqrt(np.mean((xs_est[:, 2:] - xs_true[:, 2:]) ** 2))
    print(f"Position RMSE: {pos_rmse:.4f}")
    print(f"Velocity RMSE: {vel_rmse:.4f}")

    # Trajectory plot (x-y plane)
    plt.figure(figsize=(8, 6))
    plt.plot(xs_true[:, 0], xs_true[:, 1], label="True trajectory", linewidth=2.0)
    plt.plot(xs_est[:, 0], xs_est[:, 1], "--", label="UKF estimated trajectory", linewidth=2.0)
    plt.scatter(sensor_coords[:, 0], sensor_coords[:, 1], s=25, alpha=0.6, label="Sensors")
    plt.scatter([xs_true[0, 0]], [xs_true[0, 1]], marker="o", s=80, label="Start (true)")
    plt.scatter([xs_true[-1, 0]], [xs_true[-1, 1]], marker="x", s=80, label="End (true)")
    plt.title("Trajectory: True vs UKF Estimate")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    plt.show()
