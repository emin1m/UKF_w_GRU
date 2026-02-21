from __future__ import annotations

import numpy as np
from math import sqrt
import matplotlib.pyplot as plt

# =========================
# Physical setup (UNCHANGED)
# =========================
P0 = 1e4
n = 2

num_sensors_x = 5
num_sensors_y = 5
num_sensors = num_sensors_x * num_sensors_y
rng_values = np.linspace(-20, 20, num_sensors_x)
sensor_coords = np.array([[x, y] for x in rng_values for y in rng_values], dtype=float)

def h(x_t):
    # measurement model: z_i = sqrt(P0 / (1 + d^n)), with n=2
    z_t = np.zeros(len(sensor_coords), dtype=float)
    for i, (x_i, y_i) in enumerate(sensor_coords):
        d_i_t = sqrt((x_i - x_t[0])**2 + (y_i - x_t[1])**2)
        z_t[i] = sqrt(P0 / (1.0 + d_i_t**n))
    return z_t

# =========================
# UKF (replaces EKF only)
# =========================
def _force_sym(M: np.ndarray) -> np.ndarray:
    return 0.5 * (M + M.T)

def _safe_chol(P: np.ndarray, tries: int = 8, eps: float = 1e-10) -> np.ndarray:
    P = _force_sym(P)
    I = np.eye(P.shape[0], dtype=P.dtype)
    jitter = 0.0
    for _ in range(tries):
        try:
            return np.linalg.cholesky(P + jitter * I)
        except np.linalg.LinAlgError:
            jitter = eps if jitter == 0.0 else jitter * 10.0
    raise np.linalg.LinAlgError("Cholesky failed (P not PD).")

def _safe_solve(A: np.ndarray, B: np.ndarray, tries: int = 8, eps: float = 1e-10) -> np.ndarray:
    A = _force_sym(A)
    I = np.eye(A.shape[0], dtype=A.dtype)
    jitter = 0.0
    for _ in range(tries):
        try:
            return np.linalg.solve(A + jitter * I, B)
        except np.linalg.LinAlgError:
            jitter = eps if jitter == 0.0 else jitter * 10.0
    raise np.linalg.LinAlgError("Solve failed (S singular/ill-conditioned).")

class MerweSigmaPoints:
    def __init__(self, n_dim: int, alpha: float = 0.3, beta: float = 2.0, kappa: float = 0.0):
        self.n = n_dim
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa

        self.lmbda = (alpha**2) * (n_dim + kappa) - n_dim
        self.c = n_dim + self.lmbda

        self.Wm = np.full(2 * n_dim + 1, 0.5 / self.c, dtype=float)
        self.Wc = np.full(2 * n_dim + 1, 0.5 / self.c, dtype=float)
        self.Wm[0] = self.lmbda / self.c
        self.Wc[0] = self.lmbda / self.c + (1.0 - alpha**2 + beta)

    def sigma_points(self, x: np.ndarray, P: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(self.n)
        P = np.asarray(P, dtype=float).reshape(self.n, self.n)

        U = _safe_chol(self.c * P)  # chol(cP)
        sigmas = np.zeros((2 * self.n + 1, self.n), dtype=float)
        sigmas[0] = x
        for i in range(self.n):
            sigmas[i + 1] = x + U[:, i]
            sigmas[self.n + i + 1] = x - U[:, i]
        return sigmas

def unscented_transform(sigmas: np.ndarray, Wm: np.ndarray, Wc: np.ndarray, noise_cov: np.ndarray):
    mean = Wm @ sigmas
    d = sigmas.shape[1]
    cov = np.zeros((d, d), dtype=float)
    for i in range(sigmas.shape[0]):
        e = (sigmas[i] - mean).reshape(-1, 1)
        cov += Wc[i] * (e @ e.T)
    cov += noise_cov
    cov = _force_sym(cov)
    return mean, cov

class UKF:
    def __init__(
        self,
        fx,              # fx(x) -> x'
        hx,              # hx(x) -> z
        x0: np.ndarray,
        P0: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        alpha: float = 0.3,
        beta: float = 2.0,
        kappa: float = 0.0,
    ):
        self.fx = fx
        self.hx = hx

        self.x = np.asarray(x0, dtype=float).reshape(-1)
        self.P = np.asarray(P0, dtype=float)

        self.Q = np.asarray(Q, dtype=float)
        self.R = np.asarray(R, dtype=float)

        self.n = self.x.size
        self.m = self.R.shape[0]

        self.points = MerweSigmaPoints(self.n, alpha=alpha, beta=beta, kappa=kappa)
        self.Wm = self.points.Wm
        self.Wc = self.points.Wc

        self.sigmas_f = None

    def predict(self):
        sigmas = self.points.sigma_points(self.x, self.P)
        sigmas_f = np.zeros_like(sigmas)
        for i in range(sigmas.shape[0]):
            sigmas_f[i] = np.asarray(self.fx(sigmas[i]), dtype=float).reshape(self.n)

        self.sigmas_f = sigmas_f
        self.x, self.P = unscented_transform(sigmas_f, self.Wm, self.Wc, self.Q)
        self.P = _force_sym(self.P)

    def update(self, z: np.ndarray):
        if self.sigmas_f is None:
            raise RuntimeError("Call predict() before update().")

        z = np.asarray(z, dtype=float).reshape(self.m)
        n_sig = self.sigmas_f.shape[0]

        sigmas_h = np.zeros((n_sig, self.m), dtype=float)
        for i in range(n_sig):
            sigmas_h[i] = np.asarray(self.hx(self.sigmas_f[i]), dtype=float).reshape(self.m)

        z_pred, S = unscented_transform(sigmas_h, self.Wm, self.Wc, self.R)
        S = _force_sym(S)

        Pxz = np.zeros((self.n, self.m), dtype=float)
        for i in range(n_sig):
            dx = (self.sigmas_f[i] - self.x).reshape(-1, 1)
            dz = (sigmas_h[i] - z_pred).reshape(-1, 1)
            Pxz += self.Wc[i] * (dx @ dz.T)

        # K = Pxz * inv(S)
        K = _safe_solve(S, Pxz.T).T

        y = z - z_pred
        self.x = self.x + K @ y
        self.P = self.P - K @ S @ K.T
        self.P = _force_sym(self.P)

    def step(self, z: np.ndarray):
        self.predict()
        self.update(z)
        return self.x.copy(), self.P.copy()

# =========================
# Your simulation (same physics), filter = UKF
# =========================
R_std = 1.0
tau = 1e-2
dt = 0.25

x_mean = np.array([-8.0, -8.0, 1.0, 1.0])
std_theta = 2.0
P_init = np.diag([std_theta, std_theta, 0.01, 0.01]).astype(float)

F = np.array([
    [1, 0, dt, 0],
    [0, 1, 0, dt],
    [0, 0, 1, 0 ],
    [0, 0, 0, 1 ],
], dtype=float)

Q_base = np.array([
    [(dt**3)/3, 0,         (dt**2)/2, 0        ],
    [0,         (dt**3)/3, 0,         (dt**2)/2],
    [(dt**2)/2, 0,         dt,        0        ],
    [0,         (dt**2)/2, 0,         dt       ],
], dtype=float)

Q_std = tau * Q_base
R = np.eye(num_sensors, dtype=float) * (R_std**2)

# initial true state draw (same idea as your EKF code)
position = np.random.multivariate_normal(mean=x_mean, cov=P_init)

# Filter init
x0_est = x_mean.copy()

def fx(x):
    return F @ x

ukf = UKF(
    fx=fx,
    hx=h,
    x0=x0_est,
    P0=P_init,
    Q=Q_std,   # IMPORTANT: use the same scaled process noise you simulate with
    R=R,
    alpha=0.3,
    beta=2.0,
    kappa=0.0,
)

# Simulation params
T = 20
true_targets = np.zeros((4, T), dtype=float)
measurements = np.zeros((num_sensors, T), dtype=float)
mu = np.zeros((4, T), dtype=float)

for t in range(T):
    if t == 0:
        true_targets[:, t] = position
    else:
        true_targets[:, t] = F @ true_targets[:, t-1] + np.random.multivariate_normal(np.zeros(4), Q_std)

    measurements[:, t] = h(true_targets[:, t]) + np.random.multivariate_normal(np.zeros(num_sensors), R)

    x_est, P_est = ukf.step(measurements[:, t])
    mu[:, t] = x_est

plt.figure(figsize=(10, 8))
plt.plot(true_targets[0, :], true_targets[1, :], 'r-', label='True Target', linewidth=2)
plt.scatter(true_targets[0, :], true_targets[1, :], color='r', s=50)

plt.plot(mu[0, :], mu[1, :], 'k-', label='UKF Estimation', linewidth=2)
plt.scatter(mu[0, :], mu[1, :], color='k', marker='s', s=50)

plt.scatter(sensor_coords[:, 0], sensor_coords[:, 1], c='blue', marker='^', s=100, label='Sensors')

plt.title("True Target Trajectory vs UKF Estimation")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend(loc="best")
plt.grid(True)
plt.tight_layout()
plt.show()