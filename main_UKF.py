import numpy as np
from math import sqrt
import matplotlib.pyplot as plt

from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints

# -----------------------------
# Same constants / setup
# -----------------------------
P0 = 1e4
n = 2

num_sensors_x = 5
num_sensors_y = 5
num_sensors = num_sensors_x * num_sensors_y
rng_values = np.linspace(-20, 20, num_sensors_x)

sensor_coords = np.array([[x, y] for x in rng_values for y in rng_values])

def h(x_t):
    """Measurement model: returns 25x1 vector (amplitudes)"""
    z_t = np.zeros(len(sensor_coords))
    for i, (x_i, y_i) in enumerate(sensor_coords):
        d_i_t = sqrt((x_i - x_t[0])**2 + (y_i - x_t[1])**2)
        z_t[i] = sqrt(P0 / (1 + d_i_t**n))
    return z_t

# -----------------------------
# Simulation / filter parameters
# -----------------------------
x_mean = np.array([-8, -8, 1, 1])
std_theta = 2
p_varience = np.diag([std_theta, std_theta, 0.1, 0.1])
position = np.random.multivariate_normal(x_mean, p_varience)

R_std = 1
to = 1e-2
dt = 0.25

# State transition matrix (for fx)
F = np.array([[1, 0, dt, 0],
              [0, 1, 0, dt],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])

# Process noise (same construction as your code)
Q = np.array([[((dt**3)/3), 0, ((dt**2)/2), 0],
              [0, ((dt**3)/3), 0, ((dt**2)/2)],
              [((dt**2)/2), 0, dt, 0],
              [0, ((dt**2)/2), 0, dt]])

Q_std = to * Q  # used only for truth simulation (same as your code)

R = np.eye(num_sensors) * R_std**2

# -----------------------------
# UKF definitions 
# -----------------------------
def fx(x, dt_local):
    """State transition for UKF: x_k+1 = F x_k (same model as EKF predict)"""
    return F @ x

# Sigma points
# (Typical stable defaults; can tune alpha/beta/kappa if needed)
points = MerweScaledSigmaPoints(n=4, alpha=0.3, beta=2.0, kappa=0.0)

ukf = UKF(dim_x=4, dim_z=num_sensors, fx=fx, hx=h, dt=dt, points=points)

# Initial state/covariance 
ukf.x = x_mean.copy()
ukf.P = np.diag([std_theta, std_theta, 0.01, 0.01])

ukf.Q = Q.copy()
ukf.R = R.copy()

# -----------------------------
# Simulation loop 
# -----------------------------
total_time = 20
T = total_time  
true_targets = np.zeros((4, T))
measurements = np.zeros((num_sensors, T))
binary = np.zeros((num_sensors, T))
mu = np.zeros((4, T))

for t in range(T):
    if t == 0:
        true_targets[:, t] = position
    else:
        true_targets[:, t] = F @ true_targets[:, t-1] + np.random.multivariate_normal(
            np.zeros(4), Q_std
        )

    measurements[:, t] = h(true_targets[:, t]) + np.random.multivariate_normal(
        np.zeros(num_sensors), R
    )
    binary[:, t] = (measurements[:, t] > 12).astype(int)

    # UKF predict + update
    ukf.predict()
    ukf.update(measurements[:, t])

    mu[:, t] = ukf.x

# -----------------------------
# Plot results
# -----------------------------
plt.figure(figsize=(10, 8))
plt.plot(true_targets[0, :], true_targets[1, :], 'r-', label='True Target', linewidth=2)
plt.scatter(true_targets[0, :], true_targets[1, :], color='r', s=50)

plt.plot(mu[0, :], mu[1, :], 'k-', label='Estimation (UKF)', linewidth=2)
plt.scatter(mu[0, :], mu[1, :], color='k', marker='s', s=50)

plt.scatter(sensor_coords[:, 0], sensor_coords[:, 1], c='blue', marker='^', s=100, label='Sensors')

plt.title("True Target Trajectory vs Estimation (UKF)")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend(loc="best")
plt.grid(True)
plt.tight_layout()
plt.show()
