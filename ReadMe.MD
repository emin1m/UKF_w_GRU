# Unscented Kalman Filter (UKF) for 2D Target Tracking

This repository implements a **2D target tracking system** using a **manually implemented Unscented Kalman Filter (UKF)** without relying on any external filtering libraries.

The target is tracked using **nonlinear amplitude measurements** collected from a grid of fixed sensors.

---

## Problem Description

- **State vector** : x = [x, y, vx, vy]


where:
- `(x, y)` is the target position  
- `(vx, vy)` is the target velocity  

The target follows a **constant-velocity motion model** and is observed by a grid of fixed sensors.

---

## Sensor and Measurement Model

- **Number of sensors:** 25 (5 × 5 grid)
- **Sensor layout:** Uniform grid between −20 and +20 on both axes
- **Measurement type:** Signal amplitude
- **Measurement model:** z_i = sqrt(P0 / (1 + d_i^n))



where:
- `d_i` is the distance between the target and sensor `i`
- `P0` and `n` are model parameters

The measurement model is **strongly nonlinear**, making UKF a suitable choice.

---

## Filter Overview

- **Filter type:** Unscented Kalman Filter (UKF)
- **Sigma-point formulation:** Merwe scaled sigma points
- **Implementation:** Fully manual (NumPy-based)
- **Jacobian:** Not required (unlike EKF)

The UKF propagates sigma points through both the **motion model** and the **measurement model** to capture nonlinear effects accurately.

---

## Motion Model

The target dynamics follow a constant-velocity model: x_{k+1} = F x_k + w_k


where `w_k` is zero-mean Gaussian process noise.

---

## Repository Structure


.
├── ukf_tracking.py # Main simulation and UKF implementation
├── README.md # Project documentation
└── .gitignore # Git ignore rules


---

## Requirements

Minimal dependencies:

```bash
pip install numpy matplotlib



