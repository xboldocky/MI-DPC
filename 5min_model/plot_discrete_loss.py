# %%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the MPC loss function
def mpc_loss(x, u):
    Q = np.array([[10, 0], [0, 10]])  # State cost matrix
    R = np.array([[10]])             # Control cost matrix
    x_cost = np.einsum('ij,ji->i', x @ Q, x.T)  # Quadratic state cost
    u_cost = np.einsum('ij,ji->i', u @ R, u.T)  # Quadratic control cost
    return x_cost + u_cost

# Create a meshgrid with integer control inputs
x1 = np.linspace(-2, 2, 50)  # Continuous state variable
u = np.arange(-2, 3, 1)      # Integer control input (-2, -1, 0, 1, 2)
X1, U = np.meshgrid(x1, u)

# Compute the loss function
X = np.vstack([X1.ravel(), np.zeros_like(X1.ravel())]).T  # Assume x2 = 0
U_vec = U.ravel().reshape(-1, 1)
Z = mpc_loss(X, U_vec).reshape(X1.shape)

# Plot the loss function
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Use plot_surface with discrete u values
ax.plot_surface(X1, U, Z, cmap='viridis', edgecolor='k', alpha=0.8)
ax.scatter(X1, U, Z, color="red", s=10)  # Highlight integer points

# Labels and Title
ax.set_xlabel("State (x1)")
ax.set_ylabel("Control (u) - Integer")
ax.set_zlabel("MPC Loss")
ax.set_title("MPC Loss Function Landscape (Integer Control Input)")

plt.show()
#%%

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# System dynamics: x_k+1 = A*x_k + B*u_k
A = 0.9
B = 0.08
N = 5  # Prediction horizon

# Cost matrices
Q = 1  # State cost
R = 1  # Control cost

# Define the MPC loss function considering dynamics
def mpc_loss(x0, u_seq):
    x = x0
    cost = 0
    for u in u_seq:  # Apply control inputs sequentially
        cost += Q * x**2 + R * u**2
        x = A * x + B * u  # State evolution
    return cost

# Define state x0 range and integer control inputs
x0_vals = np.linspace(-2, 2, 50)  # Initial state x0
u_vals = np.arange(-2, 3, 1)  # Integer control inputs (-2, -1, 0, 1, 2)

# Create a grid
X0, U = np.meshgrid(x0_vals, u_vals)

# Compute loss function values
Z = np.zeros_like(X0)
for i in range(X0.shape[0]):
    for j in range(X0.shape[1]):
        x0 = X0[i, j]
        u_seq = [U[i, j]] * N  # Assume constant control over horizon
        Z[i, j] = mpc_loss(x0, u_seq)

# Plot the loss function
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X0, U, Z, cmap='viridis', edgecolor='k', alpha=0.8)
ax.scatter(X0, U, Z, color="red", s=10)  # Highlight integer points

# Labels and Title
ax.set_xlabel("Initial State (x0)")
ax.set_ylabel("Control (u) - Integer")
ax.set_zlabel("MPC Loss")
ax.set_title(f"MPC Loss Landscape with Dynamics (Horizon N={N})")

plt.show()

#%%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import expit  # Sigmoid function

# System dynamics: x_k+1 = A*x_k + B*u_k
A = 0.9
B = 1.0
N = 20  # Prediction horizon

# Cost matrices
Q = 0.1  # State cost
R = 0.1  # Control cost

# Sigmoid-based STE method for integer control
def ste_sigmoid(u_cont, lambda_val=1.0, u_min=-2, u_max=2):
    u_sigmoid = expit(lambda_val * u_cont)  # Smooth step approximation
    u_ste = u_min + u_sigmoid * (u_max - u_min)  # Scale to control range
    return np.round(u_ste)  # Forward pass applies rounding

# Define the MPC loss function considering dynamics with STE
def mpc_loss(x0, u_cont):
    x = x0
    cost = 0
    u_int = ste_sigmoid(u_cont)  # Convert continuous control to discrete
    for _ in range(N):  # Apply control inputs sequentially
        cost += Q * x**2 + R * u_int**2
        x = A * x + B * u_int  # State evolution
    return cost

# Define state x0 range and continuous control values
x0_vals = np.linspace(-2, 2, 50)  # Initial state x0
u_cont_vals = np.linspace(-3, 3, 50)  # Continuous control variable

# Create a grid
X0, U_cont = np.meshgrid(x0_vals, u_cont_vals)

# Compute loss function values
Z = np.zeros_like(X0)
for i in range(X0.shape[0]):
    for j in range(X0.shape[1]):
        x0 = X0[i, j]
        u_cont = U_cont[i, j]
        Z[i, j] = mpc_loss(x0, u_cont)

# Plot the loss function
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X0, U_cont, Z, cmap='viridis', edgecolor='k', alpha=0.8)
ax.scatter(X0, ste_sigmoid(U_cont), Z, color="red", s=10)  # Highlight integer-converted points

# Labels and Title
ax.set_xlabel("Initial State (x0)")
ax.set_ylabel("Continuous Control Input (before rounding)")
ax.set_zlabel("MPC Loss")
ax.set_title(f"MPC Loss Landscape with Sigmoid-STE (Horizon N={N})")

plt.show()

# %%
