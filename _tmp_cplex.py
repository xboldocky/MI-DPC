#%%
# import os
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
import cvxpy as cp
import numpy as np
from tqdm import tqdm
from utils import initial_conditions
import cplex, scipy, torch

device = torch.device("cpu")
torch.set_default_device(device)

q_hr0, ni = 1, 0.0010
alpha_1, alpha_2 = 0.9983, 0.9966
betta_1, betta_2, betta_3, betta_4, betta_5 = 0.0750, 0.0750, 0.0825, 0.0833, 0.0833

x1_min, x2_min, input_energy_min = 0.0, 0.0, 0.0
x1_max, x2_max, input_energy_max = 8.4, 3.6, 8
d1_max, d2_max = 7, 17
u_int_min, u_int_max = -0.49, 3.49
ref1, ref2 = 4.2, 1.8

A = np.array([[alpha_1, ni],
              [0, alpha_2 - ni]])
B = np.diag([betta_1, betta_2])
B_delta = np.array([[0], [betta_3 * q_hr0]])
B = np.hstack((B, B_delta))
E = np.diag([-betta_4, -betta_5])
C = np.eye(2)

# ss_model = lambda x, u, d: x @ A.T + u @ B.T + d @ E.T # training model
d = scipy.io.loadmat("loads_matrix.mat")
# d_tensor = torch.tensor(d['newloads_matrix'], device=device)
# d1_torch, d2_torch = d_tensor[:,0], d_tensor[:,1]
d_array = d['newloads_matrix']
d1, d2 = d_array[:, 0], d_array[:, 1]
# s_length = 1873
s_length = 200

#%%
N = 35  # prediction horizon

# State and input sizes
nx, nu = 2, 3

# Cost weights
Q = np.eye(nx)        # state tracking
R = np.diag([0.5,0.5,0.1])        # input penalty
# total simulation steps
T = len(d1)

# initial state
x_current = np.array([0.0, 0.0])

# preallocate state and control trajectories
x_traj = [x_current.copy()]
u_traj = []
sol_time = []
for k in range(s_length):
    # create disturbance horizon for current step
    d_horizon = np.column_stack((d1[k:k+N], d2[k:k+N]))  # shape (N, 2)

    # CVXPY variables
    x = cp.Variable((N+1, 2))
    u = cp.Variable((N, 2))
    u_int = cp.Variable((N,), integer=True)
    u_full = cp.hstack([u, cp.reshape(u_int, (N,1), order='C')])

    # constraints & cost
    constraints = [x[0,:] == x_current]
    cost = 0
    for t in range(N):
        # constraints += [x[t+1,:] == x[t,:] @ A.T + u_full[t,:] @ B.T + d_horizon[t,:] @ E.T]
        constraints += [x[t+1,:] == A @ x[t,:] + B @ u_full[t,:] + E @ d_horizon[t,:]]
        
        constraints += [
            u_full[t,0]+u_full[t,1] >= input_energy_min, 
            u_full[t,0]+u_full[t,1] <= input_energy_max,
            u_full[t] >= 0, u_int[t] <= 3,
            x[t, 0] >= x1_min,  # x1 lower bound
            x[t, 0] <= x1_max,  # x1 upper bound
            x[t, 1] >= x2_min,  # x2 lower bound
            x[t, 1] <= x2_max   # x2 upper bound
        ]
        cost += cp.quad_form(x[t+1,:] - np.array([ref1, ref2]), Q)
        cost += cp.quad_form(u_full[t,:], R)
    
    # solve MPC
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver=cp.GUROBI, verbose=False, warm_start=True,
    # cplex_params={"mip.tolerances.absmipgap": 1e-8, "threads": 1}
    )
    print(f"step={k}",f"  time={prob.solver_stats.solve_time}")
    # apply first control
    u_opt = u_full.value[0,:]
    u_traj.append(u_opt)
    sol_time.append(prob.solver_stats.solve_time)
    # simulate system one step
    # x_current = x_current @ A.T + u_opt @ B.T + np.array([d1[k], d2[k]]) @ E.T
    x_current =  A @ x_current +  B @ u_opt + E @ np.array([d1[k], d2[k]])
    x_traj.append(x_current.copy())

#%% Stack the tensors
states = np.stack(x_traj)
inputs = np.stack(u_traj)
times = np.stack(sol_time)

x_ref = np.array([ref1, ref2])
stage_cost = [] 
for t in range(len(u_traj)): # Compute the stage costs
    x_err = states[t] - x_ref         
    u = inputs[t]                     
    cost_t = x_err.T @ Q @ x_err + u.T @ R @ u
    stage_cost.append(cost_t)
stage_cost = np.array(stage_cost)

#%% Save the data
# np.savez(f"CPLEX_inference_data/cvxpy_cplex_N{nsteps}.npz",
#          x=states,
#          u=inputs,
#          disturbances=d_array,
#          times=sol_time,
#          stage_cost=stage_cost,
#          nsteps=N
#          )
print(times.mean())

#%%
import matplotlib.pyplot as plt

plt.plot(states[:,0])
plt.plot(np.zeros(states.shape[0]))
plt.plot(x1_max*np.ones(states.shape[0]))
plt.plot(ref1*np.ones(states.shape[0]))
plt.show()

plt.plot(states[:,1])
plt.plot(np.zeros(states.shape[0]))
plt.plot(x2_max*np.ones(states.shape[0]))
plt.plot(ref2*np.ones(states.shape[0]))
plt.show()

plt.plot(inputs[:,0]+inputs[:,1])
plt.show()

plt.plot(inputs[:,-1])