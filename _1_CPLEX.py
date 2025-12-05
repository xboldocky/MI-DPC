#%%
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import cvxpy as cp
import numpy as np
from tqdm import tqdm
from utils import initial_conditions
import cplex, scipy, torch
from argparse import ArgumentParser
import time

parser = ArgumentParser()
parser.add_argument('-ic', default=0, type=int, help='Initial Condition index in range [0,19]')
args, unknown = parser.parse_known_args()

q_hr0, ni = 1, 0.0010
alpha_1, alpha_2 = 0.9983, 0.9966
betta_1, betta_2, betta_3, betta_4, betta_5 = 0.0750, 0.0750, 0.0825, 0.0833, 0.0833

x1_min, x2_min, input_energy_min = 0.0, 0.0, 0.0
x1_max, x2_max, input_energy_max = 8.4, 3.6, 8
d1_max, d2_max = 7, 17
u_int_min, u_int_max = 0., 3.
ref1, ref2 = 4.2, 1.8

A = np.array([[alpha_1, ni],
              [0, alpha_2 - ni]])
B = np.diag([betta_1, betta_2])
B_delta = np.array([[0], [betta_3 * q_hr0]])
B = np.hstack((B, B_delta))
E = np.diag([-betta_4, -betta_5])
C = np.eye(2)

d = scipy.io.loadmat("loads_matrix.mat")
d_array = d['newloads_matrix']
d1, d2 = d_array[:, 0], d_array[:, 1]
s_length = 1873

# State and input sizes
nx, nu = 2, 3
# Cost weights
Q = np.eye(nx)        # state tracking
R = np.diag([0.5,0.5,0.1])        # input penalty
# total simulation steps
T = len(d1)

initial_tensor = np.array(initial_conditions.x0_tensor)  # shape (20,2)
#%%
all_results = {}

print(f"Running simulation with initial state {initial_tensor[args.ic,:]}, index {args.ic}")
for N in [10,15,20,25,30,35]: # prediction horizon
    print(f'RUNNING EXPERIMENTS WITH HORIZON N={N}')
    results = []
    x0 = initial_tensor[args.ic,:]

    x_current = x0.copy()
    x_traj = [x_current.copy()]
    u_traj = []
    sol_time = []
    num_iterations = []
    start_time = time.time(); time_limit = 7200 # 2 hours time limit
    for k in tqdm(range(s_length), miniters=10):
        # disturbance horizon
        d_horizon = np.column_stack((d1[k:k+N], d2[k:k+N]))
        
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
            constraints += [x[t+1,:] == A @ x[t,:] +  B @ u_full[t,:] + E @ d_horizon[t,:]]
            constraints += [
                u_full[t,0]+u_full[t,1] >= input_energy_min,
                u_full[t,0]+u_full[t,1] <= input_energy_max,
                u_full[t] >= 0, u_int[t] <= 3,
                x[t, 0] >= x1_min,
                x[t, 0] <= x1_max,
                x[t, 1] >= x2_min,
                x[t, 1] <= x2_max
            ]
            cost += cp.quad_form(x[t+1,:] - np.array([ref1, ref2]), Q)
            cost += cp.quad_form(u_full[t,:], R)
        
        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(solver=cp.CPLEX, verbose=False, 
                cplex_params={"mip.tolerances.absmipgap": 1e-16,
                                "mip.tolerances.mipgap": 1e-16, 
                                "mip.tolerances.integrality": 1e-16,
                                'simplex.tolerances.feasibility': 1e-9,
                                'barrier.convergetol': 1e-12,
                                    "threads": 1})
        
        u_opt = u_full.value[0,:]
        u_traj.append(u_opt)
        sol_time.append(prob.solver_stats.solve_time)
        num_iterations.append(prob.solver_stats.num_iters)
        # x_current = x_current @ A.T + u_opt @ B.T + np.array([d1[k], d2[k]]) @ E.T
        x_current = A @ x_current + B @ u_opt + E @ np.array([d1[k], d2[k]])
        x_traj.append(x_current.copy())
        if time.time() - start_time > time_limit: # Exceeding time limit
            print(f"Time limit exceeded at {time.time()-start_time}")
            break
    
    # Compute stage cost
    states = np.stack(x_traj)
    inputs = np.stack(u_traj)
    iterations = np.array(num_iterations)
    stage_cost = np.array([ (states[t]-[ref1, ref2]).T @ Q @ (states[t]-[ref1, ref2]) 
                            + inputs[t].T @ R @ inputs[t] 
                            for t in range(len(u_traj)) ])
    
    # Store result for this initial condition
    results.append({
        "x": states,
        "u": inputs,
        "d": d_array,
        # "iterations": iterations,
        "times": np.array(sol_time),
        "stage_cost": stage_cost,
        "initial_condition": x0,
        "ic_index": args.ic,
        "nsteps": N
    })
    # all_results[f'N={N}'] = results
    #%% Save all results to a single file
    # np.savez(f"CPLEX_inference_data/cvxpy_cplex_N{N}.npz", data=all_results)
    os.makedirs(f"CPLEX_inference_data/N{N}/", exist_ok=True)
    torch.save(results, f'CPLEX_inference_data/N{N}/cvxpy_cplex_ic{args.ic}.pt')
print('...Run finished...')