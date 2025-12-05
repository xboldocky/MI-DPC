#%%
import os
import warnings
warnings.simplefilter('error', UserWarning)
import cvxpy as cp
import numpy as np
from tqdm import tqdm
import cplex, scipy, torch, time, argparse

parser = argparse.ArgumentParser(description="Example with only long option.")
parser.add_argument("--solver", type=str, help="MIP Solver [cplex, gurobi]")
parser.add_argument("--nsteps", type=int, help="Prediction horizon length")

args = parser.parse_args()

solver = cp.GUROBI if args.solver == 'gurobi' else cp.CPLEX
cplex_params = {"mip.tolerances.absmipgap": 1e-10, 'threads': 1} if solver == cp.CPLEX else None
print(f"Solving with {solver}")

torch.manual_seed(206)
for var in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ.pop(var, None)



q_hr0, ni = 1., 0.0010
alpha_1, alpha_2 = 0.9983, 0.9966
betta_1, betta_2, betta_3, betta_4, betta_5 = 0.0750, 0.0750, 0.0825, 0.0833, 0.0833

x1_min, x2_min, input_energy_min = 0.0, 0.0, 0.0
x1_max, x2_max, input_energy_max = 8.4, 3.6, 8
u_int_min, u_int_max = -0.49, 3.49
ref1, ref2 = 4.2, 1.8

A = np.array([[alpha_1, ni],
              [0, alpha_2 - ni]])
B = np.diag([betta_1, betta_2])
B_delta = np.array([[0], [betta_3 * q_hr0]])
B = np.hstack((B, B_delta))
E = np.diag([-betta_4, -betta_5])

# Parameter space sampling
num_data = 24000
dist_data_torch = torch.load('training_data/extended_disturbances_60.pt')
dist_data_torch = dist_data_torch[0:num_data]
x1_torch = torch.empty(num_data, 1, 1).uniform_(x1_min, x1_max)
x2_torch = torch.empty(num_data, 1, 1).uniform_(x2_min, x2_max)
x_data_torch = torch.cat((x1_torch, x2_torch), dim=2)
# To numpy
dist_data_np = dist_data_torch.detach().cpu().numpy()
x_data_np = x_data_torch.detach().cpu().numpy()


# State and input sizes
nx, nu = 2, 3
# Cost weights
Q = np.eye(nx)        # state tracking
R = np.diag([0.5,0.5,0.1])        # input penalty
ref_c = np.array([ref1, ref2])
# total simulation steps

STOP_TIME = 2 # Two hours maximum time per instance

#%%
# for N in [10,15,20,25,30,35,40]: # prediction horizon
N = args.nsteps
print(f'RUNNING EXPERIMENTS WITH HORIZON N={N}')
# CVXPY variables
x = cp.Variable((N+1, 2))
u = cp.Variable((N, 2))
u_int = cp.Variable((N,1), integer=True)
u_full = cp.hstack([u, cp.reshape(u_int, (N,1), order='C')])

x_list = []; u_list = []; d_list = []; status = []
infeasible = 0
start_time = time.time()

for k, x0 in enumerate(tqdm(x_data_np, total=num_data, miniters=num_data//100, maxinterval=float("inf"))):
    if time.time() - start_time > STOP_TIME * 60 * 60 :
        print(f"STOP: Reached 2-hour limit. Computed {k+1} samples for N={N}")
        break
    # disturbance horizon
    d_horizon = dist_data_np[k,:N,:]
    
    # constraints & cost
    x_current = x0.copy()
    constraints = [x[0,:] == x0[0]]
    cost = 0

    for t in range(N):
        constraints += [x[t+1,:] == A @ x[t,:] +  B @ u_full[t,:] + E @ d_horizon[t,:]]
        constraints += [
            u_full[t,0]+u_full[t,1] >= input_energy_min,
            u_full[t,0]+u_full[t,1] <= input_energy_max,
            u_full[t,0] >= 0.0,
            u_full[t,1] >= 0.0,
            u_int[t] >= 0, u_int[t] <= 3,
            x[t, 0] >= x1_min,
            x[t, 0] <= x1_max,
            x[t, 1] >= x2_min,
            x[t, 1] <= x2_max
        ]
        cost += cp.quad_form(x[t+1,:] - ref_c, Q)
        cost += cp.quad_form(u_full[t,:], R)
    
    prob = cp.Problem(cp.Minimize(cost), constraints)
    try:
        prob.solve(solver=solver, verbose=False, warm_start=False,
            cplex_params=cplex_params)
    except cp.error.SolverError:
        continue

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        infeasible += 1
        continue
    # Append computed d,x and u pairs
    u_list.append(u_full.value)
    x_list.append(x.value[[0],:]) 
    d_list.append(d_horizon)
    status.append(prob.status)

# Save Data
states = np.stack(x_list)
inputs = np.stack(u_list)
dists = np.stack(d_list)
np.savez(f"imitation_learning_data/data_N{N}.npz", 
X = states, U = inputs, D = dists, stop_time = time.time() - start_time,
    num_infeasible = infeasible, solver = solver
)