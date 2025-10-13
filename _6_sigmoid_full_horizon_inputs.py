
# %% Imports
import os, sys
import torch
from neuromancer.system import Node, System
from neuromancer.modules import blocks, functions
from neuromancer.dataset import DictDataset
from neuromancer.constraint import variable
from neuromancer.loss import PenaltyLoss, BarrierLoss, AugmentedLagrangeLoss
from neuromancer.problem import Problem
from neuromancer.trainer import Trainer
from neuromancer.plot import pltCL, pltPhase
from neuromancer.loggers import BasicLogger
import matplotlib.pyplot as plt
import numpy as np
import scipy
from utils import systempreview
import importlib
script_dir = os.path.dirname(os.path.abspath(__file__))
default_type = torch.float32
torch.set_default_dtype(default_type)

class RelaxedRoundingFunction(torch.autograd.Function): # Define sigmoid STE
    @staticmethod
    def forward(ctx, input, scale=1.0):
        ctx.save_for_backward(input+0.5) # 0.5 - torch.round threshold
        ctx.scale = scale
        # scaled sigmod to approximate rounding
        rounded = torch.round(input)
        return rounded
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        scale = ctx.scale
        # derivative of the sigmoid for backward pass
        sigmoid_approx = torch.sigmoid(scale*(input-torch.round(input)))
        grad_input = grad_output*sigmoid_approx*(1-sigmoid_approx)*scale
        return grad_input, None

temperature_coefficient = 10.0
def relaxed_round(input, scale=temperature_coefficient):
    return RelaxedRoundingFunction.apply(input, scale)
#%%
q_hr0 = 1
alpha_1 = 0.9983
alpha_2 = 0.9966
ni = 0.0010
betta_1 = 0.0750
betta_2 = 0.0750
betta_3 = 0.0825
betta_4 = 0.0833
betta_5 = 0.0833

x1_min, x2_min, input_energy_min = 0.0, 0.0, 0.0
x1_max = 8.4
x2_max = 3.6
d1_max = 7
d2_max = 17
input_energy_max = 8

u_int_max = 3.49
u_int_min = -0.49

ref1 = 4.2
ref2 = 1.8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

A = torch.tensor([[alpha_1, ni], [0, alpha_2-ni]])
B = torch.diag(torch.tensor([betta_1, betta_2]))
B_delta = torch.tensor([[0],[betta_3*q_hr0]])
B = torch.cat((B,B_delta),dim=1)
E = torch.diag(torch.tensor([-betta_4, -betta_5]))
C = torch.eye(2)
# ss_model = lambda x, u, d: x @ A.T + u @ B.T + d @ E.T # training model

# def ss_model(x, u, d):
    # return x @ A.T + u @ B.T + d @ E.T
def ss_model(x, u, d):
    return (A @ x.T + B @ u.T + E @ d.T).T

system = Node(ss_model, ['X', 'U', 'D'], ['X'], name='system')
importlib.reload(systempreview)
# cl_system = systempreview.PreviewSystem([mip_node, system], preview_keys=['D'], preview_node_name='mip_policy')
cl_system = System([system], name='cl_system')
nsteps = 35
cl_system.nsteps = nsteps # prediction horizon

test_data = {'X': torch.ones(6,1,2), 
            'U': torch.cat((torch.ones(6,nsteps,1)*.1, torch.ones(6,nsteps,1)*.2, torch.ones(6,nsteps,1)*.3), dim=-1),
            'D': torch.cat((torch.ones(6,nsteps,1)*0.0, torch.ones(6,nsteps,1)*1.0), -1)
            }
test_result = cl_system(test_data)

print(test_result['U'][0,:,0])
print(test_result['U'][0,:,1])
print(test_result['U'][0,:,2])

print(test_result['X'].shape)
print(test_result['U'].shape)
print(test_result['D'].shape)

#%%
nx = A.shape[0]
nu = B.shape[1]
nd = E.shape[1]  
nref = 0
batch_size = 2000

#%% Policy network architecture
# for nsteps in [10,15,20,25,30,40]:
for nsteps in [35]:
    torch.manual_seed(208)
    input_features = nx+nref+(nd*(nsteps))
    layer_width = 140 #500 without bn works best
    class policy(torch.nn.Module):
        def __init__(self, layer_width=layer_width):
            super(policy, self).__init__()
            self.fc_input = torch.nn.Linear(input_features, layer_width)  # Common Input Layer
        
            self.fc1_x1 = torch.nn.Linear(layer_width, layer_width)  # Layers for the first branch
            self.fc2_x1 = torch.nn.Linear(layer_width, layer_width) 
            self.fc_output_x1 = torch.nn.Linear(layer_width, nsteps, bias=True) 
            
            self.fc1_x12 = torch.nn.Linear(layer_width, layer_width)  # Layers for the first branch
            self.fc2_x12 = torch.nn.Linear(layer_width, layer_width) 
            self.fc_output_x12 = torch.nn.Linear(layer_width, nsteps, bias=True) 
        
            self.fc1_x2 = torch.nn.Linear(layer_width, layer_width)   # Layers for the second branch
            self.fc2_x2 = torch.nn.Linear(layer_width, layer_width)  
            self.fc_output_x2 = torch.nn.Linear(layer_width, 1*nsteps, bias=True) 
            
            self.bn_input = torch.nn.LayerNorm(layer_width, elementwise_affine=False) # Common Input Layer norm

            self.bn1_x1 = torch.nn.LayerNorm(layer_width, elementwise_affine=False) # Frist branch Layer norm
            self.bn2_x1 = torch.nn.LayerNorm(layer_width, elementwise_affine=False)
            
            self.bn1_x12 = torch.nn.LayerNorm(layer_width, elementwise_affine=False) # Frist branch Layer norm
            self.bn2_x12 = torch.nn.LayerNorm(layer_width, elementwise_affine=False)
            
            self.bn1_x2 = torch.nn.LayerNorm(layer_width, elementwise_affine=False) # Second branch Layer norm
            self.bn2_x2 = torch.nn.LayerNorm(layer_width, elementwise_affine=False)        
            self.dropout = torch.nn.Dropout(0.1) # Dropout object

        def forward(self, *inputs):
            inputs = list(inputs)
            if len(inputs) > 1:
                n_xi = len(inputs)
                for i in range(n_xi):
                    inputs[i]=inputs[i].reshape(inputs[0].size(0),-1)
                    # print(inputs[i].shape)
                x = torch.cat(inputs, dim=-1)
                # print(x.shape)
                # print('asdfasd')
            else:
                x = inputs[0]
            x = torch.nn.functional.tanh(self.fc_input(x))  # Common Input Layer
            x = self.bn_input(x)
            
            x1 = torch.nn.functional.tanh(self.fc1_x1(x)) # Continous input module
            x1 = self.bn1_x1(x1)
            x1 = self.dropout(x1)
            x1 = torch.nn.functional.tanh(self.fc2_x1(x1))
            x1 = self.bn2_x1(x1)
            x1 = self.dropout(x1)
            out1 = self.fc_output_x1(x1)   
           
            x12 = torch.nn.functional.tanh(self.fc1_x12(x)) # Continous input module
            x12 = self.bn1_x12(x12)
            x12 = self.dropout(x12)
            x12 = torch.nn.functional.tanh(self.fc2_x12(x12))
            x12 = self.bn2_x12(x12)
            x12 = self.dropout(x12)
            out12 = self.fc_output_x12(x12)

            x2 = torch.nn.functional.selu(self.fc1_x2(x)) # Integer input module
            x2 = self.bn1_x2(x2)
            x2 = torch.nn.functional.selu(self.fc2_x2(x2))
            x2 = self.bn2_x2(x2)
            out2 = relaxed_round(functions.bounds_clamp(self.fc_output_x2(x2), -0.49, 3.49)) # Rounding
            # out2 = relaxed_round(torch.clip(self.fc_output_x2(x2), -0.49, 3.49)) # Rounding
            # out2 = relaxed_round(self.fc_output_x2(x2)) # Rounding
            output = torch.cat((out1.reshape(inputs[0].size(0),nsteps,-1),
            out12.reshape(inputs[0].size(0),nsteps,-1), 
            out2.reshape(inputs[0].size(0),nsteps,-1)), dim=-1)
            output.shape
            return output # Return u1,u2,u3
    
    mip_policy = policy()
    mip_node = Node(mip_policy,['X', 'D'], ['U'], name='mip_policy')

    test_policy_data = mip_node(test_data)
    print(test_policy_data['U'][:,:,2])
    #%% Training Data
    num_data = 24000
    num_dev_data = 4000
    batch_size = 2000
    nref = nx
    file_path = os.path.join(script_dir, 'loads_matrix.mat')
    d = scipy.io.loadmat(file_path)
    d_tensor = torch.tensor(d['newloads_matrix'], dtype=default_type, device=device)

    d1 = d_tensor[:,0]
    d2 = d_tensor[:,1]

    x1_train = torch.empty(num_data, 1, 1, dtype=default_type).uniform_(x1_min, x1_max)
    x2_train = torch.empty(num_data, 1, 1, dtype=default_type).uniform_(x2_min, x2_max)
    x_train = torch.cat((x1_train, x2_train), dim=2)

    x1_dev = torch.empty(num_dev_data, 1, 1, dtype=default_type).uniform_(x1_min, x1_max)
    x2_dev = torch.empty(num_dev_data, 1, 1, dtype=default_type).uniform_(x2_min, x2_max)
    x_dev = torch.cat((x1_dev, x2_dev), dim=2)

    dist_data = torch.load(f'{script_dir}/training_data/extended_disturbances_60.pt')
    dist_data = torch.tensor(dist_data, dtype=default_type, device=device)
    
    train_data = DictDataset({'X': x_train.to(device), 'D': dist_data[:num_data,:nsteps,:].to(device)}, name='train')  # Split conditions into train and dev
    dev_data = DictDataset({'X': x_dev[:num_dev_data,:,:].to(device), 'D': dist_data[num_data:num_data+num_dev_data,:nsteps,:].to(device)}, name='dev')
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                            collate_fn=train_data.collate_fn, shuffle=False)
    dev_loader = torch.utils.data.DataLoader(dev_data, batch_size=batch_size,
                                            collate_fn=dev_data.collate_fn, shuffle=False)

    # %% OCP definition
    u = variable('U')
    x = variable('X')
    d = variable('D')

    action_loss1 = 0.5*(u[:,:,0] == 0.)^2  # control penalty
    action_loss2 = 0.5*(u[:,:,1] == 0.0)^2  # control penalty
    integer_loss = 0.1*(u[:,:,2] == 0.0)^2
    regulation_loss1 = 1.0*(x[:,:,[0]] == ref1)^2  # target position
    regulation_loss2 = 1.0*(x[:,:,[1]] == ref2)^2  # target position
 
    action_loss1.name, action_loss2.name, integer_loss.name = 'action_loss1', 'action_loss2', 'integer_input_loss'
    regulation_loss1.name, regulation_loss2.name = 'control_state1', 'control_state2'

    objectives = [regulation_loss1, regulation_loss2, action_loss1, action_loss2, integer_loss] 

    input1_con_l = 250.0*(u[:,:,0] >= 0.0)
    input2_con_l = 250.0*(u[:,:,1] >= 0.0)
    # input3_con_l = 250.0*(u[:,:,[2]] >= 0.0)
    # input3_con_u = 250.0*(u[:,:,[2]] <= 3.0)
    input_energy_con_u = 250.0*((u[:,:,0]+u[:,:,1]) < input_energy_max)
    input_energy_con_l = 250.0*((u[:,:,0]+u[:,:,1]) >= 0.0)

    state1_con_l = 250.0*(x[:,:,[0]] >= x1_min)
    state1_con_u = 250.0*(x[:,:,[0]] < x1_max)
    state2_con_l = 250.0*(x[:,:,[1]] >= x2_min)
    state2_con_u = 250.0*(x[:,:,[1]] < x2_max)

    input1_con_l.name, input2_con_l.name = 'int1_l', 'int2_l'
    input_energy_con_u.name, input_energy_con_l.name = 'input_energy_con_u', 'input_energy_con_l'
    # input3_con_l.name, input3_con_u.name = 'u3_l', 'u3_u'
    state1_con_l.name, state1_con_u.name = "x1_l", 'x1_u'
    state2_con_l.name, state2_con_u.name = 'state_2_lower', 'state_2_upper'

    constraints = [
                    input1_con_l,
                    input2_con_l,
                    input_energy_con_l,
                    input_energy_con_u,
                    state1_con_l,
                    state1_con_u,
                    state2_con_l,
                    state2_con_u,
                    # input3_con_l,
                    # input3_con_u,
                                ]

    loss = PenaltyLoss(objectives, constraints)
    problem = Problem([mip_node, cl_system], loss)
   
    for name, param in cl_system.named_parameters():
        if param.grad is not None:
            print(name, param.grad.shape)
    # %%
    optimizer = torch.optim.Adam(mip_node.parameters(), lr=0.0003, amsgrad=False, weight_decay=0.0)
    logger = BasicLogger(args=None, savedir=f'training_outputs/sigmoid_full_horizon/logs/log_sigmoid_N{nsteps}', verbosity=1, stdout=['train_loss', 'dev_loss', 'eltime'])

    trainer = Trainer(
        problem.to(device),
        train_loader, dev_loader,
        optimizer=optimizer,
        epochs=1000,
        # epochs=10,
        train_metric='train_loss',
        dev_metric='dev_loss',
        eval_metric='dev_loss',
        warmup=20,
        patience=80,
        epoch_verbose=1,
        device=device,
        clip=torch.inf,
        lr_scheduler=False,
        logger=logger
    )
    if __name__ == "__main__":

        best_model = trainer.train()
        trainer.model.load_state_dict(best_model) # load best trained model
        
        torch.save(cl_system, f'training_outputs/sigmoid_full_horizon/models/model_sigmoid_N{nsteps}.pt')
        problem.load_state_dict(best_model)

    # %%
    """
    Simulation
    """
    # ============================================================
    # Receding Horizon Simulator
    # ============================================================
    import time
    import numpy as np
    import matplotlib.pyplot as plt
    # s_length = 1873 # Simulation length
    s_length = 200 # Simulation length
    dists = torch.cat((d1.unsqueeze(-1),d2.unsqueeze(-1)),1)
    dists = d_tensor[:s_length+nsteps,:].unsqueeze(0)
    data = {'X': torch.zeros(1, 1, nx),
            'D': dists}
    def receding_horizon_simulation(
        x0,
        d_tensor,
        sim_length,
        nsteps,
        device,
        mip_node,
        ss_model,
        A, B, E,
        verbose=True
    ):
        """
        Run a receding-horizon (MPC-like) simulation using the learned policy `mip_node`.

        Parameters
        ----------
        x0 : torch.Tensor
            Initial state, shape (1,1,nx).
        d_tensor : torch.Tensor
            Disturbance sequence, shape (T, nd) or (1, T, nd).
        sim_length : int
            Number of simulation steps.
        nsteps : int
            Horizon length.
        device : torch.device
            Torch device.
        mip_node : Node
            Policy node, maps {'X','D'} -> {'U'}.
        ss_model : callable
            State transition function: x_{t+1} = ss_model(x, u, d).
        A,B,E : torch.Tensor
            System matrices (used for nx, nu, nd).
        verbose : bool
            Print progress every 500 steps.

        Returns
        -------
        dict with:
            'X_traj' : (sim_length+1, nx)
            'U_traj' : (sim_length, nu)
            'D_traj' : (sim_length, nd)
        """
        # Ensure batch dimension for disturbances
        if d_tensor.ndim == 2:
            d_tensor = d_tensor.unsqueeze(0)   # (1, T, nd)

        nd = d_tensor.shape[-1]
        nx = A.shape[0]
        nu = B.shape[1]

        # Initial state
        x_t = x0.to(device)

        # Storage
        X_traj = torch.zeros(sim_length+1, nx, dtype=default_type, device=device)
        U_traj = torch.zeros(sim_length, nu, dtype=default_type, device=device)
        D_traj = torch.zeros(sim_length, nd, dtype=default_type, device=device)

        X_traj[0] = x_t.reshape(-1)

        start_time = time.time()
        with torch.no_grad():
            for t in range(sim_length):

                # Disturbance preview for horizon [t:t+nsteps)
                d_preview = d_tensor[:, t:t+nsteps, :].to(device)  # (1, nsteps, nd)

                # Policy input
                policy_in = {'X': x_t, 'D': d_preview}
                out = mip_node(policy_in)   # {'U': (1,nsteps,nu)}
                U_horizon = out['U']

                # First action
                u_t = U_horizon[:, 0, :].reshape(1, 1, -1)

                # Current disturbance for dynamics
                d_t = d_tensor[:, t, :].reshape(1, 1, -1).to(device)

                # Step dynamics
                x_next = ss_model(
                    x_t.reshape(1, -1),
                    u_t.reshape(1, -1),
                    d_t.reshape(1, -1)
                )
                if x_next.ndim == 2:
                    x_next = x_next.reshape(1, 1, -1)

                # Record
                X_traj[t+1] = x_next.reshape(-1)
                U_traj[t] = u_t.reshape(-1)
                D_traj[t] = d_t.reshape(-1)

                # Update
                x_t = x_next

                if verbose and (t % 500 == 0):
                    elapsed = time.time() - start_time
                    print(f"t={t}/{sim_length} | x={X_traj[t+1].cpu().numpy()} | u={U_traj[t].cpu().numpy()} | elapsed={elapsed:.1f}s")

        total_time = time.time() - start_time
        if verbose:
            print(f"Simulation finished: {sim_length} steps in {total_time:.1f}s")

        return {
            'X_traj': X_traj.cpu(),
            'U_traj': U_traj.cpu(),
            'D_traj': D_traj.cpu()
        }

    # ============================================================
    # Run the simulator
    # ============================================================
    # Disturbance tensor (ensure correct shape: (1,T,nd))
    # You had earlier:
    # dists = d_tensor[:s_length+nsteps,:].unsqueeze(0)
    d_input = dists.to(device)   # already (1,T,nd)

    sim_result = receding_horizon_simulation(
        x0=data['X'].to(device),
        d_tensor=d_input,
        sim_length=s_length,
        nsteps=cl_system.nsteps,
        device=device,
        mip_node=mip_node,
        ss_model=ss_model,
        A=A, B=B, E=E,
        verbose=True
    )

    # ============================================================
    # Quick plots
    # ============================================================
    X = sim_result['X_traj'].numpy()
    U = sim_result['U_traj'].numpy()
    D = sim_result['D_traj'].numpy()

    t = np.arange(X.shape[0])
    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.plot(t, X[:,0], label='x1')
    plt.plot(t, X[:,1], label='x2')
    plt.title('States')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(U[:,0], label='u1')
    plt.plot(U[:,1], label='u2')
    plt.plot(U[:,2], label='u3 (int)')
    plt.title('Control inputs')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Save results
    torch.save(sim_result, f'training_outputs/sigmoid_full_horizon/sim_result_N{nsteps}.pt')
    print("Simulation result saved.")

        # torch.save(trajectories, f"authdata_nsteps_{nsteps}/sigmoid_trajectories.pt")

#%%
# ============================================================
# Warm Start MPC Simulation (DPC -> CPLEX)
# ============================================================
import cvxpy as cp
import numpy as np
from tqdm import tqdm
import torch
import scipy.io
import matplotlib.pyplot as plt

def warm_start_simulation(
    x0,
    d_array,
    s_length,
    nsteps,
    A, B, E,
    mip_node,
    device,
    ref1=4.2,
    ref2=1.8,
    Q=None,
    R=None,
    verbose=True
):
    """
    MPC with warm start from DPC policy (mip_node).

    Parameters
    ----------
    x0 : np.ndarray
        Initial state, shape (2,).
    d_array : np.ndarray
        Disturbance sequence, shape (T,2).
    s_length : int
        Simulation length.
    nsteps : int
        Prediction horizon.
    A, B, E : np.ndarray
        System matrices.
    mip_node : Node
        Learned policy Node (expects {'X','D'}).
    device : torch.device
        Torch device for DPC policy.
    ref1, ref2 : float
        State references.
    Q, R : np.ndarray
        Cost weights.
    verbose : bool
        Print/log progress.

    Returns
    -------
    dict
        X_traj, U_traj, D_traj, times, stage_cost
    """
    nx, nu = 2, 3
    if Q is None:
        Q = np.eye(nx)
    if R is None:
        R = np.diag([0.5, 0.5, 0.1])

    # unpack disturbances
    d1, d2 = d_array[:, 0], d_array[:, 1]

    x_current = x0.copy()
    x_traj = [x_current.copy()]
    u_traj = []
    sol_time = []

    for k in range(s_length):
            # === Step 1: DPC warm start proposal ===
        d_preview = d_array[k:k+nsteps, :]
        d_torch = torch.tensor(d_preview, dtype=torch.float32).unsqueeze(0).to(device)  # (1,N,2)
        x_torch = torch.tensor(x_current, dtype=torch.float32).reshape(1,1,-1).to(device)

        with torch.no_grad():
            policy_out = mip_node({'X': x_torch, 'D': d_torch})
            U_dpc = policy_out['U'].cpu().numpy().reshape(nsteps, -1)  # (N,3)
            # X_dpc = policy_out['X'].cpu().numpy().reshape(nsteps+1, -1)  # (N,3)
        X_dpc = np.zeros((nsteps+1, nx))
        X_dpc[0,:] = x_current
        for t in range(nsteps):
            X_dpc[t+1,:] = A @ X_dpc[t,:] + B @ U_dpc[t,:] + E @ d_preview[t,:]
        # === Step 2: Build CVXPY MPC problem ===
        x = cp.Variable((nsteps+1, 2))
        u = cp.Variable((nsteps, 2))
        u_int = cp.Variable((nsteps,), integer=True)
        u_full = cp.hstack([u, cp.reshape(u_int, (nsteps,1))])


        constraints = [x[0,:] == x_current]
        cost = 0
        for t in range(nsteps):
            constraints += [x[t+1,:] == A @ x[t,:] + B @ u_full[t,:] + E @ d_preview[t,:]]
            constraints += [
                u_full[t,0]+u_full[t,1] >= 0.0,
                u_full[t,0]+u_full[t,1] <= 8.0,
                u_full[t] >= 0.0,
                u_full[t,2] <= 3.0,
                x[t,0] >= 0.0,
                x[t,0] <= 8.4,
                x[t,1] >= 0.0,
                x[t,1] <= 3.6,
            ]
            cost += cp.quad_form(x[t+1,:] - np.array([ref1, ref2]), Q)
            cost += cp.quad_form(u_full[t,:], R)

        prob = cp.Problem(cp.Minimize(cost), constraints)
        # warm start assignment
        try:
            u.value = U_dpc[:, :2]
            u_int.value = np.rint(U_dpc[:, 2])  # round to nearest integer
            # u_full.value = U_dpc
            # x.value = X_dpc
        except Exception as e:
            if verbose:
                print(f"[Warm start warning] Could not assign DPC values at step {k}: {e}")
        prob.solve(solver=cp.GUROBI, verbose=False, warm_start=True,
                    #  Heuristics=0.5,
    Cuts=0,
    # MIPGap=0.001,
    # MIPFocus=3, # increases sol time
    Heuristics=0., # slightly reduces sol time
    # PreSolve=0, # increases sol time
    # PrePasses=0, # increases sol time
    # TimeLimit = 0.1
                    
    #                  GurobiParams={
    #                   "Heuristics": 0.5,  # fraction of time for heuristics
    #                     "MIPGap": 0.001,    # optional: small gap for early termination
    #                     "Cuts": 0,          # disable cuts
    #                     "WarmStart": 1      # optional: small gap for early termination
    # }
                #    cplex_params={
                #        "mip.tolerances.absmipgap": 1e-16,
                #        "mip.tolerances.mipgap": 1e-16, 
                #        "mip.tolerances.integrality": 1e-16,
                #        "simplex.tolerances.feasibility": 1e-9,
                #        "barrier.convergetol": 1e-12,
                #        "threads": 1
                #    }
                   )

        if prob.status not in ["optimal", "optimal_inaccurate"]:
            raise RuntimeError(f"Solver failed at step {k}, status {prob.status}")

        # === Step 3: Apply first input ===
        u_opt = u_full.value[0,:]
        u_traj.append(u_opt)
        sol_time.append(prob.solver_stats.solve_time)

        # update dynamics
        x_current = A @ x_current + B @ u_opt + E @ np.array([d1[k], d2[k]])
        x_traj.append(x_current.copy())

        # if verbose and (k % 1 == 0):
            # print(f"[Warm start step {k}] x={x_current}, u={u_opt}, time={sol_time[-1]:.4f}s")
        print(f"step={k}",f"  time={prob.solver_stats.solve_time}")

    # === Collect results ===
    X_traj = np.stack(x_traj)
    U_traj = np.stack(u_traj)
    D_traj = d_array[:s_length,:]

    stage_cost = np.array([
        (X_traj[t] - [ref1, ref2]).T @ Q @ (X_traj[t] - [ref1, ref2]) +
        U_traj[t].T @ R @ U_traj[t] for t in range(s_length)
    ])

    return {
        "X_traj": X_traj,
        "U_traj": U_traj,
        "D_traj": D_traj,
        "times": np.array(sol_time),
        "stage_cost": stage_cost,
        "initial_condition": x0,
        "nsteps": nsteps
    }


# ============================================================
# Example Usage
# ============================================================
if __name__ == "__main__":
    # load disturbances
    d = scipy.io.loadmat("loads_matrix.mat")
    d_array = d['newloads_matrix']
    # s_length = 500  # shorter run for test

    # system matrices
    A = np.array([[alpha_1, ni],
                  [0, alpha_2 - ni]])
    B = np.diag([betta_1, betta_2])
    B_delta = np.array([[0],[betta_3*q_hr0]])
    B = np.hstack((B,B_delta))
    E = np.diag([-betta_4, -betta_5])

    # initial condition
    x0 = np.array([0.0, 0.0])

    # run warm start MPC
    result = warm_start_simulation(
        x0=x0,
        d_array=d_array,
        s_length=s_length,
        nsteps=nsteps,
        A=A, B=B, E=E,
        mip_node=mip_node,
        device=device,
        verbose=True
    )
    print(result['times'].mean())
    # plot states and controls
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(result["X_traj"][:,0], label="x1")
    plt.plot(result["X_traj"][:,1], label="x2")
    plt.title("States (Warm Start MPC)")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(result["U_traj"][:,0], label="u1")
    plt.plot(result["U_traj"][:,1], label="u2")
    plt.plot(result["U_traj"][:,2], label="u3 (int)")
    plt.title("Controls (Warm Start MPC)")
    plt.legend()
    plt.show()

    # %%
