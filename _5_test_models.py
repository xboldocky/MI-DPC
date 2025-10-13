#%%
import torch
import neuromancer
import scipy
import numpy as np
from _2_sigmoid import policy, loss
import matplotlib.pyplot as plt

    # cl_system = torch.load(
        # 'training_outputs/sigmoid/models/model_sigmoid_N20.pt', weights_only=False
    # )
torch.manual_seed(206)
torch.set_default_dtype(torch.float32)
torch.set_default_device('cpu')
# nstep_list = [10, 15, 20, 25, 30]
nstep_list = [10, 15, 20, 25, 30, 35, 40]

for nsteps in nstep_list:
    nup = 0
    stage_cost = []
    time = []
    data = np.load(f"CPLEX_inference_data/cvxpy_cplex_N{nsteps}.npz", allow_pickle=True)
    all_results = data['data']
    for ic in range(20):
        this_ic = all_results[ic] # 0 is the initial condition
        stage_cost.append(this_ic['stage_cost'].mean())
        time.append(this_ic['times'])
        for i in range(this_ic['times'].shape[0]):
            if this_ic['times'][i] >= 30:
                nup += 1
    mean_stage_cost = np.vstack(stage_cost).mean()
    mean_inferenece_time = np.vstack(time).mean()
    print("Nsteps", nsteps)
    print("Number of samples", this_ic['times'].shape)
    print("Mean cost", mean_stage_cost)
    print("Mean inference time", mean_inferenece_time)
    print('NUP: ', nup/(1873*20)*100, '%')

# %%

for nsteps in nstep_list:
    im_data = np.load(f"imitation_learning_data/data_N{nsteps}.npz", allow_pickle=True)
    
#%%

mat_data = scipy.io.loadmat(f"CPLEX_inference_data/N{nsteps}.mat")
# Extract the main variable
optimal_data = mat_data[f"N{nsteps}_20"]
# Assign each feature to a variable
optimal_data = {f"x0_{i+1}": optimal_data[0, i] for i in range(optimal_data.shape[1])}
mat_data = scipy.io.loadmat(f"CPLEX_inference_data/N10.mat")
# Extract the main variable
optimal_data = mat_data[f"N10_20"]
# Assign each feature to a variable
optimal_data = {f"x0_{i+1}": optimal_data[0, i] for i in range(optimal_data.shape[1])}
# s_length = optimal_data[f'x0_1'][D_index].shape[0]

StageCost_index = 2
Time_index = 3
U_index = 4
X_index = 5
D_index = 6

cvxpy = np.load(f"CPLEX_inference_data/cvxpy_cplex_N{10}.npz", allow_pickle=True)['data']

# plt.plot(cvxpy[0]['x'][100:300], 'k--')
# plt.plot(optimal_data['x0_1'][X_index][100:300], ':')

keys = optimal_data.keys()

optim_loss = []
init_conditions = []
for i in keys:
    init_conditions.append(optimal_data[i][X_index][0])
    optim_loss.append(optimal_data[i][StageCost_index].mean())

print(np.vstack(optim_loss).mean())