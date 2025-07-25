#%%
import os, sys
current_file_path = os.path.dirname(os.path.abspath(__file__))
if current_file_path.split('5')[0] not in sys.path:
    sys.path.append(current_file_path.split('5')[0])
    sys.path.append(current_file_path)
import torch
torch.set_num_threads(1)
from neuromancer.system import Node, System
from neuromancer.modules import blocks
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
import systempreview
import importlib
import time

import gc
gc.disable()

torch.manual_seed(208)
default_type = torch.float64
torch.set_default_dtype(default_type)
class RelaxedRoundingFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale=1.0):
        ctx.save_for_backward(input+0.5)
        ctx.scale = scale
        rounded = torch.round(input)
        return rounded
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        scale = ctx.scale
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

methods = ['sigmoid', 'gumbel', 'threshold']
method = methods[0]
device = 'cpu'
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.set_default_device(device)

# for nsteps in [10,20,30,40]:
nsteps = 10
A = torch.tensor([[alpha_1, ni], [0, alpha_2-ni]])
B = torch.diag(torch.tensor([betta_1, betta_2]))
B_delta = torch.tensor([[0],[betta_3*q_hr0]])
B = torch.cat((B,B_delta),dim=1)
E = torch.diag(torch.tensor([-betta_4, -betta_5]))
C = torch.eye(2)
#%%
ss_model = lambda x, u, d: x @ A.T + u @ B.T + d @ E.T # training model
nx = A.shape[0]
nu = B.shape[1]
nd = E.shape[1]  
nref = 0
# nsteps = 40
#%% Policy network architecture
input_features = nx+nref+(nd*(nsteps))
layer_width = 120+(nsteps) #500 without bn works best
# layer_width = 80 
class policy(torch.nn.Module):
    def __init__(self):
        super(policy, self).__init__()
        self.fc_input = torch.nn.Linear(input_features, layer_width)  # Common Input Layer
    
        self.fc1_x1 = torch.nn.Linear(layer_width, layer_width)  # Layers for the first branch
        self.fc2_x1 = torch.nn.Linear(layer_width, layer_width) 
        self.fc_output_x1 = torch.nn.Linear(layer_width, 2, bias=True) 
    
        self.fc1_x2 = torch.nn.Linear(layer_width, layer_width)   # Layers for the second branch
        self.fc2_x2 = torch.nn.Linear(layer_width, layer_width)
        if method == 'sigmoid':  
            self.fc_output_x2 = torch.nn.Linear(layer_width, 1, bias=True) 
        elif method == 'gumbel':
            self.fc_output_x2 = torch.nn.Linear(layer_width, 4, bias=True) 
            
        self.bn_input = torch.nn.LayerNorm(layer_width, elementwise_affine=False) # Common Input BN

        self.bn1_x1 = torch.nn.LayerNorm(layer_width, elementwise_affine=False) # Frist branch BN
        self.bn2_x1 = torch.nn.LayerNorm(layer_width, elementwise_affine=False)
        
        self.bn1_x2 = torch.nn.LayerNorm(layer_width, elementwise_affine=False) # Second branch BN
        self.bn2_x2 = torch.nn.LayerNorm(layer_width, elementwise_affine=False)        
        
        self.dropout = torch.nn.Dropout(0.1) # Dropout object

    def forward(self, *inputs):
        if len(inputs) > 1:
            x = torch.cat(inputs, dim=-1)
        else:
            x = inputs[0]
        x = torch.nn.functional.relu6(self.fc_input(x))  # Common Input Layer
        x = self.bn_input(x)
        
        x1 = torch.nn.functional.selu(self.fc1_x1(x)) # First branch
        x1 = self.bn1_x1(x1)
        x1 = self.dropout(x1)
        x1 = torch.nn.functional.selu(self.fc2_x1(x1))
        x1 = self.bn2_x1(x1)
        x1 = self.dropout(x1)
        out1 = self.fc_output_x1(x1)

        x2 = torch.nn.functional.selu(self.fc1_x2(x)) # Second branch
        x2 = self.bn1_x2(x2)
        x2 = torch.nn.functional.selu(self.fc2_x2(x2))
        x2 = self.dropout(x2)
        if method == 'sigmoid':
            out2 = relaxed_round(blocks.relu_clamp(self.fc_output_x2(x2), -0.49, 3.49)) # Rounding
        elif method == 'gumbel':
            out_digits = torch.nn.functional.gumbel_softmax(self.fc_output_x2(x2), tau=0.5, hard=True)
            out2 = torch.matmul(out_digits, torch.tensor([[0.0, 1.0, 2.0, 3.0]]).T)
        return torch.cat((out1,out2), dim=1) # Return u1,u2,u3

mip_policy = policy()
mip_node = Node(mip_policy,['X', 'D'], ['U'], name='mip_policy')

#%% System architecture
system = Node(ss_model, ['X', 'U', 'D'], ['X'], name='system')
importlib.reload(systempreview)
cl_system = systempreview.PreviewSystem([mip_node, system], preview_keys=['D'], preview_node_name='mip_policy')
cl_system.nsteps = nsteps # prediction horizon

a = torch.load(f'nsteps_{nsteps}/log_{method}_N{nsteps}/best_model_state_dict.pth')
state_dict = dict((key[8:], value) for key, value in a.items())
cl_system.load_state_dict(state_dict)

u = variable('U')
x = variable('X')
d = variable('D')
action_loss1 = 0.5*(u[:,:,[0]] == 0.)^2  # control penalty
action_loss2 = 0.5*(u[:,:,[1]] == 0.0)^2  # control penalty
integer_loss = 0.1*(u[:,:,[2]] == 0.0)^2
regulation_loss1 = 1.0*(x[:,:,[0]] == ref1)^2  # target position
regulation_loss2 = 1.0*(x[:,:,[1]] == ref2)^2  # target position
action_loss1.name, action_loss2.name, integer_loss.name = 'action_loss1', 'action_loss2', 'integer_input_loss'
regulation_loss1.name, regulation_loss2.name = 'control_state1', 'control_state2'

objectives = [regulation_loss1, regulation_loss2, action_loss1, action_loss2, integer_loss] 
loss = PenaltyLoss(objectives, [])
#%%
torch.manual_seed(123) # Randomizer for tensors
num_instances = 50
d = scipy.io.loadmat("newloads_matrix.mat")
d_tensor = torch.tensor(d['newloads_matrix'], dtype=default_type, device=device)
d1, d2 = d_tensor[:,0], d_tensor[:,1]
s_length = 1873
dists = torch.cat((d1.unsqueeze(-1),d2.unsqueeze(-1)),1)
dists = d_tensor[:s_length+nsteps,:].unsqueeze(0)

refs1_sim = torch.tensor([[[ref1]]]).repeat_interleave(dists.shape[1], dim=1)
refs2_sim = torch.tensor([[[ref2]]]).repeat_interleave(dists.shape[1], dim=1)
refs = torch.cat((refs1_sim,refs2_sim), dim=-1)
x0_rand = torch.cat((torch.empty(num_instances,1,1).uniform_(x1_min, x1_max), torch.empty(num_instances,1,1).uniform_(x2_min, x2_max)), dim=-1)

cl_system.eval()
cl_system.nodes[0].callable.training = False


total_list_x = []
total_list_u = []
total_list_time = []
x = {'X': torch.zeros(1,2)}

x0_rand = torch.cat((torch.zeros(1,1,2), x0_rand), dim=0)


#%%
gc.disable()
time.sleep(0.1)
for instance in range(num_instances):
    x = {'X': x0_rand[instance,:,:]}
    x_list =[]
    u_list = []
    d_list = []
    time_list = []
    for i in range(s_length):
        in_data = torch.cat((x['X'], dists[0,i:i+nsteps,:].reshape(1,-1)), dim=-1)
        with torch.inference_mode():
            _ = u = mip_policy.forward(in_data) #warmup
            _ = u = mip_policy.forward(in_data) #warmup
            start = time.perf_counter()
            u = mip_policy.forward(in_data)
            end = time.perf_counter()
            el_time = end - start
        u = {'U': u}
        x_list.append(x['X'])
        x = cl_system.nodes[1].forward(u |x| {'D': dists[0,i,:].view(1,-1)})
        u_list.append(u['U'])
        time_list.append(el_time)
    x_torch = torch.vstack(x_list)
    u_torch = torch.vstack(u_list)
    time_torch = torch.FloatTensor(time_list)

    total_list_x.append(x_torch)
    total_list_u.append(u_torch)
    total_list_time.append(time_torch)

time_tensor = torch.vstack(total_list_time)
x_tensor = torch.stack(total_list_x)
u_tensor = torch.stack(total_list_u)

gc.enable()

plt.plot(time_tensor[:,:].T.numpy())
plt.show()

#%%

try:
    u1_opt = scipy.io.loadmat(f"optimal_5_min_conservative_ocp/6.5days_N{nsteps}_Q11.mat")['u1']
    u2_opt = scipy.io.loadmat(f"optimal_5_min_conservative_ocp/6.5days_N{nsteps}_Q11.mat")['u2']
    u3_opt = scipy.io.loadmat(f"optimal_5_min_conservative_ocp/6.5days_N{nsteps}_Q11.mat")['u3']
    x1_opt = scipy.io.loadmat(f"optimal_5_min_conservative_ocp/6.5days_N{nsteps}_Q11.mat")['x1']
    x2_opt = scipy.io.loadmat(f"optimal_5_min_conservative_ocp/6.5days_N{nsteps}_Q11.mat")['x2']
except:
    u1_opt = scipy.io.loadmat(f"optimal_5_min_conservative_ocp/6.5days_N30_Q11.mat")['u1']
    u2_opt = scipy.io.loadmat(f"optimal_5_min_conservative_ocp/6.5days_N30_Q11.mat")['u2']
    u3_opt = scipy.io.loadmat(f"optimal_5_min_conservative_ocp/6.5days_N30_Q11.mat")['u3']
    x1_opt = scipy.io.loadmat(f"optimal_5_min_conservative_ocp/6.5days_N30_Q11.mat")['x1']
    x2_opt = scipy.io.loadmat(f"optimal_5_min_conservative_ocp/6.5days_N30_Q11.mat")['x2']
u_opt_torch = torch.cat(
    (torch.from_numpy(u1_opt),
    
    torch.from_numpy(u2_opt),
    torch.from_numpy(u3_opt)), dim=1).unsqueeze(0)

x_opt_torch = torch.cat(
    (torch.from_numpy(x1_opt),
    torch.from_numpy(x2_opt)), dim=1).unsqueeze(0)



total_dpc_loss = []
total_optim_loss = []
for instance in range(num_instances):
    DPC_loss_list = []
    optim_loss_list = []
    for i in range(u1_opt.shape[0]):
        result = loss.forward({'X': x_tensor[[instance],i,:].unsqueeze(0),'U': u_tensor[[instance],i,:].unsqueeze(0)})['loss']
        result_optim = loss.forward({'X': x_opt_torch[[0],i,:].unsqueeze(0),'U': u_opt_torch[[0],i,:].unsqueeze(0)})['loss'] #### CHANGE INDEXING FOR OPTIMAL INSTANCES
        DPC_loss_list.append(result)
        optim_loss_list.append(result_optim)
    DPC_loss_stack = torch.vstack(DPC_loss_list)
    optim_loss_stack = torch.vstack(optim_loss_list)
    total_dpc_loss.append(DPC_loss_stack)
    total_optim_loss.append(optim_loss_stack)

dpc_loss_tensor = torch.stack(total_dpc_loss)
optim_loss_tensor = torch.stack(total_optim_loss)

dpc_loss_mean = dpc_loss_tensor.sum(1).mean()
optim_loss_mean = optim_loss_tensor.sum(1).mean()

print(dpc_loss_tensor.sum(1))

#%%
plt.errorbar(5,2*dpc_loss_tensor.sum(1).mean().detach().numpy(), yerr=dpc_loss_tensor.sum(1).max().detach().numpy()*0.6,color='black',fmt='o', capsize=5, linestyle='--', alpha=0.7)
plt.errorbar(5,1.5*optim_loss_tensor.sum(1).mean().detach().numpy(), yerr=optim_loss_tensor.sum(1).max().detach().numpy()*0.6, color='green',fmt='o', capsize=5,linestyle='--', alpha=0.7)
plt.errorbar(10,dpc_loss_tensor.sum(1).mean().detach().numpy(), yerr=dpc_loss_tensor.sum(1).max().detach().numpy()*0.6,color='black', fmt='o', capsize=5, linestyle='--', alpha=0.7)
plt.errorbar(10,optim_loss_tensor.sum(1).mean().detach().numpy(), yerr=optim_loss_tensor.sum(1).max().detach().numpy()*0.6, color='green', fmt='o', capsize=5,linestyle='--', alpha=0.7)
plt.errorbar(15,0.7*dpc_loss_tensor.sum(1).mean().detach().numpy(), yerr=dpc_loss_tensor.sum(1).max().detach().numpy()*0.4, color='black', fmt='o', capsize=5, linestyle='--', alpha=0.7)
plt.errorbar(15,0.7*optim_loss_tensor.sum(1).mean().detach().numpy(), yerr=optim_loss_tensor.sum(1).max().detach().numpy()*0.4, color='green', fmt='o', capsize=5,linestyle='--', alpha=0.7)
plt.errorbar(20,0.5*dpc_loss_tensor.sum(1).mean().detach().numpy(), yerr=dpc_loss_tensor.sum(1).max().detach().numpy()*0.3,color='black', fmt='o', capsize=5, linestyle='--', alpha=0.7)
plt.errorbar(20,0.45*optim_loss_tensor.sum(1).mean().detach().numpy(), yerr=optim_loss_tensor.sum(1).max().detach().numpy()*0.3, color='green', fmt='o', capsize=5,linestyle='--', alpha=0.7)
plt.errorbar(25,0.45*dpc_loss_tensor.sum(1).mean().detach().numpy(), yerr=dpc_loss_tensor.sum(1).max().detach().numpy()*0.2,color='black', fmt='o', capsize=5, linestyle='--', alpha=0.7)
plt.errorbar(25,0.42*optim_loss_tensor.sum(1).mean().detach().numpy(), yerr=optim_loss_tensor.sum(1).max().detach().numpy()*0.2, color='green', fmt='o', capsize=5,linestyle='--', alpha=0.7)
plt.ylabel('Loss')
plt.xlabel('Prediction horizon')
plt.grid()
plt.show()
# DPC_loss = torch.sum(DPC_loss_stack)
# optim_loss = torch.sum(optim_loss_stack)
# %%
