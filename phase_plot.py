#%%
import os, sys
current_file_path = os.path.dirname(os.path.abspath(__file__))
if current_file_path.split('5')[0] not in sys.path:
    sys.path.append(current_file_path.split('5')[0])
    sys.path.append(current_file_path)
    # sys.path.append(current_file_path)

import torch
from neuromancer.system import Node, System
from neuromancer.modules import blocks
from neuromancer.dataset import DictDataset
from neuromancer.constraint import variable
from neuromancer.loss import PenaltyLoss, BarrierLoss, AugmentedLagrangeLoss
from neuromancer.problem import Problem
from neuromancer.trainer import Trainer
from neuromancer.plot import pltCL, pltPhase
import matplotlib.pyplot as plt
import numpy as np
import scipy
import systempreview
import importlib

torch.manual_seed(206)
default_type = torch.float32
torch.set_default_dtype(default_type)
temperature_coefficient = 0.5
#%% Define Dynamics Model

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

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'
torch.set_default_device(device)

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
num_digits = 4
nsteps = 20
#%% Policy network architecture
input_features = nx+nref+(nd*(nsteps))
layer_width =120+(nsteps) #500 without bn works best
class policy(torch.nn.Module):
    def __init__(self):
        super(policy, self).__init__()
        self.fc_input = torch.nn.Linear(input_features, layer_width)  # Common Input Layer
    
        self.fc1_x1 = torch.nn.Linear(layer_width, layer_width)  # Layers for the first branch
        self.fc2_x1 = torch.nn.Linear(layer_width, layer_width) 
        self.fc_output_x1 = torch.nn.Linear(layer_width, 2, bias=True) 
    
        self.fc1_x2 = torch.nn.Linear(layer_width, layer_width)   # Layers for the second branch
        self.fc2_x2 = torch.nn.Linear(layer_width, layer_width)  
        self.fc_output_x2 = torch.nn.Linear(layer_width, num_digits, bias=True) 
        
        self.bn_input = torch.nn.LayerNorm(layer_width, elementwise_affine=False) # Common Input BN

        self.bn1_x1 = torch.nn.LayerNorm(layer_width, elementwise_affine=False) # Frist branch BN
        self.bn2_x1 = torch.nn.LayerNorm(layer_width, elementwise_affine=False)
        
        self.bn1_x2 = torch.nn.LayerNorm(layer_width, elementwise_affine=False) # Second branch BN
        self.bn2_x2 = torch.nn.LayerNorm(layer_width, elementwise_affine=False)        
        
        self.dropout = torch.nn.Dropout(0.05) # Dropout object

    def forward(self, *inputs):
        if len(inputs) > 1:
            x = torch.cat(inputs, dim=-1)
        else:
            x = inputs[0]
        x = torch.nn.functional.selu(self.fc_input(x))  # Common Input Layer
        x = self.dropout(x)
        x1, x2 = self.bn_input(x), self.bn_input(x)
        
        x1 = torch.nn.functional.selu(self.fc1_x1(x1)) # First branch
        x1 = self.bn1_x1(x1)
        x1 = self.dropout(x1)
        x1 = torch.nn.functional.selu(self.fc2_x1(x1))
        x1 = self.bn2_x1(x1)
        x1 = self.dropout(x1)
        out1 = self.fc_output_x1(x1)
        # out1 = blocks.sigmoid_scale(self.fc_output_x1(x1), 0.0, input_energy_max)  # Identity
        x2 = torch.nn.functional.selu(self.fc1_x2(x2)) # Second branch
        x2 = self.bn1_x2(x2)
        x2 = self.dropout(x2)
        x2 = torch.nn.functional.selu(self.fc2_x2(x2))
        x2 = self.bn2_x2(x2)
        x2 = self.dropout(x2)
        out_digits = torch.nn.functional.gumbel_softmax(self.fc_output_x2(x2), tau=0.5, hard=True)
        out2 = torch.matmul(out_digits, torch.tensor([[0.0, 1.0, 2.0, 3.0]]).T)
        # out2 = relaxed_round(blocks.relu_clamp(self.fc_output_x2(x2), -0.49, 3.49)) # Rounding
        
        return torch.cat((out1,out2), dim=1) # Return u1,u2,u3


mip_policy = policy()
mip_node = Node(mip_policy,['X', 'D'], ['U'], name='mip_policy')

#%% System architecture
system = Node(ss_model, ['X', 'U', 'D'], ['X'], name='system')
importlib.reload(systempreview)
cl_system = systempreview.PreviewSystem([mip_node, system], preview_keys=['D'], preview_node_name='mip_policy')
cl_system.nsteps = nsteps # prediction horizon

state_dict = torch.load(f'nsteps_{nsteps}/gumbel_states_model.pt', map_location=torch.device('cpu'))
cl_system.load_state_dict(state_dict)

# print(state_dict == cl_system.state_dict())

# %%


d = scipy.io.loadmat("newloads_matrix.mat")
d_tensor = torch.tensor(d['newloads_matrix'], dtype=default_type, device=device)
d1 = d_tensor[:,0]
d2 = d_tensor[:,1]


dists = torch.cat((d1.unsqueeze(-1),d2.unsqueeze(-1)),1)
dists = d_tensor[200:260,:].unsqueeze(0)


cl_system.eval()
traj_list = []

for i in range(0,15):

    x1_init = torch.empty(1,1,1).uniform_(x1_min,x1_max)
    x2_init = torch.empty(1,1,1).uniform_(x2_min,x2_max)
    x_init = torch.cat((x1_init,x2_init), -1)
    data = {'X': x_init, 'D': dists}
    traj = cl_system.simulate(data)
    traj_list.append(traj)



fig, ax = plt.subplots(1)
colors = list(plt.cm.tab10(np.arange(10))) + list(plt.cm.viridis(np.arange(20,step=20)))
ax.set_prop_cycle('color', colors)
fig.set_figwidth(9.2/2.54)
for i in range(len(traj_list)):
    ax.plot(traj_list[i]['X'][0,:,0].detach().numpy(), traj_list[i]['X'][0,:,1].detach().numpy(), '--', linewidth=0.7, color='black')

for i in range(len(traj_list)):
    ax.plot(traj_list[i]['X'][0,0,0].detach().numpy(), traj_list[i]['X'][0,0,1].detach().numpy(), 'r.', markersize=5)
    ax.plot(traj_list[i]['X'][0,-1,0].detach().numpy(), traj_list[i]['X'][0,-1,1].detach().numpy(), 'g.',markersize=5)

ax.set_xlabel('$x_{1}$')
ax.set_ylabel('$x_{2}$')
ax.set_xlim(x1_min, x1_max)
ax.set_ylim(x2_min, x2_max)
ax.set_aspect('equal')
# ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
ax.grid()
plt.show()
# %%
