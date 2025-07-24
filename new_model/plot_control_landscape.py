# %%
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

torch.manual_seed(205)

class RelaxedRoundingFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale=1.0):
        ctx.save_for_backward(input)
        ctx.scale = scale
        # scaled sigmod to approximate rounding
        rounded = torch.round(input)
        return rounded
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        scale = ctx.scale
        # derivative of the sigmoid for backward passa
        sigmoid_approx = torch.sigmoid(scale*(input-torch.round(input)))
        grad_input = grad_output*sigmoid_approx*(1-sigmoid_approx)*scale
        return grad_input, None

def relaxed_round(input, scale=1.0):
    return RelaxedRoundingFunction.apply(input, scale)
#%%
q_hr0 = 2
alpha_1 = 0.99949
ni = 0.003
alpha_2 = 0.9979
betta_1 = 0.275
betta_2 = 0.192
betta_3 = 0.248
betta_4 = 0.298
betta_5 = 0.339
x1_min = 0.0
x1_max = 8.4
x2_min = 0.0
x2_max = 3.6

d1_max = 3
d2_max = 15

input_energy_max = 11.1
input_energy_min = 0.0

u_int_max = 4.49
u_int_min = -0.49

A = torch.diag(torch.tensor([alpha_1, alpha_2-ni]))
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
nref = nx
nsteps = 20

#%% Policy network architecture
# input_features = nx+nref-1+nd
input_features = nx+nref+(nd*(nsteps+1))
layer_width = 150
class policy(torch.nn.Module):
    def __init__(self):
        super(policy, self).__init__()
        self.fc1 = torch.nn.Linear(input_features, layer_width)  # input 2
        self.fc2 = torch.nn.Linear(layer_width, layer_width) 
        self.fc3 = torch.nn.Linear(layer_width, layer_width) 
        self.fc4 = torch.nn.Linear(layer_width, layer_width) 
        self.fc5 = torch.nn.Linear(layer_width, 2) # linear -> identity
        self.fc6 = torch.nn.Linear(layer_width, 1) # linear -> softmax 
        
        self.bn1 = torch.nn.LayerNorm(layer_width)
        self.bn2 = torch.nn.LayerNorm(layer_width)
        self.bn3 = torch.nn.LayerNorm(layer_width)        
        self.bn4 = torch.nn.LayerNorm(layer_width)        

        self.dropout = torch.nn.Dropout(0.2) # Dropout object

    def forward(self, *inputs):
        if len(inputs) > 1:
            x = torch.cat(inputs, dim=-1)
        else:
            x = inputs[0]
        x = torch.nn.functional.softplus(self.fc1(x))  
        x = self.bn1(x)
        x = self.dropout(x)
        x = torch.nn.functional.softplus(self.fc2(x))
        x = self.bn2(x)
        x = self.dropout(x)
        x = torch.nn.functional.softplus(self.fc3(x))
        x = self.bn3(x)
        x = self.dropout(x)
        x = torch.nn.functional.softplus(self.fc4(x))
        x = self.bn4(x)
        out1 = blocks.sigmoid_scale(self.fc5(x), 0.0, 11.1)  # Identity
        out2 = relaxed_round(blocks.sigmoid_scale(self.fc6(x), u_int_min, u_int_max)) # softmax first discrete
        return torch.cat((out1,out2), dim=1)
        
mip_policy = policy()
mip_node = Node(mip_policy,['X', 'R', 'D'], ['U'], name='mip_policy')
#%% System architecture
system = Node(ss_model, ['X', 'U', 'D'], ['X'], name='system')
importlib.reload(systempreview)
cl_system = systempreview.PreviewSystem([mip_node, system], preview_keys=['D'], preview_node_name='mip_policy')
cl_system.nsteps = nsteps # prediction horizon
# Minor test
d1 = torch.ones(2,nsteps+1,1)
d2 = torch.zeros(2,nsteps+1,1)
dist_vector = torch.cat((d1,d2), dim=-1)

# test_input = {'X': torch.rand(2,1,nx),'R': torch.rand(2,nsteps+1,nref),'D': torch.rand(2,nsteps+1,nd)}
test_input = {'X': torch.rand(2,1,nx),'R': torch.rand(2,nsteps+1,nref),'D': dist_vector}
test_result = cl_system(test_input)
cl_system.load_state_dict(torch.load('/home/jb/git/building_control_new/new_model/softround_states_model.pt'))
QL = scipy.io.loadmat("/home/jb/git/building_control_new/with_disturbance/QL.mat")['QL']
PL = scipy.io.loadmat("/home/jb/git/building_control_new/with_disturbance/PL.mat")['PL']
QL_raw = torch.tensor(QL, dtype=torch.float32)
PL_raw = torch.tensor(PL, dtype=torch.float32)
QL = PL_raw*0.07 #switching PL for QL to mimic realistic setup
PL = QL_raw*0.07
s_length = 300
dists = torch.cat((QL,PL),1)
dists = dists[-s_length:,:]
#%%

num_points = 200
x1 = torch.linspace(x1_min, x1_max, num_points)
x2 = torch.linspace(x2_min, x2_max, num_points)
x1_grid, x2_grid = torch.meshgrid(x1, x2, indexing='ij')  # Use 'ij' for MATLAB-like indexing
# features = torch.stack([x1_grid, x2_grid], dim=-1)
references = [5.5, 2.0]
reference_vector = np.stack([np.full_like(x1_grid, i) for i in references]).reshape(-1,2)

grid_points = np.c_[x1_grid.ravel(), x2_grid.ravel()]
inputs = np.hstack([grid_points, reference_vector])

disturbances = dists[:nsteps+1,:].flatten() # first timestep for all features
# disturbance_vector = np.stack([np.full_like(x1_grid, value)] for value in disturbances).reshape(-1, len(disturbances))
disturbance_vector = np.stack([np.full_like(x1_grid, value)] for value in disturbances)
disturbance_vector = disturbance_vector.squeeze(1).swapaxes(0,-1).reshape(-1, len(disturbances))

inputs_t = torch.cat((torch.tensor(inputs),torch.tensor(disturbance_vector)), dim=-1)

outputs = cl_system.nodes[0].callable(inputs_t)
u1 = outputs[:,0].reshape(x1_grid.shape)
u2 = outputs[:,1].reshape(x1_grid.shape)
u3 = outputs[:,2].reshape(x1_grid.shape)
#%%

plt.figure(figsize=(8, 6))
plt.contourf(x1_grid, x2_grid, u1.detach().numpy(), levels=3, cmap='CMRmap', alpha=0.9)
plt.colorbar(label='NN Output')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('U1')


plt.figure(figsize=(8, 6))
plt.contourf(x1_grid, x2_grid, u2.detach().numpy(), levels=3, cmap='CMRmap', alpha=0.9)
plt.colorbar(label='NN Output')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('U2')

plt.figure(figsize=(8, 6))
plt.contourf(x1_grid, x2_grid, u3.detach().numpy(), levels=3, cmap='CMRmap', alpha=0.9)
plt.colorbar(label='NN Output')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('U3')


fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x1_grid, x2_grid, u1.detach().numpy(), cmap='CMRmap', alpha=0.9)
#%%
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
# Create a surface plot
surf = ax.contourf(x1_grid, x2_grid, u1.detach().numpy(), cmap='viridis', edgecolor='none')
