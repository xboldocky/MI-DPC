
#%%
print(3)
# %%

import os, sys
current_file_path = os.path.dirname(os.path.abspath(__file__))
if current_file_path.split('new')[0] not in sys.path:
    sys.path.append(current_file_path.split('new')[0])
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
        # derivative of the sigmoid for backward pass
        sigmoid_approx = torch.sigmoid(scale*(input-torch.round(input)))
        grad_input = grad_output*sigmoid_approx*(1-sigmoid_approx)*scale
        return grad_input, None

def relaxed_round(input, scale=8.0):
    return RelaxedRoundingFunction.apply(input, scale)
#%%
q_hr0 = 1
alpha_1 = 0.995
ni = 0.003
alpha_2 = 0.990
betta_1 = 0.225
betta_2 = 0.225
betta_3 = 0.2475
betta_4 = 0.25
betta_5 = 0.25
x1_min = 0.0
x1_max = 8.4
x2_min = 0.0
x2_max = 3.6

d1_max = 7
d2_max = 17

input_energy_max = 8
input_energy_min = 0.0

u_int_max = 3.49
u_int_min = -0.49

ref1 = 4.2
ref2 = 1.8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

# A = torch.diag(torch.tensor([alpha_1, alpha_2-ni]))

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
# nref = nx
nref = 0
nsteps = 20

#%% Policy network architecture
# input_features = nx+nref-1+nd
input_features = nx+nref+(nd*(nsteps))
layer_width = 300
class policy(torch.nn.Module):
    def __init__(self):
        super(policy, self).__init__()
        self.fc1 = torch.nn.Linear(input_features, layer_width)  # input 2
        self.fc2 = torch.nn.Linear(layer_width, layer_width) 
        self.fc_ = torch.nn.Linear(layer_width, layer_width) 
        self.fc3 = torch.nn.Linear(layer_width, layer_width) 
        self.fc4 = torch.nn.Linear(layer_width, layer_width) 
        self.fc5 = torch.nn.Linear(layer_width, 2) # linear -> identity
        self.fc6 = torch.nn.Linear(layer_width, 4) # linear -> softmax 
        
        self.bn1 = torch.nn.LayerNorm(layer_width)
        self.bn2 = torch.nn.LayerNorm(layer_width)
        self.bn3 = torch.nn.LayerNorm(layer_width)        
        self.bn4 = torch.nn.LayerNorm(layer_width)        
        self.bn = torch.nn.LayerNorm(layer_width)

        self.dropout = torch.nn.Dropout(0.03) # Dropout object

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
        x = torch.nn.functional.softplus(self.fc_(x))
        x = self.bn(x)
        
        x = self.dropout(x)
        x = torch.nn.functional.softplus(self.fc4(x))
        x = self.bn4(x)
        out1 = blocks.sigmoid_scale(self.fc5(x), -0.5, input_energy_max)  # Identity
        out2 = torch.nn.functional.gumbel_softmax(self.fc6(x), tau=0.5, hard=True)
        out3 = torch.matmul(out2, torch.tensor([[0.0, 1.0, 2.0, 3.0]]).T)
        # out = torch.cat((out1,out3), dim=1)
        # print(out)
        # out2 = relaxed_round(blocks.sigmoid_scale(self.fc6(x), u_int_min, u_int_max)) # softmax first discrete
        # out2 = relaxed_round(self.fc6(x)) # softmax first discrete
        return torch.cat((out1,out3), dim=1)

     
mip_policy = policy()
# mip_node = Node(mip_policy,['X', 'R', 'D'], ['U'], name='mip_policy')
mip_node = Node(mip_policy,['X', 'D'], ['U'], name='mip_policy')
#%% System architecture
system = Node(ss_model, ['X', 'U', 'D'], ['X'], name='system')
importlib.reload(systempreview)
cl_system = systempreview.PreviewSystem([mip_node, system], preview_keys=['D'], preview_node_name='mip_policy')
cl_system.nsteps = nsteps # prediction horizon
# Minor test
test_input = {'X': torch.rand(1,1,nx),'R': torch.rand(1,nsteps,nref),'D': torch.rand(1,nsteps,nd)}
print(test_input['D'].shape)
test_result = cl_system(test_input)

#%%
num_data = 12000
num_dev_data = 1200
batch_size = 1200
nref = nx

d = scipy.io.loadmat("../disturbances.mat")
d_tensor = torch.tensor(d['disturbances'], dtype=torch.float32)

d1 = d_tensor[:,0]
d2 = d_tensor[:,1]
 #%%
# QL = scipy.io.loadmat("/home/jb/git/building_control_new/with_disturbance/QL.mat")['QL']
# PL = scipy.io.loadmat("/home/jb/git/building_control_new/with_disturbance/PL.mat")['PL']
# QL_raw = torch.tensor(QL, dtype=torch.float32)
# PL_raw = torch.tensor(PL, dtype=torch.float32)
# QL = PL_raw*0.07 #switching PL for QL to mimic realistic setup
# PL = QL_raw*0.07

x1_train = torch.FloatTensor(num_data, 1, 1).uniform_(x1_min, x1_max)
x2_train = torch.FloatTensor(num_data, 1, 1).uniform_(x2_min, x2_max)
x_train = torch.cat((x1_train, x2_train), dim=2)
x1_dev = torch.FloatTensor(num_dev_data, 1, 1).uniform_(x1_min, x1_max)
x2_dev = torch.FloatTensor(num_dev_data, 1, 1).uniform_(x2_min, x2_max)
x_dev = torch.cat((x1_dev, x2_dev), dim=2)

# list_refs1 = [torch.rand(1, 1).uniform_(x1_min,x1_max)*torch.ones(nsteps+1, 1) for k in range(2*num_data)]
# list_refs2 = [torch.rand(1, 1).uniform_(x1_min,x2_max)*torch.ones(nsteps+1, 1) for k in range(2*num_data)]
# refs1 = torch.stack(list_refs1).reshape([2*num_data, nsteps+1, 1])
# refs2 = torch.stack(list_refs2).reshape([2*num_data, nsteps+1, 1])

# batched_ref = torch.cat((refs1, refs2), dim=-1)
# [train_ref, dev_ref] = torch.split(batched_ref, num_data, dim=0)

n_d_data = num_data//(nsteps)

# d1_train, d2_train = QL[0:n_d_data], PL[0:n_d_data]
# d1_dev, d2_dev = QL[n_d_data:2*n_d_data+1], PL[n_d_data:2*n_d_data+1]
d1_train, d2_train = d1[0:n_d_data].unsqueeze(-1), d2[0:n_d_data].unsqueeze(-1)
d1_dev, d2_dev = d1[n_d_data:2*n_d_data+1].unsqueeze(-1), d2[n_d_data:2*n_d_data+1].unsqueeze(-1)

# plt.plot(d1_train)
# plt.plot(d1_dev)
# plt.show()
# plt.plot(d2_train)
# plt.plot(d2_dev)
# plt.show()


# d1_train_ = d1_train.repeat_interleave(nsteps+1,0).unfold(0,nsteps+1,1)
# d2_train_ = d2_train.repeat_interleave(nsteps+1,0).unfold(0,nsteps+1,1)
# d_train = torch.cat((d1_train_,d2_train_), 1).swapaxes(1,2)

# d1_dev_ = d1_dev.repeat_interleave(nsteps+1,0).unfold(0,nsteps+1,1)
# d2_dev_ = d2_dev.repeat_interleave(nsteps+1,0).unfold(0,nsteps+1,1)
# d_dev = torch.cat((d1_dev_,d2_dev_), 1).swapaxes(1,2)


# d1_train__ = d1_train.repeat_interleave(nsteps+1,0).unfold(0,nsteps,1).swapaxes(1,2)
d1_train_ = d1_train.unfold(0,nsteps,1).swapaxes(1,-1)

num_extensions = 80
extended_tensors = [d1_train_]  # Start with the original tensor
for _ in range(num_extensions):
    random_indices = torch.randperm(d1_train_.size(1))  # Permutes 21 indices
    randomized_tensor = d1_train_[:, random_indices, :]
    extended_tensors.append(randomized_tensor)
d1_train__ = torch.cat(extended_tensors, dim=0)


d2_train_ = d2_train.unfold(0,nsteps,1).swapaxes(1,-1)


num_extensions = 80
extended_tensors = [d2_train_]  # Start with the original tensor
for _ in range(num_extensions):
    random_indices = torch.randperm(d2_train_.size(1))  # Permutes 21 indices
    randomized_tensor = d2_train_[:, random_indices, :]
    extended_tensors.append(randomized_tensor)
extended_tensor_train = torch.cat(extended_tensors, dim=0)

d_train = torch.cat((d1_train__[:num_data,:,:],extended_tensor_train[:num_data,:,:]), dim=-1)

d1_dev_ = d1_dev.repeat_interleave(nsteps+1,0).unfold(0,nsteps,1)
d2_dev_ = d2_dev.unfold(0,nsteps,1).swapaxes(1,2)

num_extensions = 80
extended_tensors = [d2_dev_]  # Start with the original tensor
for _ in range(num_extensions):
    random_indices = torch.randperm(d2_dev_.size(1))  # Permutes 21 indices
    randomized_tensor = d2_dev_[:, random_indices, :]
    extended_tensors.append(randomized_tensor)
extended_tensor_dev = torch.cat(extended_tensors, dim=0)

d_dev = torch.cat((d1_dev_[:num_dev_data,:,:].swapaxes(1,2),extended_tensor_train[:num_dev_data,:,:]), dim=-1)





#%%

# train_data = DictDataset({'X': x_train, 'R':torch.cat((ref1*torch.ones(num_data,nsteps+1,1),ref2*torch.ones(num_data,nsteps+1,1)), dim=-1), 'D': d_train[:num_data,:,:]}, name='train')  # Split conditions into train and dev
# dev_data = DictDataset({'X': x_dev[:d_dev.shape[0],:,:], 'R': torch.cat((ref1*torch.ones(d_dev.shape[0],nsteps+1,1),ref2*torch.ones(d_dev.shape[0],nsteps+1,1)), dim=-1), 'D': d_dev[:d_dev.shape[0],:,:]}, name='dev')
train_data = DictDataset({'X': x_train.to(device), 'D': d_train[:num_data,:,:].to(device)}, name='train')  # Split conditions into train and dev
dev_data = DictDataset({'X': x_dev[:num_dev_data,:,:].to(device), 'D': d_dev[:num_dev_data,:,:].to(device)}, name='dev')
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                           collate_fn=train_data.collate_fn, shuffle=False)
dev_loader = torch.utils.data.DataLoader(dev_data, batch_size=batch_size,
                                         collate_fn=dev_data.collate_fn, shuffle=False)

# %% OCP definition
u = variable('U')
x = variable('X')
d = variable('D')
# r = variable('R')

action_loss1 = 1.0*(u[:,:,[0]] == 0.)^2  # control penalty
action_loss2 = 1.0*(u[:,:,[1]] == 0.)^2  # control penalty
integer_loss = 0.1*(u[:,:,[2]] == 0.0)^2
regulation_loss1 = 10.0*(x[:,:,[0]] == ref1)^2  # target position
regulation_loss2 = 10.0*(x[:,:,[1]] == ref2)^2  # target position
disturbance_loss = 10.0*(d[:,[-1],:] @ E.T == 0)^2

action_loss1.name = 'action_loss1'
action_loss2.name = 'action_loss2'
integer_loss.name = 'integer_input_loss'
regulation_loss1.name = 'control_state1'
regulation_loss2.name = 'control_state2'
disturbance_loss.name = 'dist_loss'
objectives = [regulation_loss1, regulation_loss2, action_loss1, action_loss2, integer_loss] 
# objectives = [regulation_loss1, regulation_loss2] 


#%%
input1_con_l = 15.0*(u[:,:,[0]] >= 0.0)
input2_con_l = 15.0*(u[:,:,[1]] >= 0.0)


# input3_con_l = 10.0*(u[:,:,[2]] >= 0.0)
# input3_con_u = 10.0*(u[:,:,[2]] <= 3.0)
# input3_con_l.name = 'u3_l'
# input3_con_u.name = 'u3_u'

input_energy_con_u = 15.0*((u[:,[0],[0]]+u[:,[0],[1]]) < input_energy_max)
input_energy_con_l = 15.0*((u[:,[0],[0]]+u[:,[0],[1]]) >= 0.0)
# input_energy_con_u = 15.0*((u[:,:,[0]]+u[:,:,[1]]) < input_energy_max)
# input_energy_con_l = 15.0*((u[:,:,[0]]+u[:,:,[1]]) >= 0.0)

state1_con_l = 5.0*(x[:,:,[0]] >= x1_min)
state1_con_u = 5.0*(x[:,:,[0]] < x1_max)
state2_con_l = 5.0*(x[:,:,[1]] >= x2_min)
state2_con_u = 5.0*(x[:,:,[1]] < x2_max)

terminal_state1_l = 10.0 *(x[:,[-1],[0]] >= ref1-0.1)
terminal_state1_u = 10.0 *(x[:,[-1],[0]] <= ref1+0.1)

terminal_state2_l = 10.0 *(x[:,[-1],[1]] >= ref2-0.3)
terminal_state2_u = 10.0 *(x[:,[-1],[1]] <= ref2+0.3)


input1_con_l.name = 'int1_l'
input2_con_l.name = 'int2_l'
input_energy_con_u.name = 'input_energy_con_u'
input_energy_con_l.name = 'input_energy_con_l'
state1_con_l.name, state1_con_u.name = "x1_l", 'x1_u'
state2_con_l.name = 'state_2_lower'
state2_con_u.name = 'state_2_upper'

terminal_state1_l.name = 'terminal_x1_l'
terminal_state1_u.name = 'terminal_x1_u'
terminal_state2_l.name = 'terminal_x2_l'
terminal_state2_u.name = 'terminal_x2_u'

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
                # terminal_state1_l,
                # terminal_state1_u,
                # terminal_state2_l,
                # terminal_state2_u
                            ]

loss = PenaltyLoss(objectives, constraints)
# loss = BarrierLoss(objectives, constraints, barrier='softexp')
# loss = BarrierLoss(objectives, constraints)
problem = Problem([cl_system], loss)
# %%
# optimizer = torch.optim.AdamW(cl_system.parameters(), lr=0.001, amsgrad=True)
# optimizer = torch.optim.SGD(cl_system.parameters(), lr=0.0001, nesterov=True, momentum=0.9, dampening=0.0)
# optimizer = torch.optim.RMSprop(cl_system.parameters(), lr=0.0001, 
#                                 weight_decay=0.01, centered=False, momentum=0.0) # works well

optimizer = torch.optim.Adam(cl_system.parameters(), lr=0.001, weight_decay=0.0)
# optimizer = torch.optim.LBFGS(cl_system.parameters(), lr=0.0001)
# optimizer = torch.optim.Adagrad(cl_system.parameters(), lr=0.0001)
trainer = Trainer(
    problem.to(device),
    train_loader, dev_loader,
    optimizer=optimizer,
    epochs=250,
    train_metric='train_loss',
    dev_metric='dev_loss',
    eval_metric='dev_loss',
    warmup=20,
    epoch_verbose=1,
    device=device
)
best_model = trainer.train()
# load best trained model
trainer.model.load_state_dict(best_model)
torch.save(cl_system.state_dict(), 'softround_states_model.pt')

# %%
"""
Simulation
"""
# torch.set_default_device(device)

s_length = 116
refs1_sim = torch.tensor([[[ref1]]]).repeat_interleave(s_length, dim=1)
refs2_sim = torch.tensor([[[ref2]]]).repeat_interleave(s_length, dim=1)
refs = torch.cat((refs1_sim,refs2_sim), dim=-1)
dists = torch.cat((d1.unsqueeze(-1),d2.unsqueeze(-1)),1)
dists = dists[-s_length:,:].unsqueeze(0)
print(refs.shape)
print(dists.shape)
problem.load_state_dict(best_model)
data = {'X': torch.zeros(1, 1, nx, dtype=torch.float32),
        'R': refs,
        'D': dists}

# data_step = {'X': torch.zeros(1, 1, nx, dtype=torch.float32),
#         'R': refs,
#         # 'D': dists[:, :21,:]}
#         'D': dists[:, :21,:]}
# 
# cl_system.nsteps = refs.shape[1]
cl_system.nsteps = 20
trajectories = cl_system.simulate(data)
print(trajectories.keys())

x_list =[]
u_list = []
d_list = []
x = {'X': torch.zeros(1,2)}
for i in range(s_length-nsteps):
    d = {'D': dists[0,i,:].view(1,-1)}
    u = cl_system.nodes[0](x | {'D': dists[0,i:i+nsteps,:].view(1,-1)})
    x = cl_system.nodes[1](u| x| {'D': dists[0,i,:].view(1,-1)})
    # x = x['X']
    x_list.append(x['X'])
    u_list.append(u['U'])
    d_list.append(d['D'])

torch.set_default_device('cpu')

x_stack= torch.vstack(x_list)
u_stack = torch.vstack(u_list)
d_stack = torch.vstack(d_list)

# plt.plot(x_stack[:,0].cpu().detach().numpy())
# plt.plot(trajectories['X'][:,1:,0][0].to('cpu').detach().numpy())
# plt.show()
# plt.plot(x_stack[:,1].cpu().detach().numpy())
# plt.plot(trajectories['X'][:,1:,1][0].to('cpu').detach().numpy())
# plt.show()
# plt.plot(u_stack[:,2].cpu().detach().numpy())
# plt.plot(trajectories['U'][:,:,2][0].to('cpu').detach().numpy())
# plt.show()
# plt.plot(u_stack[:,0].cpu().detach().numpy())
# plt.plot(trajectories['U'][:,:,0][0].to('cpu').detach().numpy())
# plt.show()
# plt.plot(u_stack[:,2].cpu().detach().numpy())
# plt.plot(trajectories['U'][:,:,2][0].to('cpu').detach().numpy())
# plt.show()
# plt.plot(d_stack[:,1].cpu().detach().numpy())
# plt.plot(trajectories['D'][:,:,1][0].to('cpu').detach().numpy())
# plt.show()




# input_data = {'X': torch.zeros(1, nx, dtype=torch.float32),
#         'D': dists[0, :21,:].reshape(1, -1)}
    
# u = cl_system.nodes[0].forward(input_data)

# input_data_1 = {'X': torch.zeros(1, nx, dtype=torch.float32),
#         'D': dists[0, 0,:].reshape(1, -1)}

# x1 = cl_system.nodes[1].forward( input_data_1 | u)

# step_output = cl_system.forward(data_step)

u1_opt = scipy.io.loadmat("../ress/u1.mat")['u1']
u2_opt = scipy.io.loadmat("../ress/u2.mat")['u2']
u3_opt = scipy.io.loadmat("../ress/u3.mat")['u3']
x1_opt = scipy.io.loadmat("../ress/x1.mat")['x1']
x2_opt = scipy.io.loadmat("../ress/x2.mat")['x2']
load1_opt = scipy.io.loadmat("../ress/load1.mat")['load1']
load2_opt = scipy.io.loadmat("../ress/load2.mat")['load2']


#%%
fig1, ax = plt.subplots(4,2, figsize=(20, 10))
ax[0,0].plot(trajectories['X'][:,:,0][0].to('cpu').detach().numpy(), label='DPC')
ax[0,0].plot(x1_opt, 'k--', label='Optimal')
# ax[0,0].plot(x_stack[:,0].cpu().detach().numpy())
ax[0,0].plot(trajectories['R'][:,:dists.size(1)-nsteps,0][0].to('cpu').detach().numpy())
ax[0,0].set_ylabel('State1')
ax[0,0].plot(torch.zeros(dists.size(1)-nsteps))
ax[0,0].plot(x1_max*torch.ones(dists.size(1)-nsteps))
ax[0,0].legend()

ax[1,0].plot(trajectories['X'][:,:,1][0].to('cpu').detach().numpy())
ax[1,0].plot(x2_opt, 'k--')
# ax[1,0].plot(x_stack[:,1].cpu().detach().numpy())

ax[1,0].plot(trajectories['R'][:,:dists.size(1)-nsteps,1][0].to('cpu').detach().numpy())
ax[1,0].set_ylabel('State2')
ax[1,0].plot(torch.zeros(dists.size(1)-nsteps))
ax[1,0].plot(x2_max*torch.ones(dists.size(1)-nsteps))

ax[2,0].plot(trajectories['D'][:,:-nsteps,0][0].to('cpu').detach().numpy())
ax[2,0].plot(load1_opt, 'k--')
ax[3,0].plot(trajectories['D'][:,:-nsteps,1][0].to('cpu').detach().numpy())
ax[3,0].plot(load2_opt, 'k--')
ax[2,0].set_ylabel('Load1')
ax[3,0].set_ylabel('Load2')
ax[2,0].set_title('Disturbances')
ax[-1,0].set_xlabel('Sample')

ax[0,1].plot(trajectories['U'][:,:,0][0].to('cpu').detach().numpy())
ax[0,1].plot(u1_opt, 'k--')
ax[1,1].plot(trajectories['U'][:,:,1][0].to('cpu').detach().numpy())
ax[1,1].plot(u2_opt, 'k--')
ax[2,1].plot(trajectories['U'][:,:,2][0].to('cpu').detach().numpy())
ax[2,1].plot(u3_opt, 'k--')
# ax[2,1].plot(u_stack[:,2].cpu().detach().numpy())

ax[3,1].plot(trajectories['U'][:,:,0][0].to('cpu').detach().numpy()+trajectories['U'][:,:,1][0].to('cpu').detach().numpy())
ax[3,1].plot(u1_opt+u2_opt, 'k--')
ax[3,1].plot(input_energy_max*torch.ones(dists.size(1)-nsteps))
ax[-1,1].set_xlabel('Sample')
ax[0,1].set_ylabel('Input1')
ax[1,1].set_ylabel('Input2')
ax[2,1].set_ylabel('Input3')
ax[3,1].set_ylabel('Input1+Input2')

plt.show()
fig1.savefig('softround_gumbel_states.pdf')

# %%
