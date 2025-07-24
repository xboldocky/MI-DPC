
# %%
import os, sys
current_file_path = os.path.dirname(os.path.abspath(__file__))
if current_file_path.split('5')[0] not in sys.path:
    sys.path.append(current_file_path.split('5')[0])
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
from neuromancer.loggers import BasicLogger

from rnd import roundGumbelModel, roundThresholdModel, netFC


default_type = torch.float32
torch.set_default_dtype(default_type)
temperature_coefficient = 10.0

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

# for nsteps in [10,20,30,40]:
# nsteps = 20
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
layer_width = 95
for nsteps in [10,15,20,25,30,40]:
    torch.manual_seed(208)
    input_features = nx+(nd*(nsteps))
#%% Policy network architecture

    class policy(torch.nn.Module):
        def __init__(self, in_features):
            super(policy, self).__init__()
            self.fc_input = torch.nn.Linear(in_features, layer_width)  # Common Input Layer
        
            self.fc1_x1 = torch.nn.Linear(layer_width, layer_width)  # Layers for the first branch
            self.fc2_x1 = torch.nn.Linear(layer_width, layer_width) 
            self.fc3_x1 = torch.nn.Linear(layer_width, layer_width) 
            self.fc_output_x1 = torch.nn.Linear(layer_width, 2, bias=True) 
        
            self.fc1_x2 = torch.nn.Linear(layer_width, layer_width)   # Layers for the second branch
            self.fc2_x2 = torch.nn.Linear(layer_width, layer_width)  
            self.fc3_x2 = torch.nn.Linear(layer_width, layer_width)  
            self.fc_output_x2 = torch.nn.Linear(layer_width, 1, bias=True) 
            
            self.bn_input = torch.nn.LayerNorm(layer_width, elementwise_affine=False) # Common Input BN

            self.bn1_x1 = torch.nn.LayerNorm(layer_width, elementwise_affine=False) # Frist branch BN
            self.bn2_x1 = torch.nn.LayerNorm(layer_width, elementwise_affine=False)
            
            self.bn1_x2 = torch.nn.LayerNorm(layer_width, elementwise_affine=False) # Second branch BN
            self.bn2_x2 = torch.nn.LayerNorm(layer_width, elementwise_affine=False)        
            del self.fc3_x1, self.fc3_x2
            self.dropout = torch.nn.Dropout(0.1) # Dropout object
        @torch.jit.ignore
        def forward(self, *inputs):
            if len(inputs) > 1:
                x = torch.cat(inputs, dim=-1)
            else:
                x = inputs[0]
            x = torch.nn.functional.tanh(self.fc_input(x))  # Common Input Layer
            # x = self.dropout(x)
            x1, x2 = self.bn_input(x), self.bn_input(x)
            
            x1 = torch.nn.functional.tanh(self.fc1_x1(x1)) # First branch
            x1 = self.bn1_x1(x1)
            x1 = self.dropout(x1)
            x1 = torch.nn.functional.tanh(self.fc2_x1(x1))
            x1 = self.bn2_x1(x1)
            x1 = self.dropout(x1)
            out1 = self.fc_output_x1(x1)

            x2 = torch.nn.functional.selu(self.fc1_x2(x2)) # Second branch
            x2 = self.bn1_x2(x2)
            x2 = self.dropout(x2)
            x2 = torch.nn.functional.selu(self.fc2_x2(x2))
            x2 = self.bn2_x2(x2)
            # x2 = self.dropout(x2)
            # out2 = blocks.relu_clamp(self.fc_output_x2(x2), 0.0, 3.) # Rounding
            out2 = self.fc_output_x2(x2) # Rounding
            
            return torch.cat((out1,out2), dim=1) # Return u1,u2,u3
        
    mip_policy = policy(in_features=input_features)
    mip_node = Node(mip_policy,['X', 'D'], ['U_relaxed'], name='mip_policy')
    #%%
    class rounding_network(torch.nn.Module):
        def __init__(self, input_dim, layer_width):
            super(rounding_network, self).__init__()
            self.input_dim = input_dim
            self.layer_width = layer_width
            self.fc1 = torch.nn.Linear(self.input_dim, self.layer_width)  # input 2
            self.fc2 = torch.nn.Linear(self.layer_width, self.layer_width) 
            self.fc3 = torch.nn.Linear(self.layer_width, self.layer_width) 
            self.fc4 = torch.nn.Linear(self.layer_width, self.layer_width) 
            self.fc41 = torch.nn.Linear(self.layer_width, self.layer_width) 
            self.fc5 = torch.nn.Linear(self.layer_width, 2) # linear -> identity
            self.fc6 = torch.nn.Linear(self.layer_width, 1) # linear -> softmax 
            
            self.bn = torch.nn.LayerNorm(self.layer_width, elementwise_affine=False)
            self.bn1 = torch.nn.LayerNorm(self.layer_width, elementwise_affine=False)
            self.bn2 = torch.nn.LayerNorm(self.layer_width, elementwise_affine=False)
            self.bn3 = torch.nn.LayerNorm(self.layer_width, elementwise_affine=False)        
            self.bn4 = torch.nn.LayerNorm(self.layer_width, elementwise_affine=False)        
            self.bn41 = torch.nn.LayerNorm(self.layer_width, elementwise_affine=False)        

            self.dropout = torch.nn.Dropout(0.1) # Dropout object
        @torch.jit.ignore
        def forward(self, *inputs):
            if len(inputs) > 1:
                x = torch.cat(inputs, dim=-1)
            else:
                x = inputs[0]
            x = torch.nn.functional.tanh(self.fc1(x))  
            x1,x2 = self.bn1(x),self.bn1(x)

            x1 = torch.nn.functional.tanh(self.fc2(x1))
            x1 = self.bn2(x1)
            # x1 = self.dropout(x1)

            x1 = torch.nn.functional.tanh(self.fc3(x1))
            # x1 = self.bn3(x1)
            # x1 = self.dropout(x1)


            x2 = torch.nn.functional.selu(self.fc4(x2))
            # x2 = self.bn4(x2)        

            x2 = torch.nn.functional.selu(self.fc41(x2))
            # x2 = self.bn41(x2)        
            # x2 = self.dropout(x2)


            # out1 = blocks.sigmoid_scale(self.fc5(x), -1.0, input_energy_max)  # Identity
            out1 = self.fc5(x1)  # Identity
            # out2 = blocks.relu_clamp(self.fc6(x), 0.0, 3.) # softmax first discrete
            out2 = self.fc6(x2) # softmax first discrete
            return torch.cat((out1,out2), dim=1)

    rounding_nn = rounding_network(input_dim=input_features+nu, layer_width=layer_width)    
    # rounding_nn = policy(in_features=input_features+nu)    

    rounding_method = 'threshold'
    # rounding_method = 'classification'

    if rounding_method == 'threshold':
        rounding_node = roundThresholdModel(
            layers=rounding_nn, param_keys=['X','D'], 
            var_keys=['U_relaxed'], output_keys=['U'], 
            int_ind={'U_relaxed': -1}, continuous_update=True, 
            name='round', slope=temperature_coefficient)  
    else: 
        rounding_node = roundGumbelModel(
            layers=rounding_nn, param_keys=['X','D'], 
            var_keys=['U_relaxed'], output_keys=['U'], int_ind={'U_relaxed': -1}, 
            continuous_update=True, name='round', temperature=0.5)

    # r_result = rounding_node.forward({'X':torch.rand(10,2)*10.0, 'D':torch.rand(10,input_features-2)*10.0,'U_relaxed': torch.rand(10,3)*10.0})

    def clip_fn(x):
        # o = rounding_node(x)
        # print(o.keys())
        return torch.cat((blocks.relu_clamp(x[:,:-1],0.0,8.0), blocks.relu_clamp(x[:,[-1]], 0.0, 3.)), dim=-1)

    rounding_clip = Node(clip_fn, ['U'], ['U_clipped'], name='round_clip')

    # print(r_result)

    #%% System architecture
    system = Node(ss_model, ['X', 'U_clipped', 'D'], ['X'], name='system')
    importlib.reload(systempreview)
    cl_system = systempreview.PreviewSystem([mip_node, rounding_node, rounding_clip ,system], preview_keys=['D'], preview_node_name=['mip_policy','round'])
    cl_system.nsteps = nsteps # prediction horizon

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    n_parameters = count_parameters(cl_system)
    print(n_parameters)
    # %%Forward Test
    # test_input = {'X': torch.rand(1,1,nx),'D': torch.rand(1,nsteps,nd)}
    # test_result = cl_system(test_input)

    #%%

    # Simulator test
    # s_length = 116
    # refs1_sim = torch.tensor([[[ref1]]]).repeat_interleave(s_length, dim=1)
    # refs2_sim = torch.tensor([[[ref2]]]).repeat_interleave(s_length, dim=1)
    # refs = torch.cat((refs1_sim,refs2_sim), dim=-1)
    # dists = torch.rand(1,s_length,2)
    # data = {'X': torch.zeros(1, 1, nx),
    #         'R': refs,
    #         'D': dists}
    # trajectories = cl_system.simulate(data)

    #%% Training Data
    num_data = 24000
    num_dev_data = 4000
    batch_size = 2000
    nref = nx

    d = scipy.io.loadmat("newloads_matrix.mat")
    d_tensor = torch.tensor(d['newloads_matrix'], dtype=default_type, device=device)

    d1 = d_tensor[:,0]
    d2 = d_tensor[:,1]

    x1_train = torch.empty(num_data, 1, 1, dtype=default_type).uniform_(x1_min, x1_max)
    x2_train = torch.empty(num_data, 1, 1, dtype=default_type).uniform_(x2_min, x2_max)
    # x1_train_2 = torch.empty(4000, 1, 1, dtype=default_type).uniform_(ref1-1.0, ref1+1.0)
    # x2_train_2 = torch.empty(4000, 1, 1, dtype=default_type).uniform_(ref2-0.2, ref2+0.2)

    # x1_train = torch.cat((x1_train_1,x1_train_2), dim=0)
    # x2_train = torch.cat((x2_train_1,x2_train_2), dim=0)
    x_train = torch.cat((x1_train, x2_train), dim=2)

    x1_dev = torch.empty(num_dev_data, 1, 1, dtype=default_type).uniform_(x1_min, x1_max)
    x2_dev = torch.empty(num_dev_data, 1, 1, dtype=default_type).uniform_(x2_min, x2_max)
    # x1_dev_2 = torch.empty(1000, 1, 1, dtype=default_type).uniform_(ref1-1.0, ref1+1.0)
    # x2_dev_2 = torch.empty(1000, 1, 1, dtype=default_type).uniform_(ref2-0.2, ref2+0.2)

    # x1_dev = torch.cat((x1_dev_1,x1_dev_2), dim=0)
    # x2_dev = torch.cat((x2_dev_1,x2_dev_2), dim=0)

    x_dev = torch.cat((x1_dev, x2_dev), dim=2)

    dist_data = torch.load(f'extended_disturbances_60.pt')
    dist_data = torch.tensor(dist_data, dtype=default_type, device=device)
    #%% DataLoaders
    train_data = DictDataset({'X': x_train.to(device), 'D': dist_data[:num_data,:nsteps,:].to(device)}, name='train')  # Split conditions into train and dev
    dev_data = DictDataset({'X': x_dev[:num_dev_data,:,:].to(device), 'D': dist_data[num_data:num_data+num_dev_data,:nsteps,:].to(device)}, name='dev')
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                            collate_fn=train_data.collate_fn, shuffle=False)
    dev_loader = torch.utils.data.DataLoader(dev_data, batch_size=batch_size,
                                            collate_fn=dev_data.collate_fn, shuffle=False)

    # %% OCP definition
    u = variable('U_clipped')
    x = variable('X')
    d = variable('D')

    action_loss1 = 0.5*(u[:,:,[0]] == 0.)^2  # control penalty
    action_loss2 = 0.5*(u[:,:,[1]] == 0.0)^2  # control penalty
    integer_loss = 0.1*(u[:,:,[2]] == 0.0)^2
    regulation_loss1 = 1.0*(x[:,:,[0]] == ref1)^2  # target position
    regulation_loss2 = 1.0*(x[:,:,[1]] == ref2)^2  # target position
    # state_smoothing = 3.0*((x[:, 1:, [1]] == x[:, :-1, [1]])^2)

    # state_smoothing.name = 'state_smoothing'
    action_loss1.name, action_loss2.name, integer_loss.name = 'action_loss1', 'action_loss2', 'integer_input_loss'
    regulation_loss1.name, regulation_loss2.name = 'control_state1', 'control_state2'

    objectives = [regulation_loss1, regulation_loss2, action_loss1, action_loss2, integer_loss] 

    input1_con_l = 25.0*(u[:,:,[0]] >= 0.0)
    input2_con_l = 25.0*(u[:,:,[1]] >= 0.0)
    input3_con_l = 25.0*(u[:,:,[2]] >= 0.0)
    input3_con_u = 25.0*(u[:,:,[2]] <= 3.0)
    input_energy_con_u = 25.0*((u[:,:,[0]]+u[:,:,[1]]) < input_energy_max)
    input_energy_con_l = 25.0*((u[:,:,[0]]+u[:,:,[1]]) >= 0.0)

    state1_con_l = 25.0*(x[:,:,[0]] >= x1_min)
    state1_con_u = 25.0*(x[:,:,[0]] < x1_max)
    state2_con_l = 25.0*(x[:,:,[1]] >= x2_min)
    state2_con_u = 25.0*(x[:,:,[1]] < x2_max)

    input1_con_l.name, input2_con_l.name = 'int1_l', 'int2_l'
    input_energy_con_u.name, input_energy_con_l.name = 'input_energy_con_u', 'input_energy_con_l'
    input3_con_l.name, input3_con_u.name = 'u3_l', 'u3_u'
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

                    input3_con_l,
                    input3_con_u,
                                ]

    loss = PenaltyLoss(objectives, constraints)
    # loss = BarrierLoss(objectives, constraints)
    problem = Problem([cl_system], loss)
    # %%
    optimizer = torch.optim.Adam(cl_system.parameters(), lr=0.0003, amsgrad=False)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.7)
    logger = BasicLogger(args=None, savedir=f'authdata_nsteps_{nsteps}/log_{rounding_method}_N{nsteps}', verbosity=1, stdout=['train_loss', 'dev_loss', 'eltime'])

    trainer = Trainer(
        problem.to(device),
        train_loader, dev_loader,
        optimizer=optimizer,
        epochs=1000,
        train_metric='train_loss',
        dev_metric='dev_loss',
        eval_metric='dev_loss',
        warmup=20,
        patience=80,
        epoch_verbose=1,
        device=device,
        clip=2.0,
        lr_scheduler=False,
        logger=logger
    )
    best_model = trainer.train()

    trainer.model.load_state_dict(best_model) # load best trained model
    torch.save(cl_system.state_dict(), f'authdata_nsteps_{nsteps}/{rounding_method}_model.pt')

    # %%
    """
    Simulation
    """
    s_length = 1873
    dists = torch.cat((d1.unsqueeze(-1),d2.unsqueeze(-1)),1)
    dists = d_tensor[:s_length+nsteps,:].unsqueeze(0)

    refs1_sim = torch.tensor([[[ref1]]]).repeat_interleave(dists.shape[1], dim=1)
    refs2_sim = torch.tensor([[[ref2]]]).repeat_interleave(dists.shape[1], dim=1)
    refs = torch.cat((refs1_sim,refs2_sim), dim=-1)
    print(refs.shape)
    print(dists.shape)
    problem.load_state_dict(best_model)
    data = {'X': torch.zeros(1, 1, nx),
            'R': refs,
            'D': dists}

    cl_system.nsteps = nsteps
    cl_system.eval()
    # cl_system.nodes[0].callable.eval()
    # cl_system.training = False
    # cl_system.nodes[0].callable.training = False


    trajectories = cl_system.simulate(data)
    print(trajectories.keys())

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

    # u1_opt = scipy.io.loadmat(f"optimal_5_min/Hp40.mat")['u1']
    # u2_opt = scipy.io.loadmat(f"optimal_5_min/Hp40.mat")['u2']
    # u3_opt = scipy.io.loadmat(f"optimal_5_min/Hp40.mat")['u3']
    # x1_opt = scipy.io.loadmat(f"optimal_5_min/Hp40.mat")['x1']
    # x2_opt = scipy.io.loadmat(f"optimal_5_min/Hp40.mat")['x2']

    u_opt_torch = torch.cat(
        (torch.from_numpy(u1_opt),
        
        torch.from_numpy(u2_opt),
        torch.from_numpy(u3_opt)), dim=1).unsqueeze(0)

    x_opt_torch = torch.cat(
        (torch.from_numpy(x1_opt),
        torch.from_numpy(x2_opt)), dim=1).unsqueeze(0)


    DPC_loss_list = []
    optim_loss_list = []
    for i in range(u1_opt.shape[0]-1):
        result = loss.forward({'X':trajectories['X'][[0],i,:].unsqueeze(0),'U_clipped': trajectories['U_clipped'][[0],i,:].unsqueeze(0)})['loss']
        result_optim = loss.forward({'X': x_opt_torch[[0],i,:].unsqueeze(0).to(device),'U_clipped': u_opt_torch[[0],i,:].unsqueeze(0).to(device)})['loss']
        DPC_loss_list.append(result)
        optim_loss_list.append(result_optim)
    DPC_loss_stack = torch.vstack(DPC_loss_list)
    optim_loss_stack = torch.vstack(optim_loss_list)
    DPC_loss = torch.sum(DPC_loss_stack)
    optim_loss = torch.sum(optim_loss_stack)

    # plt.plot(DPC_loss_stack.to('cpu').detach().numpy()-optim_loss_stack.to('cpu').detach().numpy())
    # plt.show()

    # result = loss.forward({'X':trajectories['X'][[0],i,:].unsqueeze(0),'U': trajectories['U'][[0],i,:].unsqueeze(0)})['loss']
    # result_optim = loss.forward({'X': x_opt_torch[[0],i,:].unsqueeze(0).to(device),'U': u_opt_torch[[0],i,:].unsqueeze(0).to(device)})['loss']
    #%%
    plt.rcParams['font.family'] = 'DejaVu Serif'
    plt.rcParams['font.size'] = 11

    fig1, ax = plt.subplots(4,2, figsize=(20, 10))
    ax[0,0].plot(trajectories['X'][:,:-1,0][0].to('cpu').detach().numpy(), color='dodgerblue', label='DPC')
    ax[0,0].plot(x1_opt, '--', color='black',label='Optimal')
    # ax[0,0].plot(x_stack[:,0].to('cpu').detach().numpy(), 'r')
    ax[0,0].plot(trajectories['R'][:,:dists.size(1)-nsteps,0][0].to('cpu').detach().numpy(), '-',linewidth=0.5, color='red', label='Reference')
    ax[0,0].set_ylabel('$x_1$')
    ax[0,0].plot(torch.zeros(dists.size(1)-nsteps).to('cpu'), 'k-')
    ax[0,0].plot(x1_max*torch.ones(dists.size(1)-nsteps).to('cpu'), 'k-', label='Constraints')
    ax[0,0].legend()

    ax[1,0].plot(trajectories['X'][:,:-1,1][0].to('cpu').detach().numpy(), color='dodgerblue')
    # ax[1,0].plot(x_stack[:,1].to('cpu').detach().numpy(), 'r')

    ax[1,0].plot(x2_opt, 'k--')

    ax[1,0].set_ylabel('$x_2$')
    ax[1,0].plot(torch.zeros(dists.size(1)-nsteps).to('cpu'), 'k-')
    ax[1,0].plot(x2_max*torch.ones(dists.size(1)-nsteps).to('cpu'),'k-')
    ax[1,0].plot(trajectories['R'][:,:dists.size(1)-nsteps,1][0].to('cpu').detach().numpy(), 'r', linewidth=0.5)

    ax[2,0].plot(trajectories['D'][:,:-nsteps,0][0].to('cpu').detach().numpy(), 'k-')
    # ax[2,0].plot(load1_opt, 'r')
    ax[3,0].plot(trajectories['D'][:,:-nsteps,1][0].to('cpu').detach().numpy(), 'k-')
    # ax[3,0].plot(load2_opt, 'r')
    ax[2,0].set_ylabel('$d_1$')
    ax[3,0].set_ylabel('$d_2$')
    # ax[2,0].set_title('Disturbances')
    ax[-1,0].set_xlabel('Sample $k$')

    # ax[0,1].step(trajectories['U'][:,:,0].to('cpu').detach().numpy())
    ax[0,1].step(np.arange(trajectories['U'].shape[1]), trajectories['U_clipped'][0,:,0].to('cpu').detach().numpy(), color='dodgerblue')
    ax[0,1].step(u1_opt, 'k--')
    # ax[1,1].plot(trajectories['U'][:,:,1][0].to('cpu').detach().numpy())

    ax[1,1].step(np.arange(trajectories['U_clipped'].shape[1]), trajectories['U_clipped'][0,:,1].to('cpu').detach().numpy(), color='dodgerblue')
    ax[1,1].step(u2_opt, 'k--')

    # ax[2,1].plot(trajectories['U'][:,:,2][0].to('cpu').detach().numpy())
    ax[2,1].step(np.arange(trajectories['U_clipped'].shape[1]), trajectories['U_clipped'][0,:,2].to('cpu').detach().numpy(), color='dodgerblue')
    ax[2,1].step(u3_opt, 'k--')

    # ax[3,1].plot(trajectories['U'][:,:,0][0].to('cpu').detach().numpy()+trajectories['U'][:,:,1][0].to('cpu').detach().numpy())
    ax[3,1].plot(input_energy_max*torch.ones(dists.size(1)-nsteps).to('cpu'),'k-', markersize=4)
    ax[3,1].step(np.arange(trajectories['U_clipped'].shape[1]), trajectories['U_clipped'][0,:,0].to('cpu').detach().numpy()+trajectories['U_clipped'][0,:,1].to('cpu').detach().numpy(), color='dodgerblue')
    ax[3,1].step(u1_opt+u2_opt, 'k--')

    ax[-1,1].set_xlabel('Sample $k$')
    ax[0,1].set_ylabel('$u_1$')
    ax[1,1].set_ylabel('$u_2$')
    ax[2,1].set_ylabel('$\Delta_1$')
    ax[3,1].set_ylabel('$u_1+u_2$')


    ax[0,0].grid()
    ax[1,0].grid()
    ax[2,0].grid()
    ax[3,0].grid()
    ax[0,1].grid()
    ax[1,1].grid()
    ax[2,1].grid()
    ax[3,1].grid()
    fig1.tight_layout()
    # plt.grid()
    fig1.savefig(f'authdata_nsteps_{nsteps}/{rounding_method}_eval.pdf', bbox_inches='tight')

    # plt.show()
    #%%


    file_path = f"authdata_nsteps_{nsteps}/output_{rounding_method}_nsteps.txt"

    with open(file_path, 'w') as file:
        # Write values into the file
        file.write(f"Nsteps: {nsteps}\n")
        file.write(f"Layer_width: {layer_width}\n")
        file.write(f"Stage cost: {DPC_loss}\n")
        file.write(f"Optimal Stage cost: {optim_loss}\n")
        file.write(f"Suboptimality: {(DPC_loss/optim_loss-1)*100:.3f}%\n")
        file.write(f"Scale coeff: {temperature_coefficient}\n")
        file.write(f"Number of parameters: {n_parameters}\n")

    torch.save(trajectories, f"authdata_nsteps_{nsteps}/{rounding_method}_trajectories.pt")

# %%
