#%%
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
from neuromancer.loggers import BasicLogger
import matplotlib.pyplot as plt
import numpy as np
import scipy
import systempreview
import importlib
import gc
import time
import matplotlib

default_type = torch.float32
torch.set_default_dtype(default_type)
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

def softmax(
    logits: torch.Tensor,
    tau: float = 1,
    hard: bool = False,
    eps: float = 1e-10,
    dim: int = -1,
) -> torch.Tensor:
    r"""
    
    """
    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
    )  # ~Gumbel(0,1)
    gumbels = (logits) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(
            logits, memory_format=torch.legacy_contiguous_format
        ).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret



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

methods = ['sigmoid','gumbel']
method = methods[1]
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device('cpu')
list_out =[]
with torch.no_grad():
    # for nsteps in [10,15,20,25,30,40]:
    # for nsteps in [20]:
        nsteps = 25
        # torch.manual_seed(208)

        # for nsteps in [20]:
        # nsteps = 10
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
        # layer_width = 120+(nsteps) #500 without bn works best
        layer_width = 140 #500 without bn works best
        # layer_width = 80 
        class policy(torch.nn.Module):
            def __init__(self):
                super(policy, self).__init__()
                self.fc_input = torch.nn.Linear(input_features, layer_width)  # Common Input Layer
            
                self.fc1_x1 = torch.nn.Linear(layer_width, layer_width)  # Layers for the first branch
                self.fc2_x1 = torch.nn.Linear(layer_width, layer_width) 
                # if method == 'sigmoid':
                #     self.fc3_x1 = torch.nn.Linear(layer_width, layer_width)
                #     for param in self.fc3_x1.parameters():
                #         param.requires_grad = False 
                self.fc_output_x1 = torch.nn.Linear(layer_width, 2, bias=True) 
            
                self.fc1_x2 = torch.nn.Linear(layer_width, layer_width)   # Layers for the second branch
                self.fc2_x2 = torch.nn.Linear(layer_width, layer_width)  
                # if method == 'sigmoid':
                #     self.fc3_x2 = torch.nn.Linear(layer_width, layer_width)  
                #     for param in self.fc3_x2.parameters():
                #         param.requires_grad = False
                if method == 'sigmoid':
                    self.fc_output_x2 = torch.nn.Linear(layer_width, 1, bias=True) 
                else:
                    self.fc_output_x2 = torch.nn.Linear(layer_width, 4, bias=True) 


                self.bn_input = torch.nn.LayerNorm(layer_width, elementwise_affine=False) # Common Input BN

                self.bn1_x1 = torch.nn.LayerNorm(layer_width, elementwise_affine=False) # Frist branch BN
                self.bn2_x1 = torch.nn.LayerNorm(layer_width, elementwise_affine=False)
                # self.bn3_x1 = torch.nn.LayerNorm(layer_width, elementwise_affine=False)
                
                self.bn1_x2 = torch.nn.LayerNorm(layer_width, elementwise_affine=False) # Second branch BN
                self.bn2_x2 = torch.nn.LayerNorm(layer_width, elementwise_affine=False)        
                # self.bn3_x2 = torch.nn.LayerNorm(layer_width, elementwise_affine=False)        
                
                self.dropout = torch.nn.Dropout(0.1) # Dropout object

            def forward(self, *inputs):
                if len(inputs) > 1:
                    x = torch.cat(inputs, dim=-1)
                else:
                    x = inputs[0]
                x = torch.nn.functional.tanh(self.fc_input(x))  # Common Input Layer
                # x = self.fc_input(x)  # Common Input Layer
                # x = self.dropout(x)
                # x1, x2 = self.bn_input(x), self.bn_input(x)
                x = self.bn_input(x)
                
                x1 = torch.nn.functional.tanh(self.fc1_x1(x)) # First branch
                x1 = self.bn1_x1(x1)
                x1 = self.dropout(x1)
                x1 = torch.nn.functional.tanh(self.fc2_x1(x1))
                x1 = self.bn2_x1(x1)
                x1 = self.dropout(x1)
                out1 = self.fc_output_x1(x1)
                # if method == 'gumbel':
                out1 = blocks.relu_clamp(self.fc_output_x1(x1),0.0,8.0)

                x2 = torch.nn.functional.selu(self.fc1_x2(x)) # Second branch
                x2 = self.bn1_x2(x2)
                x2 = torch.nn.functional.selu(self.fc2_x2(x2))
                x2 = self.bn2_x2(x2)
                if method == 'sigmoid':
                    out2 = relaxed_round(blocks.relu_clamp(self.fc_output_x2(x2), -0.49, 3.49)) # Rounding
                elif method == 'gumbel':
                    out_digits = softmax(self.fc_output_x2(x2), tau=temperature_coefficient, hard=True)
                    # out_digits = torch.nn.functional.gumbel_softmax(self.fc_output_x2(x2), tau=temperature_coefficient, hard=True)
                    out2 = torch.matmul(out_digits, torch.tensor([[0.0, 1.0, 2.0, 3.0]]).T)

                return torch.cat((out1,out2), dim=1) # Return u1,u2,u3

        mip_policy = policy()
        mip_node = Node(mip_policy,['X', 'D'], ['U'], name='mip_policy')

        #%% System architecture
        system = Node(ss_model, ['X', 'U', 'D'], ['X'], name='system')
        importlib.reload(systempreview)
        cl_system = systempreview.PreviewSystem([mip_node, system], preview_keys=['D'], preview_node_name='mip_policy')
        cl_system.nsteps = nsteps # prediction horizon
        cl_system.eval()

        a = torch.load(f'authdata_nsteps_{nsteps}/log_{method}_N{nsteps}/best_model_state_dict.pth')
        # state_dict = torch.load(f'nsteps_{nsteps}/log_{method}_N{nsteps}/best_model_state_dict.pth')
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
        # state_smoothing = 1.0*((x[:, 1:, [1]] == x[:, :-1, [1]])^2)

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
        if method == 'gumbel':

            constraints = [
                        input1_con_l,
                        input2_con_l,
                        input_energy_con_l,
                        input_energy_con_u,
                        state1_con_l,
                        state1_con_u,
                        state2_con_l,
                        state2_con_u,
                                ]

        loss = PenaltyLoss(objectives, constraints)

        #%%
        ##### IMPORT OPTIMAL TRAJECTORIES
        StageCost_index = 2
        Time_index = 3
        U_index = 4
        X_index = 5
        D_index = 6
        # Load the .mat file
        try:
            mat_data = scipy.io.loadmat(f"final_data/N{nsteps}.mat")
            # Extract the main variable
            optimal_data = mat_data[f"N{nsteps}_20"]
            # Assign each feature to a variable
            optimal_data = {f"x0_{i+1}": optimal_data[0, i] for i in range(optimal_data.shape[1])}
        except:
            mat_data = scipy.io.loadmat(f"final_data/N30.mat")
            # Extract the main variable
            optimal_data = mat_data[f"N30_20"]
            # Assign each feature to a variable
            optimal_data = {f"x0_{i+1}": optimal_data[0, i] for i in range(optimal_data.shape[1])}
        s_length = optimal_data[f'x0_1'][D_index].shape[0]

        ##### LOAD DISTURBANCES
        d = scipy.io.loadmat("newloads_matrix.mat")
        d_tensor = torch.tensor(d['newloads_matrix'], dtype=default_type)
        d1, d2 = d_tensor[:,0], d_tensor[:,1]
        dists = torch.cat((d1.unsqueeze(-1),d2.unsqueeze(-1)),1)
        dists = d_tensor[:s_length+nsteps,:].unsqueeze(0)
        mip_policy.eval()

        #%%

        # dists = torch.from_numpy(optimal_data[f'x0_1'][D_index]).unsqueeze(0)
        # dists = torch.tensor(dists, dtype=torch.float32)
        ##### Simulation
        # gc.disable()
        total_list_x = []
        total_list_u = []
        total_list_time = []
        for instance in range(len(optimal_data.keys())):
            x = {'X': torch.tensor(torch.from_numpy(optimal_data[f'x0_{instance+1}'][1]),dtype=torch.float32)}
            x_list =[]
            u_list = []
            d_list = []
            time_list = []
            for i in range(s_length):
                in_data = torch.cat((x['X'], dists[0,i:i+nsteps,:].reshape(1,-1)), dim=-1)
                with torch.inference_mode():
                    # _ = u = mip_policy.forward(in_data) #warmup
                    # _ = u = mip_policy.forward(in_data) #warmup
                    # start = time.perf_counter()
                    u = mip_policy.forward(in_data)
                    # end = time.perf_counter()
                    # el_time = end - start
                u = {'U': u}
                x_list.append(x['X'])
                x = cl_system.nodes[1].forward(u |x| {'D': dists[0,i,:].view(1,-1)})
                u_list.append(u['U'])
                # time_list.append(el_time)
            x_torch = torch.vstack(x_list)
            u_torch = torch.vstack(u_list)
            time_torch = torch.FloatTensor(time_list)

            total_list_x.append(x_torch)
            total_list_u.append(u_torch)
            total_list_time.append(time_torch)

        time_tensor = torch.vstack(total_list_time)
        x_tensor = torch.stack(total_list_x)
        u_tensor = torch.stack(total_list_u)
        # gc.enable()

        # %%

        optim_x_list = []
        optim_u_list = []
        optim_ell_list = []
        for key in optimal_data.keys():
            optim_x_list.append(torch.tensor(optimal_data[key][X_index]))
            optim_u_list.append(torch.tensor(optimal_data[key][U_index]))
            optim_ell_list.append(torch.tensor(optimal_data[key][StageCost_index]))
            
        optim_x = torch.stack(optim_x_list)
        optim_u = torch.stack(optim_u_list)
        optim_ell = torch.stack(optim_ell_list)

#%%

matplotlib.use("pgf")

plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "text.usetex": True,  # Use LaTeX for text
    "font.family": "serif",  # Use a serif font
    "font.size": 10,  # Set font size
    "pgf.rcfonts": False,  # Don't override with default matplotlib fonts
    "legend.fontsize": 8
})



x1_max = 8.4
x1_min = 0.0
x2_max = 3.6
x2_min = 0.0
input_energy_max = 8.0

fig1, ax = plt.subplots(1,1, figsize=(3.5,2.),sharex=False)

length = 80
# indexes = [0,1,3,5,9,11,12,4]
indexes = [0,1,3,5,9,11,12,4,13]



colors = plt.cm.binary(np.linspace(1.0, 1.0, len(indexes)))


for i, color in zip(indexes, colors):
    ax.plot(x_tensor[i,:length,0].detach().numpy(),x_tensor[i,:length,1].detach().numpy(),'-',linewidth=1, color='royalblue', label='DPC')
    ax.plot(optim_x[i,:length,0].detach().numpy(),optim_x[i,:length,1].detach().numpy(),'--',linewidth=1, color='crimson', label='Optimal', dashes=(1.5, 1.5))
    ax.plot(optim_x[i,0,0].detach().numpy(),optim_x[i,0,1].detach().numpy(),'.',markersize=5, color=color)
    if i == indexes[0]:
        plt.legend(framealpha=1.0,edgecolor='gray',fancybox=False, bbox_to_anchor=(1.00,0.5))
# ax.plot(x_tensor[:,length-1,0].detach().numpy(),x_tensor[:,length-1,1].detach().numpy(),'x',markersize=5, color='black', alpha=0.5, markerfacecolor=color)

ax.set_xlabel('$x_1$ [kWh]')
ax.set_ylabel('$x_2$ [kWh]')
# ax.margins(x=0,y=0)
fig1.subplots_adjust(left=0, right=1, top=1, bottom=0)

# ax.set_xticks([0,2,4.2,6,8.2])
# ax.set_yticks([0,0.9,1.8,2.7,3.6])
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)

plt.tight_layout(pad=0.0)
fig1.tight_layout(pad=0.0)
plt.grid()
fig1.show()
plt.gcf().set_tight_layout(True)

fig1.savefig(f'phase_plot.pdf', bbox_inches='tight',pad_inches=0.05,transparent=True)
fig1.savefig(f'phase_plot.pgf', bbox_inches='tight', pad_inches=0.05,transparent=True)