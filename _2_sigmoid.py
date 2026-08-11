
# %% Imports
import os, sys
import torch
from neuromancer.system import Node, System, SystemPreview
from neuromancer.modules import blocks, functions
from neuromancer.dataset import DictDataset
from neuromancer.constraint import variable
from neuromancer.loss import PenaltyLoss, BarrierLoss, AugmentedLagrangeLoss
from neuromancer.problem import Problem
from neuromancer.trainer import Trainer
import matplotlib.pyplot as plt
import numpy as np
import scipy
import importlib, time

script_dir = os.path.dirname(os.path.abspath(__file__))
default_type = torch.float32
torch.set_default_dtype(default_type)

#Deprecated softrounding
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


def _relaxed_round(x, slope=10.0): # differentiable nearest integer rounding via Sigmoid STE
    backward = (x-torch.floor(x)-0.5) # fractional value with rounding threshold
    return torch.round(x) + (torch.sigmoid(slope*backward) - torch.sigmoid(slope*backward).detach())
#%%
q_hr0 = 1
alpha_1 = 0.9983; alpha_2 = 0.9966; ni = 0.0010
betta_1 = 0.0750; betta_2 = 0.0750
betta_3 = 0.0825
betta_4 = 0.0833; betta_5 = 0.0833

x1_min, x2_min, input_energy_min = 0.0, 0.0, 0.0
x1_max = 8.4; x2_max = 3.6
d1_max = 7; d2_max = 17
input_energy_max = 8

u_int_max = 3.49; u_int_min = -0.49

ref1 = 4.2; ref2 = 1.8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

A = torch.tensor([[alpha_1, ni], [0, alpha_2-ni]])
B = torch.diag(torch.tensor([betta_1, betta_2]))
B_delta = torch.tensor([[0],[betta_3*q_hr0]])
B = torch.cat((B,B_delta),dim=1)
E = torch.diag(torch.tensor([-betta_4, -betta_5]))
C = torch.eye(2)

def ss_model(x, u, d):
    return x @ A.T + u @ B.T + d @ E.T

nx = A.shape[0]; nu = B.shape[1]; nd = E.shape[1]  

#%% Policy network architecture
for nsteps in [10,15,20,25,30,35,40]:
# for nsteps in [20]:
    torch.manual_seed(208)
    input_features = nx+(nd*(nsteps))
    layer_width = 140
    class policy(torch.nn.Module):
        def __init__(self, layer_width=layer_width):
            super(policy, self).__init__()
            self.fc_input = torch.nn.Linear(input_features, layer_width)  # Common Input Layer
        
            self.fc1_x1 = torch.nn.Linear(layer_width, layer_width)  # Layers for the first branch
            self.fc2_x1 = torch.nn.Linear(layer_width, layer_width) 
            self.fc_output_x1 = torch.nn.Linear(layer_width, 2, bias=True) 
        
            self.fc1_x2 = torch.nn.Linear(layer_width, layer_width)   # Layers for the second branch
            self.fc2_x2 = torch.nn.Linear(layer_width, layer_width)  
            self.fc_output_x2 = torch.nn.Linear(layer_width, 1, bias=True) 
            
            self.ln_input = torch.nn.LayerNorm(layer_width, elementwise_affine=False) # Common Input Layer norm

            self.ln1_x1 = torch.nn.LayerNorm(layer_width, elementwise_affine=False) # Frist branch Layer norm
            self.ln2_x1 = torch.nn.LayerNorm(layer_width, elementwise_affine=False)
            
            self.ln1_x2 = torch.nn.LayerNorm(layer_width, elementwise_affine=False) # Second branch Layer norm
            self.ln2_x2 = torch.nn.LayerNorm(layer_width, elementwise_affine=False)        
            self.dropout = torch.nn.Dropout(0.1) # Dropout object

        def forward(self, *inputs):
            if len(inputs) > 1:
                x = torch.cat(inputs, dim=-1)
            else:
                x = inputs[0]
            x = torch.nn.functional.tanh(self.fc_input(x))  # Common Input Layer
            x = self.ln_input(x)
            
            x1 = torch.nn.functional.tanh(self.fc1_x1(x)) # Continous input module
            x1 = self.ln1_x1(x1)
            x1 = self.dropout(x1)
            x1 = torch.nn.functional.tanh(self.fc2_x1(x1))
            x1 = self.ln2_x1(x1)
            x1 = self.dropout(x1)
            out1 = self.fc_output_x1(x1)

            x2 = torch.nn.functional.selu(self.fc1_x2(x)) # Integer input module
            x2 = self.ln1_x2(x2)
            x2 = torch.nn.functional.selu(self.fc2_x2(x2))
            x2 = self.ln2_x2(x2)
            out2 = _relaxed_round(functions.bounds_clamp(self.fc_output_x2(x2), -0.49, 3.49)) # Rounding
            
            return torch.cat((out1,out2), dim=1) # Return u1,u2,u3
    
    mip_policy = policy()
    mip_node = Node(mip_policy,['X', 'D'], ['U'], name='mip_policy')

    #%% System architecture
    system = Node(ss_model, ['X', 'U', 'D'], ['X'], name='system')

    cl_system = SystemPreview([mip_node,system],  
                            preview_keys_map={'D': ['mip_policy']},
                            preview_length={'D': nsteps-1},
                            pad_mode='constant', pad_constant=0, nsteps=nsteps)
    
    # cl_system.nsteps = nsteps # prediction horizon

    #%% Training Data
    num_data = 24000
    num_dev_data = 4000
    batch_size = 2000
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

    action_loss1 = 0.5*(u[:,:,[0]] == 0.)^2  # control penalty
    action_loss2 = 0.5*(u[:,:,[1]] == 0.0)^2  # control penalty
    integer_loss = 0.1*(u[:,:,[2]] == 0.0)^2
    regulation_loss1 = 1.0*(x[:,:,[0]] == ref1)^2  # target position
    regulation_loss2 = 1.0*(x[:,:,[1]] == ref2)^2  # target position
 
    action_loss1.name, action_loss2.name, integer_loss.name = 'action_loss1', 'action_loss2', 'integer_input_loss'
    regulation_loss1.name, regulation_loss2.name = 'control_state1', 'control_state2'

    objectives = [regulation_loss1, regulation_loss2, action_loss1, action_loss2, integer_loss] 

    input1_con_l = 25.0*(u[:,:,[0]] >= 0.0)
    input2_con_l = 25.0*(u[:,:,[1]] >= 0.0)
    # input3_con_l = 25.0*(u[:,:,[2]] >= 0.0)
    # input3_con_u = 25.0*(u[:,:,[2]] <= 3.0)
    input_energy_con_u = 25.0*((u[:,:,[0]]+u[:,:,[1]]) < input_energy_max)
    input_energy_con_l = 25.0*((u[:,:,[0]]+u[:,:,[1]]) >= 0.0)

    state1_con_l = 25.0*(x[:,:,[0]] >= x1_min)
    state1_con_u = 25.0*(x[:,:,[0]] < x1_max)
    state2_con_l = 25.0*(x[:,:,[1]] >= x2_min)
    state2_con_u = 25.0*(x[:,:,[1]] < x2_max)

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
    problem = Problem([cl_system], loss)
    #%% 
    for name, param in cl_system.named_parameters():
        if param.grad is not None:
            print(name, param.grad.shape)
    # %%
    optimizer = torch.optim.Adam(cl_system.parameters(), lr=0.0003, amsgrad=False, weight_decay=0.0)

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
        epoch_verbose=10,
        device=device,
        clip=torch.inf,
        lr_scheduler=False,
    )

    if __name__ == "__main__":
        
        start_time = time.time()
        best_model = trainer.train()
        training_time = time.time() - start_time
        
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_parameters = count_parameters(cl_system)

        trainer.model.load_state_dict(best_model) # load best trained model
        problem.load_state_dict(best_model)
        
        torch.save(cl_system, f'training_outputs/sigmoid/models/model_sigmoid_N{nsteps}.pt')
        
        training_data = {}
        training_data['NTP'] = n_parameters; training_data['TT'] = training_time
        torch.save(training_data, f'training_outputs/sigmoid/models/training_data_N{nsteps}.pt')
