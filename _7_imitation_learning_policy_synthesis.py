#%%
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
from utils import systempreview
import importlib, time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

def _relaxed_round(x, slope=10.0): # differentiable nearest integer rounding via Sigmoid STE
    backward = (x-torch.floor(x)-0.5) # fractional value with rounding threshold
    return torch.round(x) + (torch.sigmoid(slope*backward) - torch.sigmoid(slope*backward).detach())

#%% Policy network architecture
for nsteps in [10,15,20,25,30,35,40]:
# for nsteps in [20]:
    torch.manual_seed(208)
    labeled_data = np.load(f'imitation_learning_data/data_N{nsteps}.npz')
    x_data = torch.tensor(labeled_data['X'], dtype=torch.float32)
    u_data = torch.tensor(labeled_data['U'], dtype=torch.float32)
    d_data = torch.tensor(labeled_data['D'], dtype=torch.float32)
    num_data = x_data.size(0)

    nx=2;nd=2
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

    # system = Node(ss_model, ['X', 'U', 'D'], ['X'], name='system')
    
    cl_system = SystemPreview([mip_node],  
                            preview_keys_map={'D': ['mip_policy']},
                            preview_length={'D': nsteps-1},
                            pad_mode='constant', pad_constant=0, nsteps=1)

    num_dev_data = 4000
    num_train_data = num_data - num_dev_data

    x_train = x_data[:num_train_data]
    u_train = u_data[:num_train_data]
    d_train = d_data[:num_train_data]

    x_dev = x_data[num_train_data:]
    u_dev = u_data[num_train_data:]
    d_dev = d_data[num_train_data:]

    batch_size = 2000

    train_data = DictDataset({'X': x_train.to(device), 'D': d_train.to(device), 'U_hat': u_train.to(device)}, name='train')
    dev_data = DictDataset({'X': x_dev.to(device), 'D': d_dev.to(device), 'U_hat': u_dev.to(device)}, name='dev')
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                            collate_fn=train_data.collate_fn, shuffle=False)
    dev_loader = torch.utils.data.DataLoader(dev_data, batch_size=batch_size,
                                            collate_fn=dev_data.collate_fn, shuffle=False)


    u = variable('U')
    u_hat = variable('U_hat')

    imitation_loss = (u == u_hat[:,[0],:])^2.
    imitation_loss.name = 'imitation_loss'
    objectives = [imitation_loss]
    loss=PenaltyLoss(objectives, [])
    problem = Problem([cl_system], loss)
    optimizer = torch.optim.Adam(cl_system.parameters(), lr=0.0003, amsgrad=False, weight_decay=0.0)

    trainer = Trainer(
        problem.to(device),
        train_loader, dev_loader,
        optimizer=optimizer,
        epochs=500,
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
        
        torch.save(cl_system, f'training_outputs/imitation_learning/models/model_imitation_N{nsteps}.pt')
        
        training_data = {}
        training_data['NTP'] = n_parameters; training_data['TT'] = training_time
        training_data['num_data'] = num_data
        torch.save(training_data, f'training_outputs/imitation_learning/models/training_data_N{nsteps}.pt')

