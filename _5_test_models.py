#%%
import torch
import neuromancer
import scipy, time
import numpy as np
import _2_sigmoid
from _2_sigmoid import A,B,E, policy, loss, ss_model
from utils import initial_conditions
import matplotlib.pyplot as plt
import gc

torch.set_num_threads(1)
with torch.no_grad():
    s_length = 1873
    _2_sigmoid.A = A.cpu()
    _2_sigmoid.B = B.cpu()
    _2_sigmoid.E = E.cpu()

    torch.manual_seed(206)
    torch.set_default_dtype(torch.float32)
    torch.set_default_device('cpu')
    # nstep_list = [10, 15]
    nstep_list = [10, 15, 20, 25, 30, 40]


    # Load CPLEX data
    cplex_data = {}
    for nsteps in nstep_list[:-1]:
        cplex_data[f'N={nsteps}'] = []
        for ic in range(20):
            t_data = torch.load(f"CPLEX_inference_data/N{nsteps}/cvxpy_cplex_ic{ic}.pt", weights_only=False)[0]
            cplex_data[f'N={nsteps}'].append(t_data)

    # Evaluate cplex data
    MIT_hist, FUP_hist, ell_hist = {},{},{}
    MIT_list, FUP_list, ell_list = [],[],[]
    nup = 0
    for nsteps in nstep_list[:-1]:
        for ic in range(20):
            MIT_list.append(torch.tensor(cplex_data[f'N={nsteps}'][ic]['times']).mean())
            ell_list.append(torch.tensor(cplex_data[f'N={nsteps}'][ic]['stage_cost']).mean())
            for i in range(torch.tensor(cplex_data[f'N={nsteps}'][ic]['times']).shape[0]):
                if torch.tensor(cplex_data[f'N={nsteps}'][ic]['times'])[i] >= 15:
                    nup += 1
            FUP_list.append(torch.tensor(nup))
            nup=0
        MIT_hist[f'N={nsteps}'] = torch.stack(MIT_list).mean()
        ell_hist[f'N={nsteps}'] = torch.stack(ell_list).mean()
        FUP_hist[f'N={nsteps}'] = \
        torch.stack(FUP_list).sum()/(20*torch.tensor(cplex_data[f'N={nsteps}'][ic]['times']).shape[0])*100
        MIT_list = []; ell_list = []; FUP_list = []

    # Printing CPLEX data
    print('#'*100)
    print('#'*100)
    print('CPLEX data analysis')
    for nsteps in nstep_list[:-1]:
        print("-"*100)
        print(f'N={nsteps}')
        print(f"L mean = {ell_hist[f'N={nsteps}']}")
        print(f"MIT = {MIT_hist[f'N={nsteps}']}")
        print(f"FUP = {FUP_hist[f'N={nsteps}']}")
        

    #%% MI-DPC sigmoid STE analysis
    print('#'*100)
    print('#'*100)
    print('Sigmoid STE data analysis')

    x0 = initial_conditions.x0_tensor.unsqueeze(0)
    d = scipy.io.loadmat("loads_matrix.mat")
    d_array = d['newloads_matrix']
    d_torch = torch.tensor(d_array, dtype=torch.float32).unsqueeze(0)

    sigmoid_data = {}
    # simulation
    for nsteps in nstep_list:
        sigmoid_data[f"N={nsteps}"] = []
        cl_system = torch.load(f'training_outputs/sigmoid/models/model_sigmoid_N{nsteps}.pt', weights_only=False, map_location=torch.device("cpu"))
        cl_system.eval()
        start_time = time.time()
        gc.disable()
        for ic in range(20):
            input_data = {'X': x0[:,[ic],:].cpu(), 'D': d_torch.cpu()}
            cl_system.nsteps = s_length
            trajectories = cl_system.forward(input_data)
            sigmoid_data[f"N={nsteps}"].append(trajectories)
        end_time = time.time() - start_time
        sigmoid_data[f"N={nsteps}"][0]['MIT']=torch.tensor(end_time)/(20*s_length)
        gc.enable()
    # cost computation
    for nsteps in nstep_list:
        cost_list = []
        for ic in range(20):
            stage_cost = loss.calculate_objectives(sigmoid_data[f'N={nsteps}'][ic])
            cost_list.append(stage_cost['objective_loss'])
        sigmoid_data[f'N={nsteps}'][0]['cost'] = torch.stack(cost_list).mean() 

    torch.save(sigmoid_data, 'simulation_data/sigmoid.pt')
    # training data loading
    #%%
    # printing
    for nsteps in nstep_list:
        print("-"*100)
        print(f"N = {nsteps}")
        print(f"L mean = {sigmoid_data[f'N={nsteps}'][0]['cost']:.5f}")
        try:
            RSM = (sigmoid_data[f'N={nsteps}'][0]['cost'] -  ell_hist[f'N={nsteps}']) / sigmoid_data[f'N={nsteps}'][0]['cost']
            print(f"RSM = {RSM*100:.4f}")
        except:
            print(f"RSM = None")
        print(f"MIT = {sigmoid_data[f'N={nsteps}'][0]['MIT']:.5f}")
        
        sigmoid_training_data = torch.load(f'training_outputs/sigmoid/models/training_data_N{nsteps}.pt')
        print(f"NTP = {sigmoid_training_data['NTP']}")
        print(f"TT = {sigmoid_training_data['TT']}")
    del sigmoid_data

    #%%
    ##
    ## IMITATION LEARNING EVAL
    ##
    print('#'*100)
    print('#'*100)
    print('Imitation Learning data analysis')
    from neuromancer.system import SystemPreview, Node
    # from _5_test_models import policy
    im_data = {}
    # simulation
    for nsteps in nstep_list:
        im_data[f"N={nsteps}"] = []
        im_policy = torch.load(f'training_outputs/imitation_learning/models/model_imitation_N{nsteps}.pt', weights_only=False, map_location=torch.device("cpu"))
        system = Node(ss_model, ['X', 'U', 'D'], ['X'], name='system')
        
        cl_system = SystemPreview([im_policy.nodes[0],system],  
                                preview_keys_map={'D': ['mip_policy']},
                                preview_length={'D': nsteps-1})
        
        # im_cl_system_nodes = [im_cl_system.nodes[0], cl_system.nodes[1]]
        cl_system.eval()
        gc.disable()
        start_time = time.time()
        for ic in range(20):
            input_data = {'X': x0[:,[ic],:].cpu(), 'D': d_torch.cpu()}
            cl_system.nsteps = s_length
            trajectories = cl_system.forward(input_data)
            im_data[f"N={nsteps}"].append(trajectories)
        end_time = time.time() - start_time
        gc.enable()
        im_data[f"N={nsteps}"][0]['MIT']=torch.tensor(end_time)/(20*s_length)

    # cost computation
    for nsteps in nstep_list:
        cost_list = []
        for ic in range(20):
            stage_cost = loss.calculate_objectives(im_data[f'N={nsteps}'][ic])
            cost_list.append(stage_cost['objective_loss'])
        im_data[f'N={nsteps}'][0]['cost'] = torch.stack(cost_list).mean() 


    torch.save(im_data, 'simulation_data/imitation.pt')
    #%%
    for nsteps in nstep_list:
        print("-"*100)
        print(f"N = {nsteps}")
        print(f"L mean = {im_data[f'N={nsteps}'][0]['cost']:.5f}")
        try:
            RSM = (im_data[f'N={nsteps}'][0]['cost'] -  ell_hist[f'N={nsteps}']) / im_data[f'N={nsteps}'][0]['cost']
            print(f"RSM = {RSM*100:.4f}")
        except:
            print(f"RSM = None")
        print(f"MIT = {im_data[f'N={nsteps}'][0]['MIT']:.5f}")
        
        imitation_training_data = torch.load(f'training_outputs/imitation_learning/models/training_data_N{nsteps}.pt')
        print(f"NTP = {imitation_training_data['NTP']}")
        print(f"TT = {imitation_training_data['TT']}")
        print(f"Number of data samples = {imitation_training_data['num_data']}")

    del im_data
    #%%
    ###
    ### Softmax
    ###
    print('#'*100)
    print('#'*100)
    print('Softmax STE data analysis')
    import _3_softmax
    from _3_softmax import A,B,E, integers, policy, loss, ss_model

    _3_softmax.A = A.cpu()
    _3_softmax.B = B.cpu()
    _3_softmax.E = E.cpu()
    _3_softmax.integers = integers.cpu()
    del cl_system
    softmax_data = {}
    # simulation
    for nsteps in nstep_list:
        softmax_data[f"N={nsteps}"] = []
        cl_system = torch.load(f'training_outputs/softmax/models/model_softmax_N{nsteps}.pt', weights_only=False, map_location=torch.device("cpu"))
        cl_system.eval()
        cl_system.nodes[0].callable.enable_gumbels = False
        gc.disable()
        start_time = time.time()
        for ic in range(20):
            input_data = {'X': x0[:,[ic],:].cpu(), 'D': d_torch.cpu()}
            cl_system.nsteps = s_length
            trajectories = cl_system.forward(input_data)
            softmax_data[f"N={nsteps}"].append(trajectories)
        end_time = time.time() - start_time
        softmax_data[f"N={nsteps}"][0]['MIT']=torch.tensor(end_time)/(20*s_length)
        gc.enable()
    # cost computation
    for nsteps in nstep_list:
        cost_list = []
        for ic in range(20):
            stage_cost = loss.calculate_objectives(softmax_data[f'N={nsteps}'][ic])
            cost_list.append(stage_cost['objective_loss'])
        softmax_data[f'N={nsteps}'][0]['cost'] = torch.stack(cost_list).mean() 

    torch.save(softmax_data, 'simulation_data/softmax.pt')

    #%%
    # printing
    for nsteps in nstep_list:
        print("-"*100)
        print(f"N = {nsteps}")
        print(f"L mean = {softmax_data[f'N={nsteps}'][0]['cost']:.5f}")
        try:
            RSM = (softmax_data[f'N={nsteps}'][0]['cost'] -  ell_hist[f'N={nsteps}']) / softmax_data[f'N={nsteps}'][0]['cost']
            print(f"RSM = {RSM*100:.4f}")
        except:
            print(f"RSM = None")
        print(f"MIT = {softmax_data[f'N={nsteps}'][0]['MIT']:.5f}")
        
        softmax_training_data = torch.load(f'training_outputs/softmax/models/training_data_N{nsteps}.pt')
        print(f"NTP = {softmax_training_data['NTP']}")
        print(f"TT = {softmax_training_data['TT']}")

    del softmax_data
    #%%
    print('#'*100)
    print('#'*100)
    print('Learnable Threshold data analysis')
    import _4_learnable_threshold
    from _4_learnable_threshold import A,B,E, policy, loss, ss_model, rounding_network, clip_fn

    _4_learnable_threshold.A = A.cpu()
    _4_learnable_threshold.B = B.cpu()
    _4_learnable_threshold.E = E.cpu()
    lt_data = {}
    # simulation
    for nsteps in nstep_list:
        lt_data[f"N={nsteps}"] = []
        cl_system = torch.load(f'training_outputs/lt/models/model_lt_N{nsteps}.pt', weights_only=False, map_location=torch.device("cpu"))
        cl_system.eval()
        gc.disable()
        start_time = time.time()
        for ic in range(20):
            input_data = {'X': x0[:,[ic],:].cpu(), 'D': d_torch.cpu()}
            cl_system.nsteps = s_length
            trajectories = cl_system.forward(input_data)
            lt_data[f"N={nsteps}"].append(trajectories)
        end_time = time.time() - start_time
        lt_data[f"N={nsteps}"][0]['MIT']=torch.tensor(end_time)/(20*s_length)
        gc.enable()
    # cost computation
    for nsteps in nstep_list:
        cost_list = []
        for ic in range(20):
            stage_cost = loss.calculate_objectives(lt_data[f'N={nsteps}'][ic])
            cost_list.append(stage_cost['objective_loss'])
        lt_data[f'N={nsteps}'][0]['cost'] = torch.stack(cost_list).mean() 
    torch.save(lt_data, 'simulation_data/lt.pt')

    #%%
    # printing
    for nsteps in nstep_list:
        print("-"*100)
        print(f"N = {nsteps}")
        print(f"L mean = {lt_data[f'N={nsteps}'][0]['cost']:.5f}")
        try:
            RSM = (lt_data[f'N={nsteps}'][0]['cost'] -  ell_hist[f'N={nsteps}']) / lt_data[f'N={nsteps}'][0]['cost']
            print(f"RSM = {RSM*100:.4f}")
        except:
            print(f"RSM = None")
        print(f"MIT = {lt_data[f'N={nsteps}'][0]['MIT']:.5f}")
        
        lt_training_data = torch.load(f'training_outputs/lt/models/training_data_N{nsteps}.pt')
        print(f"NTP = {lt_training_data['NTP']}")
        print(f"TT = {lt_training_data['TT']}")

    del lt_data

    print('...DONE...')
#%% SIGMOID MANUAL COST CALCULATION
# cost = []
# for ic in range(20):
#     for i in range(s_length):
#         cost_ = (sigmoid_data['N=25'][ic]['X'][0,[i],:]-torch.tensor([4.2,1.8]).view(1,-1))@(sigmoid_data['N=25'][ic]['X'][0,[i],:]-torch.tensor([4.2,1.8]).view(1,-1)).T \
#         + sigmoid_data['N=25'][ic]['U'][0,[i],:]@torch.diag(torch.tensor([0.5,0.5,0.1]))@sigmoid_data['N=25'][ic]['U'][0,[i],:].T
#         cost.append(cost_)
# print("manual cost: ", torch.stack(cost).mean())
