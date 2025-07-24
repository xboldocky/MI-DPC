#%%
from neuromancer.system import System
from torch import nn as nn
class PreviewSystem(System):

    def __init__(self, nodes, name=None, nstep_key='X', init_func=None, nsteps=None, preview_keys=list[str], preview_node_name='policy'):
        """
        :param nodes: (list of Node objects)
        :param name: (str) Unique identifier for system class.
        :param nstep_key: (str) Key is used to infer number of rollout steps from input_data
        :param init_func: (callable(input_dict) -> input_dict) This function is used to set initial conditions of the system
        :param nsteps: (int) prediction horizon (rollout steps) length
        """
        super().__init__(nodes=nodes, name=name)
        self.nstep_key = nstep_key
        self.nsteps = nsteps
        self.nodes, self.name = nn.ModuleList(nodes), name
        if init_func is not None:
            self.init = init_func
        self.input_keys = set().union(*[c.input_keys for c in nodes])
        self.output_keys = set().union(*[c.output_keys for c in nodes])
        self.system_graph = self.graph()
        self.preview_keys = preview_keys
        self.preview_node_name = preview_node_name
   
    def forward(self, input_dict):
        """
        :param input_dict: (dict: {str: Tensor}) Tensor shapes in dictionary are asssumed to be (batch, time, dim)
                                           If an init function should be written to assure that any 2-d or 1-d tensors
                                           have 3 dims.
        :return: (dict: {str: Tensor}) data with outputs of nstep rollout of Node interactions
        """
        data = input_dict.copy()
        nsteps = self.nsteps if self.nsteps is not None else data[self.nstep_key].shape[1]  # Infer number of rollout steps
        data = self.init(data)  # Set initial conditions of the system
        for i in range(nsteps): # number 2 works TODO : implement with sliding window for the preview
            # print(i)
            for node in self.nodes:
                indata = {
                    # encoding --- first feature for whole horizon, next feature, etc..
                    # k: (data[k].swapaxes(1,2).reshape(data[k].size(0), -1)
                    # encoding --- first timestep for all features, next timestep, etc..
                    # k: (data[k].reshape(data[k].size(0), -1) 
                    # k: (nn.functional.pad(data[k].reshape(data[k].size(0), -1), (0,20), mode='constant', value=0)[:,i*2:i*2+nsteps*2] 
                    k: (nn.functional.pad(data[k].reshape(data[k].size(0), -1), (0,nsteps*2), mode='constant', value=0)[:,i*2:i*2+nsteps*2] 
                    if (k in self.preview_keys and node.name in self.preview_node_name)
                    else data[k][:, i]) for k in node.input_keys
                }
                # print(indata['X'].shape)     
                # try:
                # #     # print(indata['U'].shape)  
                #     print(node.name)   
                #     print(indata['D'].shape)     
                # except:
                #     pass
                outdata = node(indata)  # compute
                data = self.cat(data, outdata)  # feed the data nodes
        return data  # return recorded system measurements
   
    def simulate(self, input_dict):
        nsteps = self.nsteps
        sim_data = input_dict.copy()
        simulation_length = sim_data[self.preview_keys[0]].size(1)
        sim_data = self.init(sim_data)
        for i in range(simulation_length-nsteps): # length of simulation
            for node in self.nodes:
                indata = {
                    # k: (sim_data[k][:,i:i+nsteps,:].swapaxes(1,2).reshape(sim_data[k].size(0), -1)
                    k: (sim_data[k][:,i:i+nsteps,:].reshape(sim_data[k].size(0), -1)
                    if (k in self.preview_keys and node.name in self.preview_node_name)
                    else sim_data[k][:, i]) for k in node.input_keys
                }
                # try:
                # #     # print(indata['U'].shape)  
                #     print(i)
                #     print(node.name)   
                #     print(indata['D'])     
                # except:
                #     pass
                outdata = node(indata)
                sim_data = self.cat(sim_data, outdata)
        return sim_data
# %%
