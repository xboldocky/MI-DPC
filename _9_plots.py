#%%
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import torch, scipy, os
script_dir = os.path.dirname(os.path.abspath(__file__))

###
### Phase plot
###
nsteps = 25
indexes = [0,1,3,5,9,11,12,4,13]
length = 80

cplex_data = {}
cplex_data[f'N={nsteps}'] = []
for ic in range(20):
    t_data = torch.load(f"CPLEX_inference_data/N{nsteps}/cvxpy_cplex_ic{ic}.pt", weights_only=False)[0]
    cplex_data[f'N={nsteps}'].append(t_data)

dpc_data = torch.load('simulation_data/softmax.pt')[f'N={nsteps}']
x_data_list = []; u_data_list = []
optim_x_list = []; optim_u_list = []

for i in range(20):
    x_data_list.append(dpc_data[i]['X'][:,:,:])
    u_data_list.append(dpc_data[i]['U'][:,:,:])
    optim_x_list.append(torch.tensor(cplex_data[f'N={nsteps}'][i]['x'][:,:], dtype=torch.float32).view(1,-1,2))
    optim_u_list.append(torch.tensor(cplex_data[f'N={nsteps}'][i]['u'][:,:], dtype=torch.float32).view(1,-1,3))

x_tensor = torch.vstack(x_data_list)
u_tensor = torch.vstack(u_data_list)
optim_x = torch.vstack(optim_x_list)
optim_u = torch.vstack(optim_u_list)


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
# original 3.5, 2.5
fig1, ax = plt.subplots(1,1, figsize=(3.5,2),sharex=False)

# indexes = [0,1,3,5,9,11,12,4]



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

fig1.savefig(f'plots/phase_plotv2_large.pdf', bbox_inches='tight',pad_inches=0.05,transparent=True)
fig1.savefig(f'plots/phase_plotv2_large.pgf', bbox_inches='tight', pad_inches=0.05,transparent=True)
# %%
###
### Control Plot
###

nsteps = 20

cplex_data = {}
cplex_data[f'N={nsteps}'] = []
for ic in range(20):
    t_data = torch.load(f"CPLEX_inference_data/N{nsteps}/cvxpy_cplex_ic{ic}.pt", weights_only=False)[0]
    cplex_data[f'N={nsteps}'].append(t_data)

dpc_data = torch.load('simulation_data/sigmoid.pt')[f'N={nsteps}']
x_data_list = []; u_data_list = []
optim_x_list = []; optim_u_list = []

for i in range(20):
    x_data_list.append(dpc_data[i]['X'][:,:,:])
    u_data_list.append(dpc_data[i]['U'][:,:,:])
    optim_x_list.append(torch.tensor(cplex_data[f'N={nsteps}'][i]['x'][:,:], dtype=torch.float32).view(1,-1,2))
    optim_u_list.append(torch.tensor(cplex_data[f'N={nsteps}'][i]['u'][:,:], dtype=torch.float32).view(1,-1,3))

x_tensor = torch.vstack(x_data_list)
u_tensor = torch.vstack(u_data_list)
optim_x = torch.vstack(optim_x_list)
optim_u = torch.vstack(optim_u_list)

matplotlib.use("pgf")

plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "text.usetex": True,  # Use LaTeX for text
    "font.family": "serif",  # Use a serif font
    "font.size": 10,  # Set font size
    "pgf.rcfonts": False,  # Don't override with default matplotlib fonts
    "legend.fontsize": 6,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8
})

s_length = 1873

file_path = os.path.join(script_dir, 'loads_matrix.mat')
d = scipy.io.loadmat(file_path)
d_tensor = torch.tensor(d['newloads_matrix'], dtype=torch.float32, device='cpu')


# d = scipy.io.loadmat("newloads_matrix.mat")
# d_tensor = torch.tensor(d['newloads_matrix'], dtype=torch.float32)
dists = d_tensor[:s_length+nsteps,:].unsqueeze(0)
time_vector = np.arange(0, 1873)*5/60
until=864

x1_opt = optim_x[0,:,[0]]
x2_opt = optim_x[0,:,[1]]
u1_opt = optim_u[0,:,[0]]
u2_opt = optim_u[0,:,[1]]
u3_opt = optim_u[0,:,[2]]

trajectories = {
    'X': x_tensor[[0],:,:],
    'D': dists,
    'U': u_tensor[[0],:,:],
    'R': torch.cat((torch.ones(1,s_length,1)*4.2, torch.ones(1,s_length,1)*1.8), -1).view(1,-1,2)
}

fig1, ax = plt.subplots(4,2, figsize=(7.16, 4),sharex=True) # original 7.16, 4.5


ax[0,0].plot(time_vector[:until],trajectories['R'][:,:until,0][0].to('cpu').detach().numpy(), '-',linewidth=1, color='black')
ax[0,0].plot(time_vector[:until],trajectories['X'][:,:until,0][0].to('cpu').detach().numpy(), color='royalblue', label='DPC')
ax[0,0].plot(time_vector[:until],x1_opt[:until], '--', color='crimson',label='Optimal',alpha=1, dashes=(1.5, 1.5))
# ax[0,0].plot(x_stack[:,0].to('cpu').detach().numpy(), 'r')
ax[0,0].set_ylabel('$x_1$ [kWh]')
ax[0,0].plot(time_vector[:until],torch.zeros(dists.size(1)-nsteps).to('cpu')[:until], 'k--')
ax[0,0].plot(time_vector[:until],x1_max*torch.ones(dists.size(1)-nsteps)[:until].to('cpu'), 'k--')
# ax[0,0].legend(frameon=True,framealpha=1)

# ax[0,0].legend()

ax[1,0].plot(time_vector[:until],trajectories['R'][:,:dists.size(1)-nsteps,1][0].to('cpu').detach().numpy()[:until], 'k', linewidth=1)
ax[1,0].plot(time_vector[:until],trajectories['X'][:,:until,1][0].to('cpu').detach().numpy(), color='royalblue')
ax[1,0].plot(time_vector[:until],x2_opt[:until], '--', color='crimson', alpha=1, dashes=(1.5, 1.5))
ax[1,0].set_ylabel('$x_2$ [kWh]')
ax[1,0].plot(time_vector[:until],torch.zeros(dists.size(1)-nsteps).to('cpu')[:until], 'k--')
ax[1,0].plot(time_vector[:until],x2_max*torch.ones(dists.size(1)-nsteps).to('cpu')[:until],'k--')


ax[2,0].set_ylabel('$e$ [kWh]')

ax[2,0].plot(time_vector[:until],x1_opt[:until,0]-trajectories['X'][0,:until,0].to('cpu').detach().numpy(), color='dimgray', alpha=0.9, label='$e_1$')
ax[2,0].plot(time_vector[:until],x2_opt[:until,0]-trajectories['X'][0,:until,1].to('cpu').detach().numpy(), color='gray',alpha=0.7, label='$e_2$')
ax[2,0].legend(frameon=True,framealpha=0.8,fancybox=False,edgecolor='w', loc='upper right', bbox_to_anchor=(1,1))



ax[3,0].plot(time_vector[:until],trajectories['D'][:,:until,0][0].to('cpu').detach().numpy(), '-', color='gray', alpha=0.7, label='$d_1$')
ax[3,0].plot(time_vector[:until],trajectories['D'][:,:until,1][0].to('cpu').detach().numpy(), '-', color='dimgray', label='$d_2$', alpha=0.9)
ax[3,0].set_ylabel('$d$ [kW]')
ax[3,0].legend(frameon=True,framealpha=0.8, fancybox=False, edgecolor='w', loc='best')



# ax[0,1].step(trajectories['U'][:,:,0].to('cpu').detach().numpy())
ax[0,1].step(time_vector[:until], trajectories['U'][0,:until,0].to('cpu').detach().numpy(), color='royalblue')
ax[0,1].step(time_vector[:until],u1_opt[:until], '--', color='crimson', alpha=1, dashes=(1.5, 1.5))
# ax[1,1].plot(trajectories['U'][:,:,1][0].to('cpu').detach().numpy())

ax[1,1].step(time_vector[:until], trajectories['U'][0,:until,1].to('cpu').detach().numpy(), color='royalblue')
ax[1,1].step(time_vector[:until],u2_opt[:until], '--', color='crimson', alpha=1, dashes=(1.5, 1.5))

# ax[2,1].plot(trajectories['U'][:,:,2][0].to('cpu').detach().numpy())
ax[2,1].step(time_vector[:until], trajectories['U'][0,:until,2].to('cpu').detach().numpy(), color='royalblue')
ax[2,1].step(time_vector[:until],u3_opt[:until], '--', color='crimson', alpha=1, dashes=(1.5, 1.5))

# ax[3,1].plot(trajectories['U'][:,:,0][0].to('cpu').detach().numpy()+trajectories['U'][:,:,1][0].to('cpu').detach().numpy())
ax[3,1].plot(time_vector[:until],input_energy_max*torch.ones(dists.size(1)-nsteps).to('cpu')[:until],'k--', markersize=4)
ax[3,1].plot(time_vector[:until],0.0*torch.ones(dists.size(1)-nsteps).to('cpu')[:until],'k--', markersize=4)
ax[3,1].step(time_vector[:until], trajectories['U'][0,:until,0].to('cpu').detach().numpy()+trajectories['U'][0,:until,1].to('cpu').detach().numpy(), color='royalblue')
ax[3,1].step(time_vector[:until], u1_opt[:until]+u2_opt[:until], '--', color='crimson', alpha=1, dashes=(1.5, 1.5))
# line, = ax[3,1].step(u1_opt+u2_opt, 'k--')


# fig1.legend(loc="upper center", bbox_to_anchor=(0.45, 1.01), ncol=8, frameon=False)


# ax[3,1].set_xdata()

ax[-1,1].set_xlabel('Time [h]')
ax[-1,0].set_xlabel('Time [h]')
ax[0,1].set_ylabel('$u_1$ [kW]')
ax[1,1].set_ylabel('$u_2$ [kW]')
ax[2,1].set_ylabel('$\delta_1$ [kW]')
ax[3,1].set_ylabel('$u_1\!+\!u_2$ [kW]')

ax[3,1].set_xlim(xmin=-1, xmax=time_vector[until]+1)
ax[3,0].set_xlim(xmin=-1, xmax=time_vector[until]+1)

for axes, (func, label) in zip(ax.flat, []):
    # axes.xticks(time_vector)
    axes.xaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: rf"${val:.1f}$"))
    axes.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: rf"${val:.1f}$"))
    axes.set_xlim(xmin=4, xmax=time_vector[until])
    axes.grid()

ax[0,0].grid()
ax[1,0].grid()
ax[2,0].grid()
ax[3,0].grid()
ax[0,1].grid()
ax[1,1].grid()
ax[2,1].grid()
ax[0,0].set_yticks([0,4.2,8.4])
ax[1,0].set_yticks([0,1.8,3.6])
#Stary graf
# ax[2,0].set_yticks([0,3.5])
ax[2,1].set_yticks([0,1,2,3])
ax[3,1].set_yticks([0,4,8])
# ax[3,1].grid()
# plt.xticks(fontsize=8)
# plt.yticks(fontsize=8)

fig1.tight_layout(h_pad=0.1)
plt.grid()
fig1.show()
fig1.savefig(f'plots/fig1v2.pdf', bbox_inches='tight',transparent=True, pad_inches=0.05)
fig1.savefig(f'plots/fig1v2.pgf', bbox_inches='tight',transparent=True, pad_inches=0.05)

# %%
