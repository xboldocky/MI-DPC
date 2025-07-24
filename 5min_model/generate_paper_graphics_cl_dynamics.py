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
import matplotlib
# import tikzplotlib

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

#%%

x1_max = 8.4
x1_min = 0.0
x2_max = 3.6
x2_min = 0.0
input_energy_max = 8.0

time_vector = np.arange(0, 1873)*5/60

nsteps = 20
trajectories = torch.load('authdata_nsteps_20/sigmoid_trajectories.pt')
u1_opt = scipy.io.loadmat(f"optimal_5_min_conservative_ocp/6.5days_N{nsteps}_Q11.mat")['u1']
u2_opt = scipy.io.loadmat(f"optimal_5_min_conservative_ocp/6.5days_N{nsteps}_Q11.mat")['u2']
u3_opt = scipy.io.loadmat(f"optimal_5_min_conservative_ocp/6.5days_N{nsteps}_Q11.mat")['u3']
x1_opt = scipy.io.loadmat(f"optimal_5_min_conservative_ocp/6.5days_N{nsteps}_Q11.mat")['x1']
x2_opt = scipy.io.loadmat(f"optimal_5_min_conservative_ocp/6.5days_N{nsteps}_Q11.mat")['x2']

d = scipy.io.loadmat("newloads_matrix.mat")
d_tensor = torch.tensor(d['newloads_matrix'], dtype=torch.float32)
s_length = 1873
dists = d_tensor[:s_length+nsteps,:].unsqueeze(0)

until=864
#%%
# fig1, ax = plt.subplots(4,2, figsize=(7.16, 3.5),sharex=True)
fig1, ax = plt.subplots(4,2, figsize=(7.16, 4.5),sharex=True)
# fig1, ax = plt.subplots(4,2, figsize=(20, 8),sharex=True)


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

#Stary graf
# ax[2,0].plot(time_vector[:until],trajectories['D'][:,:until,0][0].to('cpu').detach().numpy(), '-', color='gray')
# ax[3,0].plot(time_vector[:until],trajectories['D'][:,:until,1][0].to('cpu').detach().numpy(), '-', color='gray', label='Disturbances')
# ax[2,0].set_ylabel('$d_1$ [kW]')
# ax[3,0].set_ylabel('$d_2$ [kW]')

#Novy graf
ax[2,0].set_ylabel('$e$ [kWh]')

ax[2,0].plot(time_vector[:until],x1_opt[:until,0]-trajectories['X'][:,:until,0][0].to('cpu').detach().numpy(), color='dimgray', alpha=0.9, label='$e_1$')
ax[2,0].plot(time_vector[:until],x2_opt[:until,0]-trajectories['X'][:,:until,1][0].to('cpu').detach().numpy(), color='gray',alpha=0.7, label='$e_2$')
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
fig1.savefig(f'fig1.pdf', bbox_inches='tight',transparent=True, pad_inches=0.05)
fig1.savefig(f'fig1.pgf', bbox_inches='tight',transparent=True, pad_inches=0.05)



fig2, ax = plt.subplots(1,1, figsize=(3.5,2),sharex=False)

length = 80
indexes = [0,1,3,5,9,11,12]



colors = plt.cm.hsv(np.linspace(0, 0.9, len(indexes)))


# for i, color in zip(indexes, colors):
#     ax.plot(x_tensor[i,:length,0].detach().numpy(),x_tensor[i,:length,1].detach().numpy(),'k-',linewidth=0.3)
#     ax.plot(optim_x[i,:length,0].detach().numpy(),optim_x[i,:length,1].detach().numpy(),'-',linewidth=0.3, color='royalblue')
#     ax.plot(optim_x[i,0,0].detach().numpy(),optim_x[i,0,1].detach().numpy(),'.',markersize=5, color=color)
#     ax.plot(x_tensor[i,length-1,0].detach().numpy(),x_tensor[i,length-1,1].detach().numpy(),'*',markersize=5, color=color)


sigmoid_mean = np.array([6.7101263999938965,4.731839179992676,4.087917327880859,3.9580652713775635,3.870213270187378,3.8436098098754883])

sigmoid_max = np.array([ 6.83247184753418, 4.8472418785095215,4.206162452697754, 4.072006702423096,3.984553575515747, 3.9594171047210693])

sigmoid_min= np.array([6.629976272583008, 4.655569076538086,4.01269006729126,3.884345054626465,3.7958505153656006,3.7688417434692383])

opt_mean = np.array([5.828334840913413,4.41193979000925,4.0152404162578526,3.88889393958868,3.84705246507289,3.847052465072])

opt_max = np.array([5.94675158878063,4.5261037186447,4.12868570120952,4.0022055482411,3.9603388125827,3.96033881258])

opt_min = np.array([5.75026053037973,4.337856639820,3.94178534416469,3.81554629261794,3.773718959379074,3.7737189593790])

gumbel_mean = np.array([6.793025016784668,4.785126686096191,4.132749080657959,3.9483015537261963,3.8721327781677246,3.8656599521636963,])
gumbel_max = np.array([6.914600372314453,4.899981498718262,4.247660636901855,4.06406354904174,3.98913192749023,3.98153400421142,])
gumbel_min = np.array([6.7120542526245,4.7099981307983,4.058752059936,3.874103307723,3.79755473136,3.790627479553])

lt_mean = np.array([6.4613647460937,4.5480279922485,4.0909347534179,3.9153988361358,3.8753361701965,3.8443620204925,])
lt_max = np.array([6.583170890808105,4.66363382339477,4.20526170730590,4.0280656814575,3.98811745643615,3.9579191207885,])
lt_min = np.array([6.3796014785766,4.4731950759887,4.0163388252258,3.84141826629,3.802076578140,3.771082878112,])




pred_hor = np.array([10,15,20,25,30,40])
# Plot Sigmoid

plt.errorbar(
    pred_hor[:-1], sigmoid_mean[:-1], yerr=[sigmoid_mean[:-1]-sigmoid_min[:-1], sigmoid_max[:-1]-sigmoid_mean[:-1]], fmt='.', 
    markersize=4, capsize=3, capthick=1, 
    elinewidth=1, color='royalblue', label='STE with Sigmoid', alpha=0.7
)
plt.plot(pred_hor[:-1], sigmoid_mean[:-1],'--',color='royalblue',linewidth=1,alpha=0.5 )




plt.errorbar(
    pred_hor[:-1], gumbel_mean[:-1], yerr=[gumbel_mean[:-1]-gumbel_min[:-1], gumbel_max[:-1]-gumbel_mean[:-1]], fmt='.', 
    markersize=4, capsize=3, capthick=1, 
    elinewidth=1, color='crimson', label='STE with Softmax', alpha=0.5
)
plt.plot(pred_hor[:-1], gumbel_mean[:-1],'--',color='crimson',linewidth=1,alpha=0.5 )


plt.errorbar(
    pred_hor[:-1], lt_mean[:-1], yerr=[lt_mean[:-1]-lt_min[:-1], lt_max[:-1]-lt_mean[:-1]], fmt='.', 
    markersize=4, capsize=3, capthick=1, 
    elinewidth=1, color='green', label='Learnable Threshold', alpha=0.5
)
plt.plot(pred_hor[:-1], lt_mean[:-1],'--',color='green',linewidth=1,alpha=0.5 )

plt.errorbar(
    pred_hor[:-1], opt_mean[:-1], yerr=[opt_mean[:-1]-opt_min[:-1], opt_max[:-1]-opt_mean[:-1]], fmt='.', 
    markersize=4, capsize=3, capthick=1, 
    elinewidth=1, color='k', label='Optimal', alpha=0.5
)
plt.plot(pred_hor[:-1], opt_mean[:-1],'--',color='black',linewidth=1,alpha=0.5 )

ax.legend(framealpha=1,edgecolor='gray',fancybox=False)

ax.set_xlabel('$N$')
ax.set_ylabel('$\ell_\mathrm{mean}$')
# ax.margins(x=0,y=0)
fig2.subplots_adjust(left=0, right=1, top=1, bottom=0)

# ax.set_xticks([0,2,4.2,6,8.2])
# ax.set_yticks([0,0.9,1.8,2.7,3.6])


plt.tight_layout(pad=0.0)
fig2.tight_layout(pad=0.0)
plt.grid()
fig2.show()
plt.gcf().set_tight_layout(True)

fig2.savefig(f'loss_plot.pdf', bbox_inches='tight',pad_inches=0.05,transparent=True)
fig2.savefig(f'loss_plot.pgf', bbox_inches='tight', pad_inches=0.05,transparent=True)

# %%
