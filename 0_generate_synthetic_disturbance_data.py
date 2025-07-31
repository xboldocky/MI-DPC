# %% Generate synthetic disturbance data

import scipy
import torch

nsteps_list = [10, 20, 30, 40, 60]
num_data = 20000

for nsteps in nsteps_list:

    d = scipy.io.loadmat("newloads_matrix.mat")
    d_tensor = torch.tensor(d['newloads_matrix'], dtype=torch.float32)

    d1 = d_tensor[:,0]
    d2 = d_tensor[:,1]


    d1_2d = d1.unsqueeze(-1).unfold(0,nsteps,1).swapaxes(1,-1)
    d2_2d = d2.unsqueeze(-1).unfold(0,nsteps,1).swapaxes(1,-1)


    rows_with_all_zeros = torch.all(d2_2d == 0, dim=1)

    # Count the number of such rows
    # num_rows_with_all_zeros = torch.sum(rows_with_all_zeros).item()
    # print(f"Number of rows filled with all zeros: {num_rows_with_all_zeros}")



    num_extensions = 20
    extended_tensors = [d2_2d]  # Start with the original tensor
    for _ in range(num_extensions):
        random_indices = torch.randperm(d2_2d.size(1))  # Permutes 21 indices
        randomized_tensor = d2_2d[:, random_indices, :]*torch.empty(d2_2d.shape).uniform_(0.1, 0.8)
        extended_tensors.append(randomized_tensor)
    d2_extended = torch.cat(extended_tensors, dim=0)

    num_extensions = 20
    extended_tensors = [d1_2d]  # Start with the original tensor
    for _ in range(num_extensions):
        random_indices = torch.randperm(d1_2d.size(1))  # Permutes 21 indices
        randomized_tensor = d1_2d[:, random_indices, :]*torch.empty(d1_2d.shape).uniform_(0.1, 0.8)
        extended_tensors.append(randomized_tensor)
    d1_extended = torch.cat(extended_tensors, dim=0)

    rows_with_all_zeros = torch.all(d2_extended == 0, dim=1)

    # Count the number of such rows
    num_rows_with_all_zeros = torch.sum(rows_with_all_zeros).item()

    d_extended = torch.cat((d1_extended,d2_extended), dim=-1)
    # torch.save(d_extended, f'extended_disturbances_{nsteps}.pt')
    print(d_extended.shape)


    print(f"Number of rows filled with all zeros: {num_rows_with_all_zeros}")





#%%
from matplotlib import pyplot as plt
d2_art1 = (d2+torch.empty(d2.shape).uniform_(0.1, 1.5))*torch.empty(d2.shape).uniform_(-1.0, 1.0)

plt.plot(torch.abs(d2_art1).to('cpu').detach().numpy())
plt.plot(d2.to('cpu').detach().numpy(), 'k--')
plt.show()








#%% mixing without windows

repeated_indices_d2 = torch.randint(0, len(d2), (40000, 60))
repeated_indices_d1 = torch.randint(0, len(d1), (40000, 60))
  
# Create the 2D tensor by reordering
tensor_2d_d2 = d2[repeated_indices_d2]
tensor_2d_d1 = d1[repeated_indices_d1]

# Modulate the amplitude with random values in range [0.7, 1.1]
random_modulation = torch.empty(40000, 60).uniform_(0.9, 1.1)
tensor_2d_modulated_2d = tensor_2d_d2 * random_modulation
tensor_2d_modulated_1d = tensor_2d_d1 * random_modulation

tensor_2d_modulated = torch.cat((tensor_2d_modulated_1d.unsqueeze(-1),tensor_2d_modulated_2d.unsqueeze(-1)), dim=-1)
torch.save(tensor_2d_modulated, 'extended_disturbances_60.pt')


rows_with_all_zeros = torch.all(tensor_2d_modulated == 0, dim=1)

# Count the number of such rows
num_rows_with_all_zeros = torch.sum(rows_with_all_zeros).item()

print(f"Number of rows filled with all zeros: {num_rows_with_all_zeros}")



# %%


import numpy as np
import matplotlib.pyplot as plt
import scipy
import torch

nsteps = 40
# Parameters for synthetic data
num_days = 140  # Number of days to simulate
samples_per_day = int(1440/5)  # 1440 minutes / 5 minutes
num_samples = num_days * samples_per_day  # Total samples
baseline = 0.  # Average water consumption
noise_level = 0.0
num_peaks = 8 * num_days  # More peaks over multiple days
peak_amplitude_range = (1, 12)
peak_width_range = (2, 5)
daily_cycle_amplitude = 0.0

# Generate baseline consumption with noise
time_series = baseline + noise_level * np.random.randn(num_samples)

# Add daily cycle (24-hour periodicity)
time = np.arange(num_samples)
daily_cycle = daily_cycle_amplitude * np.sin(2 * np.pi * time / samples_per_day)
time_series += daily_cycle

# Add random peaks to simulate bursts of water usage
for _ in range(num_peaks):
    peak_center = np.random.randint(0, num_samples)
    peak_width = np.random.randint(*peak_width_range)
    peak_amplitude = np.random.uniform(*peak_amplitude_range)
    peak = peak_amplitude * np.exp(-0.5 * ((np.arange(num_samples) - peak_center) / peak_width)**2)
    # print(peak.shape)
    time_series += peak

# Introduce zero consumption periods
for day in range(num_days):
    # Define nighttime zero consumption (11 PM to 6 AM)
    start_night = day * samples_per_day + int(23 * samples_per_day / 24)
    end_night = day * samples_per_day + int(6 * samples_per_day / 24)
    time_series[start_night:end_night] = 0

    # Randomly set zero consumption during the day
    for _ in range(np.random.randint(4, 8)):  # 1-2 zero intervals during the day
        day_start = day * samples_per_day
        day_end = day_start + samples_per_day
        zero_start = np.random.randint(day_start, day_end - 12)  # Ensure room for 1 hour
        zero_duration = np.random.randint(6, 72)  # 30 minutes to 6 hours
        time_series[zero_start:zero_start + zero_duration] = 0

time_series = np.clip(time_series,0,16)


d = scipy.io.loadmat("newloads_matrix.mat")
d_tensor = torch.tensor(d['newloads_matrix'], dtype=torch.float32)

d1 = d_tensor[:,0]
d2 = d_tensor[:,1]


d1_2d = d1.unsqueeze(-1).unfold(0,nsteps,1).swapaxes(1,-1)

d1_beta=np.random.beta(0.6,1.4, time_series.shape[0])*7.0

d1_torch = torch.tensor(d1_beta).unsqueeze(-1).unfold(0,nsteps,1).swapaxes(1,-1)
d2_torch = torch.tensor(time_series).unsqueeze(-1).unfold(0,nsteps,1).swapaxes(1,-1)
tensor_2d_data = torch.cat((d1_torch,d2_torch), dim=-1)

torch.save(tensor_2d_data, 'extended_disturbances_60.pt')


print(time_series.shape)
# Plot the synthetic data
plt.figure(figsize=(12, 6))
plt.plot(time_series[:d2.shape[0]], linestyle='--', color='black')
plt.plot(d2)
plt.xlabel('Sample k (ts = 5 min)')
plt.ylabel('Water Consumption')
# plt.title('Synthetic Water Consumption Data with Zero Consumption Periods')
plt.grid(True)
plt.show()

#%%

plt.hist(d1.to('cpu').detach().numpy(),60, alpha=0.5, label='GT data')
plt.hist(np.random.beta(0.6,1.4, 2000)*7.0,60, alpha=0.5, label='Beta generated')
plt.xlabel('x')
plt.ylabel('D(x)')
plt.legend()
plt.show()

plt.hist(d2.to('cpu').detach().numpy(),60, alpha=0.5, label='GT data')
plt.hist(time_series[:d2.shape[0]],60, alpha=0.5, label='Beta generated')
plt.xlabel('x')
plt.ylabel('D(x)')
plt.legend()
plt.show()


# plt.plot(np.clip(np.random.beta(0.7,1.4, 2000),0.0,7.5)*7.0)
# plt.plot(d1.to('cpu').detach().numpy())
# plt.show()