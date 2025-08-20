# %% Generate synthetic disturbance data
import numpy as np
import matplotlib.pyplot as plt
import scipy
import torch

nsteps = 40 # Longest prediction horizon
# Parameters for d2 signals
num_days = 140  # Number of days to simulate
samples_per_day = int(1440/5)  # 1440 minutes / 5 minutes (Ts=5minutes)
num_samples = num_days * samples_per_day  # Total samples
baseline = 0.  
noise_level = 0.0 # Optional
num_peaks = 8 * num_days  # More peaks over multiple days
peak_amplitude_range = (1, 12) # d2 signal
peak_width_range = (2, 5) # d2 signal
daily_cycle_amplitude = 0.0 # Optional

# Generate baseline consumption with noise
time_series = baseline + noise_level * np.random.randn(num_samples)

# Add daily cycle (24-hour periodicity) -- Optional
time = np.arange(num_samples)
daily_cycle = daily_cycle_amplitude * np.sin(2 * np.pi * time / samples_per_day)
time_series += daily_cycle

# Add random peaks to simulate bursts of energy usage
for _ in range(num_peaks):
    peak_center = np.random.randint(0, num_samples)
    peak_width = np.random.randint(*peak_width_range)
    peak_amplitude = np.random.uniform(*peak_amplitude_range)
    peak = peak_amplitude * np.exp(-0.5 * ((np.arange(num_samples) - peak_center) / peak_width)**2)
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

# Generate d2 signal samples
time_series = np.clip(time_series,0,16)

# Generate d1 signal samples
d1_beta=np.random.beta(0.6,1.4, time_series.shape[0])*7.0

# Numpy to torch, unfold the temporal dimension
d1_torch = torch.tensor(d1_beta).unsqueeze(-1).unfold(0,nsteps,1).swapaxes(1,-1)
d2_torch = torch.tensor(time_series).unsqueeze(-1).unfold(0,nsteps,1).swapaxes(1,-1)
tensor_2d_data = torch.cat((d1_torch,d2_torch), dim=-1)

# Overwriting the existing dataset may yield different results to those in paper
# torch.save(tensor_2d_data, 'training_dist_data/extended_disturbances_40.pt')

#%% Load test data
d = scipy.io.loadmat("loads_matrix.mat") 
d_tensor = torch.tensor(d['newloads_matrix'], dtype=torch.float32)
d1 = d_tensor[:,0]
d2 = d_tensor[:,1]

#%% Compare synthetic and test data in time domain

# Compare the d1 time series data
plt.figure(figsize=(12, 6))
plt.plot(d1_beta[:d1.shape[0]], linestyle='--', color='black', label='Synthetic data')
plt.plot(d1, label='Test data')
plt.xlabel('Sample k (ts = 5 min)')
plt.ylabel('d1 Signal')
plt.grid(True)
plt.legend()
plt.show()

# Compare the d2 time series data
plt.figure(figsize=(12, 6))
plt.plot(time_series[:d2.shape[0]], linestyle='--', color='black', label='Synthetic data')
plt.plot(d2, label='Test data')
plt.xlabel('Sample k (ts = 5 min)')
plt.ylabel('d2 Signal')
plt.grid(True)
plt.legend()
plt.show()

#%% Compare probability distributions of training (synthetic) and test data

plt.hist(d1.to('cpu').detach().numpy(),60, alpha=0.5, label='Test data d1')
plt.hist(np.random.beta(0.6,1.4, 2000)*7.0,60, alpha=0.5, label='Synthetic data d1 - Beta distribution')
plt.xlabel('Amplitude')
plt.ylabel('Number of samples')
plt.legend()
plt.show()

plt.hist(d2.to('cpu').detach().numpy(),60, alpha=0.5, label='Test data d2')
plt.hist(time_series[:d2.shape[0]],60, alpha=0.5, label='Synthetic data d2 - Algorithmically generated')
plt.xlabel('Amplitude')
plt.ylabel('Number of samples')
plt.legend()
plt.show()
# %%
