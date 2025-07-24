# %% Generate synthetic disturbance data

import scipy
import torch

nsteps = 20
num_data = 20000

d = scipy.io.loadmat("../disturbances.mat")
d_tensor = torch.tensor(d['disturbances'], dtype=torch.float32)

d1 = d_tensor[:,0]
d2 = d_tensor[:,1]


d1_2d = d1.unsqueeze(-1).unfold(0,nsteps,1).swapaxes(1,-1)
d2_2d = d2.unsqueeze(-1).unfold(0,nsteps,1).swapaxes(1,-1)


rows_with_all_zeros = torch.all(d2_2d == 0, dim=1)

# Count the number of such rows
num_rows_with_all_zeros = torch.sum(rows_with_all_zeros).item()
print(f"Number of rows filled with all zeros: {num_rows_with_all_zeros}")


#%% mixing without windows

repeated_indices = torch.randint(0, len(d2), (20000, 20))
  
# Create the 2D tensor by reordering
tensor_2d = d2[repeated_indices]

# Modulate the amplitude with random values in range [0.7, 1.1]
random_modulation = torch.empty(20000, 20).uniform_(0.5, 0.99)
tensor_2d_modulated = tensor_2d * random_modulation



rows_with_all_zeros = torch.all(tensor_2d_modulated == 0, dim=1)

# Count the number of such rows
num_rows_with_all_zeros = torch.sum(rows_with_all_zeros).item()

print(f"Number of rows filled with all zeros: {num_rows_with_all_zeros}")


#%% mixing with windows
num_extensions = 40
extended_tensors = [d2_2d]  # Start with the original tensor
for _ in range(num_extensions):
    random_indices = torch.randperm(d2_2d.size(1))  # Permutes 21 indices
    randomized_tensor = d2_2d[:, random_indices, :]
    extended_tensors.append(randomized_tensor)
d2_extended = torch.cat(extended_tensors, dim=0)

num_extensions = 40
extended_tensors = [d1_2d]  # Start with the original tensor
for _ in range(num_extensions):
    random_indices = torch.randperm(d1_2d.size(1))  # Permutes 21 indices
    randomized_tensor = d1_2d[:, random_indices, :]
    extended_tensors.append(randomized_tensor)
d1_extended = torch.cat(extended_tensors, dim=0)

rows_with_all_zeros = torch.all(d2_extended == 0, dim=1)

# Count the number of such rows
num_rows_with_all_zeros = torch.sum(rows_with_all_zeros).item()

d_extended = torch.cat((d1_extended,d2_extended), dim=-1)
torch.save(d_extended, 'extended_disturbances.pt')



print(f"Number of rows filled with all zeros: {num_rows_with_all_zeros}")