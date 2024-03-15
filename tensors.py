#!/usr/bin/env python3

import torch
import numpy as np


data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
print(x_data)

np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print(x_np)

x_ones = torch.ones_like(x_data)
print(f"Ones Tensor: \n{x_ones}\n")
x_rand = torch.rand_like(x_data, dtype=torch.float)
print(f"Random Tensor: \n{x_rand}\n")

shape = (2, 3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor:\n{rand_tensor}\n")
print(f"Ones Tensor:\n{ones_tensor}\n")
print(f"Zeros Tensor:\n{zeros_tensor}")

tensor = torch.rand(3, 4)
print(f"Shape of Tensor: {tensor.shape}")
print(f"Datatype of Tensor: {torch.dtype}")
print(f"Device Tensor is stored on: {torch.device}")

if torch.cuda.is_available():
    tensor = tensor.to("cuda")

tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First Column: {tensor[:, 0]}")
print(f"Last Column: {tensor[..., -1]}")
tensor[:, 1] = 0
print(tensor)

t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)
