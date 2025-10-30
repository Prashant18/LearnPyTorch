'''
Day 2: Working with Tensors
'''

import torch
import numpy as np
from torch.cuda import is_available

# ======================================================================
# SECTION 1: CREATING TENSORS
# ======================================================================
# Tensors are multi-dimensional arrays that can be used to store and manipulate data.
# They are similar to NumPy arrays, but can be used on GPUs and CPUs.
# Tensors can be created from NumPy arrays, lists, and other data types.
# Tensors can be created from random numbers, ones, zeros, and other data types.
# Tensors can be created from existing tensors.
data = [[1,2], [3,4]]
x_data = torch.tensor(data)
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
x_ones = torch.ones_like(x_data)
x_rand = torch.rand_like(x_data, dtype=torch.float)

print(f"Tensor from data: {x_data}")
print(f"Tensor from numpy array: {x_np}")
print(f"Tensor from ones: {x_ones}")
print(f"Tensor from random: {x_rand}")

shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: {rand_tensor}")
print(f"Ones Tensor: {ones_tensor}")
print(f"Zeros Tensor: {zeros_tensor}")

# ======================================================================
# SECTION 2: TENSOR PROPERTIES
# ======================================================================
# Tensors have a shape, a dtype, and a device.
# The shape is the number of elements in each dimension.
# The dtype is the data type of the elements.
# The device is the device on which the tensor is located.
tensor = torch.rand(3,4)
print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")


# ======================================================================
# SECTION 3: OPERATIONS ON TENSORS
# ======================================================================
# Tensors can be added, subtracted, multiplied, and divided.
# Tensors can be transposed, reshaped, and broadcasted.
# Tensors can be indexed and sliced.
# Tensors can be concatenated and split.
# Tensors can be transformed using various functions.
tensor = torch.rand(3,4)
## Moving the tensor to the GPU fo rnow
## Find number of accelerators available 
accelerators = torch.accelerator.device_count()
print(f"Number of accelerators available: {accelerators}")
if accelerators > 0:
    tensor = tensor.to(torch.accelerator.current_accelerator())
    print(f"Tensor is now on the {tensor.device}")
else:
    print("No accelerator available")

tensor = torch.ones(4,4)
print(f"First row: {tensor[0]}")
print(f"Last row: {tensor[-1]}")
print(f"First column: {tensor[:,0]}")
print(f"Last column: {tensor[:,-1]}")

tensor[:,1] = 0
print(tensor)

t1 = torch.cat([tensor, tensor], dim=1)
print(f"Concatenated tensor: {t1}")

y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)
y3 = torch.rand_like(tensor.T)
torch.matmul(tensor, tensor.T, out=y3)
print(f"y1: {y1}")
print(f"y2: {y2}")
print(f"y3: {y3}")

z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)
print(f"z1: {z1}")
print(f"z2: {z2}")
print(f"z3: {z3}")
