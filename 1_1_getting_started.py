# Getting Started


# ====Tensors====

from __future__ import print_function
import torch

# Construct a 5x3 matrix, uninitialized:
x = torch.Tensor(5, 3)
print(x)

# Construct a randomly initialized matrix:
x = torch.rand(5, 3)
print(x)

# Construct a matrix filled zeros and of dtype long:


print(x.size())


# ====Operations====

# 1
y = torch.rand(5, 3)
print(x + y)

# 2
print(torch.add(x, y))

# result
result = torch.Tensor(5, 3)
torch.add(x, y, out=result)
print(result)