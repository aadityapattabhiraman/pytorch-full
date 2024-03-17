#!/usr/bin/env python3

import torch


x = torch.empty(3, 4)
print(type(x))
print(x)

zeros = torch.zeros(2, 3)
print(zeros)

ones = torch.ones(2, 3)
print(ones)

torch.manual_seed(47)
random = torch.rand(2, 3)
print(random)
