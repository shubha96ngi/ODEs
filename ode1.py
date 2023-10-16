# this is taken from 
# https://github.com/analysiscenter/pydens/blob/torch/examples/_torch_examples.ipynb
# I am just checking it 

import sys
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import pydens
from pydens.model_torch import Solver, D, V, TorchModel
from pydens.batchflow import NumpySampler as NS

def ode(f, x):
    return D(f, x) - 2 * np.pi * torch.cos(2 * np.pi * x)

solver = Solver(ode, ndims=1, initial_condition=torch.tensor(.5))
solver.fit(niters=1500, batch_size=400)

xs = torch.tensor(np.linspace(0, 1, 20)).float()

approxs = solver.model(xs)  # problem with this line .. mat multiplication 

plt.plot(xs.detach().numpy(), approxs.detach().numpy(), label='Approx',
         linewidth=5, alpha=0.8)
plt.plot(xs.detach().numpy(), np.sin(2 * np.pi * xs.detach().numpy()) + .5,
         label='True', linewidth=2)
plt.grid()
plt.legend()





