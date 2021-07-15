from nflows.flows.base import Flow

import openturns as ot
import numpy as np
import torch
from torch import optim

from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.permutations import ReversePermutation

from torch.distributions.studentT import StudentT
from torch.distributions.laplace import Laplace
from torch.distributions.normal import Normal

from utils.distributions import ProdDist
from utils.lipschitz_estimation import plot_spectral_surface

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--base_dist", type=str, default=exactMarginals) # choose base_dist from: normal, heavierTails, correctFamily, exactMarginals
args = parser.parse_args()

base_dist = args.base_dist

def train_process(marginals, sample_size=10000, batch_size=128, num_layers=3):
    """Fit a MAF with given marginals (of the base distribution)to a Gumbel
        Copula Distribution. Return the trained flow."""
    base_dist = ProdDist([2], marginals)
    # Generate data from a copula distribution:
    copula = ot.GumbelCopula(2.5)
    x1 = ot.Student(2, 0, 1)
    x2 = ot.Student(2, 0, 1)
    X = ot.ComposedDistribution([x1, x2], copula)


    sample = X.getSample(sample_size)
    sample_test = X.getSample(10000)


    transforms = []
    for _ in range(num_layers):
        transforms.append(ReversePermutation(features=2))
        transforms.append(MaskedAffineAutoregressiveTransform(features=2,
                                                              hidden_features=4))
    transform = CompositeTransform(transforms)

    flow = Flow(transform, base_dist)
    optimizer = optim.Adam(flow.parameters())

    # train the flow:
    data = np.array(sample)
    num_iter = 2500
    counter_column = 0
    for i in range(num_iter):
        flow.train()
        batch_ind = np.random.choice(sample_size, batch_size, replace=False)
        x = data[batch_ind]
        x = torch.tensor(x, dtype=torch.float32)
        optimizer.zero_grad()
        loss = -flow.log_prob(inputs=x).mean()
        loss.backward()
        optimizer.step()

        if (i + 1) % 500 == 0:
            flow.eval()
            loss_train = np.around(loss.detach().numpy(), decimals=2)
            loss_test = np.around(torch.mean(-flow.log_prob(sample_test)).detach().numpy(), decimals=2)
            counter_column += 1
            print(f'Iteration {i+1}/{num_iter}: Train loss = {loss_train}, Test loss = {loss_test}')
    return(flow)



if base_dist == "normal":
    marginals = [Normal(0, 1), Normal(0, 1)]
elif base_dist == "heavierTails":
    marginals = [Laplace(0, 4), StudentT(5, 0, 2)]
elif base_dist == "correctFamily":
    marginals = [StudentT(5, 0, 1), StudentT(5, 0, 1)]
elif base_dist == "exactMarginals":
    marginals = [StudentT(2, 0, 1), StudentT(2, 0, 1)]

# train a flow
flow = train_process(marginals)

# plot the resulting spectral surface
plot_spectral_surface(flow, base_dist)
