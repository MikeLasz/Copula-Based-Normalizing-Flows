import openturns as ot
import numpy as np
import torch
from torch import optim

from nflows.flows.base import Flow
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.permutations import ReversePermutation

from torch.distributions.studentT import StudentT
from torch.distributions.laplace import Laplace
from torch.distributions.normal import Normal

from utils.distributions import ProdDist
from utils.quantiles import generate_quantiles, generate_quantiles_2models

def train(marginals, title, batch_size=128, num_layers=3, sample_size=10000, num_iter=2500):
    """Train a MAF to fit a Gumbel Copula Distribution. Returns the resulting flow and
        saves a Q-plot that compares the quantiles of the trained flow with the quantiles
        of the Gumbel Copula Distribution."""
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
    for i in range(num_iter):
        batch_ind = np.random.choice(sample_size, batch_size, replace=False)
        x = data[batch_ind]
        x = torch.tensor(x, dtype=torch.float32)
        optimizer.zero_grad()

        loss = -flow.log_prob(inputs=x).mean()
        loss.backward()
        optimizer.step()

        if (i + 1) % 200 == 0:
            with torch.no_grad():
                loss_test = -flow.log_prob(sample_test).mean()
                print(loss_test)
    with torch.no_grad():
        sample_test = np.array(sample_test)
        flow.eval()
        generate_quantiles(sample_test, np.array(flow.sample(1000)), title)
    return flow

# train a flow for each base distribution
flow_normal = train([Normal(0, 1), Normal(0, 1)], title="normal")
flow_exact = train([StudentT(2, 0, 1), StudentT(2, 0, 1)], title="exactMarginals")
flow_tails = train([Laplace(0, 4), StudentT(5, 0, 2)], title="heavierTails")
flow_family = train([StudentT(5, 0, 1), StudentT(5, 0, 1)], title="correctFamily")


# plot vanilla NF and the NF with exact marginals in one Q-plot
generate_quantiles_2models(flow_normal, flow_exact, "normalandexact", num_components=2)

# plot the cases correctTails and correctFamily in one Q-plot
generate_quantiles_2models(flow_tails, flow_family, "heaviertailsandfamily", num_components=2)