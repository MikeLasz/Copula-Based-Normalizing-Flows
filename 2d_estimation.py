import openturns as ot
import matplotlib.pyplot as plt
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

import pandas as pd
import seaborn as sns
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--compute", type=bool, default=False)
args = parser.parse_args()

run_calculations = args.compute

def train_process(marginals, str_marg, sample_size=10000, batch_size=128, num_layers=3):
    """Given the marginals, this function runs a training procedure for fitting a
        MAF to a Gumbel Copula Distribution. If the resulting test loss after training
        is below 25, we return a DataFrame with train loss, test loss, and the
        corresponding iteration."""
    base_dist = ProdDist([2], marginals)
    # Generate data from a copula distribution:
    copula = ot.GumbelCopula(2.5)
    x1 = ot.Student(2, 0, 1)
    x2 = ot.Student(2, 0, 1)
    X = ot.ComposedDistribution([x1, x2], copula)

    train_losses = np.zeros(25)
    test_losses = np.zeros(25)
    counter_losses = 0

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
    for i in range(num_iter):
        batch_ind = np.random.choice(sample_size, batch_size, replace=False)
        x = data[batch_ind]
        x = torch.tensor(x, dtype=torch.float32)
        optimizer.zero_grad()
        loss = -flow.log_prob(inputs=x).mean()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            with torch.no_grad():
                flow.eval()
                loss_train = np.around(loss.detach().numpy(), decimals=2)
                train_losses[counter_losses] = loss_train
                loss_test = np.around(torch.mean(-flow.log_prob(sample_test)).detach().numpy(), decimals=2)
                test_losses[counter_losses] = loss_test
                counter_losses += 1
                #print(f'Iteration {i+1}/{num_iter}: Train loss = {loss_train}, Test loss = {loss_test}')

    if(loss_test <= 25):
        results_df = pd.DataFrame()
        results_df["train"] = train_losses
        results_df["test"] = test_losses
        results_df["# iter."] = np.linspace(100, 2500, num=25)
        results_df["marginals"] = np.repeat(str_marg, 25)
    else:
        print(str_marg + ": final test loss was above 100")
        results_df = pd.DataFrame()
        results_df["train"] = np.nan
        results_df["test"] = np.nan
        results_df["# iter."] = np.nan
        results_df["marginals"] = [str_marg]
    return results_df

# generate csv including the losses
if run_calculations:
    df_all = pd.DataFrame(columns=["train", "test", "marginals"])
    for marginals in ["normal", "heavierTails", "correctFamily", "exactMarginals"]:
        print(marginals)
        for rep in range(100):
            # normal, heavierTails, correctFamily, exactMarginals
            if marginals=="normal":
                base_marg = [Normal(0, 1), Normal(0, 1)]
            elif marginals == "heavierTails":
                base_marg = [Laplace(0, 4), StudentT(5, 0, 2)]
            elif marginals == "correctFamily":
                base_marg = [StudentT(5, 0, 1), StudentT(5, 0, 1)]
            elif marginals == "exactMarginals":
                base_marg = [StudentT(2, 0, 1), StudentT(2, 0, 1)]

            df_iter = train_process(base_marg, str_marg=marginals)
            df_all = pd.concat([df_all, df_iter])

    df_all.to_csv("data/2d_estimation.csv")

# read csv that contains all training and test losses.
df_all = pd.read_csv("data/2d_estimation.csv")

sns.set_context("paper", font_scale=2)

# Generate plot train performance
h = sns.lineplot(data=df_all, x="# iter.", y="train", hue="marginals")
h.set(ylabel="loss", ylim=(3, 10))
h.legend_.remove()
plt.tight_layout()
plt.savefig("plots/2d_estimation/train_performance.pdf")
plt.clf()

# Generate plot test performance
h = sns.lineplot(data=df_all, x="# iter.", y="test", hue="marginals")
h.set(ylabel="loss", ylim=(3, 10))
h.legend_.remove()
plt.tight_layout()
plt.savefig("plots/2d_estimation/test_performance.pdf")
