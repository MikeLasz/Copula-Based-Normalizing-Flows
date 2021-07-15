import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import seaborn as sns

def estimate_localLip(flow, inputs, num_samps=100):
    """Computes the local Lipschitz constant of the flow in samples given by input."""
    batch_size = inputs.shape[0]
    dim_size = inputs.shape[1]

    # 1. sample from normal distribution
    normal_dist = MultivariateNormal(torch.zeros(dim_size), torch.eye(dim_size))
    normal_samps = normal_dist.sample([num_samps * batch_size]).reshape([batch_size, num_samps, dim_size])

    # 2. obtain unit sphere sample of dimensionality [batch_size, num_samples, dim_size]
    unit_sphere_samps = torch.div(normal_samps, torch.norm(normal_samps, dim=2)[:, :, None])

    # 3. Compute 1/eps \Vert F(x) - F(x-eps*u)\Vert where u is a unit sphere sample for
    # each sample n in 1,...,num_samples
    eps = 0.001
    inputs_stacked = inputs[:, None, :].repeat(1, num_samps, 1).reshape(num_samps*batch_size, dim_size)
    sphere_stacked = unit_sphere_samps.reshape(num_samps*batch_size, dim_size)
    flow_dif = flow._transform(inputs_stacked)[0] - flow._transform(inputs_stacked + eps*sphere_stacked)[0]
    flow_dif = flow_dif.reshape(batch_size, num_samps, dim_size)
    norm_dif = 1/eps * torch.norm(flow_dif, dim=2) # take the norm with respect to dim 2, i.e. over each sample

    max_dif = torch.max(norm_dif, dim=1)[0]
    # take the maximum of each batch sample, i.e. max over dim=1
    # which is the dimension over the samples num_samps
    return(torch.log(max_dif))

def estimate_invlocalLip(flow, inputs, num_samps=100):
    """Computes the local Lipschitz constant of the inverse flow in samples given by input."""
    batch_size = inputs.shape[0]
    dim_size = inputs.shape[1]

    # 1. sample from normal distribution
    normal_dist = MultivariateNormal(torch.zeros(dim_size), torch.eye(dim_size))
    normal_samps = normal_dist.sample([num_samps * batch_size]).reshape([batch_size, num_samps, dim_size])

    # 2. obtain unit sphere sample of dimensionality [batch_size, num_samples, dim_size]
    unit_sphere_samps = torch.div(normal_samps, torch.norm(normal_samps, dim=2)[:, :, None])

    # 3. Compute 1/eps \Vert F(x) - F(x-eps*u)\Vert where u is a unit sphere sample for
    # each sample n in 1,...,num_samples
    eps = 0.001
    inputs_stacked = inputs[:, None, :].repeat(1, num_samps, 1).reshape(num_samps*batch_size, dim_size)
    sphere_stacked = unit_sphere_samps.reshape(num_samps*batch_size, dim_size)
    inv_flow_dif = flow._transform.inverse(inputs_stacked)[0] - flow._transform.inverse(inputs_stacked + eps*sphere_stacked)[0]
    inv_flow_dif = inv_flow_dif.reshape(batch_size, num_samps, dim_size)
    # take the norm with respect to dim 2, i.e. over each sample
    norm_dif = 1/eps * torch.norm(inv_flow_dif, dim=2)
    # take the maximum of each batch sample, i.e. max over dim=1
    # which is the dimension over the samples num_samps
    max_dif = torch.max(norm_dif, dim=1)[0]
    return(torch.log(max_dif))


def plot_spectral_surface(flow, title):
    nx, ny = (100, 100)
    x = np.linspace(-10, 10, nx)
    y = np.linspace(-10, 10, ny)

    xv, yv = np.meshgrid(x, y)
    inputs_numpy = np.vstack(np.dstack([xv, yv]))

    inputs = torch.tensor(inputs_numpy, dtype=torch.float32)
    # Estimate the spectrum, i.e. the local Lipschitz constant for each input in the grid.
    Z = torch.reshape(estimate_localLip(flow, inputs), (nx, ny)).detach().numpy()

    sns.set_context("paper", font_scale=2)
    fig, ax = plt.subplots()
    im = ax.pcolormesh(x, y, Z, rasterized=True)
    fig.colorbar(im, ax=ax)
    im.set_clim(-2, 3)
    PATH = "plots/lipschitz_surface/" + title + ".pdf"
    plt.savefig(PATH)
    plt.clf()

    # do the same or the inverse to get a feeling for the bi-Lipschitz constant
    Z = torch.reshape(estimate_invlocalLip(flow, inputs), (nx, ny)).detach().numpy()

    fig, ax = plt.subplots()
    im = ax.pcolormesh(x, y, Z, rasterized=True)
    fig.colorbar(im, ax=ax)
    im.set_clim(-2, 3)
    PATH_inv = "plots/lipschitz_surface/inv_" + title + ".pdf"
    plt.savefig(PATH_inv)