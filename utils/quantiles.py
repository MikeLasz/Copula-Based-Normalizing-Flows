import openturns as ot
import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns



def generate_quantiles(samples_true, samples_model, title, num_components=2):
    """Generates a Q-Plot given samples from 2 distributions."""
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    axs = axs.ravel()
    sns.set_context("paper", font_scale=3)
    for component in range(num_components):
        #discard NANs:
        data_true = samples_true[:, component]
        data_true = data_true[~np.isnan(data_true)]
        print('True sampels: amount of discarded NAN-values: {}'.format(np.isnan(data_true).sum()))

        data_model = samples_model[:, component]
        data_model = data_model[~np.isnan(data_model)]
        print('Model sampels: amount of discarded NAN-values: {}'.format(np.isnan(data_model).sum()))

        # compute CDFs
        data_true_sorted = np.sort(data_true)
        data_model_sorted = np.sort(data_model)

        u_true = 1. * np.arange(len(data_true)) / (len(data_true) - 1) # cumulative probability
        u_model = 1. * np.arange(len(data_model)) / (len(data_model) - 1)

        axs[component].plot(u_true, data_true_sorted, label="truth", linewidth=4.0)
        axs[component].plot(u_model, data_model_sorted, label="model", linewidth=4.0, linestyle="--")
        axs[component].set_xlim([0, 1])
        axs[component].set_ylim([-10, 10])
        axs[component].set_xlabel("u", fontsize=25.0)
        axs[component].tick_params(axis="x", labelsize=20)
        axs[component].set_ylabel("Q(u)", fontsize=25.0)
        axs[component].tick_params(axis="y", labelsize=20)

    plt.tight_layout()
    plt.savefig("plots/quantiles/Qplot" + title + ".pdf")
    plt.clf()

    # quantiles of ||x|| instead of marginal quantiles
    data_true = np.linalg.norm(samples_true, axis=1)
    data_true = data_true[~np.isnan(data_true)]
    print('True sampels: amount of discarded NAN-values: {}'.format(np.isnan(data_true).sum()))

    data_model = np.linalg.norm(samples_model, axis=1)
    data_model = data_model[~np.isnan(data_model)]
    print('Model samples: amount of discarded NAN-values: {}'.format(np.isnan(data_model).sum()))

    # compute CDFs
    data_true_sorted = np.sort(data_true)
    data_model_sorted = np.sort(data_model)

    u_true = 1. * np.arange(len(data_true)) / (len(data_true) - 1)  # cumulative probability
    u_model = 1. * np.arange(len(data_model)) / (len(data_model) - 1)

    plt.figure(figsize=(7.5, 6))
    plt.plot(u_true, data_true_sorted, label="truth", linewidth=4.0)
    plt.plot(u_model, data_model_sorted, label="model", linewidth=4.0, linestyle="--")
    plt.xlim(0, 1)
    plt.ylim(0, 20)
    plt.xlabel("u")
    plt.ylabel("Q(u)")
    plt.tight_layout()
    plt.savefig("plots/quantiles/Qplot" + title + "_norm.pdf")

def generate_quantiles_2models(flow1, flow2, title, num_components=2):
    """Generates a Q-Plot, which compares the samples from flow1 and flow2 with the
        quantiles of a Gumbel Copula Distribution."""
    copula = ot.GumbelCopula(2.5)
    x1 = ot.Student(2, 0, 1)
    x2 = ot.Student(2, 0, 1)
    X = ot.ComposedDistribution([x1, x2], copula)

    samples_true = np.array(X.getSample(1000))
    samples_model1 = np.array(flow1.sample(1000).detach().numpy())
    samples_model2 = np.array(flow2.sample(1000).detach().numpy())

    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    axs = axs.ravel()
    sns.set_context("paper", font_scale=3)
    for component in range(num_components):
        #discard NANs:
        data_true = samples_true[:, component]
        data_true = data_true[~np.isnan(data_true)]
        print('True sampels: amount of discarded NAN-values: {}'.format(np.isnan(data_true).sum()))

        data_model1 = samples_model1[:, component]
        data_model1 = data_model1[~np.isnan(data_model1)]
        print('Model 1: amount of discarded NAN-values: {}'.format(np.isnan(data_model1).sum()))

        data_model2 = samples_model2[:, component]
        data_model2 = data_model2[~np.isnan(data_model2)]
        print('Model 2: amount of discarded NAN-values: {}'.format(np.isnan(data_model2).sum()))

        # compute CDFs
        data_true_sorted = np.sort(data_true)
        data_model1_sorted = np.sort(data_model1)
        data_model2_sorted = np.sort(data_model2)

        u_true = 1. * np.arange(len(data_true)) / (len(data_true) - 1) # cumulative probability
        u_model1 = 1. * np.arange(len(data_model1)) / (len(data_model1) - 1)
        u_model2 = 1. * np.arange(len(data_model2)) / (len(data_model2) - 1)

        axs[component].plot(u_true, data_true_sorted, label="truth", linewidth=4.0)
        axs[component].plot(u_model1, data_model1_sorted, label="model1", linewidth=4.0, linestyle="--")
        axs[component].plot(u_model2, data_model2_sorted, label="model2", linewidth=4.0, linestyle="dotted")
        axs[component].set_xlim([0, 1])
        axs[component].set_ylim([-10, 10])
        axs[component].set_xlabel("u", fontsize=25.0)
        axs[component].tick_params(axis="y", labelsize=20)
        axs[component].set_ylabel("Q(u)", fontsize=25.0)
        axs[component].tick_params(axis="y", labelsize=20)
    plt.tight_layout()
    plt.savefig("plots/quantiles/2in1Qplot" + title + ".pdf")
    plt.clf()

    # quantiles of ||x|| instead of marginal quantiles
    data_true = np.linalg.norm(samples_true, axis=1)
    data_true = data_true[~np.isnan(data_true)]
    print('True sampels: amount of discarded NAN-values: {}'.format(np.isnan(data_true).sum()))

    data_model1 = np.linalg.norm(samples_model1, axis=1)
    data_model1 = data_model1[~np.isnan(data_model1)]
    print('Model 1: amount of discarded NAN-values: {}'.format(np.isnan(data_model1).sum()))

    data_model2 = np.linalg.norm(samples_model2, axis=1)
    data_model2 = data_model2[~np.isnan(data_model2)]
    print('Model 2: amount of discarded NAN-values: {}'.format(np.isnan(data_model2).sum()))

    # compute CDFs
    data_true_sorted = np.sort(data_true)
    data_model1_sorted = np.sort(data_model1)
    data_model2_sorted = np.sort(data_model2)

    u_true = 1. * np.arange(len(data_true)) / (len(data_true) - 1)  # cumulative probability
    u_model1 = 1. * np.arange(len(data_model1)) / (len(data_model1) - 1)
    u_model2 = 1. * np.arange(len(data_model2)) / (len(data_model2) - 1)

    plt.figure(figsize=(7.5, 6))
    plt.plot(u_true, data_true_sorted, label="truth", linewidth=4.0)
    plt.plot(u_model1, data_model1_sorted,  label="model1", linewidth=4.0, linestyle="--")
    plt.plot(u_model2, data_model2_sorted, label="model2", linewidth=4.0, linestyle="dotted")
    plt.ylim(0, 20)
    plt.xlabel("u")
    plt.ylabel("Q(u)")
    plt.tight_layout()
    plt.savefig("plots/quantiles/2in1Qplot" + title + "_norm.pdf")