"""
https://arxiv.org/pdf/1505.05770
"""
# Les données récupérées proviennent de https://github.com/e-hulten/planar-flows/blob/master/target_distribution.py

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from torch.distributions import MultivariateNormal
from typing import Callable, List


# Define TargetDistribution (identique à ton code)
class TargetDistribution:
    def __init__(self, name: str):
        self.func = self.get_target_distribution(name)

    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        return self.func(z)

    @staticmethod
    def get_target_distribution(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
        w1 = lambda z: torch.sin(2 * torch.pi * z[:, 0] / 4)
        w2 = lambda z: 3 * torch.exp(-0.5 * ((z[:, 0] - 1) / 0.6) ** 2)
        w3 = lambda z: 3 * torch.sigmoid((z[:, 0] - 1) / 0.3)

        if name == "U_1":
            def U_1(z):
                u = 0.5 * ((torch.norm(z, 2, dim=1) - 2) / 0.4) ** 2
                u -= torch.log(
                    torch.exp(-0.5 * ((z[:, 0] - 2) / 0.6) ** 2)
                    + torch.exp(-0.5 * ((z[:, 0] + 2) / 0.6) ** 2)
                )
                return u
            return U_1

        elif name == "U_2":
            def U_2(z):
                u = 0.5 * ((z[:, 1] - w1(z)) / 0.4) ** 2
                return u
            return U_2

        elif name == "U_3":
            def U_3(z):
                u = -torch.log(
                    torch.exp(-0.5 * ((z[:, 1] - w1(z)) / 0.35) ** 2)
                    + torch.exp(-0.5 * ((z[:, 1] - w1(z) + w2(z)) / 0.35) ** 2)
                    + 1e-6
                )
                return u
            return U_3

        elif name == "U_4":
            def U_4(z):
                u = -torch.log(
                    torch.exp(-0.5 * ((z[:, 1] - w1(z)) / 0.4) ** 2)
                    + torch.exp(-0.5 * ((z[:, 1] - w1(z) + w3(z)) / 0.35) ** 2)
                    + 1e-6
                )
                return u
            return U_4


# Define PlanarFlow and NormalizingFlowModel (identique à ton code)
class PlanarFlow(nn.Module):
    def __init__(self, latent_dim):
        super(PlanarFlow, self).__init__()
        self.weight = nn.Parameter(torch.randn(1, latent_dim) * 0.01)
        self.scale = nn.Parameter(torch.randn(1, latent_dim) * 0.01)
        self.bias = nn.Parameter(torch.randn(1))

    def forward(self, z):
        linear = torch.mm(z, self.weight.t()) + self.bias
        activation = torch.tanh(linear)
        activation_derivative = 1 - activation.pow(2)
        psi = activation_derivative * self.weight
        z_transformed = z + self.scale * activation
        det_jacobian = torch.abs(1 + torch.mm(self.scale, psi.t()))
        log_det_jacobian = torch.log(1e-4 + det_jacobian)
        return z_transformed, log_det_jacobian


class NormalizingFlowModel(nn.Module):
    def __init__(self, flows, target_dist):
        super(NormalizingFlowModel, self).__init__()
        self.base_distribution = MultivariateNormal(torch.zeros(2), torch.eye(2))
        self.target_distribution = target_dist
        self.flows = nn.ModuleList(flows)

    def forward(self, z):
        log_det_jacobian_sum = 0
        for flow in self.flows:
            z, log_det_jacobian = flow(z)
            log_det_jacobian_sum += log_det_jacobian
        return z, log_det_jacobian_sum

    def loss(self, z_0, z, log_det_jacobian_sum):
        log_prob_base = self.base_distribution.log_prob(z_0)
        target_density_log_prob = -self.target_distribution(z)
        return (log_prob_base - target_density_log_prob - log_det_jacobian_sum).mean()


# Plotting function for a grid
def plot_comparison(target_names: List[str], flows: List[int], steps=5000, batch_size=128, lr=1e-3):
    fig, axes = plt.subplots(len(target_names), len(flows), figsize=(15, 15))

    for i, target_name in enumerate(target_names):
        target_dist = TargetDistribution(target_name)

        for j, K in enumerate(flows):
            # Define and train the model
            model = NormalizingFlowModel([PlanarFlow(2) for _ in range(K)], target_dist)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            # Training loop
            for step in range(steps):
                z0 = model.base_distribution.sample((batch_size,))
                zk, log_jacobian = model(z0)
                loss = model.loss(z0, zk, log_jacobian)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Generate samples for plotting
            with torch.no_grad():
                z0 = model.base_distribution.sample((1000,))
                zk, _ = model(z0)
                zk = zk.detach().numpy()

            # Create density plot
            ax = axes[i, j]
            sns.kdeplot(x=zk[:, 0], y=zk[:, 1], ax=ax, fill=True, cmap="viridis")
            ax.set_title(f"Target: {target_name}, K={K}")
            ax.set_xlim(-4, 4)
            ax.set_ylim(-4, 4)

    plt.tight_layout()
    plt.show()


# Main script
if __name__ == "__main__":
    target_names = ["U_1", "U_2", "U_3", "U_4"]
    flows = [2, 8, 32]
    plot_comparison(target_names, flows)
