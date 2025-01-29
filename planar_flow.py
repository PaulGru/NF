import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns


# Planar Flow Transformation
class PlanarFlow(nn.Module):
    def __init__(self, latent_dim):
        super(PlanarFlow, self).__init__()
        self.weight = nn.Parameter(torch.randn(1, latent_dim))
        self.scale = nn.Parameter(torch.randn(1, latent_dim))
        self.bias = nn.Parameter(torch.randn(1))

    def forward(self, z):
        linear = torch.mm(z, self.weight.t()) + self.bias
        activation = torch.tanh(linear)
        activation_derivative = 1 - activation.pow(2)
        psi = activation_derivative * self.weight
        z_transformed = z + self.scale * activation
        det_jacobian = torch.abs(1 + torch.mm(self.scale, psi.t()))
        log_det_jacobian = torch.log(det_jacobian)
        return z_transformed, log_det_jacobian


# Normalizing Flow Model
class NormalizingFlowModel(nn.Module):
    def __init__(self, base_dist, flows):
        super(NormalizingFlowModel, self).__init__()
        self.base_distribution = base_dist
        self.flows = nn.ModuleList(flows)

    def forward(self, z):
        log_prob_base = self.base_distribution.log_prob(z)
        log_det_jacobian_sum = 0
        for flow in self.flows:
            z, log_det_jacobian = flow(z)
            log_det_jacobian_sum += log_det_jacobian
        log_prob_final = log_prob_base - log_det_jacobian_sum
        return z, log_prob_final

    def loss(self, z_samples, target_samples):
        """
        Calcul de la perte basée sur la divergence entre les échantillons transformés et ceux de la cible.
        """
        z_transformed, _ = self.forward(z_samples)
        # Utilisation de la MMD (Maximum Mean Discrepancy) pour comparer les distributions
        return mmd_loss(z_transformed, target_samples)


# MMD Loss (Maximum Mean Discrepancy)
def mmd_loss(samples1, samples2, sigma=1.0):
    """
    Calcul de la perte MMD entre deux ensembles d'échantillons.
    - samples1, samples2 : Tenseurs contenant les échantillons.
    - sigma : Paramètre pour le noyau Gaussien.
    """
    def gaussian_kernel(x, y, sigma):
        dist = torch.cdist(x, y) ** 2
        return torch.exp(-dist / (2 * sigma ** 2))

    k_xx = gaussian_kernel(samples1, samples1, sigma).mean()
    k_yy = gaussian_kernel(samples2, samples2, sigma).mean()
    k_xy = gaussian_kernel(samples1, samples2, sigma).mean()

    return k_xx + k_yy - 2 * k_xy


# Distribution en forme d'anneau (cible)
def sample_ring(num_samples, radius=5.0, noise=0.1):
    angles = 2 * torch.pi * torch.rand(num_samples)
    radii = radius + noise * torch.randn(num_samples)
    x = radii * torch.cos(angles)
    y = radii * torch.sin(angles)
    return torch.stack([x, y], dim=1)


# Visualisation
def visualize_flow_comparison(model, target_samples, num_samples=1000):
    z_base = model.base_distribution.sample((num_samples,))
    z_transformed, _ = model.forward(z_base)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.kdeplot(
        x=z_transformed[:, 0].detach().numpy(),
        y=z_transformed[:, 1].detach().numpy(),
        fill=True, ax=axes[0]
    )
    axes[0].set_title("Échantillons Transformés")

    sns.kdeplot(
        x=target_samples[:, 0].detach().numpy(),
        y=target_samples[:, 1].detach().numpy(),
        fill=True, ax=axes[1]
    )
    axes[1].set_title("Distribution Cible (Anneau)")

    plt.tight_layout()
    plt.show()


# Main Script
if __name__ == "__main__":
    latent_dim = 2

    # Distribution de base
    base_dist = torch.distributions.MultivariateNormal(
        loc=torch.zeros(latent_dim),
        covariance_matrix=torch.eye(latent_dim)
    )

    # Modèle avec plusieurs flows
    num_flows = 5
    flows = [PlanarFlow(latent_dim) for _ in range(num_flows)]
    model = NormalizingFlowModel(base_dist, flows)

    # Optimiseur
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Données cibles
    num_samples = 1000
    target_samples = sample_ring(num_samples)

    # Entraînement
    def train(model, optimizer, steps=1000, batch_size=100):
        model.train()
        for step in range(steps):
            optimizer.zero_grad()

            # Échantillons de la distribution de base
            z_samples = model.base_distribution.sample((batch_size,))

            # Calcul de la perte
            loss = model.loss(z_samples, target_samples)
            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                print(f"Step {step}, Loss: {loss.item()}")

    train(model, optimizer, steps=1000)

    # Visualisation après entraînement
    visualize_flow_comparison(model, target_samples, num_samples=1000)
