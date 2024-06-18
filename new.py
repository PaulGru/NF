import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import numpy as np

# Data generation function for Two Moons dataset
def generate_twomoons_data(n_samples=1000, noise=0.05):
    data, _ = make_moons(n_samples=n_samples, noise=noise)
    return torch.tensor(data, dtype=torch.float32)

# Define an affine coupling layer
class AffineCoupling(nn.Module):
    def __init__(self, dim):
        super(AffineCoupling, self).__init__()
        self.scale = nn.Parameter(torch.randn(1, dim))
        self.shift = nn.Parameter(torch.randn(1, dim))

    def forward(self, x, reverse=False):
        if not reverse:
            y = x * torch.exp(self.scale) + self.shift
            return y, self.scale
        else:
            y = (x - self.shift) * torch.exp(-self.scale)
            return y

# Real NVP Model
class RealNVP(nn.Module):
    def __init__(self, dims, num_couplings):
        super(RealNVP, self).__init__()
        self.couplings = nn.ModuleList([AffineCoupling(dims) for _ in range(num_couplings)])

    def forward(self, x, reverse=False):
        log_det_jacobian = 0
        if not reverse:
            for coupling in self.couplings:
                x, scale = coupling(x)
                log_det_jacobian += scale.sum(dim=1)
            return x, log_det_jacobian
        else:
            for coupling in reversed(self.couplings):
                x = coupling(x, reverse=True)
            return x

# Log-likelihood loss function
def loss_function(transformed_data, log_det_jacobian):
    log_pz = -0.5 * torch.sum(transformed_data**2, axis=1)  # Assuming standard normal distribution
    log_px = log_pz + log_det_jacobian
    return -torch.mean(log_px)

# Training setup
dims = 2
num_couplings = 6
model = RealNVP(dims, num_couplings)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
data = generate_twomoons_data(1000)

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    transformed_data, log_det_jacobian = model(data)
    loss = loss_function(transformed_data, log_det_jacobian)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")
        with torch.no_grad():
            sample_z = torch.randn(100, dims)
            transformed_sample, _ = model(sample_z)

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title("Original Two Moons Distribution")
        plt.scatter(data[:, 0], data[:, 1], alpha=0.6)
        plt.subplot(1, 2, 2)
        plt.title("Transformed Distribution")
        plt.scatter(transformed_data[:, 0], transformed_data[:, 1], alpha=0.6)
        plt.show()
