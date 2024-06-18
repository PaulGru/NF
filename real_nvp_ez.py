import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class AffineCoupling(nn.Module):
    def __init__(self, dim):
        super(AffineCoupling, self).__init__()
        self.scale = nn.Parameter(torch.randn(1, dim))  # Ensuring it’s at least 2D
        self.shift = nn.Parameter(torch.randn(1, dim))

    def forward(self, x, reverse=False):
        exp_scale = torch.exp(self.scale)  # Calculate exponential of scale for transformation
        if not reverse:
            y = x * exp_scale + self.shift
            return y, exp_scale
        else:
            y = (x - self.shift) * torch.exp(-self.scale)
            return y

        

# Real NVP Model construction
class RealNVP(nn.Module):
    def __init__(self, dims, num_couplings):
        super(RealNVP, self).__init__()
        self.couplings = nn.ModuleList([AffineCoupling(dims) for _ in range(num_couplings)])

    def forward(self, x, reverse=False):
        log_det_jacobian = 0
        if not reverse:
            for coupling in self.couplings:
                x, scale = coupling(x)
                # Ensure scale has the correct dimensions and perform sum operation
                if scale.dim() > 1:  # Checking if scale is at least 2D
                    log_det_jacobian += scale.sum(dim=1)
                else:
                    log_det_jacobian += scale.sum()  # Fallback if scale is unexpectedly 1D
            return x, log_det_jacobian
        else:
            for coupling in reversed(self.couplings):
                x = coupling(x, reverse=True)
            return x



# Log-likelihood loss function
def loss_function(x, transformed_x, log_det_jacobian):
    log_pz = -0.5 * torch.sum(transformed_x**2, axis=1)  # Assuming standard normal distribution
    log_px = log_pz + log_det_jacobian
    return -torch.mean(log_px)  # Maximize log_px, minimize -log_px

# Parameters
dims = 2  # Dimension of the data space
num_couplings = 6  # Number of affine coupling layers

# Model initialization
model = RealNVP(dims, num_couplings)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training data: samples from a normal distribution
data = torch.randn(1000, dims)

# Training
num_epochs = 1000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    transformed_data, log_det_jacobian = model(data)
    loss = loss_function(data, transformed_data, log_det_jacobian)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")
        with torch.no_grad():
            generated_data, _ = model(data)  # Transform data

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title("Original Distribution")
        plt.hist2d(data[:, 0].numpy(), data[:, 1].numpy(), bins=30, density=True)
        plt.colorbar()

        plt.subplot(1, 2, 2)
        plt.title("Transformed Distribution")
        plt.hist2d(generated_data[:, 0].numpy(), generated_data[:, 1].numpy(), bins=30, density=True)
        plt.colorbar()
        plt.show()

# To generate new data
sampled_z = torch.randn(100, dims)
generated_data, _ = model(sampled_z, reverse=False)  # Using the forward transformation to generate new data
