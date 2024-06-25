import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import numpy as np

class AffineCouplingLayer(nn.Module):
    def __init__(self, input_dim):
        super(AffineCouplingLayer, self).__init__()
        self.input_dim = input_dim
        self.scale_net = nn.Sequential(
            nn.Linear(input_dim // 2, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim // 2),
        )
        self.translate_net = nn.Sequential(
            nn.Linear(input_dim // 2, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim // 2)
        )

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        s = self.scale_net(x1)
        t = self.translate_net(x1)
        y2 = x2 * torch.exp(s) + t
        return torch.cat([x1, y2], dim=1), s, s.sum(dim=1)
    
    def inverse(self, y):
        y1, y2 = y.chunk(2, dim=1)
        s = self.scale_net(y1)
        t = self.translate_net(y1)
        x2 = (y2 - t) * torch.exp(-s)
        return torch.cat([y1, x2], dim=1)


model = AffineCouplingLayer(input_dim=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

base_dist = torch.distributions.Normal(torch.zeros(2), torch.ones(2))

gaussian_samples = base_dist.sample((1000,))

num_epochs = 500
for epoch in range(num_epochs):
    optimizer.zero_grad()
    transformed_samples, s, log_det_jacobian = model(gaussian_samples)
    
    # Calculate log probability under the Gaussian base distribution
    log_prob = -0.5 * (transformed_samples**2).sum(dim=1) - 0.5 * transformed_samples.size(1) * np.log(2 * np.pi)
    # Compute the loss as the negative log-likelihood
    loss = -torch.mean(log_prob + log_det_jacobian)
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    if epoch % 5 == 0:
        # Your plotting and logging code here
        print(f'Epoch {epoch}, Loss: {loss.item()}')
        print(f"Max scale: {s.max().item()}, Min scale: {s.min().item()}")
        print(f"Transformed samples max: {transformed_samples.max().item()}, min: {transformed_samples.min().item()}")
        print(f"Log Det Jacobian: {log_det_jacobian.mean().item()}")
        plt.figure(figsize=(6, 4))
        plt.scatter(transformed_samples.detach().numpy()[:, 0], transformed_samples.detach().numpy()[:, 1], alpha=0.5, color='blue')
        plt.title(f'Transformed Gaussian Distribution at Epoch {epoch}')
        plt.show()
