import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import numpy as np
from sklearn import cluster, datasets, mixture
from sklearn.preprocessing import StandardScaler

class AffineCouplingLayer(nn.Module):
    def __init__(self, d, k, hidden):
        super(AffineCouplingLayer, self).__init__()
        self.d, self.k = d, k
        self.scale_net = nn.Sequential(
            nn.Linear(k, hidden),
            nn.ReLU(),
            nn.Linear(hidden, d-k),
        )
        self.translate_net = nn.Sequential(
            nn.Linear(k, hidden),
            nn.ReLU(),
            nn.Linear(hidden, d-k)
        )

    def forward(self, x, flip=False):
        x1, x2 = x[:, :self.k], x[:, self.k:] 
        s = self.scale_net(x1)
        t = self.translate_net(x1)
        y2 = x2 * torch.exp(s) + t

        if flip:
            z2, z1 = z1, z2
       
        transformed_samples = torch.cat([x1, y2], dim=1)
        log_pz = base_dist.log_prob(transformed_samples)

        log_jac = s.sum(dim=1)
        
        return transformed_samples, log_pz, log_jac
    
    def inverse(self, y):
        y1, y2 = y.chunk(2, dim=1)
        s = self.scale_net(y1)
        t = self.translate_net(y1)
        x2 = (y2 - t) * torch.exp(-s)
        return torch.cat([y1, x2], dim=1)


model = AffineCouplingLayer(input_dim=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

base_dist = torch.distributions.Normal(torch.zeros(2), torch.ones(2))

num_epochs = 500

def train(model, epochs, batch_size, optim):
    losses = []
    for _ in range(epochs):

        # get batch 
        X, _ = datasets.make_moons(n_samples=batch_size, noise=.05)
        X = torch.from_numpy(StandardScaler().fit_transform(X)).float()

        optim.zero_grad()
        z, log_pz, log_jacob = model(X)
        loss = (-log_pz - log_jacob).mean()
        losses.append(loss)

        loss.backward()
        optim.step()
    return losses

def view(model, losses):
    plt.plot([loss.item() for loss in losses])
    plt.title("Model Loss vs Epoch")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()

    X_hat = model.inverse(Z).detach().numpy()
    plt.scatter(X_hat[:, 0], X_hat[:, 1])
    plt.title("Inverse of Normal Samples Z: X = F^-1(Z)")
    plt.show()

    n_samples = 3000
    X, _ = datasets.make_moons(n_samples=n_samples, noise=.05)
    X = torch.from_numpy(StandardScaler().fit_transform(X)).float()
    z, _, _ = model(X)
    z = z.detach().numpy()
    plt.scatter(z[:, 0], z[:, 1])
    plt.title("Transformation of Data Samples X: Z = F(X)")
    plt.show()

n_samples = 512
losses = train(model, 10000, n_samples, optimizer)
view(model, losses)