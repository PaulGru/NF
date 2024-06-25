import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Distribution, Normal
import numpy as np
import torch.optim as optim
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

cfg = {
  'MODEL_SAVE_PATH': './saved_models/',
  'USE_CUDA': False,
  'TRAIN_BATCH_SIZE': 256,
  'TRAIN_EPOCHS': 1000,
  'NUM_COUPLING_LAYERS': 10,
  'NUM_NET_LAYERS': 10,  # neural net layers for each coupling layer
  'NUM_HIDDEN_UNITS': 100
}

class CouplingLayer(nn.Module):
    def __init__(self, data_dim, hidden_dim, mask, num_layers=4):
        super().__init__()

        assert data_dim % 2 == 0

        self.mask = mask

        modules = [nn.Linear(data_dim, hidden_dim), nn.LeakyReLU(0.2)]

        for i in range(num_layers - 2):
            modules.append(nn.Linear(hidden_dim, hidden_dim))
            modules.append(nn.LeakyReLU(0.2))
        modules.append(nn.Linear(hidden_dim, data_dim))

        self.m = nn.Sequential(*modules)

    def forward(self, x, logdet, invert=False):
        if not invert:
            x1, x2 = self.mask * x, (1. - self.mask) * x
            y1, y2 = x1, x2 + (self.m(x1) * (1. - self.mask))
            return y1 + y2, logdet

        # Inverse additive coupling layer
        y1, y2 = self.mask * x, (1. - self.mask) * x
        x1, x2 = y1, y2 - (self.m(y1) * (1. - self.mask))
        return x1 + x2, logdet


class ScalingLayer(nn.Module):
    def __init__(self, data_dim):
        super().__init__()
        self.log_scale_vector = nn.Parameter(torch.randn(1, data_dim, requires_grad=True))

    def forward(self, x, logdet, invert=False):
        log_det_jacobian = torch.sum(self.log_scale_vector)

        if invert:
            return torch.exp(- self.log_scale_vector) * x, logdet - log_det_jacobian

        return torch.exp(self.log_scale_vector) * x, logdet + log_det_jacobian


class GaussianDistribution(Distribution):
    def __init__(self):
        super().__init__()
        self.dist = Normal(0, 1)

    def log_prob(self, x):
        return self.dist.log_prob(x).sum(dim=1)

    def sample(self, size):
        return self.dist.sample(size)


class NICE(nn.Module):
    def __init__(self, data_dim, num_coupling_layers=3):
        super().__init__()

        self.data_dim = data_dim

        # alternating mask orientations for consecutive coupling layers
        masks = [self._get_mask(data_dim, orientation=(i % 2 == 0))
                                                for i in range(num_coupling_layers)]

        self.coupling_layers = nn.ModuleList([CouplingLayer(data_dim=data_dim,
                                    hidden_dim=cfg['NUM_HIDDEN_UNITS'],
                                    mask=masks[i], num_layers=cfg['NUM_NET_LAYERS'])
                                for i in range(num_coupling_layers)])

        self.scaling_layer = ScalingLayer(data_dim=data_dim)

        self.prior = GaussianDistribution()

    def forward(self, x, invert=False):
        if not invert:
            z, log_det_jacobian = self.f(x)
            log_likelihood = self.prior.log_prob(z) + log_det_jacobian
            return z, log_likelihood

        return self.f_inverse(x)

    def f(self, x):
        z = x
        log_det_jacobian = 0
        for i, coupling_layer in enumerate(self.coupling_layers):
            z, log_det_jacobian = coupling_layer(z, log_det_jacobian)
        z, log_det_jacobian = self.scaling_layer(z, log_det_jacobian)
        return z, log_det_jacobian

    def f_inverse(self, z):
        x = z
        x, _ = self.scaling_layer(x, 0, invert=True)
        for i, coupling_layer in reversed(list(enumerate(self.coupling_layers))):
            x, _ = coupling_layer(x, 0, invert=True)
        return x

    def sample(self, num_samples):
        z = self.prior.sample([num_samples, self.data_dim]).view(num_samples, self.data_dim)
        return self.f_inverse(z)

    def _get_mask(self, dim, orientation=True):
        mask = np.zeros(dim)
        mask[::2] = 1.
        if orientation:
            mask = 1. - mask     # flip mask orientation
        mask = torch.tensor(mask)

        return mask.float()


# Data
X, y = make_moons(n_samples=1000, noise=0.1)
scaler = StandardScaler()
X = scaler.fit_transform(X)
X = torch.tensor(X, dtype=torch.float32)

dataset = TensorDataset(X)
dataloader = DataLoader(dataset, batch_size=cfg['TRAIN_BATCH_SIZE'], shuffle=True)

model = NICE(data_dim=2, num_coupling_layers=cfg['NUM_COUPLING_LAYERS'])

# Train the model
model.train()

opt = optim.Adam(model.parameters())

for i in range(cfg['TRAIN_EPOCHS']):
    mean_likelihood = 0.0
    num_minibatches = 0

    for batch_id, (x,) in enumerate(dataloader):
        z, likelihood = model(x)
        loss = -torch.mean(likelihood)   # NLL

        loss.backward()
        opt.step()
        model.zero_grad()

        mean_likelihood -= loss.item()
        num_minibatches += 1

    mean_likelihood /= num_minibatches
    print('Epoch {} completed. Log Likelihood: {}'.format(i, mean_likelihood))

# Visualize the transformation from Gaussian distribution to two moons
num_samples = 1000
with torch.no_grad():
    z = model.prior.sample([num_samples, model.data_dim]).view(num_samples, model.data_dim)
    x_transformed = model.f_inverse(z).numpy()

plt.figure(figsize=(8, 6))
plt.scatter(x_transformed[:, 0], x_transformed[:, 1], c='blue', s=5, label='Transformed Points')
plt.title('Transformed Points from Gaussian Distribution to Two Moons')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.grid(True)
plt.show()
