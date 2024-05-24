import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Distribution, Uniform

cfg = {
  'MODEL_SAVE_PATH': './saved_models/',

  'USE_CUDA': True,

  'TRAIN_BATCH_SIZE': 256,

  'TRAIN_EPOCHS': 75,

  'NUM_COUPLING_LAYERS': 4,

  'NUM_NET_LAYERS': 6,  # neural net layers for each coupling layer

  'NUM_HIDDEN_UNITS': 1000
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


class LogisticDistribution(Distribution):
    def __init__(self):
        super().__init__()

    def log_prob(self, x):
        return -(F.softplus(x) + F.softplus(-x))

    def sample(self, size):
        if cfg['USE_CUDA']:
            z = Uniform(torch.cuda.FloatTensor([0.]), torch.cuda.FloatTensor([1.])).sample(size)
        else:
            z = Uniform(torch.FloatTensor([0.]), torch.FloatTensor([1.])).sample(size)

        return torch.log(z) - torch.log(1. - z)



import numpy as np
import torch
import torch.nn as nn


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

        self.prior = LogisticDistribution()

    def forward(self, x, invert=False):
        if not invert:
            z, log_det_jacobian = self.f(x)
            log_likelihood = torch.sum(self.prior.log_prob(z), dim=1) + log_det_jacobian
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
        z = self.prior.sample([num_samples, self.data_dim]).view(self.samples, self.data_dim)
        return self.f_inverse(z)

    def _get_mask(self, dim, orientation=True):
        mask = np.zeros(dim)
        mask[::2] = 1.
        if orientation:
            mask = 1. - mask     # flip mask orientation
        mask = torch.tensor(mask)
        if cfg['USE_CUDA']:
            mask = mask.cuda()
        return mask.float()
  

import os
import torch
import torch.optim as optim
from torchvision import transforms, datasets

# Data
transform = transforms.ToTensor()
dataset = datasets.MNIST(root='./data/mnist', train=True, transform=transform, download=True)
dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=cfg['TRAIN_BATCH_SIZE'],
                                         shuffle=True, pin_memory=True)

model = NICE(data_dim=784, num_coupling_layers=cfg['NUM_COUPLING_LAYERS'])
if cfg['USE_CUDA']:
    device = torch.device('cuda')
    model = model.to(device)

# Train the model
model.train()

opt = optim.Adam(model.parameters())

epoch = cfg['TRAIN_EPOCHS']

for i in range(epoch):
    mean_likelihood = 0.0
    num_minibatches = 0

    for batch_id, (x, _) in enumerate(dataloader):
        x = x.view(-1, 784) + torch.rand(784) / 256.
        if cfg['USE_CUDA']:
            x = x.cuda()

        x = torch.clamp(x, 0, 1)

        z, likelihood = model(x)
        loss = -torch.mean(likelihood)   # NLL

        loss.backward()
        opt.step()
        model.zero_grad()

        mean_likelihood -= loss
        num_minibatches += 1

    mean_likelihood /= num_minibatches
    print('Epoch {} completed. Log Likelihood: {}'.format(i, mean_likelihood))

    if epoch % 5 == 0:
        save_path = os.path.join(cfg['MODEL_SAVE_PATH'], '{}.pt'.format(epoch))
        torch.save(model.state_dict(), save_path)
