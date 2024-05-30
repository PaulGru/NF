import os
import torch
import numpy as np

from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, Lambda
import matplotlib.pyplot as plt

mean = [0, 0]
cov = [[1, 0.8], [0.8, 1]]

X = np.random.multivariate_normal(mean, cov, 5000)

temp = X.T
plt.plot(temp[0], temp[1], 'x')
plt.axis('equal')
plt.show()


class Dataset2D(Dataset):
    def __init__(self, data):
        mean = [0, 0]
        cov = [[1, 0.8], [0.8, 1]]
        self.size = len(data)
        self.origX = data
        self.X = torch.tensor(self.origX).float()

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.X[idx, :]



def backbone(input_width, network_width=10):
    return nn.Sequential(
            nn.Linear(input_width, network_width),
            nn.ReLU(),
            nn.Linear(network_width, network_width),
            nn.ReLU(),
            nn.Linear(network_width, network_width),
            nn.ReLU(),
            nn.Linear(network_width, input_width),
            nn.Tanh(),
    )


class NormalizingFlow2D(nn.Module):
    def __init__(self, num_coupling, width):
        super(NormalizingFlow2D, self).__init__()
        self.num_coupling = num_coupling
        self.s = nn.ModuleList([backbone(1, width) for x in range(num_coupling)])
        self.t = nn.ModuleList([backbone(1, width) for x in range(num_coupling)])

        # Learnable scaling parameters for outputs of S
        self.s_scale = torch.nn.Parameter(torch.randn(num_coupling))
        self.s_scale.requires_grad = True

    def forward(self, x):
        original_x = x.clone()  # Create a copy of x to avoid modifying the original tensor
        s_vals = []
        if self.training:
            for i in range(self.num_coupling):
                if i % 2 == 0:
                    s = self.s_scale[i] * self.s[i](original_x[:, :1])
                    temp = torch.exp(s) * original_x[:, 1:] + self.t[i](original_x[:, :1])
                    original_x = torch.cat((original_x[:, :1], temp), dim=1)
                else:
                    s = self.s_scale[i] * self.s[i](original_x[:, 1:])
                    temp = torch.exp(s) * original_x[:, :1] + self.t[i](original_x[:, 1:])
                    original_x = torch.cat((temp, original_x[:, 1:]), dim=1)
                s_vals.append(s)
            return original_x, torch.cat(s_vals)
        else:
            for i in reversed(range(self.num_coupling)):
                if i % 2 == 0:
                    s = self.s_scale[i] * self.s[i](original_x[:, :1])
                    temp = (original_x[:, 1:] - self.t[i](original_x[:, :1])) * torch.exp(-s)
                    original_x = torch.cat((original_x[:, :1], temp), dim=1)
                else:
                    s = self.s_scale[i] * self.s[i](original_x[:, 1:])
                    temp = (original_x[:, :1] - self.t[i](original_x[:, 1:])) * torch.exp(-s)
                    original_x = torch.cat((temp, original_x[:, 1:]), dim=1)
            return original_x




def train_loop(dataloader, model, loss_fn, optimizer, report_iters=10):
    size = len(dataloader)
    for batch, X in enumerate(dataloader):

        # Compute prediction and loss
        y, s = model(X)
        loss = loss_fn(y, s, batch_size)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % report_iters == 0:
            loss, current = loss.item(), batch
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader)
    num_batches = len(dataloader)
    test_loss = 0

    with torch.no_grad():
        for X in dataloader:
            y, s = model(X)
            test_loss += loss_fn(y, s, batch_size)

    test_loss /= num_batches
    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")

def loss_fn(y, s, batch_size):
    # -log(zero-mean gaussian) + log determinant
    # -log p_x = log(pz(f(x))) + log(det(\partial f/\partial x))
    # -log p_x = 0.5 * y**2 + s1 + s2
    logpx = -torch.sum(0.5 * y**2)
    det = torch.sum(s)

    ret = -(logpx + det)
    return torch.div(ret, batch_size)



learning_rate = 0.001
batch_size = 1000
epochs = 10

model = NormalizingFlow2D(16, 10)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

mean = [0, 0]
cov = [[1, 0.8], [0.8, 1]]
training_data = Dataset2D(np.random.multivariate_normal(mean, cov, 20000))
test_data = Dataset2D(np.random.multivariate_normal(mean, cov, 5000))
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)

print("Done!")

fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(16,8))

# subplot1
ax1.set_title("Original")
ax1.set_xlabel("x1")
ax1.set_ylabel("x2")
ax1.set_xlim(-4, 4)
ax1.set_ylim(-4, 4)

data = training_data.origX.T
ax1.plot(data[0], data[1], 'x', color='blue')

# subplot2
model.eval()
ax2.set_title("NF Sampled")
ax2.set_xlabel("x1")
ax2.set_ylabel("x2")
ax2.set_xlim(-4, 4)
ax2.set_ylim(-4, 4)
mean = [0, 0]
cov = [[1, 0], [0, 1]]
with torch.no_grad():
    X = torch.Tensor(np.random.multivariate_normal(mean, cov, 20000))
    Y = model(X)
samples = Y.numpy().T
ax2.plot(samples[0], samples[1], 'x', color='red')

