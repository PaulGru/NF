import torch
import torch.nn as nn


class PlanarFlow(nn.Module):
    def __init__(self, dim):
        super(PlanarFlow, self).__init__()
        self.w = nn.Parameter(torch.randn(1, dim))
        self.u = nn.Parameter(torch.randn(1, dim))
        self.b = nn.Parameter(torch.randn(1))

    def forward(self, z):
        linear = torch.mm(z, self.w.t()) + self.b
        activation = torch.tanh(linear)
        psi = (1 - activation.pow(2)) * self.w  # derivative of tanh
        z_new = z + self.u * activation
        log_det_jacobian = torch.log(torch.abs(1 + torch.mm(self.u, psi.t())))
        return z_new, log_det_jacobian.sum(1, keepdim=True)


class NormalizingFlowModel(nn.Module):
    def __init__(self, base_distribution, flows, prior_distribution):
        super(NormalizingFlowModel, self).__init__()
        self.base_distribution = base_distribution
        self.prior_distribution = prior_distribution
        self.flows = nn.ModuleList(flows)

    def forward(self, z):
        log_qz0 = self.base_distribution.log_prob(z)
        sum_log_det_jacobians = 0
        for flow in self.flows:
            z, log_det_jacobian = flow(z)
            sum_log_det_jacobians += log_det_jacobian
        log_qzK = log_qz0 - sum_log_det_jacobians
        return z, log_qzK, sum_log_det_jacobians

    def loss(self, z, x):
        z_transformed, log_qzK, _ = self.forward(z)

        # Assurez-vous que la dim de sortie est [batch_size, 1] pour log_pzK
        log_pzK = self.prior_distribution.log_prob(z_transformed).unsqueeze(1)

        # Assurez-vous que log_px_zK est également de dimension [batch_size, 1]
        log_px_zK = log_likelihood_fn(x, z_transformed).unsqueeze(1)

        # Assurez-vous que log_qzK est correctement dimensionné
        log_qzK = log_qzK.unsqueeze(1)

        # Calcul de la perte comme la négative de l'ELBO moyenné
        return -(log_px_zK + log_pzK - log_qzK).mean()


def log_likelihood_fn(x, z):
    # Placeholder for the actual likelihood function of data given z
    return -((x - z) ** 2).sum(dim=1)  # Example assuming Gaussian likelihood


# Example usage
dim = 5
base_dist = torch.distributions.Normal(torch.zeros(dim), torch.ones(dim))
prior_dist = torch.distributions.Normal(torch.zeros(dim), torch.ones(dim))
flows = [PlanarFlow(dim) for _ in range(3)]  # Stack multiple flows for deeper transfo
model = NormalizingFlowModel(base_dist, flows, prior_dist)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop not included, refer to previous example


# Training loop
def train(model, optimizer, steps=10000, batch_size=100):
    model.train()
    for step in range(steps):
        optimizer.zero_grad()

        z = model.base_distribution.sample((batch_size,))
        x = torch.randn(batch_size, dim)  # Assuming x is generated for demo
        loss = model.loss(z, x)

        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss.item()}")


train(model, optimizer)