import numpy as np
import torch
from torch.distributions import LowRankMultivariateNormal


class BayesianPosterior(torch.nn.Module):
    def __init__(self, nr_xs, nr_out, params):
        super().__init__()
        eps = 1e-4
        self.nr_input = nr_xs
        self.nr_out = nr_out

        mu_z, sd_z, *_ = params
        [mu_z, sd_z] = [torch.Tensor(mu_z.T), torch.Tensor(sd_z.T)]
        [mu_u, sd_u] = [torch.zeros_like(mu_z), torch.sqrt(0.1 * torch.ones_like(sd_z))]
        [mu, sd] = [torch.cat([mu_u, mu_z], 1), torch.cat([sd_u, sd_z], 1)]

        sd_diag = torch.diag_embed(sd)
        [cov_factor, cov_ldiag] = [
            eps * torch.randn(sd_diag.shape) + sd_diag, 
            np.log(eps) * torch.ones(nr_out, self.nr_input * 2)
        ]

        self.mu = torch.nn.Parameter(mu)
        self.cov_factor = torch.nn.Parameter(cov_factor)
        self.register_buffer("cov_ldiag", cov_ldiag)

    @property
    def distribution(self):
        dist = LowRankMultivariateNormal(
            self.mu, self.cov_factor, torch.exp(self.cov_ldiag))
        return dist
    
    def get_beta(self, n_samples, is_test):
        if is_test:
            mu_T = self.mu.T
            beta_u = mu_T[:self.nr_input].unsqueeze(2)
            beta_z = mu_T[self.nr_input:].unsqueeze(2)
        else:
            sampling_results = self.distribution.rsample([n_samples])
            reshaped_samples = sampling_results.permute([2, 1, 0])
            beta_u = reshaped_samples[:self.nr_input, :, :]
            beta_z = reshaped_samples[self.nr_input:, :, :]
        return beta_u, beta_z

