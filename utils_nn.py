import torch
import numpy as np
import torch.nn as nn
import scipy.linalg as la

from dataloader import setup_scatter
from utils import args
from utils_classifier import SoftmaxRegression


class FeatureExtractor(nn.Module):
    def __init__(self, params=args):
        super(FeatureExtractor, self).__init__()
        self.args = params
        self.ds_name = self.args.ds 
        self.ds_suffix = self.args.ds_suffix
        self.nr_fea = self.args.nr_fea
        if 'mnist' in self.ds_name.lower(): # for MNIST, FMNIST dataset
            self.feature_extractor_part1 = nn.Sequential(
                nn.Conv2d(1, 20, kernel_size=5),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2),
                nn.Conv2d(20, 50, kernel_size=5),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2)
            )
            [self.input_dim, self.H, self.L] = [50 * 4 * 4, 256, 128]

        elif 'sival' in self.ds_name.lower(): # for SIVAL_MIPL dataset
            [self.input_dim, self.H1, self.H2, self.L] = [self.nr_fea, 512, 1024, 256]

        elif 'crc' in self.ds_name.lower() and 'sift' in self.ds_suffix.lower(): # for CRC-MIPL-sift dataset
            [self.input_dim, self.H1, self.H2, self.L] = [self.nr_fea, 1024, 512, 64]
            
        else: # for Birdsong_MIPL, CRC-MIPL (row, sbn, kmeanSegs) dataset
            [self.input_dim, self.H, self.L] = [self.nr_fea, 512, 256]
        
        if 'sival' in self.ds_name.lower() or (
            'crc' in self.ds_name.lower() and 'sift' in self.ds_suffix.lower()):
            self.feature_extractor_part2 = nn.Sequential(
                nn.Linear(self.input_dim, self.H1),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(self.H1, self.H2),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(self.H2, self.L),
                nn.ReLU(),
                nn.Dropout(),
            )
        else:
            self.feature_extractor_part2 = nn.Sequential(
                nn.Linear(self.input_dim, self.H),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(self.H, self.L),
                nn.ReLU(),
                nn.Dropout(),
            )

    def forward(self, xs):
        x = torch.cat(xs, dim=0)
        x, i, i_ptr = setup_scatter(xs)
        x = x.squeeze(0).float()
        if 'mnist' in self.ds_name.lower():
            x = x.reshape(x.shape[0], 1, 28, 28)
            h = self.feature_extractor_part1(x)
            h = h.view(-1, 50 * 4 * 4)
            h = self.feature_extractor_part2(h)
        else:   # for Birdsong_MIPL, SIVAL_MIPL, CRC-MIPL datasets
            x = x.float()
            h = self.feature_extractor_part2(x)
        hs = []
        for start_idx, end_idx in zip(i_ptr[:-1], i_ptr[1:]):
            hs.append(h[start_idx:end_idx])
        return hs

def zb_regress(x, f):
    """
    regresses out x from f
    """
    f_inv = la.pinv(f) 
    b = f_inv.dot(x) 
    x_out = x - f.dot(b)
    out = [x_out, b, f_inv]
    return out

def init_sr(x, s, b, fiv):
    model = SoftmaxRegression(
        eta=0.1,
        epochs=1000,
        minibatches=1,
        random_seed=args.seed,
        print_progress=0,
        n_classes=args.nr_class
    )

    model.fit(x, s)

    alpha = model.b_[None]
    gamma = model.w_ 

    # Compute bag prediction u and reparametrize
    u = x.dot(gamma) 
    [um, us] = [u.mean(0)[None], u.std(0)[None]]
    alpha = alpha + um
    mu_gamma = us * gamma / np.sqrt((gamma**2).mean(0)[None])
    sd_gamma = np.sqrt(0.1 * (mu_gamma**2).mean()) * np.ones_like(mu_gamma)
    alpha = fiv.dot(np.ones((fiv.shape[1], 1))).dot(alpha) - b.dot(mu_gamma)

    # init prior
    var_z = (mu_gamma**2 + sd_gamma**2).mean(0).reshape(1, -1)

    return [torch.Tensor(el) for el in (mu_gamma, sd_gamma, var_z, alpha)]

def generate_init_params(x, fe, s):
    eps = 1e-8
    z_b = np.concatenate([x.mean(0, keepdims=True) for x in x], axis=0) # generating features of each bag
    [fe_array, s_array] = [fe.numpy(), s.numpy()]

    zb, b, fe_inv = zb_regress(z_b, fe_array)
    nor_zb = (zb - zb.mean(0, keepdims=True)) / (zb.std(0, keepdims=True) * np.sqrt(zb.shape[-1]) + eps)

    mu_z, sd_z, var_z, alpha = init_sr(nor_zb, s_array, b, fe_inv)
    return mu_z, sd_z, var_z, alpha
