import math
import numpy as np

import torch
import torch.sparse
import torch.nn as nn
from torch.nn import Parameter
from torch.nn import Module
import scipy
import scipy.sparse as sp
import torch.nn.functional as F
from torch.autograd import Function
from utils import projection_norm_inf, projection_norm_inf_and_1, SparseDropout
from functions import ImplicitFunction, IDMFunction
import ipdb
import os


class ImplicitGraph(Module):
    """
    A Implicit Graph Neural Network Layer (IGNN)
    """

    def __init__(self, in_features, out_features, num_node, kappa=0.99, b_direct=False):
        super(ImplicitGraph, self).__init__()
        self.p = in_features
        self.m = out_features
        self.n = num_node
        print(f'p = {self.p}, m = {self.m}, n = {self.n}')
        self.k = kappa  # if set kappa=0, projection will be disabled at forward feeding.
        self.b_direct = b_direct

        self.W = Parameter(torch.FloatTensor(self.m, self.m))
        self.Omega_1 = Parameter(torch.FloatTensor(self.m, self.p))
        self.Omega_2 = Parameter(torch.FloatTensor(self.m, self.p))
        self.bias = Parameter(torch.FloatTensor(self.m, 1))
        self.init()

    def init(self):
        stdv = 1. / math.sqrt(self.W.size(1))
        self.W.data.uniform_(-stdv, stdv)
        self.Omega_1.data.uniform_(-stdv, stdv)
        self.Omega_2.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, X_0, A, U, phi, A_rho=1.0, fw_mitr=300, bw_mitr=300, A_orig=None):
        """Allow one to use a different A matrix for convolution operation in equilibrium equ"""
        if self.k is not None:  # when self.k = 0, A_rho is not required
            self.W = projection_norm_inf(self.W, kappa=self.k / A_rho)
        # print(f'U: {U}')
        support_1 = torch.spmm(torch.transpose(U, 0, 1), self.Omega_1.T).T
        support_1 = torch.spmm(torch.transpose(A, 0, 1), support_1.T).T
        support_2 = torch.spmm(torch.transpose(U, 0, 1), self.Omega_2.T).T
        b_Omega = support_1  # + support_2
        # b_Omega = U
        return ImplicitFunction.apply(self.W, X_0, A if A_orig is None else A_orig, b_Omega, phi, fw_mitr, bw_mitr)


class IDM_SGC(nn.Module):
    def __init__(self, adj, sp_adj, m, num_eigenvec, gamma, adj_preload_file=None):
        super(IDM_SGC, self).__init__()
        self.F = nn.Parameter(torch.FloatTensor(m, m), requires_grad=True)
        self.S = adj
        self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float), requires_grad=False)
        sy = (abs(sp_adj - sp_adj.T) > 1e-10).nnz == 0
        if sy:
            self.Lambda_S, self.Q_S = scipy.linalg.eigh(sp_adj.toarray())
        else:
            self.Lambda_S, self.Q_S = scipy.linalg.eig(sp_adj.toarray())
        self.Lambda_S = torch.from_numpy(self.Lambda_S).type(torch.FloatTensor).cuda()
        self.Q_S = torch.from_numpy(self.Q_S).type(torch.FloatTensor).cuda()
        self.Lambda_S = self.Lambda_S.view(-1, 1)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.F)

    def forward(self, X):
        return IDMFunction.apply(X, self.F, self.S, self.Q_S, self.Lambda_S, self.gamma)


epsilon_F = 10 ** (-12)


def g(F):
    FF = F.t() @ F
    FF_norm = torch.norm(FF, p='fro')
    return (1 / (FF_norm + epsilon_F)) * FF


def get_G(Lambda_F, Lambda_S, gamma):
    G = 1.0 - gamma * Lambda_F @ Lambda_S.t()
    G = 1 / G
    return G
