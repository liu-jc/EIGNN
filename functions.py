import torch
import numpy as np
import scipy.sparse as sp
from torch.autograd import Function
from utils import sparse_mx_to_torch_sparse_tensor
import ipdb

class ImplicitFunction(Function):
    #ImplicitFunction.apply(input, A, U, self.X_0, self.W, self.Omega_1, self.Omega_2)
    @staticmethod
    def forward(ctx, W, X_0, A, B, phi, fd_mitr=300, bw_mitr=300):
        X_0 = B if X_0 is None else X_0
        X, err, status, D = ImplicitFunction.inn_pred(W, X_0, A, B, phi, mitr=fd_mitr, compute_dphi=True)
        ctx.save_for_backward(W, X, A, B, D, X_0, torch.tensor(bw_mitr))
        if status not in "converged":
            print("Iterations not converging!", err, status)
        return X

    @staticmethod
    def backward(ctx, *grad_outputs):

        #import pydevd
        #pydevd.settrace(suspend=False, trace_only_current_thread=True)

        W, X, A, B, D, X_0, bw_mitr = ctx.saved_tensors
        bw_mitr = bw_mitr.cpu().numpy()
        grad_x = grad_outputs[0]

        dphi = lambda X: torch.mul(X, D)
        grad_z, err, status, _ = ImplicitFunction.inn_pred(W.T, X_0, A, grad_x, dphi, mitr=bw_mitr, trasposed_A=True)
        #grad_z.clamp_(-1,1)

        grad_W = grad_z @ torch.spmm(A, X.T)
        grad_B = grad_z

        # Might return gradient for A if needed
        return grad_W, None, torch.zeros_like(A), grad_B, None, None, None

    @staticmethod
    def inn_pred(W, X, A, B, phi, mitr=300, tol=3e-6, trasposed_A=False, compute_dphi=False):
        # TODO: randomized speed up
        At = A if trasposed_A else torch.transpose(A, 0, 1)
        #X = B if X is None else X

        err = 0
        status = 'max itrs reached'
        for i in range(mitr):
            # WXA
            X_ = W @ X
            support = torch.spmm(At, X_.T).T
            X_new = phi(support + B)
            err = torch.norm(X_new - X, np.inf)
            if err < tol:
                status = 'converged'
                break
            X = X_new

        dphi = None
        if compute_dphi:
            with torch.enable_grad():
                support = torch.spmm(At, (W @ X).T).T
                Z = support + B
                Z.requires_grad_(True)
                X_new = phi(Z)
                dphi = torch.autograd.grad(torch.sum(X_new), Z, only_inputs=True)[0]

        return X_new, err, status, dphi


epsilon_F = 10**(-12)
def g(F):
    FF = F.t() @ F
    FF_norm = torch.norm(FF, p='fro')
    return (1/(FF_norm+epsilon_F)) * FF


def get_G(Lambda_F, Lambda_S, gamma):
    G = 1.0 - gamma * Lambda_F @ Lambda_S.t()
    G = 1 / G
    return G


class IDMFunction(Function):
    @staticmethod
    def forward(ctx, X, F, S, Q_S, Lambda_S, gamma):
        Lambda_F, Q_F = torch.symeig(g(F), eigenvectors=True)
        Lambda_F = Lambda_F.view(-1,1)
        G = get_G(Lambda_F, Lambda_S, gamma)
        Z = Q_F @ (G * (Q_F.t() @ X @ Q_S)) @ Q_S.t()
        ctx.save_for_backward(F, S, Q_F, Q_S, Z, G, X, gamma)
        return Z

    @staticmethod
    def backward(ctx, grad_output):
        grad_Z = grad_output
        F, S, Q_F, Q_S, Z, G, X, gamma = ctx.saved_tensors
        FF = F.t() @ F
        FF_norm = torch.norm(FF, p='fro')
        R = G * (Q_F.t() @ grad_Z @ Q_S)
        R = Q_F @ R @ Q_S.t() @ torch.sparse.mm(S, Z.t())
        scalar_1 = gamma * (1/(FF_norm+epsilon_F))
        scalar_2 = torch.sum(FF * R)
        scalar_2 = 2 * scalar_2 * (1/(FF_norm**2 + epsilon_F * FF_norm))
        grad_F = (R + R.t()) - scalar_2 * FF
        grad_F = scalar_1 * (F @ grad_F)
        grad_X = None
        return grad_X, grad_F, None, None, None, None