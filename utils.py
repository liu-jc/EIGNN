import sys

import numpy as np
import scipy.sparse as sp
import torch
# import torch_sparse
import pickle as pkl
import networkx as nx
from normalization import fetch_normalization, row_normalize, aug_normalized_adjacency
from time import perf_counter
import ipdb

from sklearn import metrics

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def preprocess_citation(adj, features, normalization="FirstOrderGCN"):
    adj_normalizer = fetch_normalization(normalization)
    adj = adj_normalizer(adj)
    features = row_normalize(features)
    return adj, features

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def Evaluation(output, labels):
    preds = output.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    '''
    binary_pred = preds
    binary_pred[binary_pred > 0.0] = 1
    binary_pred[binary_pred <= 0.0] = 0
    '''
    num_correct = 0
    binary_pred = np.zeros(preds.shape).astype('int')
    for i in range(preds.shape[0]):
        k = labels[i].sum().astype('int')
        topk_idx = preds[i].argsort()[-k:]
        binary_pred[i][topk_idx] = 1
        for pos in list(labels[i].nonzero()[0]):
            if labels[i][pos] and labels[i][pos] == binary_pred[i][pos]:
                num_correct += 1

    print('total number of correct is: {}'.format(num_correct))
    #print('preds max is: {0} and min is: {1}'.format(preds.max(),preds.min()))
    #'''
    return metrics.f1_score(labels, binary_pred, average="micro"), metrics.f1_score(labels, binary_pred, average="macro")


def sparse_mx_to_torch_sparse_tensor(sparse_mx, device=None):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    tensor = torch.sparse.FloatTensor(indices, values, shape)
    if device is not None:
        tensor = tensor.to(device)
    return tensor
def get_spectral_rad(sparse_tensor, tol=1e-5):
    """Compute spectral radius from a tensor"""
    A = sparse_tensor.data.coalesce().cpu()
    A_scipy = sp.coo_matrix((np.abs(A.values().numpy()), A.indices().numpy()), shape=A.shape)
    return np.abs(sp.linalg.eigs(A_scipy, k=1, return_eigenvectors=False)[0]) + tol

def projection_norm_inf(A, kappa=0.99, transpose=False):
    """ project onto ||A||_inf <= kappa return updated A"""
    # TODO: speed up if needed
    v = kappa
    if transpose:
        A_np = A.T.clone().detach().cpu().numpy()
    else:
        A_np = A.clone().detach().cpu().numpy()
    x = np.abs(A_np).sum(axis=-1)
    for idx in np.where(x > v)[0]:
        # read the vector
        a_orig = A_np[idx, :]
        a_sign = np.sign(a_orig)
        a_abs = np.abs(a_orig)
        a = np.sort(a_abs)

        s = np.sum(a) - v
        l = float(len(a))
        for i in range(len(a)):
            # proposal: alpha <= a[i]
            if s / l > a[i]:
                s -= a[i]
                l -= 1
            else:
                break
        alpha = s / l
        a = a_sign * np.maximum(a_abs - alpha, 0)
        # verify
        assert np.isclose(np.abs(a).sum(), v, atol=1e-4)
        # write back
        A_np[idx, :] = a
    A.data.copy_(torch.tensor(A_np.T if transpose else A_np, dtype=A.dtype, device=A.device))
    return A

def projection_norm_inf_and_1(A, kappa_inf=0.99, kappa_1=None, inf_first=True):
    """ project onto ||A||_inf <= kappa return updated A"""
    # TODO: speed up if needed
    v_inf = kappa_inf
    v_1 = kappa_inf if kappa_1 is None else kappa_1
    A_np = A.clone().detach().cpu().numpy()
    if inf_first:
        A_np = projection_inf_np(A_np, v_inf)
        A_np = projection_inf_np(A_np.T, v_1).T
    else:
        A_np = projection_inf_np(A_np.T, v_1).T
        A_np = projection_inf_np(A_np, v_inf)
    A.data.copy_(torch.tensor(A_np, dtype=A.dtype, device=A.device))
    return A

def projection_inf_np(A_np, v):
    x = np.abs(A_np).sum(axis=-1)
    for idx in np.where(x > v)[0]:
        # read the vector
        a_orig = A_np[idx, :]
        a_sign = np.sign(a_orig)
        a_abs = np.abs(a_orig)
        a = np.sort(a_abs)

        s = np.sum(a) - v
        l = float(len(a))
        for i in range(len(a)):
            # proposal: alpha <= a[i]
            if s / l > a[i]:
                s -= a[i]
                l -= 1
            else:
                break
        alpha = s / l
        a = a_sign * np.maximum(a_abs - alpha, 0)
        # verify
        assert np.isclose(np.abs(a).sum(), v, atol=1e-6)
        # write back
        A_np[idx, :] = a
    return A_np

def clip_gradient(model, clip_norm=10):
    """ clip gradients of each parameter by norm """
    for param in model.parameters():
        torch.nn.utils.clip_grad_norm(param, clip_norm)
    return model

def l_1_penalty(model, alpha=0.1):
    regularization_loss = 0
    for param in model.parameters():
        regularization_loss += alpha * torch.sum(torch.abs(param))
    return regularization_loss

class AdditionalLayer(torch.nn.Module):
    def __init__(self, model, num_input, num_output, activation=torch.nn.ReLU()):
        super().__init__()
        self.model = model
        self.add_module("model", self.model)
        self.activation = activation
        if isinstance(activation, torch.nn.Module):
            self.add_module("activation", self.activation)
        self.func = torch.nn.Linear(num_input, num_output, bias=False)

    def forward(self, *input):
        x = self.model(*input)
        x = self.activation(x)
        return self.func(x)

class SparseDropout(torch.nn.Module):
    def __init__(self, dprob=0.5):
        super(SparseDropout, self).__init__()
        # dprob is ratio of dropout
        # convert to keep probability
        self.kprob=1-dprob

    def forward(self, x, training):
        if training:
            mask=((torch.rand(x._values().size())+(self.kprob)).floor()).type(torch.bool)
            rc=x._indices()[:,mask]
            val=x._values()[mask]*(1.0/self.kprob)
            return torch.sparse.FloatTensor(rc, val, torch.Size(x.shape))
        else:
            return x