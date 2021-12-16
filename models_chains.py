import torch
import torch.nn as nn
import torch.nn.functional as F
from functions import *
from layers import ImplicitGraph, IDM_SGC
from torch.nn import Parameter
from utils import get_spectral_rad, SparseDropout
import torch.sparse as sparse
from torch_geometric.nn import GCNConv, GATConv, SGConv, APPNP, GCN2Conv, JumpingKnowledge, MessagePassing
from torch_geometric.utils import remove_self_loops, to_scipy_sparse_matrix
import numpy as np
from utils import *
import time
import ipdb
import scipy

class IGNN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, num_node, dropout, kappa=0.9, adj_orig=None):
        super(IGNN, self).__init__()

        self.adj = None
        self.adj_rho = None
        self.adj_orig = adj_orig

        #one layer with V
        self.ig1 = ImplicitGraph(nfeat, nhid, num_node, kappa)
        self.dropout = dropout
        self.X_0 = Parameter(torch.zeros(nhid, num_node), requires_grad=False)
        self.V = nn.Linear(nhid, nclass, bias=False)

    def forward(self, features, adj):
        if adj is not self.adj:
            self.adj = adj
            self.adj_rho = get_spectral_rad(adj)

        x = features
        x = self.ig1(self.X_0, adj, x, F.relu, self.adj_rho, A_orig=self.adj_orig).T
        x = F.normalize(x, dim=-1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.V(x)
        return x


class IGNN_finite(nn.Module):
    def __init__(self, m, m_y, nhid, K, dropout):
        super(IGNN_finite, self).__init__()
        self.lin1 = nn.Linear(m, nhid, bias=False)
        self.lin2 = nn.Linear(nhid, m_y, bias=False)
        self.num_layers = K
        self.hid_layer = nn.Linear(nhid, nhid, bias=False)
        self.dropout = dropout
        # self.prop1 = APPNP(K=K, alpha=alpha)

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.hid_layer.reset_parameters()

    def forward(self, x, adj):
        # x: (f,n), lin1: (f->h)
        # lin2 (h -> m_y), hid_layer(h -> h)
        x_first = self.lin1(x.T).T
        x = x_first
        # ipdb.set_trace()
        for _ in range(self.num_layers):
            # tmp = torch.spmm(self.hid_layer(x.T).T, adj) + x_first
            tmp = torch.spmm(torch.transpose(adj, 0, 1), self.hid_layer(x.T)).T + x_first
            x = F.relu(tmp)
        x = x.T
        x = F.normalize(x, dim=-1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lin2(x)
        return x


class EIGNN_Linear(nn.Module):
    def __init__(self, adj, sp_adj, m, m_y, num_eigenvec, gamma):
        super(EIGNN_Linear, self).__init__()
        self.EIGNN = IDM_SGC(adj, sp_adj, m, num_eigenvec, gamma)
        self.B = nn.Linear(m, m_y, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        self.B.reset_parameters()
        self.EIGNN.reset_parameters()

    def forward(self, X):
        output = self.EIGNN(X).t()
        output = F.normalize(output, dim=-1)
        output = F.dropout(output, 0.5, training=self.training)
        output = self.B(output)
        return output


epsilon_F = 10**(-12)
def g(F):
    FF = F.t() @ F
    FF_norm = torch.norm(FF, p='fro')
    return (1/(FF_norm+epsilon_F)) * FF


def get_G(Lambda_F, Lambda_S, gamma):
    G = 1.0 - gamma * Lambda_F @ Lambda_S.t()
    G = 1 / G
    return G


class GCN(nn.Module):
    def __init__(self, m, m_y, hidden):
        super(GCN, self).__init__()
        self.gc1 = GCNConv(m, hidden)
        self.gc2 = GCNConv(hidden, m_y)

    def forward(self, x, edge_index):
        out = self.gc1(x, edge_index)
        out = F.relu(out)
        out = F.dropout(out, p=0.5, training=self.training)
        out = self.gc2(out, edge_index)
        return out

class GAT(nn.Module):
    def __init__(self, m, m_y, hidden, heads):
        super(GAT, self).__init__()
        self.gat1 = GATConv(m, hidden, heads=heads)
        self.gat2 = GATConv(heads*hidden, m_y, heads=heads)

    def forward(self, x, edge_index):
        out = self.gat1(x, edge_index)
        out = F.elu(out)
        out = F.dropout(out, p=0.8, training=self.training)
        out = self.gat2(out, edge_index)
        return out

class SGC(nn.Module):
    def __init__(self, m, m_y, K):
        super(SGC, self).__init__()
        self.sgc = SGConv(m, m_y, K)
        self.reset_parameters()

    def reset_parameters(self):
        self.sgc.reset_parameters()

    def forward(self, x, edge_index):
        out = self.sgc(x, edge_index)
        return out

class APPNP_Net(nn.Module):
    def __init__(self, m, m_y, nhid, K, alpha):
        super(APPNP_Net, self).__init__()
        self.lin1 = nn.Linear(m, nhid)
        self.lin2 = nn.Linear(nhid, m_y)
        self.prop1 = APPNP(K=K, alpha=alpha)

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        out = self.prop1(x, edge_index)
        return out



class GCN_JKNet(torch.nn.Module):
    def __init__(self, m, m_y, hidden, layers=8):
        in_channels = m
        out_channels = m_y

        super(GCN_JKNet, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden))
        for _ in range(layers-1):
            self.convs.append(GCNConv(hidden, hidden))
        # self.conv1 = GCNConv(in_channels, hidden)
        # self.conv2 = GCNConv(hidden, hidden)
        self.lin1 = nn.Linear(layers*hidden, out_channels)
        # self.lin1 = torch.nn.Linear(64, out_channels)
        # self.one_step = APPNP(K=1, alpha=0)
        # self.JK = JumpingKnowledge(mode='lstm',
        #                            channels=64,
        #                            num_layers=4)
        self.JK = JumpingKnowledge(mode='cat')

    def forward(self, x, edge_index):

        final_xs = []
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=0.5, training=self.training)
            final_xs.append(x)

        x = self.JK(final_xs)
        x = self.lin1(x)
        return x

class GCNII_Model(torch.nn.Module):
    def __init__(self, m, m_y, hidden=64, layers=64, alpha=0.5, theta=1.):
        super(GCNII_Model, self).__init__()
        self.lin1 = nn.Linear(m, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(layers):
            self.convs.append(GCN2Conv(channels=hidden,
                                       alpha=alpha, theta=theta, layer=i+1))
        self.lin2 = nn.Linear(hidden, m_y)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin1(x))
        x_0 = x
        for conv in self.convs:
            x = F.dropout(x, p=0.5, training=self.training)
            x = F.relu(conv(x, x_0, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        out = self.lin2(x)
        return out


class H2GCN_Prop(MessagePassing):
    def __init__(self):
        super(H2GCN_Prop, self).__init__()

    def forward(self, h, norm_adj_1hop, norm_adj_2hop):
        h_1 = torch.sparse.mm(norm_adj_1hop, h) # if OOM, consider using torch-sparse
        h_2 = torch.sparse.mm(norm_adj_2hop, h)
        h = torch.cat((h_1, h_2), dim=1)
        return h


class H2GCN(torch.nn.Module):
    def __init__(self, m, m_y, hidden, edge_index, dropout=0.5, act='relu'):
        super(H2GCN, self).__init__()
        self.dropout = dropout
        self.lin1 = nn.Linear(m, hidden, bias=False)
        self.act = torch.nn.ReLU() if act == 'relu' else torch.nn.Identity()
        self.H2GCN_layer = H2GCN_Prop()
        self.num_layers = 1
        self.lin_final = nn.Linear((2**(self.num_layers+1)-1)*hidden, m_y, bias=False)
        # self.lin_final = nn.Linear((self.num_layers+1)*hidden, m_y, bias=False)

        adj = to_scipy_sparse_matrix(remove_self_loops(edge_index)[0])
        adj_2hop = adj.dot(adj)
        adj_2hop = adj_2hop - sp.diags(adj_2hop.diagonal())
        adj = indicator_adj(adj)
        adj_2hop = indicator_adj(adj_2hop)
        norm_adj_1hop = get_normalized_adj(adj)
        self.norm_adj_1hop = sparse_mx_to_torch_sparse_tensor(norm_adj_1hop, 'cuda')
        norm_adj_2hop = get_normalized_adj(adj_2hop)
        self.norm_adj_2hop = sparse_mx_to_torch_sparse_tensor(norm_adj_2hop, 'cuda')

    def forward(self, x, edge_index=None):
        hidden_hs = []
        h = self.act(self.lin1(x))
        hidden_hs.append(h)
        for i in range(self.num_layers):
            h = self.H2GCN_layer(h, self.norm_adj_1hop, self.norm_adj_2hop)
            hidden_hs.append(h)
        h_final = torch.cat(hidden_hs, dim=1)
        # print(f'lin_final.size(): {self.lin_final.weight.size()}, h_final.size(): {h_final.size()}')
        h_final = F.dropout(h_final, p=self.dropout, training=self.training)
        output = self.lin_final(h_final)
        return output