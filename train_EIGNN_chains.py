from __future__ import division
from __future__ import print_function

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import ipdb

from utils import accuracy, clip_gradient
from models_chains import IGNN, EIGNN_Linear
from datasets_utils import *
from copy import deepcopy

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=2000,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--kappa', type=float, default=0.9,
                    help='Projection parameter. ||W|| <= kappa/lpf(A)')
parser.add_argument('--dataset', type=str, default="chains-10",
                        help='Dataset to use.')
parser.add_argument('--feature', type=str, default="mul",
                    choices=['mul', 'cat', 'adj'],
                    help='feature-type')
parser.add_argument('--normalization', type=str, default='AugNormAdj',
                   choices=['AugNormAdj'],
                   help='Normalization method for the adjacency matrix.')
parser.add_argument('--degree', type=int, default=2,
                    help='degree of the approximation.')
parser.add_argument('--per', type=int, default=-1,
                    help='Number of each nodes so as to balance.')
parser.add_argument('--experiment', type=str, default="base-experiment",
                    help='feature-type')
# IDM-SGC arguments
parser.add_argument('--gamma', type=float, default=0.8)
parser.add_argument('--num_eigenvec', type=int, default=100)

parser.add_argument('--path', type=str, default='./results/')
parser.add_argument('--num_chains', type=int, default=20, help='num of chains')
parser.add_argument('--chain_len', type=int, default=10, help='the length of each chain')
parser.add_argument('--num_class', type=int, default=2, help='num of class')
parser.add_argument('--patience', type=int, default=200, help='early stop patience')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
print(args)

if not os.path.exists(args.path):
    os.mkdir(args.path)

result_name = '_'.join(['EIGNN', str(args.epochs), str(args.lr), str(args.weight_decay),
                        str(args.num_chains), str(args.chain_len)]) + '.txt'
result_path = os.path.join(args.path, result_name)
filep = open(result_path, 'w')
filep.write(str(args) + '\n')

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
adj, sp_adj, features, labels, idx_train, idx_val, idx_test = load_citation_syn_chain_IDM(args.normalization, args.cuda,
                                                                                          args.num_chains,
                                                                                          args.chain_len,
                                                                                          num_class=args.num_class)


if not os.path.exists(args.path):
    os.mkdir(args.path)


features = features.t()
Y = labels
m = features.shape[0]
m_y = torch.max(Y).int().item() + 1

S = adj
# input(f'adj: {adj}')
print(f'adj.shape: {adj.shape}, m_y: {m_y}, m: {m}')
model = EIGNN_Linear(adj, sp_adj, m, m_y, args.num_eigenvec, args.gamma)
# ipdb.set_trace()

# Model and optimizer
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()#[:10]
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    # output = model(features, adj)
    output = model(features)
    output = F.log_softmax(output, dim=1)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        # output = model(features, adj)
        output = model(features)
        output = F.log_softmax(output, dim=1)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])

    outstr = 'Epoch: {:04d} '.format(epoch+1) + \
             'loss_train: {:.4f} '.format(loss_train.item()) + \
             'acc_train: {:.4f} '.format(acc_train.item()) + \
             'loss_val: {:.4f} '.format(loss_val.item()) + \
             'acc_val: {:.4f} '.format(acc_val.item()) + \
             'loss_test: {:.4f} '.format(loss_test.item()) + \
             'acc_test: {:.4f} '.format(acc_test.item()) + \
             'time: {:.4f}s'.format(time.time() - t)
    print(outstr)
    filep.write(outstr + '\n')
    return loss_val, acc_val, loss_test, acc_test

def test():
    model.eval()
    # output = model(features, adj)
    output = model(features)
    output = F.log_softmax(output, dim=1)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Dataset: " + args.dataset)
    filep.write("Dataset: " + args.dataset + '\n')
    outstr = "Test set results:" + \
             "loss= {:.4f}".format(loss_test.item()) + \
             "accuracy= {:.4f}".format(acc_test.item())
    print(outstr)
    filep.write(outstr + '\n')


# Train model
t_total = time.time()
best_val_loss = 1000
cnt = 0
for epoch in range(args.epochs):
    loss_val, acc_val, loss_test, acc_test = train(epoch)
    if loss_val < best_val_loss:
        cnt = 0
        best_val_loss = loss_val
        weights = deepcopy(model.state_dict())
    else:
        cnt += 1
    if cnt == args.patience:
        print(f'Early stop @ Epoch {epoch}, loss_val: {loss_val}, acc_val: {acc_val}, '
              f'loss_test: {loss_test}, acc_test: {acc_test}')
        filep.write(f'Early stop @ Epoch {epoch}, loss_val: {loss_val}, acc_val: {acc_val}, '
                    f'loss_test: {loss_test}, acc_test: {acc_test}\n')
        break

print("Optimization Finished!")
filep.write("Optimization Finished!\n")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
filep.write("Total time elapsed: {:.4f}s\n".format(time.time() - t_total))
# Testing
model.load_state_dict(weights)
test()