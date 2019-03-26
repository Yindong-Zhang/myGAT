from __future__ import division
from __future__ import print_function

import os
import glob
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from utils import load_data, accuracy
from models import GAT, SpGAT

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16, help='Number of hidden units.')
parser.add_argument('--nb_heads_1', type=int, default=2, help='Number of head attentions in layer 1.')
parser.add_argument('--nb_heads_2', type=int, default=2, help='Number of head attentions in layer 2.')
parser.add_argument('--nb_heads_3', type=int, default= None, help='Number of head attentions in layer 3.')
parser.add_argument('--nb_heads_4', type=int, default= None, help='Number of head attentions in layer 4.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument('--diffused_attention', action= 'store_true', default= False,
                    help= "Whether to use diffused attention in model")
parser.add_argument('--improved_attention', action= 'store_true', default= False,
                    help= "Whether to use improved attention in model")

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

print(args)
# Load data
adj, features, labels, idx_train, idx_val, idx_test = load_data()

att_type= None
if args.diffused_attention and not args.improved_attention:
    att_type= 'diffused'
elif args.improved_attention and not args.diffused_attention:
    att_type= 'improved'
elif not args.improved_attention and not args.diffused_attention:
    att_type= None
else:
    raise RuntimeError("Attention type hyperparameter not understood!!")

# Model and optimizer
if args.sparse:
    model = SpGAT(nfeat=features.shape[1], 
                nhid=args.hidden, 
                nclass=int(labels.max()) + 1, 
                dropout=args.dropout, 
                nheads=args.nb_heads, 
                alpha=args.alpha)
else:
    if args.nb_heads_4 and args.nb_heads_3 and args.nb_heads_2:
        model = GAT(nfeat=features.shape[1],
                    nhid_list=[args.hidden, ] * 4,
                    nclass=int(labels.max()) + 1,
                    dropout=args.dropout,
                    nheads_list=[args.nb_heads_1, args.nb_heads_2, args.nb_heads_3, args.nb_heads_4 ],
                    alpha=args.alpha,
                    att_type = att_type,
                    )
    elif args.nb_heads_3 and args.nb_heads_2 and not args.nb_heads_4:
        model = GAT(nfeat=features.shape[1],
                    nhid_list=[args.hidden, ] * 3,
                    nclass=int(labels.max()) + 1,
                    dropout=args.dropout,
                    nheads_list=[args.nb_heads_1, args.nb_heads_2, args.nb_heads_3, ],
                    alpha=args.alpha,
                    att_type= att_type,
                    )
    elif args.nb_heads_2 and not args.nb_heads_3 and not args.nb_heads_4:
        model = GAT(nfeat=features.shape[1],
                    nhid_list=[args.hidden, ] * 2,
                    nclass=int(labels.max()) + 1,
                    dropout=args.dropout,
                    nheads_list=[args.nb_heads_1, args.nb_heads_2],
                    alpha=args.alpha,
                    att_type= att_type,
                    )
    else:
        raise RuntimeError("Model hyperparameters not understood!!")


optimizer = optim.Adam(model.parameters(), 
                       lr=args.lr, 
                       weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

features, adj, labels = Variable(features), Variable(adj), Variable(labels)


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.data.item()),
          'acc_train: {:.4f}'.format(acc_train.data.item()),
          'loss_val: {:.4f}'.format(loss_val.data.item()),
          'acc_val: {:.4f}'.format(acc_val.data.item()),
          'time: {:.4f}s'.format(time.time() - t))

    return loss_val.data.item()


def compute_test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.data.item()),
          "accuracy= {:.4f}".format(acc_test.data.item()))

# Train model
t_total = time.time()
loss_values = []
bad_counter = 0
best = 1E9
best_epoch = 0
for epoch in range(args.epochs):
    loss_values.append(train(epoch))

    torch.save(model.state_dict(), '{}.pkl'.format(epoch))
    if loss_values[-1] < best:
        best = loss_values[-1]
        best_epoch = epoch
        bad_counter = 0
    else:
        bad_counter += 1

    if bad_counter == args.patience:
        break

    files = glob.glob('*.pkl')
    for file in files:
        epoch_nb = int(file.split('.')[0])
        if epoch_nb < best_epoch:
            os.remove(file)

files = glob.glob('*.pkl')
for file in files:
    epoch_nb = int(file.split('.')[0])
    if epoch_nb > best_epoch:
        os.remove(file)

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Restore best model
print('Loading {}th epoch'.format(best_epoch))
model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))

# Testing
compute_test()
