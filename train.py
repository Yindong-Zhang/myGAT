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

from dataset.make_dataset import get_dataset_and_split_planetoid, get_dataset, get_train_val_test_split
from utils import accuracy
from models import GAT, SpGAT

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no_cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--gpu_ids', type= int, nargs= '+', default= [0, ], help= "Specify GPU ids to move model on.")
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16, help='Number of hidden units.')
parser.add_argument('--nb_heads_1', type=int, default= 4, help='Number of head attentions in layer 1.')
parser.add_argument('--nb_heads_2', type=int, default= None, help='Number of head attentions in layer 2.')
parser.add_argument('--nb_heads_3', type=int, default= None, help='Number of head attentions in layer 3.')
parser.add_argument('--nb_heads_4', type=int, default= None, help='Number of head attentions in layer 4.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument('--order1_attention', action= 'store_true', default= False,
                    help= "Whether to use diffused attention in model")
parser.add_argument('--order2_attention', action= 'store_true', default= False,
                    help= "Whether to use improved attention in model")
parser.add_argument('--dataset', type= str, default= 'cora', help= "Dataset to use for training.")
parser.add_argument('--train_size', type= int, default= 20, help= "Size of training dataset.")
parser.add_argument('--val_size', type= int, default= 20, help= "Size of validation dataset.")

print('test')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

print(args)

configStr= "dataset~%s-hidden~%s-nheads_1~%s-nheads_2~%s-nheads_3~%s-nheads_4~%s-learning_rate~%s-weight_decay~%s-dropout~%s-train_size~%s-val_size~%s-order1_attention~%s-order2_attention~%s-patience~%s" \
    %(args.dataset, args.hidden, args.nb_heads_1, args.nb_heads_2, args.nb_heads_3, args.nb_heads_4, args.lr, args.weight_decay, args.dropout, args.train_size, args.val_size, args.order1_attention, args.order2_attention, args.patience)
dump_dir = os.path.join('./output', configStr)
if not os.path.exists(dump_dir):
    os.makedirs(dump_dir)

# Load data
# adj, features, labels, idx_train, idx_val, idx_test = get_dataset_and_split_planetoid(args.dataset, './data',)
adj, features, labels = get_dataset(args.dataset, './data/npz/{}.npz'.format(args.dataset), standardize= True, train_examples_per_class= 40, val_examples_per_class= 100)
random_state = np.random.RandomState(args.seed)
idx_train, idx_val, idx_test = get_train_val_test_split(random_state, labels, train_examples_per_class= 40, val_examples_per_class= 80)

# convert numpy/scipy to torch tensor
adj = torch.FloatTensor(adj.todense())
features = torch.FloatTensor(features.todense())
# convert one-hot encoding back.
labels = torch.LongTensor(np.where(labels)[1])

idx_train = torch.LongTensor(idx_train)
idx_val = torch.LongTensor(idx_val)
idx_test = torch.LongTensor(idx_test)

# add batchsize axia
adj         = adj[None, ...]
features    = features[None, ...]
labels      = labels[None, ...]

# create node_mask tensor
node_masks= torch.ones_like(labels)
node_masks= node_masks[None, ...]


att_type= None
if args.order1_attention and not args.order2_attention:
    att_type= 'order1'
elif args.order2_attention and not args.order1_attention:
    att_type= 'order2'
elif not args.order2_attention and not args.order1_attention:
    att_type= None
else:
    raise RuntimeError("Attention type hyperparameter not understood!!")

# Model and optimizer
if args.nb_heads_4 and args.nb_heads_3 and args.nb_heads_2:
    model = GAT(nfeat=features.shape[-1],
                nhid_list=[args.hidden, ] * 4,
                nclass=int(labels.max()) + 1,
                dropout=args.dropout,
                nheads_list=[args.nb_heads_1, args.nb_heads_2, args.nb_heads_3, args.nb_heads_4 ],
                alpha=args.alpha,
                att_type = att_type,
                )
elif args.nb_heads_3 and args.nb_heads_2 and not args.nb_heads_4:
    model = GAT(nfeat=features.shape[-1],
                nhid_list=[args.hidden, ] * 3,
                nclass=int(labels.max()) + 1,
                dropout=args.dropout,
                nheads_list=[args.nb_heads_1, args.nb_heads_2, args.nb_heads_3, ],
                alpha=args.alpha,
                att_type= att_type,
                )
elif args.nb_heads_2 and not args.nb_heads_3 and not args.nb_heads_4:
    model = GAT(nfeat=features.shape[-1],
                nhid_list=[args.hidden, ] * 2,
                nclass=int(labels.max()) + 1,
                dropout=args.dropout,
                nheads_list=[args.nb_heads_1, args.nb_heads_2],
                alpha=args.alpha,
                att_type= att_type,
                )
elif not args.nb_heads_2 and not args.nb_heads_3 and not args.nb_heads_4:
    model= GAT(nfeat=features.shape[-1],
                nhid_list=[args.hidden, ],
                nclass=int(labels.max()) + 1,
                dropout=args.dropout,
                nheads_list=[args.nb_heads_1, ],
                alpha=args.alpha,
                att_type= att_type,
                )
else:
    raise RuntimeError("Model hyperparameters not understood!!")

print(model)

optimizer = optim.Adam(model.parameters(), 
                       lr=args.lr, 
                       weight_decay=args.weight_decay)

if args.cuda:
    devices= args.gpu_ids
    model.cuda()
    model= nn.DataParallel(model, device_ids= devices)

    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    node_masks = node_masks.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output_logits = model(features, adj, node_masks)
    output= F.log_softmax(output_logits, dim= -1)
    loss_train = F.nll_loss(output[0, idx_train], labels[0, idx_train])
    acc_train = accuracy(output[0, idx_train], labels[0, idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output_logits = model(features, adj, node_masks)
        output= F.log_softmax(output_logits, -1)

    loss_val = F.nll_loss(output[0, idx_val], labels[0, idx_val])
    acc_val = accuracy(output[0, idx_val], labels[0, idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.data.item()),
          'acc_train: {:.4f}'.format(acc_train.data.item()),
          'loss_val: {:.4f}'.format(loss_val.data.item()),
          'acc_val: {:.4f}'.format(acc_val.data.item()),
          'time: {:.4f}s'.format(time.time() - t))

    return loss_val.data.item()


# Train model
t_total = time.time()
loss_values = []
bad_counter = 0
best = 1E9
best_epoch = 0
for epoch in range(args.epochs):
    loss_values.append(train(epoch))

    torch.save(model.state_dict(), os.path.join(dump_dir, '{}.pkl'.format(epoch)))
    if loss_values[-1] < best:
        best = loss_values[-1]
        best_epoch = epoch
        bad_counter = 0
    else:
        bad_counter += 1

    if bad_counter == args.patience:
        break

files = glob.glob(os.path.join(dump_dir, '*.pkl'))
for file in files:
    filename= os.path.split(file)[-1]
    epoch_nb = int(filename.split('.')[0])
    if epoch_nb != best_epoch:
        os.remove(file)

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

print("Model arguments: ")
print(args)
print(model)

# Restore best model
print('Loading {}th epoch'.format(best_epoch))
model.load_state_dict(torch.load(os.path.join(dump_dir, '{}.pkl'.format(best_epoch))))

# Testing
model.eval()
output_logits = model(features, adj, node_masks)
output = F.log_softmax(output_logits, dim=-1)
loss_test = F.nll_loss(output[0, idx_test], labels[0, idx_test])
acc_test = accuracy(output[0, idx_test], labels[0, idx_test])
print("Test set results:",
      "loss= {:.4f}".format(loss_test.data.item()),
      "accuracy= {:.4f}".format(acc_test.data.item()))

with open(os.path.join('./result', "%s.txt" %(configStr, )), 'a') as f:
    f.write("Test graph results: \tmean loss: %.4f \tmean acc: %4f \n" %(loss_test, acc_test))

