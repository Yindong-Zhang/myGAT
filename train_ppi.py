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
from models import GAT, SpGAT, MultiLabelGAT
from process_ppi import load_p2p, create_data

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
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
parser.add_argument('--print_every', type= int, default= 2,
                    help= "Interval to print results.")

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

print(args)


# Load data
train_adj, val_adj, test_adj, \
train_feat, val_feat, test_feat, \
train_labels, val_labels, test_labels, \
train_nodes, val_nodes, test_nodes, \
tr_msk, vl_msk, ts_msk = load_p2p('./data/ppi')
#
# load test data
# train_adj, val_adj, test_adj, \
# train_feat, val_feat, test_feat, \
# train_labels, val_labels, test_labels, \
# train_nodes, val_nodes, test_nodes, \
# tr_msk, vl_msk, ts_msk = create_data()


att_type= None
if args.diffused_attention and not args.improved_attention:
    att_type= 'diffused'
elif args.improved_attention and not args.diffused_attention:
    att_type= 'improved'
elif not args.improved_attention and not args.diffused_attention:
    att_type= None
else:
    raise RuntimeError("Attention type hyperparameter not understood!!")

nb_features= train_feat.shape[-1]
nb_class= train_labels.shape[-1]
# Model and optimizer
if args.nb_heads_4 and args.nb_heads_3 and args.nb_heads_2:
    model = MultiLabelGAT(nfeat=nb_features,
                nhid_list=[args.hidden, ] * 4,
                nclass=nb_class,
                dropout=args.dropout,
                nheads_list=[args.nb_heads_1, args.nb_heads_2, args.nb_heads_3, args.nb_heads_4 ],
                alpha=args.alpha,
                att_type = att_type,
                )
elif args.nb_heads_3 and args.nb_heads_2 and not args.nb_heads_4:
    model = MultiLabelGAT(nfeat= nb_features,
                nhid_list=[args.hidden, ] * 3,
                nclass= nb_class,
                dropout=args.dropout,
                nheads_list=[args.nb_heads_1, args.nb_heads_2, args.nb_heads_3, ],
                alpha=args.alpha,
                att_type= att_type,
                )
elif args.nb_heads_2 and not args.nb_heads_3 and not args.nb_heads_4:
    model = MultiLabelGAT(nfeat= nb_features,
                nhid_list=[args.hidden, ] * 2,
                nclass= nb_class,
                dropout=args.dropout,
                nheads_list=[args.nb_heads_1, args.nb_heads_2],
                alpha=args.alpha,
                att_type= att_type,
                )
else:
    raise RuntimeError("Model hyperparameters not understood!!")

if args.cuda:
    # devices= [0, 1]
    model.cuda()
    # model= nn.DataParallel(model, device_ids= devices)

optimizer = optim.Adam(model.parameters(),
                       lr=args.lr,
                       weight_decay=args.weight_decay)


def prepare_input(batch_graphs, gpu):
    # restrict batchsize to 1
    batch_features, batch_adj, batch_labels, batch_nodes, batch_masks= batch_graphs
    features, adj, labels= batch_features[:batch_nodes, :], batch_adj[:batch_nodes, :batch_nodes], batch_labels[:batch_nodes, :]
    if gpu:
        features= features.cuda()
        adj= adj.cuda()
        labels= labels.cuda()
    return features, adj, labels


def loop_dataset(dataset, classifier, criterion, optimizer=None, batchsize= 1, cuda= False, print_every= 5, shuffle= True):

    features, adj, labels, nb_nodes, masks= dataset
    total_count= features.shape[0]
    sample_idxes= list(range(total_count))
    if shuffle:
        random.shuffle(sample_idxes)
    total_iters = ( total_count + (batchsize - 1) * (optimizer is None)) // batchsize

    mean_loss= 0.0
    mean_acc= 0.0

    # n_samples = 0
    for it in range(total_iters):
        selected_idxes = sample_idxes[it]

        # restrict batchsize to 1
        batch_features  = features[selected_idxes]
        batch_adj       = adj[selected_idxes]
        batch_labels    = labels[selected_idxes]
        batch_nodes     = nb_nodes[selected_idxes]
        batch_masks     = masks[selected_idxes]

        batch_graphs= (batch_features, batch_adj, batch_labels, batch_nodes, batch_masks)
        batch_features, batch_adj, batch_labels= prepare_input(batch_graphs, cuda)

        batch_logits= classifier(batch_features, batch_adj)
        loss = criterion(batch_logits, batch_labels)

        batch_prob= torch.sigmoid(batch_logits)
        batch_pred= (batch_prob > 0.5)
        batch_pred= batch_pred.float()

        acc_item= torch.all(batch_pred.eq(batch_labels), dim= -1)
        acc = torch.mean(acc_item.float())

        loss        = loss.mean()
        acc         = acc.mean()

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss = loss.data.cpu().item()
        acc= acc.data.cpu().item()

        mean_loss= (it * mean_loss + loss ) / (it + 1)
        mean_acc= (it * mean_acc + acc ) / (it + 1)
        if it % print_every == 0:
            print('mean_loss: %5.3f \t mean_acc: %5.3f \t %s \t %s' % (mean_loss, mean_acc, batch_pred[0, :8].cpu().numpy(), batch_labels[0, :8].cpu().numpy()))

    return mean_loss, mean_acc

# Train model
t_total = time.time()
loss_values = []
bad_counter = 0
best = 1E9
best_epoch = 0
train_data  = train_feat, train_adj, train_labels, train_nodes, tr_msk
val_data    = val_feat, val_adj, val_labels, val_nodes, vl_msk
test_data   = test_feat, test_adj, test_labels, test_nodes, ts_msk

bce_loss = torch.nn.BCEWithLogitsLoss()

for epoch in range(args.epochs):
    print("Training epoch %s:" %(epoch, ))
    model.train()
    train_loss, train_acc= loop_dataset(train_data, model, criterion= bce_loss, optimizer= optimizer, batchsize= 1, shuffle= True, cuda= args.cuda, print_every= args.print_every)

    print("Validation epoch %s:" %(epoch, ))
    model.eval()
    val_loss, val_acc= loop_dataset(val_data, model, criterion= bce_loss, batchsize= 1, shuffle= False, cuda= args.cuda, print_every= args.print_every)
    loss_values.append(val_loss)

    torch.save(model.state_dict(), './output/{}.pkl'.format(epoch))
    if loss_values[-1] < best:
        best = loss_values[-1]
        best_epoch = epoch
        bad_counter = 0
    else:
        bad_counter += 1

    if bad_counter == args.patience:
        break

files = glob.glob('./ouput/*.pkl')
for file in files:
    epoch_nb = int(file.split('.')[0])
    if epoch_nb != best_epoch:
        os.remove(file)

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Restore best model
print('Loading {}th epoch'.format(best_epoch))
model.load_state_dict(torch.load('./output/{}.pkl'.format(best_epoch)))
model.eval()
test_loss, test_acc = loop_dataset(test_data, model, criterion= bce_loss, batchsize=1, shuffle= False, cuda=args.cuda, print_every=args.print_every)
print("Test graph results: \tmean loss: %.4f \tmean acc: %4f" %(test_loss, test_acc))