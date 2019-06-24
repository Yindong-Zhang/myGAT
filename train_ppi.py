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
from models import GAT, SpGAT, SumTailGAT, FullyConnectedGAT
from process_ppi import load_p2p, create_data

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no_cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--gpu_ids', type= int, nargs= '+', default= [0, 1], help= "Specify GPU ids to move model on.")
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default= 0, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default= 256, help='Number of hidden units.')
parser.add_argument('--nb_heads_1', type=int, default= 4, help='Number of head attentions in layer 1.')
parser.add_argument('--nb_heads_2', type=int, default= 4, help='Number of head attentions in layer 2.')
parser.add_argument('--nb_heads_3', type=int, default= 4, help='Number of head attentions in layer 3.')
parser.add_argument('--nb_heads_4', type=int, default= None, help='Number of head attentions in layer 4.')
parser.add_argument('--nheads_last', type= int, default= 6,
                    help= 'Number of heads in the last layer using means of their output for multilabel classification')
parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--batch_size', type= int, default= 2,
                    help= "Training batchsize for model")
parser.add_argument('--patience', type=int, default=1000, help='Patience')
parser.add_argument('--order1_attention', action= 'store_true', default= False,
                    help= "Whether to use diffused attention in model")
parser.add_argument('--order2_attention', action= 'store_true', default= False,
                    help= "Whether to use improved attention in model")
parser.add_argument('--print_every', type= int, default= 1,
                    help= "Interval to print results.")

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

print(args)

configStr= "hidden~%s-nheads_1~%s-nheads_2~%s-nheads_3~%s-nheads_4~%s-nheads_last~%s-learning_rate~%s-weight_decay~%s-order1_attention~%s-order2_attention~%s-patience~%s" \
    %(args.hidden, args.nb_heads_1, args.nb_heads_2, args.nb_heads_3, args.nb_heads_4, args.nheads_last, args.lr, args.weight_decay, args.order1_attention, args.order2_attention, args.patience)
dump_dir = os.path.join('./output', configStr)
if not os.path.exists(dump_dir):
    os.makedirs(dump_dir)

# Load data
train_adj, val_adj, test_adj, \
train_feat, val_feat, test_feat, \
train_labels, val_labels, test_labels, \
train_nodes, val_nodes, test_nodes, \
tr_msk, vl_msk, ts_msk = load_p2p('./data/ppi')

# load test data
# train_adj, val_adj, test_adj, \
# train_feat, val_feat, test_feat, \
# train_labels, val_labels, test_labels, \
# train_nodes, val_nodes, test_nodes, \
# tr_msk, vl_msk, ts_msk = create_data()
#

att_type= None
if args.order1_attention and not args.order2_attention:
    att_type= 'order1'
elif args.order2_attention and not args.order1_attention:
    att_type= 'order2'
elif not args.order2_attention and not args.order1_attention:
    att_type= None
else:
    raise RuntimeError("Attention type hyperparameter not understood!!")

nb_features= train_feat.shape[-1]
nb_class= train_labels.shape[-1]
# Model and optimizer
if args.nb_heads_4 and args.nb_heads_3 and args.nb_heads_2:
    model = FullyConnectedGAT(nfeat=nb_features,
                nhid_list=[args.hidden, ] * 4,
                nclass=nb_class,
                dropout=args.dropout,
                nheads_list=[args.nb_heads_1, args.nb_heads_2, args.nb_heads_3, args.nb_heads_4 ],
                alpha=args.alpha,
                att_type = att_type,
                )
elif args.nb_heads_3 and args.nb_heads_2 and not args.nb_heads_4:
    model = FullyConnectedGAT(nfeat= nb_features,
                nhid_list=[args.hidden, ] * 3,
                nclass= nb_class,
                dropout=args.dropout,
                nheads_list=[args.nb_heads_1, args.nb_heads_2, args.nb_heads_3, ],
                       alpha=args.alpha,
                att_type= att_type,
                )
elif args.nb_heads_2 and not args.nb_heads_3 and not args.nb_heads_4:
    model = FullyConnectedGAT(nfeat= nb_features,
                nhid_list=[args.hidden, ] * 2,
                nclass= nb_class,
                dropout=args.dropout,
                nheads_list=[args.nb_heads_1, args.nb_heads_2],
                       alpha=args.alpha,
                att_type= att_type,
                )
elif not args.nb_heads_2 and not args.nb_heads_3 and not args.nb_heads_4:
    model = FullyConnectedGAT(nfeat=nb_features,
                          nhid_list=[args.hidden, ],
                          nclass=nb_class,
                          dropout=args.dropout,
                          nheads_list=[args.nb_heads_1, ],
                       alpha=args.alpha,
                       att_type=att_type,
                          )
else:
    raise RuntimeError("Model hyperparameters not understood!!")

print("Model: \n", model)

if args.cuda:
    devices= args.gpu_ids
    model.cuda()
    model= nn.DataParallel(model, device_ids= devices)

optimizer = optim.Adam(model.parameters(),
                       lr=args.lr,
                       weight_decay=args.weight_decay)


def prepare_input(batch_graphs, gpu):
    # restrict batchsize to 1
    batch_features, batch_adj, batch_labels, batch_nodes, batch_masks= batch_graphs
    max_nodes= batch_adj.shape[-1]
    batchsize= batch_features.shape[0]
    # features, adj, labels= batch_features[:batch_nodes, :], batch_adj[:batch_nodes, :batch_nodes], batch_labels[:batch_nodes, :]
    batch_adj[torch.eye(max_nodes).repeat(batchsize, 1, 1).byte()]= 0
    if gpu:
        batch_features= batch_features.cuda()
        batch_adj= batch_adj.cuda()
        batch_labels= batch_labels.cuda()
        batch_masks = batch_masks.cuda()
    return batch_features, batch_adj, batch_masks, batch_labels


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
        selected_idxes = sample_idxes[it * batchsize : (it + 1) * batchsize]

        # restrict batchsize to 1
        batch_features  = features[selected_idxes]
        batch_adj       = adj[selected_idxes]
        batch_labels    = labels[selected_idxes]
        batch_nodes     = nb_nodes[selected_idxes]
        batch_masks     = masks[selected_idxes]

        batch_graphs= (batch_features, batch_adj, batch_labels, batch_nodes, batch_masks)
        batch_features, batch_adj, batch_masks, batch_labels= prepare_input(batch_graphs, cuda)

        batch_logits= classifier(batch_features, batch_adj, batch_masks)
        loss = criterion(batch_logits, batch_labels)

        batch_prob= torch.sigmoid(batch_logits)
        batch_pred= (batch_prob > 0.5)
        batch_pred= batch_pred.float()

        acc_item= batch_pred.eq(batch_labels)
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
            print('mean_loss: %5.3f \t mean_acc: %5.3f \t %s \t %s' % (mean_loss, mean_acc, batch_pred[0, 0, :8].cpu().numpy(), batch_labels[0, 0, :8].cpu().numpy()))

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
    train_loss, train_acc= loop_dataset(train_data, model, criterion= bce_loss, optimizer= optimizer, batchsize= args.batch_size, shuffle= True, cuda= args.cuda, print_every= args.print_every)
    print("Training epoch %s: \t loss: %.4f \t acc: %.4f.\n" %(epoch, train_loss, train_acc))

    print("Validation epoch %s:" %(epoch, ))
    model.eval()
    val_loss, val_acc= loop_dataset(val_data, model, criterion= bce_loss, batchsize= args.batch_size, shuffle= False, cuda= args.cuda, print_every= args.print_every)
    print("Validation epoch %s: \t loss: %.4f \t acc: %.4f.\n" %(epoch, val_loss, val_acc))
    loss_values.append(val_loss)

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
        filename = os.path.split(file)[-1]
        epoch_nb = int(filename.split('.')[0])
        if epoch_nb < best_epoch:
            os.remove(file)

files = glob.glob(os.path.join(dump_dir, '*.pkl'))
for file in files:
    filename= os.path.split(file)[-1]
    epoch_nb = int(filename.split('.')[0])
    if epoch_nb > best_epoch:
        os.remove(file)

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

print("Model arguments: ")
print(args)
print(model)

# Restore best model
print('Loading {}th epoch'.format(best_epoch))
model.load_state_dict(torch.load(os.path.join(dump_dir, '{}.pkl'.format(best_epoch))))
model.eval()
test_loss, test_acc = loop_dataset(test_data, model, criterion= bce_loss, batchsize= args.batch_size, shuffle= False, cuda=args.cuda, print_every=args.print_every)
print("Test graph results: \tmean loss: %.4f \tmean acc: %4f" %(test_loss, test_acc))

with open(os.path.join('./result', "%s.txt" %(configStr, )), 'a') as f:
    f.write("Test graph results: \tmean loss: %.4f \tmean acc: %4f \n" %(test_loss, test_acc))