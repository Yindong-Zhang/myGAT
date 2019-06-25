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
from torch.utils.data import DataLoader
from dataset.make_dataset import get_dataset_and_split_planetoid, get_dataset, \
    get_train_val_test_split
from myDataset import SubGraph, custom_collate
from utils import accuracy, load_reddit
from models import GAT, SpGAT

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no_cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--gpu_ids', type= int, nargs= '+', default= [0, 1], help= "Specify GPU ids to move model on.")
parser.add_argument("--batchsize", type= int, default= 32, help = "Size of each min batch.")
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.05, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16, help='Number of hidden units.')
parser.add_argument('--nb_heads_1', type=int, default= 4, help='Number of head attentions in layer 1.')
parser.add_argument('--nb_heads_2', type=int, default= None, help='Number of head attentions in layer 2.')
parser.add_argument('--nb_heads_3', type=int, default= None, help='Number of head attentions in layer 3.')
parser.add_argument('--nb_heads_4', type=int, default= None, help='Number of head attentions in layer 4.')
parser.add_argument('--num_basis', type= int, default= 5, help= "Number of basis in second order approximation.")
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument('--order1_attention', action= 'store_true', default= False,
                    help= "Whether to use diffused attention in model")
parser.add_argument('--order2_attention', action= 'store_true', default= True,
                    help= "Whether to use improved attention in model")
parser.add_argument("--print_every", type= int, default= 5, help = "Print info for print_every epochs")

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

print(args)

configStr= "dataset~%s-hidden~%s-nheads_1~%s-nheads_2~%s-nheads_3~%s-nheads_4~%s-learning_rate~%s-weight_decay~%s-dropout~%s-train_size~%s-val_size~%s-order1_attention~%s-order2_attention~%s-patience~%s" \
    %("reddit", args.hidden, args.nb_heads_1, args.nb_heads_2, args.nb_heads_3, args.nb_heads_4, args.lr, args.weight_decay, args.dropout, args.train_size, args.val_size, args.order1_attention, args.order2_attention, args.patience)
dump_dir = os.path.join('./output', configStr)
if not os.path.exists(dump_dir):
    os.makedirs(dump_dir)

# Load data
# adj, features, labels, idx_train, idx_val, idx_test = get_dataset_and_split_planetoid(args.dataset, './data',)
adj, features, labels, idx_train, idx_val, idx_test = load_reddit()
print("load data conclude.")

# convert numpy/scipy to torch tensor
# adj = torch.FloatTensor(adj.todense())
# features = torch.FloatTensor(features.todense())
# convert one-hot encoding back.
# labels = torch.LongTensor(np.where(labels)[1])




att_type= None
if args.order1_attention and not args.order2_attention:
    att_type= 'order1'
elif args.order2_attention and not args.order1_attention:
    att_type= 'order2'
elif not args.order2_attention and not args.order1_attention:
    att_type= None
else:
    raise RuntimeError("Attention type hyperparameter not understood!!")

num_layers = 0
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
    num_layers = 5
    sample_per_layer = [15, 10, 5, 5, 5]
elif args.nb_heads_3 and args.nb_heads_2 and not args.nb_heads_4:
    model = GAT(nfeat=features.shape[-1],
                nhid_list=[args.hidden, ] * 3,
                nclass=int(labels.max()) + 1,
                dropout=args.dropout,
                nheads_list=[args.nb_heads_1, args.nb_heads_2, args.nb_heads_3, ],
                alpha=args.alpha,
                att_type= att_type,
                )
    num_layers = 4
    sample_per_layer = [25, 10, 5, 5]
elif args.nb_heads_2 and not args.nb_heads_3 and not args.nb_heads_4:
    model = GAT(nfeat=features.shape[-1],
                nhid_list=[args.hidden, ] * 2,
                nclass=int(labels.max()) + 1,
                dropout=args.dropout,
                nheads_list=[args.nb_heads_1, args.nb_heads_2],
                alpha=args.alpha,
                att_type= att_type,
                )
    num_layers = 3
    sample_per_layer = [25, 10, 5]
elif not args.nb_heads_2 and not args.nb_heads_3 and not args.nb_heads_4:
    model= GAT(nfeat=features.shape[-1],
                nhid_list=[args.hidden, ],
                nclass=int(labels.max()) + 1,
                dropout=args.dropout,
                nheads_list=[args.nb_heads_1, ],
                alpha=args.alpha,
                att_type= att_type,
                )
    num_layers = 2
    sample_per_layer = [25, 10]
else:
    raise RuntimeError("Model hyperparameters not understood!!")

dataset_train = SubGraph(adj, features, labels, idx_train, num_layers, sample_per_layer)
dataset_val = SubGraph(adj, features, labels, idx_val, num_layers, sample_per_layer)
dataset_test = SubGraph(adj, features, labels, idx_test, num_layers, sample_per_layer)
train_batch = DataLoader(dataset_train, batch_size= args.batchsize, shuffle= True, num_workers= 32, collate_fn= custom_collate)
val_batch = DataLoader(dataset_val, batch_size= args.batchsize, shuffle= True, num_workers= 32, collate_fn= custom_collate)
test_batch = DataLoader(dataset_test, batch_size= args.batchsize, shuffle= True, num_workers= 32, collate_fn= custom_collate)

print(model)

optimizer = optim.Adam(model.parameters(),
                       lr=args.lr,
                       weight_decay=args.weight_decay)

if args.cuda:
    devices= args.gpu_ids
    model.cuda()
    model= nn.DataParallel(model, device_ids= devices)


def loop_dataset(dataloader, optimizer,  print_every = 5, use_gpu = False):
    t = time.time()
    mean_loss = 0
    mean_acc = 0
    for it, data in enumerate(dataloader):
        (adj, features), labels = data
        labels = labels.squeeze()

        if use_gpu:
            features = features.cuda()
            adj = adj.cuda()
            labels = labels.cuda()

        output_logits = model(features, adj)
        # choose the only valid node
        output_logits = output_logits[:, 0, :].squeeze()

        output= F.log_softmax(output_logits, dim= -1)
        loss = F.nll_loss(output, labels)
        # print(output[0, :],labels[0])
        acc = accuracy(output, labels)

        # for multi gpu training
        loss = loss.mean()
        acc = acc.mean()

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_item = loss.data.cpu().item()
        acc_item = acc.data.cpu().item()

        mean_loss= (it * mean_loss + loss_item ) / (it + 1)
        mean_acc= (it * mean_acc + acc_item ) / (it + 1)

        if it % print_every == 0:
            print('loss: {:.4f}'.format(mean_loss),
                  'acc: {:.4f}'.format(mean_acc),
                  'time: {:.4f}s'.format(time.time() - t))

    return mean_loss, mean_acc


# Train model
t_total = time.time()
loss_values = []
bad_counter = 0
best = 1E9
best_epoch = 0
for epoch in range(args.epochs):
    # TODO:
    model.train()
    train_loss, train_acc = loop_dataset(train_batch, optimizer= optimizer, print_every= args.print_every,
                                         use_gpu= args.cuda)

    model.eval()
    val_loss, val_acc = loop_dataset(val_batch, optimizer= None, print_every= args.print_every, use_gpu= args.cuda)
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
test_loss, test_acc = loop_dataset(test_batch, optimizer= None, print_every= args.print_every, use_gpu= args.cuda)
print("Test set results:",
      "loss= {:.4f}".format(test_loss),
      "accuracy= {:.4f}".format(test_acc))

with open(os.path.join('./result', "%s.txt" %(configStr, )), 'a') as f:
    f.write("Test graph results: \tmean loss: %.4f \tmean acc: %4f \n" %(test_loss, test_acc))

