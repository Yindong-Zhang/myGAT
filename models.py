import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphDiffusedAttentionLayer, SpGraphAttentionLayer, GraphAttentionLayer, \
    Order1GraphAttentionLayer, Order2GraphAttentionLayer, MLPGraphAttentionLayer
from functools import reduce

class GAT(nn.Module):
    def __init__(self, nfeat, nclass, dropout, alpha, nheads_list, nhid_list, att_type, num_basis):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.att_type= att_type

        if self.att_type == 'diffused':
            BaseLayer= GraphDiffusedAttentionLayer
        elif self.att_type == 'order2':
            BaseLayer= Order2GraphAttentionLayer
        elif self.att_type == 'order1':
            BaseLayer= Order1GraphAttentionLayer
        elif self.att_type == 'mlp':
            BaseLayer = MLPGraphAttentionLayer
        elif self.att_type == None:
            BaseLayer= GraphAttentionLayer
        else:
            raise RuntimeError("model attention type not understood.")
        print("Use attention type %s." %(BaseLayer, ))

        assert len(nheads_list) == len(nhid_list), "Length of nheads should be equal to length of nhidden list"
        nlayers= len(nheads_list)
        self.layers= nn.ModuleList()

        self.layers.append(nn.ModuleList([BaseLayer(nfeat, nhid_list[0], dropout=dropout, alpha=alpha, num_basis = num_basis, activation= F.elu) for _ in range(nheads_list[0])]))
        for l in range(1, nlayers):
            self.layers.append(nn.ModuleList([BaseLayer(nhid_list[l - 1] * nheads_list[l - 1], nhid_list[l], dropout=dropout, alpha=alpha, num_basis= num_basis, activation= F.elu)
                                              for _ in range(nheads_list[l]) ] ) )
        self.out_att = BaseLayer(nhid_list[-1] * nheads_list[-1], nclass, dropout=dropout, alpha=alpha, num_basis = num_basis, activation= lambda x: x)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        for layer in self.layers:
            x = torch.cat([att(x, adj) for att in layer], dim= -1)
            x = F.dropout(x, self.dropout, training= self.training)
        x = self.out_att(x, adj)
        return x

class SumTailGAT(nn.Module):
    def __init__(self, nfeat, nclass, dropout, alpha, nheads_list, nhid_list, nheads_last, att_type, num_basis):
        """Dense version of GAT."""
        super(SumTailGAT, self).__init__()
        self.dropout = dropout
        self.att_type= att_type

        if self.att_type == 'diffused':
            BaseLayer= GraphDiffusedAttentionLayer
        elif self.att_type == 'order2':
            BaseLayer= Order2GraphAttentionLayer
        elif self.att_type == 'order1':
            BaseLayer= Order1GraphAttentionLayer
        elif self.att_type == 'mlp':
            BaseLayer = MLPGraphAttentionLayer
        elif self.att_type == None:
            BaseLayer= GraphAttentionLayer
        else:
            raise RuntimeError("model attention type not understood.")
        print("Use attention type %s." %(BaseLayer, ))

        assert len(nheads_list) == len(nhid_list), "Length of nheads should be equal to length of nhidden list"
        nlayers= len(nheads_list)
        self.layers= nn.ModuleList()

        self.layers.append(nn.ModuleList([BaseLayer(nfeat, nhid_list[0], dropout=dropout, alpha=alpha, num_basis = num_basis, activation= F.elu) for _ in range(nheads_list[0])]))
        for l in range(1, nlayers):
            self.layers.append(nn.ModuleList([BaseLayer(nhid_list[l - 1] * nheads_list[l - 1], nhid_list[l], dropout=dropout, alpha=alpha, num_basis= num_basis, activation= F.elu)
                                              for _ in range(nheads_list[l]) ] ) )
        self.out_att = nn.ModuleList([BaseLayer(nhid_list[-1] * nheads_list[-1], nclass, dropout=dropout, alpha=alpha, activation= None) for _ in range(nheads_last)])

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        for layer in self.layers:
            x = torch.cat([att(x, adj) for att in layer], dim= -1)
            x = F.dropout(x, self.dropout, training= self.training)
        x_list = [att(x, adj) for att in  self.out_att]
        x= reduce(torch.add, x_list) / len(self.out_att)

        return x

class FullyConnectedGAT(nn.Module):
    def __init__(self, nfeat, nclass, dropout, alpha, nheads_list, nhid_list, att_type, num_basis, ):
        """Dense version of GAT."""
        super(FullyConnectedGAT, self).__init__()
        self.dropout = dropout
        self.att_type= att_type

        if self.att_type == 'order2':
            BaseLayer= Order2GraphAttentionLayer
        elif self.att_type == 'order1':
            BaseLayer= Order1GraphAttentionLayer
        elif self.att_type == 'mlp':
            BaseLayer = MLPGraphAttentionLayer
        elif self.att_type == None:
            BaseLayer= GraphAttentionLayer
        else:
            raise RuntimeError("model attention type not understood.")
        print("Use attention type %s." %(BaseLayer, ))

        assert len(nheads_list) == len(nhid_list), "Length of nheads should be equal to length of nhidden list"
        nlayers= len(nheads_list)
        self.layers= nn.ModuleList()

        self.layers.append(nn.ModuleList([BaseLayer(nfeat, nhid_list[0], dropout=dropout, alpha=alpha, num_basis = num_basis, activation= F.elu) for _ in range(nheads_list[0])]))
        for l in range(1, nlayers):
            self.layers.append(nn.ModuleList([BaseLayer(nhid_list[l - 1] * nheads_list[l - 1], nhid_list[l], dropout=dropout, alpha=alpha, num_basis= num_basis, activation= F.elu)
                                              for _ in range(nheads_list[l]) ] ) )
        self.linear= nn.Linear(nhid_list[-1] * nheads_list[-1], nclass)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        for layer in self.layers:
            x = torch.cat([att(x, adj) for att in layer], dim= -1)
            x = F.dropout(x, self.dropout, training= self.training)
        x= self.linear(x)
        return x

# TODO:
class ResMultiLabelGAT(nn.Module):
    def __init__(self, nfeat, nclass, dropout, alpha, nheads_list, nhid_list, att_type):
        """Dense version of GAT."""
        super(ResMultiLabelGAT, self).__init__()
        self.dropout = dropout
        self.att_type= att_type

        if self.att_type == 'diffused':
            BaseLayer= GraphDiffusedAttentionLayer
        elif self.att_type == 'improved':
            BaseLayer= Order2GraphAttentionLayer
        elif self.att_type == None:
            BaseLayer= GraphAttentionLayer
        else:
            raise RuntimeError("model attention type not understood.")
        print("Use attention type %s." %(BaseLayer, ))

        assert len(nheads_list) == len(nhid_list), "Length of nheads should be equal to length of nhidden list"
        nlayers= len(nheads_list)
        self.layers= nn.ModuleList()

        self.layers.append(nn.ModuleList([BaseLayer(nfeat, nhid_list[0], dropout=dropout, alpha=alpha) for _ in range(nheads_list[0])]))
        for l in range(1, nlayers):
            self.layers.append(nn.ModuleList([BaseLayer(nhid_list[l - 1] * nheads_list[l - 1], nhid_list[l], dropout=dropout, alpha=alpha)
                                              for _ in range(nheads_list[l]) ] ) )
        self.out_att = BaseLayer(nhid_list[-1] * nheads_list[-1], nclass, dropout=dropout, alpha=alpha)

    def forward(self, x, adj):
        x_list= [x, ]
        for layer in self.layers:
            x_in = torch.cat(x_list, dim= -1)
            x_in = F.dropout(x_in, self.dropout, training=self.training)
            x_out = torch.cat([att(x_in, adj) for att in layer], dim= -1)
            x_list.append(x_out)
        x = F.dropout(x, self.dropout, training= self.training)
        x = self.out_att(x, adj)
        return x

class SpGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        self.dropout = dropout

        self.attentions = [SpGraphAttentionLayer(nfeat, 
                                                 nhid, 
                                                 dropout=dropout, 
                                                 alpha=alpha, 
                                                 concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = SpGraphAttentionLayer(nhid * nheads, 
                                             nclass, 
                                             dropout=dropout, 
                                             alpha=alpha, 
                                             concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)

