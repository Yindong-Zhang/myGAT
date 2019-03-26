import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphDiffusedAttentionLayer, SpGraphAttentionLayer, GraphAttentionLayer, HigherOrderGraphAttentionLayer, ImprovedGraphAttentionLayer


class GAT(nn.Module):
    def __init__(self, nfeat, nclass, dropout, alpha, nheads_list, nhid_list, att_type):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.att_type= att_type

        if self.att_type == 'diffused':
            BaseLayer= GraphDiffusedAttentionLayer
        elif self.att_type == 'improved':
            BaseLayer= ImprovedGraphAttentionLayer
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
        x = F.dropout(x, self.dropout, training=self.training)
        for layer in self.layers:
            x = torch.cat([att(x, adj) for att in layer], dim= -1)
            x = F.dropout(x, self.dropout, training= self.training)
        x = self.out_att(x, adj)
        return F.log_softmax(x, dim=1)


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

