import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features), dtype= torch.float))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a_1 = nn.Parameter(torch.zeros(size=(out_features, 1), dtype= torch.float))
        nn.init.xavier_uniform_(self.a_1.data, gain=1.414)
        self.a_2 = nn.Parameter(torch.zeros(size=(out_features, 1), dtype= torch.float))
        nn.init.xavier_uniform_(self.a_2.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.matmul(input, self.W)

        logit_1= torch.matmul(h, self.a_1)
        logit_2= torch.matmul(h, self.a_2)
        logits= logit_1 + logit_2.permute(1, 0)
        e= self.leakyrelu(logits)
        zero_vec = -9e15* e.new_tensor([1., ])
        e = torch.where(adj > 0, e, zero_vec)

        attention = F.softmax(e, dim= -1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_out = torch.mm(attention, h)

        return F.elu(h_out)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class GraphDiffusedAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha):
        super(GraphDiffusedAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features), dtype= torch.float))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a_1 = nn.Parameter(torch.zeros(size=(out_features, 1), dtype= torch.float))
        nn.init.xavier_uniform_(self.a_1.data, gain=1.414)
        self.a_2 = nn.Parameter(torch.zeros(size=(out_features, 1), dtype= torch.float))
        nn.init.xavier_uniform_(self.a_2.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.matmul(input, self.W)

        logit_1= torch.matmul(h, self.a_1)
        logit_2= torch.matmul(h, self.a_2)
        logits= logit_1 + logit_2.permute(1, 0)
        e= self.leakyrelu(logits)
        zero_vec = -9e15* e.new_tensor([1., ])
        e = torch.where(adj > 0, e, zero_vec)


        mean_h = torch.mean(h, dim= 0, keepdim= True)
        h_all= torch.cat([h, mean_h], 0)

        glob_logit_2= torch.mm(mean_h, self.a_2)
        glob_logit= logit_1 + glob_logit_2
        e_diffused= self.leakyrelu(glob_logit)
        e_all= torch.cat([e, e_diffused], -1)

        attention = F.softmax(e_all, dim= -1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_out = torch.mm(attention, h_all)

        return F.elu(h_out)


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)

    
class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)
                
        self.a = nn.Parameter(torch.zeros(size=(1, 2*out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def forward(self, input, adj):
        N = input.size()[0]
        edge = adj.nonzero().t()

        h = torch.mm(input, self.W)
        # h: N x out
        assert not torch.isnan(h).any()

        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        # edge: 2*D x E

        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N,1)).cuda())
        # e_rowsum: N x 1

        edge_e = self.dropout(edge_e)
        # edge_e: E

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out
        
        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()

        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
