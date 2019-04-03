import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, activation):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features), dtype= torch.float))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a_1 = nn.Parameter(torch.zeros(size=(out_features, 1), dtype= torch.float))
        nn.init.xavier_uniform_(self.a_1.data, gain=1) # how to choose a proper gain number
        self.a_2 = nn.Parameter(torch.zeros(size=(out_features, 1), dtype= torch.float))
        nn.init.xavier_uniform_(self.a_2.data, gain=1)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

        self.activation= activation

    def forward(self, input, adj, nd_flags):
        h = torch.matmul(input, self.W)

        logit_1= torch.matmul(h, self.a_1)
        logit_2= torch.matmul(h, self.a_2)
        logits= logit_1 + logit_2.permute(0, 2, 1)
        e= self.leakyrelu(logits)

        zero_vec = -9e15* e.new_tensor([1., ])
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim= -1)
        # attention= torch.where(adj > 0, attention, attention.new_tensor([0., ]))
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_out = torch.bmm(attention, h)

        if self.activation != None:
            return self.activation(h_out)
        else:
            return h_out

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

# TODO: change to batch training
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

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class Order1GraphAttentionLayer(nn.Module):
    """
    Improved GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, activation):
        super(Order1GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

        self.W= nn.Parameter(torch.zeros(size=(in_features, out_features), ))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.W_1= nn.Parameter(torch.zeros(size=(in_features, out_features), ))
        nn.init.xavier_uniform_(self.W_1.data, gain=1.414)

        self.W_2 = nn.Parameter(torch.zeros(size=(in_features, out_features), ))
        nn.init.xavier_uniform_(self.W_2.data, gain=1.414)

        self.a_1 = nn.Parameter(torch.zeros(size=(out_features, 1), dtype=torch.float))
        nn.init.xavier_uniform_(self.a_1.data, gain=1.414)
        self.a_2 = nn.Parameter(torch.zeros(size=(out_features, 1), dtype=torch.float))
        nn.init.xavier_uniform_(self.a_2.data, gain=1.414)

        self.a_12 = nn.Parameter(torch.zeros(size=(out_features, out_features)))
        nn.init.xavier_uniform_(self.a_12.data, gain=1.414)

        self.W_xy= nn.Parameter(torch.zeros(size= (out_features, 1)))
        nn.init.xavier_uniform_(self.W_xy.data, gain= 1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

        self.activation= activation


    def forward(self, input, adj, nd_flags):
        h = torch.matmul(input, self.W)

        Ax = torch.matmul(h, self.a_1)
        Ay = torch.matmul(h, self.a_2)
        A_xy_1= torch.matmul(h, self.a_12)
        A_xy= torch.matmul(A_xy_1, h.permute(0, 2, 1))

        Ax_prime= torch.matmul(nd_flags, Ax.permute(0, 2, 1))
        nd_flags_T= nd_flags.permute(0, 2, 1)
        Ay_prime= torch.matmul(Ay, nd_flags_T)
        logits = Ax_prime + Ay_prime + A_xy

        e = self.leakyrelu(logits)
        zero_vec = -9e15 * e.new_tensor([1., ])
        e = torch.where(adj > 0, e, zero_vec)

        attention = F.softmax(e, dim=-1)
        # attention= torch.where(adj > 0, attention, attention.new_tensor([0., ]))
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, input)

        h_1= torch.matmul(input, self.W_1)
        h_2= torch.matmul(h_prime, self.W_2)
        h_out= h_1 + h_2

        if not self.activation:
            return h_out
        else:
            return self.activation(h_out)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'



class Order2GraphAttentionLayer(nn.Module):
    """
    Improved GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, activation):
        super(Order2GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

        self.W= nn.Parameter(torch.zeros(size=(in_features, out_features), ))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.W_1= nn.Parameter(torch.zeros(size=(in_features, out_features), ))
        nn.init.xavier_uniform_(self.W_1.data, gain=1.414)

        self.W_2 = nn.Parameter(torch.zeros(size=(in_features, out_features), ))
        nn.init.xavier_uniform_(self.W_2.data, gain=1.414)

        self.a_1 = nn.Parameter(torch.zeros(size=(out_features, 1), dtype=torch.float))
        nn.init.xavier_uniform_(self.a_1.data, gain=1.414)
        self.a_2 = nn.Parameter(torch.zeros(size=(out_features, 1), dtype=torch.float))
        nn.init.xavier_uniform_(self.a_2.data, gain=1.414)

        self.a_12 = nn.Parameter(torch.zeros(size=(out_features, out_features)))
        nn.init.xavier_uniform_(self.a_12.data, gain=1.414)

        self.W_xy= nn.Parameter(torch.zeros(size= (out_features, 1)))
        nn.init.xavier_uniform_(self.W_xy.data, gain= 1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

        self.bilinear= nn.Bilinear(in1_features= in_features, in2_features= in_features, out_features= out_features)

        self.activation= activation

    def forward(self, input, adj, nd_flags):
        h = torch.matmul(input, self.W)

        Ax = torch.matmul(h, self.a_1)
        Ay = torch.matmul(h, self.a_2)
        A_xy_1= torch.matmul(h, self.a_12)
        A_xy= torch.matmul(A_xy_1, h.permute(0, 2, 1))
        # A_xy = torch.chain_matmul(h, self.a_12, h.permute(1, 0))

        Ax_prime= torch.matmul(nd_flags, Ax.permute(0, 2, 1))
        nd_flags_T= nd_flags.permute(0, 2, 1)
        Ay_prime= torch.matmul(Ay, nd_flags_T)
        logits = Ax_prime + Ay_prime + A_xy

        e = self.leakyrelu(logits)
        zero_vec = -9e15 * e.new_tensor([1., ])
        e = torch.where(adj > 0, e, zero_vec)

        attention = F.softmax(e, dim=-1)
        # attention= torch.where(adj > 0, attention, attention.new_tensor([0., ]))
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, input)

        h_1= torch.matmul(input, self.W_1)
        h_2= torch.matmul(h_prime, self.W_2)
        h_12= self.bilinear(input, h_prime)
        h_out= h_1 + h_2 + h_12

        if not self.activation:
            return h_out
        else:
            return self.activation(h_out)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'



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
