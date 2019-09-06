import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math

class GraphConvolutionLayer(nn.Module):
    """
    Base graph convolution layer.
    """
    def __init__(self):
        super(GraphConvolutionLayer, self).__init__()
        pass

    def forward(self, feat, adj):
        """
        a overview of logic, can be override
        :param adj:
        :param feat:
        :return:
        """
        h_prime = self._aggregate(feat, adj)
        return self._update(feat, h_prime)

    def _aggregate(self, feat, adj):
        print("Unimplemented!")

    def _update(self, feat, feat_prime):
        print("Unimplemented!")

class BaseGraphAttentionLayer(GraphConvolutionLayer):
    def __init__(self):
        super(BaseGraphAttentionLayer, self).__init__()
        pass

    def _attention(self, feat, adj):
        print("Unimplemented!")

    def _aggregate(self, feat, adj):
        """
        a overview of logic, can be override.
        :param adj: 
        :param feat: 
        :return: 
        """
        weight = self._attention(feat, adj)
        h_prime = torch.matmul(weight, feat)
        return h_prime



class GraphAttentionLayer(BaseGraphAttentionLayer):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, activation, residual_connection = False, num_basis= True):
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

    def _attention(self, h, adj):

        logit_1= torch.matmul(h, self.a_1)
        logit_2= torch.matmul(h, self.a_2)
        logits= logit_1 + logit_2.permute(0, 2, 1)
        e= self.leakyrelu(logits)

        zero_vec = -9e15* e.new_tensor([1., ])
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim= -1)
        return attention

    def _aggregate(self, feat, adj):
        h = torch.matmul(feat, self.W)
        attention = self._attention(h, adj)
        attention = F.dropout(attention, self.dropout, training=self.training)

        h_out = torch.bmm(attention, h)
        return h_out

    def _update(self, feat, feat_prime):
        if self.activation != None:
            return self.activation(feat_prime)
        else:
            return feat_prime

        return feat_prime


    def forward(self, input, adj):
        h_prime = self._aggregate(input, adj)
        return self._update(input, h_prime)


    def extra_repr(self):
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

# TODO:
class Order1GraphMLPAttentionLayer(nn.Module):
    """
    Improved GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, activation, num_basis = 5):
        super(Order1GraphMLPAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

        self.W_1= nn.Parameter(torch.zeros(size=(in_features, out_features), ))
        nn.init.xavier_uniform_(self.W_1.data, gain=1.414)

        self.W_2 = nn.Parameter(torch.zeros(size=(in_features, out_features), ))
        nn.init.xavier_uniform_(self.W_2.data, gain=1.414)

        self.attention_layer = BiInteractionLayer()

        self.leakyrelu = nn.LeakyReLU(self.alpha)

        self.activation= activation

    def _attention(self, feat, adj):
        h = torch.matmul(feat, self.W)

        Ax = torch.matmul(h, self.a_1)
        Ay = torch.matmul(h, self.a_2)
        A_xy_1= torch.matmul(h, self.a_12)
        A_xy= torch.matmul(A_xy_1, h.permute(0, 2, 1))

        # Ax_prime= torch.matmul(nd_flags, Ax.permute(0, 2, 1))
        # nd_flags_T= nd_flags.permute(0, 2, 1)
        # Ay_prime= torch.matmul(Ay, nd_flags_T)
        Ax_prime= Ax.permute(0, 2, 1)
        Ay_prime= Ay
        logits = Ax_prime + Ay_prime + A_xy

        e = self.leakyrelu(logits)
        zero_vec = -9e15 * e.new_tensor([1., ])
        e = torch.where(adj > 0, e, zero_vec)

        attention = F.softmax(e, dim=-1)
        # attention= torch.where(adj > 0, attention, attention.new_tensor([0., ]))
        attention = F.dropout(attention, self.dropout, training=self.training)
        return attention

    def _aggregate(self, feat, adj):
        attention = self._attention(feat, adj)
        h_prime = torch.matmul(attention, feat)
        return h_prime

    def _update(self, feat, feat_agg):
        h_1 = torch.matmul(feat, self.W_1)
        h_2 = torch.matmul(feat_agg, self.W_2)
        h_out = h_1 + h_2

        if not self.activation:
            return h_out
        else:
            return self.activation(h_out)

    def forward(self, feat, adj):
        feat_agg = self._aggregate(feat, adj)
        return self._update(feat, feat_agg)

    def extra_repr(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class Order1GraphAttentionLayer(nn.Module):
    """
    Improved GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, activation, num_basis = 5):
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
        bound = 1 / math.sqrt(self.a_12.size(0))
        nn.init.uniform_(self.a_12, -bound, bound)
        # nn.init.xavier_uniform_(self.a_12.data, gain=1.414)

        self.W_xy= nn.Parameter(torch.zeros(size= (out_features, 1)))
        nn.init.xavier_uniform_(self.W_xy.data, gain= 1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

        self.activation= activation

    def _attention(self, feat, adj):
        h = torch.matmul(feat, self.W)

        Ax = torch.matmul(h, self.a_1)
        Ay = torch.matmul(h, self.a_2)
        # A_xy_1= torch.matmul(h, self.a_12)
        # A_xy= torch.matmul(A_xy_1, h.permute(0, 2, 1))

        Ax_prime= Ax.permute(0, 2, 1)
        Ay_prime= Ay
        logits = Ax_prime + Ay_prime
        # logits = Ax_prime + Ay_prime + A_xy

        e = self.leakyrelu(logits)
        zero_vec = -9e15 * e.new_tensor([1., ])
        e = torch.where(adj > 0, e, zero_vec)

        attention = F.softmax(e, dim=-1)
        # attention= torch.where(adj > 0, attention, attention.new_tensor([0., ]))
        attention = F.dropout(attention, self.dropout, training=self.training)
        return attention

    def _aggregate(self, feat, adj):
        attention = self._attention(feat, adj)
        h_prime = torch.matmul(attention, feat)
        return h_prime

    def _update(self, feat, feat_agg):
        # h_1 = torch.matmul(feat, self.W_1)
        h_2 = torch.matmul(feat_agg, self.W)
        h_out = h_2
        # h_out = h_1 + h_2
        if not self.activation:
            return h_out
        else:
            return self.activation(h_out)

    def forward(self, feat, adj):
        feat_agg = self._aggregate(feat, adj)
        return self._update(feat, feat_agg)

    def extra_repr(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class SVDBilinear(nn.Module):
    """
    my bilinear matmul but reducing parameter dimension using peusodu-SVD
    """
    def __init__(self, num_basis, in1_features, in2_features, out_features):
        super(SVDBilinear, self).__init__()
        self.num_basis = num_basis
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        self.left_singular = nn.Parameter(torch.Tensor(out_features, in1_features, num_basis))
        self.right_singular = nn.Parameter(torch.Tensor(out_features, num_basis, in2_features))
        self.diag = nn.Parameter(torch.Tensor(out_features, 1, num_basis))
        self.reset_parameter()

    def reset_parameter(self):
        init.xavier_uniform_(self.left_singular, gain = 1.414)
        init.xavier_uniform_(self.right_singular, gain= 1.414)
        init.normal_(self.diag, 0, 1/ math.sqrt(self.diag.size(-1)))

    def forward(self, in1, in2):
        us = self.left_singular * self.diag
        usv = torch.matmul(us, self.right_singular)
        return F.bilinear(in1, in2, weight= usv)

    def __repr__(self):
        return "SVDBilinear Layer: in1_features={}, in2_features={}, out_features={}, num_basis={}".format(
            self.in1_features, self.in2_features, self.out_features, self.num_basis
        )

class EmbedBilinear(nn.Module):
    """
    binlinear module but reduce dimenion first to reduce complexity.
    """
    def __init__(self, embed_size, in1_features, in2_features, out_features, bias = False):
        super(EmbedBilinear, self).__init__()
        self.embed_size = embed_size
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        self.use_bias = bias
        self.left_embed_layer = nn.Linear(in_features= in1_features, out_features = embed_size, bias = bias)
        self.right_embed_layer = nn.Linear(in_features= in2_features, out_features = embed_size, bias = bias)
        self.Bilinear = nn.Bilinear(in1_features= embed_size, in2_features = embed_size, out_features = out_features, bias= bias)
        self.reset_parameters()

    def reset_parameters(self):
        self.left_embed_layer.reset_parameters()
        self.right_embed_layer.reset_parameters()
        self.Bilinear.reset_parameters()

    def forward(self, in1, in2):
        embed1 = self.left_embed_layer(in1)
        embed2 = self.right_embed_layer(in2)
        return self.Bilinear(embed1, embed2)

    def __repr__(self):
        return "EmbedBilinear Layer: in1_features={}, in2_features={}, out_features={}, embed_size={}".format(
            self.in1_features, self.in2_features, self.out_features, self.embed_size
        )

class BiInteractionLayer(nn.Module):
    def __init__(self, in1_features, in2_features, out_features, embed_size, intermediate_size= None, activation = F.relu, use_bias = True):
        """

        :param in1_features:
        :param in2_features:
        :param out_features:
        :param embed_size: embed size specific embedding vector size of input features
        :param intermediate: a list specify intermediate size in mlp
        """
        super(BiInteractionLayer, self).__init__()
        self.in1_features= in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        self.embed_size = embed_size
        self.embed_layer_1 = nn.Linear(in1_features, embed_size, bias = True)
        self.embed_layer_2 = nn.Linear(in2_features, embed_size, bias = True)
        self.activation = activation
        self.bias= use_bias
        self.interaction = nn.ModuleList()
        if not intermediate_size:
            self.interaction.append(nn.Linear(embed_size * 2, out_features, bias= True))
        else:
            self.interaction.append(nn.Linear(embed_size * 2, intermediate_size[0], bias= True))
            for i in range(1, len(intermediate_size)):
                self.interaction.append(nn.Linear(intermediate_size[i - 1], intermediate_size[i], bias= True))
            self.interaction.append(nn.Linear(intermediate_size[-1], out_features, bias= True))

            self.num_layers= len(self.interaction)
        self.reset_parameters()

    def reset_parameters(self):
        self.embed_layer_1.reset_parameters()
        self.embed_layer_2.reset_parameters()
        for layer in self.interaction:
            layer.reset_parameters()

    def forward(self, in1, in2):
        embed1 = self.activation(self.embed_layer_1(in1))
        embed2 = self.activation(self.embed_layer_2(in2))
        embed_concat = torch.cat([embed1, embed2], -1)
        for layer in self.interaction:
            embed_concat = self.activation(layer(embed_concat))
        return embed_concat

    def extra_repr(self):
        return "Multi-layer perception: in1_features: %s, in2_features: %s, out_features: %s, bias: %s, layers: %s, activation: %s" \
    %(self.in1_features, self.in2_features, self.out_features, self.bias, self.num_layers, self.activation)

class MLPGraphAttentionLayer(BaseGraphAttentionLayer):
    """
    Improved GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, activation, num_basis = 5):
        super(MLPGraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.num_basis = num_basis

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

        self.biInteraction= BiInteractionLayer(in1_features= in_features, in2_features= in_features, out_features= out_features,
                                               embed_size= int(math.sqrt(in_features)), intermediate_size= [out_features, ],
                                               activation= self.leakyrelu)

        self.activation= activation

    def _attention(self, feat, adj):
        h = torch.matmul(feat, self.W)

        Ax = torch.matmul(h, self.a_1)
        Ay = torch.matmul(h, self.a_2)
        A_xy_1= torch.matmul(h, self.a_12)
        A_xy= torch.matmul(A_xy_1, h.permute(0, 2, 1))
        # A_xy = torch.chain_matmul(h, self.a_12, h.permute(1, 0))

        # Ax_prime= torch.matmul(nd_flags, Ax.permute(0, 2, 1))
        # nd_flags_T= nd_flags.permute(0, 2, 1)
        # Ay_prime= torch.matmul(Ay, nd_flags_T)
        Ax_prime= Ax.permute(0, 2, 1)
        Ay_prime= Ay
        logits = Ax_prime + Ay_prime + A_xy

        e = self.leakyrelu(logits)
        zero_vec = -9e15 * e.new_tensor([1., ])
        e = torch.where(adj > 0, e, zero_vec)

        attention = F.softmax(e, dim=-1)
        # attention= torch.where(adj > 0, attention, attention.new_tensor([0., ]))
        attention = F.dropout(attention, self.dropout, training=self.training)
        return attention

    def _aggregate(self, feat, adj):
        attention = self._attention(feat, adj)
        h_prime = torch.matmul(attention, feat)
        return h_prime

    def _update(self, feat, feat_prime):
        h_out = self.biInteraction(feat, feat_prime)
        return h_out

    def forward(self, feat, adj):
        feat_prime = self._aggregate(feat, adj)
        h_out = self._update(feat, feat_prime)
        if not self.activation:
            return h_out
        else:
            return self.activation(h_out)

    def extra_repr(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class Order2GraphAttentionLayer(nn.Module):
    """
    Improved GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, activation, num_basis = 5):
        super(Order2GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.num_basis = num_basis

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

        self.bilinear= EmbedBilinear(num_basis, in1_features= in_features, in2_features= in_features, out_features= out_features)

        self.activation= activation

    def _attention(self, feat, adj):
        h = torch.matmul(feat, self.W)

        Ax = torch.matmul(h, self.a_1)
        Ay = torch.matmul(h, self.a_2)
        A_xy_1= torch.matmul(h, self.a_12)
        A_xy= torch.matmul(A_xy_1, h.permute(0, 2, 1))
        # A_xy = torch.chain_matmul(h, self.a_12, h.permute(1, 0))

        # Ax_prime= torch.matmul(nd_flags, Ax.permute(0, 2, 1))
        # nd_flags_T= nd_flags.permute(0, 2, 1)
        # Ay_prime= torch.matmul(Ay, nd_flags_T)
        Ax_prime= Ax.permute(0, 2, 1)
        Ay_prime= Ay
        logits = Ax_prime + Ay_prime + A_xy

        e = self.leakyrelu(logits)
        zero_vec = -9e15 * e.new_tensor([1., ])
        e = torch.where(adj > 0, e, zero_vec)

        attention = F.softmax(e, dim=-1)
        # attention= torch.where(adj > 0, attention, attention.new_tensor([0., ]))
        attention = F.dropout(attention, self.dropout, training=self.training)
        return attention

    def _aggregate(self, feat, adj):
        attention = self._attention(feat, adj)
        h_prime = torch.matmul(attention, feat)
        return h_prime

    def _update(self, feat, feat_agg):
        h_1= torch.matmul(feat, self.W_1)
        h_2= torch.matmul(feat_agg, self.W_2)
        h_12= self.bilinear(feat, feat_agg)
        h_out= h_1 + h_2 + h_12
        if not self.activation:
            return h_out
        else:
            return self.activation(h_out)

        return h_out

    def forward(self, feat, adj):
        feat_agg = self._aggregate(feat, adj)
        return self._update(feat, feat_agg)

    def extra_repr(self):
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
