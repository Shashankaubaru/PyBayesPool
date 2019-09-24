import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from gae.layers import GraphConvolution
from pyGAT.layers import GraphAttentionLayer
#from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

class GATcoarseVAE(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout, alpha):
        super(GATcoarseVAE, self).__init__()
        self.gc0 = GraphConvolution(input_feat_dim, input_feat_dim, dropout, act=lambda x: x)
        self.gat1 =GraphAttentionLayer(input_feat_dim, hidden_dim2, dropout=dropout, alpha=alpha, concat=True)
        self.gc2 = GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=lambda x: x)
        self.gc3 = GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=lambda x: x)
        self.dc = InnerProductDecoder(dropout, act=lambda x: x)

    def encode(self, x, adj, adj0):
       # hid0 = self.gc0(x, adj0)
        hidden1 = self.gat1(x, adj)
        return self.gc2(x, hidden1), self.gc3(x, hidden1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, adj, adj0):
        mu, logvar = self.encode(x, adj, adj0)
        z = self.reparameterize(mu, logvar)
        return self.dc(z), mu, logvar


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        return adj

class MLP(nn.Module):
    def __init__(self,nhid,num_classes):
        super(MLP, self).__init__()
        self.nhid = nhid
        self.num_classes = num_classes

        self.lin1 = torch.nn.Linear(self.nhid*2, self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid, self.nhid)
        self.lin3 = torch.nn.Linear(self.nhid, self. num_classes)
        
    def Pools(self, X):
        n_nodes = X.shape[0]
        X = X.unsqueeze(0)
        X = X.transpose(2,1)
        mxp = nn.MaxPool1d(n_nodes)
        avp = nn.AvgPool1d(n_nodes)
        out1 = mxp(X)
        out2 = avp(X)
        return out1.transpose(1,2), out2.transpose(1,2)
        
    def forward(self, mu):
        Mxp, Avp = self.Pools(mu)
        x = torch.cat([Mxp, Avp], dim=2)
        x = x.squeeze(0)

        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.log_softmax(self.lin3(x), dim=-1)

        return x
    