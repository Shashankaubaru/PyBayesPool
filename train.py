from __future__ import division
from __future__ import print_function

import argparse
import time

import numpy as np
from scipy import sparse
import torch
from torch import optim

import matplotlib.pyplot as plt
from pygsp import graphs, filters, plotting
from coarsening.coarsening_utils import *
import coarsening.graph_utils

from model import GATcoarseVAE
from model import MLP
from gae.optimizer import loss_function
from gae.utils import load_data, mask_test_edges, preprocess_graph, get_roc_score, sparse_mx_to_torch_sparse_tensor

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gcn_vae', help="models used")
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--hidden1', type=int, default=32, help='Number of units in hidden layer 1.')
parser.add_argument('--hidden2', type=int, default=16, help='Number of units in hidden layer 2.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset-str', type=str, default='cora', help='type of dataset.')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--ratio', type=float, default=0.5, help='Ratio for coarsening')
parser.add_argument('--K', type=float, default=5, help='size of subspace')
parser.add_argument('--method', type=str, default='variation_neighborhood', help='coarsening method')

args = parser.parse_args()

def coarsening(args, adj):
        G = graphs.Graph(adj)
        C, Gc, Call, Gall = coarsen(G, K=args.K, r=args.ratio, method=args.method) 
        adj_coarse = Gc.W
        n_nodes = adj_coarse.shape[0]
        D = sp.sparse.diags(np.array(1/np.sum(C,0))[0])    
        Pinv = C.dot(D)
        adj_temp =  Pinv.dot(G.W)
        return torch.FloatTensor(np.array(adj_temp.todense())), torch.FloatTensor(adj_coarse.toarray()), n_nodes
    
def test(model,loader):
    model.eval()
    correct = 0.
    loss = 0.
    for data in loader:
        data = data.to(args.device)
        out = model(data)
        pred = out.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
        loss += F.nll_loss(out,data.y,reduction='sum').item()
    return correct / len(loader.dataset),loss / len(loader.dataset)

def gae_for(args):
    print("Using {} dataset".format(args.dataset_str))
    adj, features = load_data(args.dataset_str)
    n_nodes, feat_dim = features.shape

    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = adj
    adj_orig = adj_orig - sparse.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()

    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
    adj = adj_train

    # Some preprocessing
    #adj_norm = preprocess_graph(adj)
   # adj_label = adj_train + sparse.eye(adj_train.shape[0])
    # adj_label = sparse_to_tuple(adj_label)
    #adj_label = torch.FloatTensor(adj_label.toarray())

    pos_weight = (adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    
    #G = graphs.Graph(adj)

    #method = 'variation_neighborhood'  

    # Parameters
    #r    = 0.6 # the extend of dimensionality reduction (r=0 means no reduction)
    #k    = 5  
    #kmax = int(3*k)
        
    #C, Gc, Call, Gall = coarsen(G, K=k, r=r, method=method) 
    #adj_coarse = Gc.W
    #adj_label = torch.FloatTensor(adj_coarse.toarray())
    #D = sp.sparse.diags(np.array(1/np.sum(C,0))[0])    
    #Pinv = C.dot(D)
    #adj_temp =  Pinv.dot(G.W)
    #adj_norm = sparse_mx_to_torch_sparse_tensor(adj_temp)
    #adj_norm = torch.FloatTensor(np.array(adj_temp.todense()))
    #n_nodes = adj_coarse.shape[0]
    

    model = GATcoarseVAE(feat_dim, args.hidden1, args.hidden2, args.dropout, args.alpha)
    model2 = MLP(args.hidden2,args.num_classes)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    adj_coarse, adj_label, n_nodes = coarsening(args, adj)
    pos_weight = torch.FloatTensor(np.repeat(pos_weight, n_nodes))
    hidden_emb = None
    for epoch in range(args.epochs):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        recovered, mu, logvar = model(features, adj_coarse)
        loss = loss_function(preds=recovered, labels=adj_label,
                             mu=mu, logvar=logvar, n_nodes=n_nodes,
                             norm=norm, pos_weight=pos_weight)
        loss.backward()
        cur_loss = loss.item()
        optimizer.step()

        hidden_emb = mu.data.numpy()
       # roc_curr, ap_curr = get_roc_score(hidden_emb, adj_orig, val_edges, val_edges_false)

        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(cur_loss),
             # "val_ap=", "{:.5f}".format(ap_curr),
              "time=", "{:.5f}".format(time.time() - t)
              )

    print("Optimization Finished!")
    
    for epoch in range(args.epochs):
    model2.train()
    for i, data in enumerate(train_loader):
        data = data.to(args.device)
        out = model2(data)
        loss = F.nll_loss(out, data.y)
        print("Training loss:{}".format(loss.item()))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    val_acc,val_loss = test(model,val_loader)
    print("Validation loss:{}\taccuracy:{}".format(val_loss,val_acc))
    if val_loss < min_loss:
        torch.save(model.state_dict(),'latest.pth')
        print("Model saved at epoch{}".format(epoch))
        min_loss = val_loss
        patience = 0
    else:
        patience += 1
    if patience > args.patience:
        break 


    test_acc,test_loss = test(model,test_loader)
    print("Test accuarcy:{}".fotmat(test_acc))

  #  roc_score, ap_score = get_roc_score(hidden_emb, adj_orig, test_edges, test_edges_false)
  #  print('Test ROC score: ' + str(roc_score))
 #   print('Test AP score: ' + str(ap_score))


if __name__ == '__main__':
    gae_for(args)
