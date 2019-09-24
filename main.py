from __future__ import division
from __future__ import print_function

import argparse
import os
import pickle
import random
import shutil
import time

import numpy as np
from scipy import sparse
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import normalize

import matplotlib.pyplot as plt
from pygsp import graphs, filters, plotting
from coarsening.coarsening_utils import *
import coarsening.graph_utils

from model import GATcoarseVAE
from model import MLP
from gae.optimizer import loss_function
from gae.utils import sparse_mx_to_torch_sparse_tensor
from diffpool import load_data
from diffpool.graph_sampler import GraphSampler
from diffpool.gen import feat as featgen

def arg_parse():
    parser = argparse.ArgumentParser(description='GATcoarseVAE arguments.')

    parser.add_argument('--datadir', dest='datadir',
            help='Directory where benchmark is located')
    parser.add_argument('--dataset', dest='dataset', 
            help='Input dataset.')
    parser.add_argument('--max-nodes', dest='max_nodes', type=int,
            help='Maximum number of nodes (ignore graghs with nodes exceeding the number.')
    parser.add_argument('--lr', dest='lr', type=float,
            help='Learning rate.')
    parser.add_argument('--clip', dest='clip', type=float,
            help='Gradient clipping.')
    parser.add_argument('--batch-size', dest='batch_size', type=int,
            help='Batch size.')
    parser.add_argument('--epochs', dest='num_epochs', type=int,
            help='Number of epochs to train.')
    parser.add_argument('--train-ratio', dest='train_ratio', type=float,
            help='Ratio of number of graphs training set to all graphs.')
    parser.add_argument('--num_workers', dest='num_workers', type=int,
            help='Number of workers to load data.')
    parser.add_argument('--feature', dest='feature_type',
            help='Feature used for encoder. Can be: id, deg')
    parser.add_argument('--input-dim', dest='input_dim', type=int,
            help='Input feature dimension')
    parser.add_argument('--hidden-dim', dest='hidden_dim', type=int,
            help='Hidden dimension')
    parser.add_argument('--GAT-hid-dim', dest='GAT_hid_dim', type=int,
            help='Hidden dimension')
    parser.add_argument('--output-dim', dest='output_dim', type=int,
            help='Output dimension')
    parser.add_argument('--num-classes', dest='num_classes', type=int,
            help='Number of label classes')
    parser.add_argument('--num-gc-layers', dest='num_gc_layers', type=int,
            help='Number of graph convolution layers before each pooling')
    parser.add_argument('--nobn', dest='bn', action='store_const',
            const=False, default=True,
            help='Whether batch normalization is used')
    parser.add_argument('--dropout', dest='dropout', type=float,
            help='Dropout rate.')
    parser.add_argument('--nobias', dest='bias', action='store_const',
            const=False, default=True,
            help='Whether to add bias. Default to True.')
    parser.add_argument('--docoarsen', dest='docoarsen', type = int,
            help='Whether to do coarsening. Default to 1.')
    parser.add_argument('--doVAE', dest='doVAE', type = int,
            help='Whether to do VAE. Default to 1.')
    parser.add_argument('--no-log-graph', dest='log_graph', action='store_const',
            const=False, default=True,
            help='Whether disable log graph')
    parser.add_argument('--alpha', dest='alpha', type=float, help='Alpha for the leaky_relu.')
    parser.add_argument('--ratio', dest='ratio',type=float,  help='Ratio for coarsening')
    parser.add_argument('--K', dest='K', type=float,  help='size of subspace')
    parser.add_argument('--method', dest='method',
            help='coarsening method')
    parser.add_argument('--name-suffix', dest='name_suffix',
            help='suffix added to the output filename')

    parser.set_defaults(datadir='data',
                        dataset='NCI109',
                        max_nodes=150,
                        feature_type='struct',
                        lr=0.001,
                        clip=2.0,
                        batch_size=1,
                        num_epochs=50,
                        train_ratio=0.8,
                        test_ratio=0.1,
                        num_workers=1,
                        input_dim=10,
                        hidden_dim=32,
                        GAT_hid_dim = 32,
                        output_dim=20,
                        num_classes=2,
                        num_gc_layers=3,
                        dropout=0.0,
                        docoarsen = 1,
                        doVAE = 1,
                        method='variation_neighborhood',
                        alpha=0.01,
                        ratio=0.8,
                        K=7,
                        name_suffix='',
                        assign_ratio=0.1,
                        num_pool=1
                       )
    return parser.parse_known_args()[0]

def prepare_data(graphs, args, test_graphs=None, max_nodes=0):

    random.shuffle(graphs)
    if test_graphs is None:
        train_idx = int(len(graphs) * args.train_ratio)
        test_idx = int(len(graphs) * (1-args.test_ratio))
        train_graphs = graphs[:train_idx]
        val_graphs = graphs[train_idx: test_idx]
        test_graphs = graphs[test_idx:]
    else:
        train_idx = int(len(graphs) * args.train_ratio)
        train_graphs = graphs[:train_idx]
        val_graphs = graph[train_idx:]
    print('Num training graphs: ', len(train_graphs), 
          '; Num validation graphs: ', len(val_graphs),
          '; Num testing graphs: ', len(test_graphs))

    print('Number of graphs: ', len(graphs))
    print('Number of edges: ', sum([G.number_of_edges() for G in graphs]))
    print('Max, avg, std of graph size: ', 
            max([G.number_of_nodes() for G in graphs]), ', '
            "{0:.2f}".format(np.mean([G.number_of_nodes() for G in graphs])), ', '
            "{0:.2f}".format(np.std([G.number_of_nodes() for G in graphs])))

    # minibatch
    dataset_sampler = GraphSampler(train_graphs, normalize=False, max_num_nodes=max_nodes,
            features=args.feature_type)
    train_dataset_loader = torch.utils.data.DataLoader(
            dataset_sampler, 
            batch_size=args.batch_size, 
            shuffle=True,
            num_workers=args.num_workers)

    dataset_sampler = GraphSampler(val_graphs, normalize=False, max_num_nodes=max_nodes,
            features=args.feature_type)
    val_dataset_loader = torch.utils.data.DataLoader(
            dataset_sampler, 
            batch_size=args.batch_size, 
            shuffle=False,
            num_workers=args.num_workers)

    dataset_sampler = GraphSampler(test_graphs, normalize=False, max_num_nodes=max_nodes,
            features=args.feature_type)
    test_dataset_loader = torch.utils.data.DataLoader(
            dataset_sampler, 
            batch_size=args.batch_size, 
            shuffle=False,
            num_workers=args.num_workers)

    return train_dataset_loader, val_dataset_loader, test_dataset_loader, \
            dataset_sampler.max_num_nodes, dataset_sampler.feat_dim, dataset_sampler.assign_feat_dim


def coarsening(args, adj):
        G = graphs.Graph(adj)
        C, Gc, Call, Gall = coarsen(G, K=args.K, r=args.ratio, method=args.method) 
        adj_coarse = Gc.W
        n_nodes = adj_coarse.shape[0]
        D = sp.sparse.diags(np.array(1/(np.sum(C,0)+1))[0])    
        Pinv = C.dot(D)
        adj_temp =  Pinv.dot(G.W)
        return adj_temp, adj_coarse, n_nodes
    
def data_coarsen(dataset, test_dataset, args):
    
    adj_O = []
    adj_C = []
    adj_lbl = []
    adj_feat = []
    num_nodes =[]
    G_lbl = []
    test_G_lbl = []
    for batch_idx, data in enumerate(dataset):
        print("Coarsening graph:", batch_idx)
        adj = data['adj'].float()
        adj= adj.numpy()
        adj=adj.squeeze(axis=0)
        if data['num_nodes'] <= args.K:
            continue
        tcoarse, tlabel, tn_nodes = coarsening(args, adj)
        feat = data['feats'].float()
        lab = data['label'].long()
        #blab = label_binary(lab)
        G_lbl.append(lab)
        feat= feat.numpy()
        feat=feat.squeeze(axis=0)
        feat= np.delete(feat,np.where(~feat.any(axis=1))[0], axis=0)
        adj_O.append(adj)
        adj_C.append(tcoarse)
        adj_lbl.append(tlabel)
        num_nodes.append(tn_nodes)
        adj_feat.append(feat)
        
    for batch_idx, data in enumerate(test_dataset):
        if batch_idx ==356:
            break
        print("Coarsening test graph:", batch_idx)
        if data['num_nodes'] <= args.K:
            continue
        adj = data['adj'].float()
        adj= adj.numpy()
        adj=adj.squeeze(axis=0)
        tcoarse, tlabel, tn_nodes = coarsening(args, adj)
        feat = data['feats'].float()
        lab = data['label'].long()
        #blab = label_binary(lab)
        test_G_lbl.append(lab)
        feat= feat.numpy()
        feat=feat.squeeze(axis=0)
        feat= np.delete(feat,np.where(~feat.any(axis=1))[0], axis=0)
        adj_O.append(adj)
        adj_C.append(tcoarse)
        adj_lbl.append(tlabel)
        num_nodes.append(tn_nodes)
        adj_feat.append(feat)
    
    return adj_O, adj_C, adj_lbl, adj_feat, G_lbl, test_G_lbl, num_nodes

def evaluateGCV(dataset, model, args):
    model.eval()
    
    adj_C, adj_L, adj_X, G_lbl, num_nodes = data_coarsen(dataset, args)
    
    hidden_emb = []
    ylabel = []
    for idx in range(len(adj_C)):
        adj_coarse = torch.FloatTensor(np.array(adj_C[idx].todense()))
        features = torch.FloatTensor(np.array(adj_X[idx]))
        recovered, mu, logvar = model(features, adj_coarse)
        hidden_emb.append(mu.data.numpy())
        ylabel.append(G_lbl[idx])
        
    return hidden_emb, ylabel

def testMLP(Embed, Label, model):
    model.eval()
    correct = 0.
    loss = 0.
    for idx in range(len(Embed)):
        data = torch.FloatTensor(np.array(Embed[idx]))
        y = Label[idx]
        out = model(data)
        pred = out.max(dim=1)[1]
        correct += pred.eq(y).sum().item()
        loss += F.nll_loss(out,y,reduction='sum').item()
    return correct / len(Embed),loss / len(Embed)
    
def trainGCV(dataset, test_dataset, model, args):
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    if args.docoarsen == 1:
        adj_O, adj_C, adj_L, adj_X, G_lbl, test_Gbl, num_nodes = data_coarsen(dataset, test_dataset, args)
        with open('NCI109_coarse_graph.pickle','wb') as f:
            pickle.dump([adj_O, adj_C, adj_L, adj_X, G_lbl, test_Gbl, num_nodes], f)

    else:
        with open('NCI109_coarse_graph.pickle','rb') as f:
            adj_O, adj_C, adj_L, adj_X, G_lbl, test_Gbl, num_nodes = pickle.load(f)
            
    ntrain = len(G_lbl)
    
    hidden_emb = []
    ylabel = []
    test_hidden_emb = []
    test_ylabel = []
    for epoch in range(args.num_epochs):
        t = time.time()
        model.train()
        for idx in range(len(adj_C)):
            if np.max(adj_L[idx].todense())==0:
                continue     
            else:
                optimizer.zero_grad()
                adj_orig = torch.FloatTensor(np.array(adj_O[idx]))
                adj_coarse = torch.FloatTensor(np.array(adj_C[idx].todense()))
                adj_label = torch.FloatTensor(np.array(adj_L[idx].todense()))
                features = torch.FloatTensor(np.array(normalize(adj_X[idx], axis=0, norm='max')))
                recovered, mu, logvar = model(features, adj_coarse, adj_orig)
                pos_weight = (adj_label.shape[0] * adj_label.shape[0] - adj_label.sum()) / adj_label.sum()
               # norm = adj_label.shape[0] * adj_label.shape[0]/max(float((adj_label.shape[0] * adj_label.shape[0] - adj_label.sum()) * 2),0.01)
                norm =1;
                pos_weight = torch.FloatTensor(np.repeat(pos_weight, num_nodes[idx]))
                loss = loss_function(preds=recovered, labels=adj_label,
                             mu=mu, logvar=logvar, n_nodes=num_nodes[idx],
                             norm=norm, pos_weight=pos_weight)
                loss.backward()
                cur_loss = loss.item()
                optimizer.step()
        #    print("Graph:", '%04d' % (idx + 1), "Training loss:{}", "{:.5f}".format(cur_loss)) 
                if epoch == args.num_epochs-1:
                    if idx < ntrain:
                        lab = G_lbl[idx]
                        hidden_emb.append(mu.data.numpy())
                        ylabel.append(lab)
                
                    else:
                        lab = test_Gbl[idx-ntrain]
                        test_hidden_emb.append(mu.data.numpy())
                        test_ylabel.append(lab)

        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(cur_loss),
             # "val_ap=", "{:.5f}".format(ap_curr),
              "time=", "{:.5f}".format(time.time() - t)
              )

    print("Optimization Finished!")
    
    #test_hidden_emb, test_ylabel = evaluateGCV(test_dataset, model, args)
    
    return hidden_emb, ylabel, test_hidden_emb, test_ylabel 

    
def trainMLP(Embed, Label, t_embed, t_label, model, args):
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)  
    
    for epoch in range(args.num_epochs*3):
        model.train()
        for idx in range(len(Embed)):
            optimizer.zero_grad()
            data = torch.FloatTensor(np.array(Embed[idx]))
            y = Label[idx]
            out = model(data)
            loss = F.nll_loss(out, y)
            loss.backward()
            cur_loss = loss.item()
            optimizer.step()
            
        print("Epoch:", '%04d' % (epoch + 1), "Training loss:{}", "{:.5f}".format(cur_loss))  
            
    print("Optimization Finished!")
    
    acc, loss = testMLP(t_embed, t_label, model)
    
    return acc, loss

def main():
    args = arg_parse()
    
    graphs = load_data.read_graphfile(args.datadir, args.dataset, max_nodes=args.max_nodes)
    
#     if  'feat_dim' in graphs[0].graph:
#         print('Using node features')
#         input_dim = graphs[0].graph['feat_dim']
    
#     else:
#         print('Using constant features')
#         featgen_const = featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float))
#         for G in graphs:
#             featgen_const.gen_node_features(G)

    train_dataset, val_dataset, test_dataset, max_num_nodes, input_dim, assign_input_dim = \
            prepare_data(graphs, args, max_nodes=args.max_nodes)
    
    if args.doVAE == 1:
        modelGCV = GATcoarseVAE(input_dim, args.hidden_dim, args.GAT_hid_dim, args.dropout, args.alpha)
        train_embed, train_label, test_embed, test_label = trainGCV(train_dataset, test_dataset, modelGCV, args)
        with open('NCI109_VAE.pickle','wb') as f:
            pickle.dump([train_embed, train_label, test_embed, test_label], f)

    else:
        with open('NCI109_VAE.pickle','rb') as f:
            train_embed, train_label, test_embed, test_label = pickle.load(f)
    
    modelMLP = MLP(args.hidden_dim,args.num_classes)
    
    test_acc, test_loss = trainMLP(train_embed, train_label, test_embed, test_label, modelMLP, args)
    
    print("Test accuarcy:", test_acc)


if __name__ == "__main__":
    main()
