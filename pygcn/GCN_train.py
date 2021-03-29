from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from pygcn.models import GCN
from pygcn.utils import normalize,sparse_mx_to_torch_sparse_tensor,load_data,accuracy
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.linalg as la
from modelutils import GCN_inputs


def update_graph(n,m,args,model,optimizer,features,labels,states,adj,degree):

    features, adj, deg, laplacian, labels, idx_train = GCN_inputs(features,labels,states,adj,degree)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda and torch.cuda.is_available():
        model.cuda()
        features = features.cuda()
        adj = adj.cuda()
        laplacian =laplacian.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()

    t_total = time.time()
    for epoch in range(args.gcn_epochs):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        output = model(features, adj)
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        soft_out= torch.unsqueeze(torch.nn.functional.softmax(output,dim=1)[:,1],1)
        loss_reg  = torch.mm(torch.mm(torch.transpose(soft_out, 0, 1),laplacian),soft_out)
        print('Epoch: {:04d}'.format(epoch+1),
        'loss_train: {:.4f}'.format(loss_train.item()),
        'time: {:.4f}s'.format(time.time() - t))
        loss_train +=  args.gcn_lambda * loss_reg.squeeze()
        loss_train.backward()
        optimizer.step()



        
