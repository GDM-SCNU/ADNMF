import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from torch.optim import Adam
from dgl.nn import GraphConv
import torch.nn as nn
import torch.nn.functional as F
from DualDNMF.TransDGL import *
import warnings
warnings.filterwarnings("ignore")
from dgl.nn import SAGEConv
from DNMF import *
from parser import *
from KnnDGL import *
from KnnDGL_fromDGL import *
import itertools
from sklearn import metrics

def compute_nmi(pred, labels):
    return metrics.normalized_mutual_info_score(labels, pred)

def compute_ac(pred, labels):
    return metrics.accuracy_score(labels, pred)

def computer_f1(pred, labels):
    return metrics.f1_score(labels, pred, average='macro')

def computer_ari(true_labels, pred_labels):
    return metrics.adjusted_rand_score(true_labels, pred_labels)



class Attention(nn.Module):
    def __init__(self, emb_dim, hidden_size= 16):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(emb_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z) # n * 1 # [265 2 32]
        beta = torch.softmax(w, dim=1) # eq. 9
        return (beta * z).sum(1), beta

class DualDNMF(nn.Module):
    def __init__(self, emb_dim, component):
        super(DualDNMF, self).__init__() # dim:32 ; 7
        self.attention = Attention(emb_dim)

        self.MLP = nn.Sequential(
            nn.Linear(emb_dim, component),
            nn.Softmax(dim=1)
        )

    def forward(self, x1, x2): 

        emb = torch.stack([x1, x2], dim=1) 
        emb, att = self.attention(emb)
        output = self.MLP(emb)
        return output, att 

import torch
from sklearn.decomposition import NMF
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import dgl
import numpy as np
from scipy import sparse
if __name__ == "__main__":

    np.random.seed(826)
    torch.manual_seed(826)
    args = parameter_parser()
    components = 6


    adjacency = TrasnDGL("citeseer", components)
    knn_feature = KnnDGL("citeseer", 7, components)



    model = DualDNMF(emb_dim = args.layers[-1],
                     component = components)

    adjDNMF = DNMF(adjacency.graph, args)
    knnDNMF = DNMF(knn_feature.graph, args)


    adjDNMF.train()
    knnDNMF.train()

    labels = adjacency.graph.ndata['label']

    optimizer = Adam(itertools.chain(model.parameters()), lr = 0.01)


    test_nmi_list = []
    test_ac_list = []
    test_f1_list = []
    test_ari_list = []
    def train(model, epoch):

        emb1 = adjDNMF.training().t()
        emb2 = knnDNMF.training().t()

        output, att = model(emb1, emb2)


        loss = - 0.1 * torch.trace(output.t().matmul(adjDNMF.B + knnDNMF.B).matmul(output))
        loss += 0.1 * torch.linalg.norm(adjDNMF.A + knnDNMF.A - output @ output.t())  ** 2

        A_1 = torch.sigmoid(output.matmul(output.t()))
        A_0 = 1 - A_1
        A = (adjDNMF.A + knnDNMF.A) * torch.log(A_1) + (1 - (adjDNMF.A + knnDNMF.A)) * torch.log(A_0)
        loss += 1 / adjacency.num_nodes * -(A.sum().sum())


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 1 == 0:
            model.eval()
            pred = output.argmax(dim=1)
            nmi = compute_nmi(pred.numpy(), labels.numpy())
            ac = compute_ac(pred.numpy(), labels.numpy())
            f1 = computer_f1(pred.numpy(), labels.numpy())
            ari = computer_ari(labels.numpy(), pred.numpy())
            test_nmi_list.append(nmi)
            test_ac_list.append(ac)
            test_f1_list.append(f1)
            test_ari_list.append(ari)
            print(
                'epoch={}, loss={:.3f},  nmi: {:.3f}, f1_score={:.3f},  ac = {:.3f}, ari= {:.3f}, MAX_NMI={:.3f}, MAX_F1={:.3f}, MAX_AC = {:.3f}, MAX ARI = {:.3f}'.format(
                    epoch,
                    loss,
                    nmi,
                    f1,
                    ac,
                    ari,
                    max(test_nmi_list),
                    max(test_f1_list),
                    max(test_ac_list),
                    max(test_ari_list)
                ))
    for epoch in range(700): 
        train(model, epoch)
