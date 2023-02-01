# coding=utf-8
# Author: Jung
# Time: 2022/4/30 21:14

"""生成KNN图并转化为DGL格式"""
from dgl.data import DGLDataset
import dgl
import torch
import torch.nn.functional as F
import dgl.data
import numpy as np
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
file_path = "datasets\\"
class KnnDGLFromDGL(DGLDataset):
    def __init__(self, name, g, feat_data, num_nodes , topK = 2):
        self.feat_data = feat_data
        self.topk = topK
        self.num_nodes = num_nodes
        self.g = g
        super(KnnDGLFromDGL, self).__init__(name = name)
    def process(self):

        adj_str = []
        adj_end = []

        # 通过属性信息构建KNN图
        sim_feat = cosine_similarity(self.feat_data)
        inds = []
        for i in range(sim_feat.shape[0]):
            ind = np.argpartition(sim_feat[i, :], -(self.topk + 1))[-(self.topk + 1):]
            inds.append(ind)
        for i, vs in enumerate(inds):
            for v in vs:
                if v == i: # 自己与自己不构成环
                    pass
                else:
                    adj_str.append(i)
                    adj_end.append(v)
        self.estr = adj_str + adj_end
        self.eend = adj_end + adj_str
        self.graph = dgl.graph((self.estr, self.eend), num_nodes = self.num_nodes) # 构建KNN图
        self.graph.ndata['feat'] = self.g.ndata['feat']
        self.graph.ndata['label'] = self.g.ndata['feat']

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1
    # @property
    def edges(self):
        return torch.from_numpy(np.array(self.estr)), torch.from_numpy(np.array(self.eend))
    # @property
    def number_of_edges(self):
        return self.graph.adjacency_matrix().to_dense().float().sum().sum()
