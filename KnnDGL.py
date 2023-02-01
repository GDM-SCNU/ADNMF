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
class KnnDGL(DGLDataset):
    def __init__(self, name = "", topK = 2, num_classes = 0):
        self.path = name # 文件路径
        self.components = num_classes
        self.topk = topK
        super(KnnDGL, self).__init__(name = name)
    def process(self):
        topology = file_path + self.path+"\\"+ self.path + ".cites" # 拓扑信息
        attribute = file_path + self.path+"\\" + self.path +".content" # 属性信息

        feat_data = []  # 用于存放每个节点特征向量的列表
        node_map = {}  # 将节点进行重新编码
        adj_lists = defaultdict(set)
        labels = []  # 用于存放每个节点对应类别的列表
        label_map = {}  # 将label映射为数字

        # 属性信息
        with open(attribute, "r", encoding='utf-8') as fp:
            for i, line in enumerate(fp):
                info = line.strip().split()
                feat_data.append([float(x) for x in info[1:-1]])
                node_map[info[0]] = i
                if not info[-1] in label_map:
                    label_map[info[-1]] = len(label_map)
                labels.append(label_map[info[-1]])
        feat_data = np.asarray(feat_data)
        labels = np.asarray(labels, dtype=np.int64)
        self.num_nodes = len(feat_data)
        adj_str = []
        adj_end = []

        # 通过属性信息构建KNN图
        sim_feat = cosine_similarity(feat_data)
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
        feat_data = torch.tensor(feat_data)
        labels = torch.tensor(labels)
        self.graph = dgl.graph((self.estr, self.eend), num_nodes = self.num_nodes) # 构建KNN图
        # self.graph = dgl.add_self_loop(self.graph)  # 加自环
        self.graph.ndata['feat'] = feat_data # feature
        self.graph.ndata['label'] = labels # 标签



    def __getitem__(self, item):
        return self.graph

    @property
    def num_classes(self):
        return self.components

    def __len__(self):
        return 1
    # @property
    def edges(self):
        return torch.from_numpy(np.array(self.estr)), torch.from_numpy(np.array(self.eend))
    # @property
    def number_of_edges(self):
        return self.graph.adjacency_matrix().to_dense().float().sum().sum()
    def number_of_nodes(self):
        return self.num_nodes