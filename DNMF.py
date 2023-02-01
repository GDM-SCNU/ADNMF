import torch
from sklearn.decomposition import NMF
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import dgl
import numpy as np
from scipy import sparse
class DNMF(object):
    def __init__(self, graph, args):
        super(DNMF, self).__init__()
        self.graph = graph
        self.args = args
        self.p = len(self.args.layers)
        self.A = graph.adjacency_matrix().to_dense().float().numpy()
        self.A = np.where(self.A >=1 ,1 ,0)
        self.A = torch.from_numpy(self.A).float()
        self.sparse_adj = sparse.csr_matrix(self.A.numpy())

        self.B = self.compute_B_matrix()
        self.setup_D()
        self.nmi = []
        self.ac = []
        self.f1 = []
        # self.labels = graph.ndata['label']
    def compute_B_matrix(self):
        nx_graph = dgl.to_networkx(self.graph)
        degree = nx.degree(nx_graph)
        deg = torch.FloatTensor([d for id, d in degree]).reshape(-1, 1)
        sum_deg = deg.sum()
        B = self.A - (deg.matmul(deg.t()) / sum_deg)
        return B

    def setup_D(self):
        self.L = torch.diag(self.A.sum(axis=1)) - self.A
        self.D = self.L + self.A
    def setup_z(self, i):
        if i == 0:
            self.Z = self.A
        else:
            self.Z = self.V_s[i-1]

    def sklearn_pretain(self, i):
        nmf_model = NMF(n_components= self.args.layers[i],
                        init = "random",
                        random_state= self.args.seed,
                        max_iter=self.args.pre_iterations)
        U = nmf_model.fit_transform(self.Z)
        V = nmf_model.components_
        return torch.from_numpy(U).float(), torch.from_numpy(V).float()

    def train(self):
        self.V_s = []
        self.U_s = []
        # for i in tqdm(range(self.p), desc = "Layers trained: ", leave= False):
        for i in range(self.p):
            self.setup_z(i)
            U, V = self.sklearn_pretain(i)
            self.V_s.append(V)
            self.U_s.append(U)
    def clear_feat(self):
        self.V_s.clear()
        self.U_s.clear()


    # DANMF iterator
    def setup_Q(self):
        """
        Setting up Q matrices.
        """
        self.Q_s = [None for _ in range(self.p+1)]
        # 最后一层
        self.Q_s[self.p] = torch.eye(self.args.layers[self.p-1]).float()
        # 逆序
        for i in range(self.p-1, -1, -1):
            self.Q_s[i] = self.U_s[i].matmul(self.Q_s[i+1])

    def update_U(self, i):
        """
        Updating left hand factors.
        :param i: Layer index.
        """
        if i == 0:
            R = self.U_s[0].matmul(self.Q_s[1].matmul(self.VpVpT).matmul(self.Q_s[1].t()))
            R = R+ self.A_sq.matmul(self.U_s[0].matmul(self.Q_s[1].matmul(self.Q_s[1].t()))) + 1e-3
            Ru = 2 * self.A.matmul(self.V_s[self.p-1].t().matmul(self.Q_s[1].t()))
            self.U_s[0] = (self.U_s[0] * Ru) / R
        else:
            R = self.P.t().matmul(self.P).matmul(self.U_s[i]).matmul(self.Q_s[i+1]).matmul(self.VpVpT).matmul(self.Q_s[i+1].t())
            R = R + self.A_sq.matmul(self.P).t().matmul(self.P).matmul(self.U_s[i]).matmul(self.Q_s[i+1]).matmul(self.Q_s[i+1].t()) + 1e-3
            Ru = 2 * self.A.matmul(self.P).t().matmul(self.V_s[self.p-1].t()).matmul(self.Q_s[i+1].t())
            self.U_s[i] = (self.U_s[i]*Ru)/ R

    def update_P(self, i):
        """
        Setting up P matrices.
        :param i: Layer index.
        """
        if i == 0:
            self.P = self.U_s[0]
        else:
            self.P = self.P.matmul(self.U_s[i])

    def update_V(self, i):
        """
        Updating right hand factors.
        :param i: Layer index.
        """
        if i < self.p-1:
            Vu = 2*self.A.matmul(self.P).t()
            Vd = self.P.t().matmul(self.P).matmul(self.V_s[i])+self.V_s[i] + 1e-3
            self.V_s[i] = self.V_s[i] * Vu/ Vd
        else:
            Vu = 2*self.A.matmul(self.P).t() + (self.args.lamb * self.A.matmul(self.V_s[i].t())).t()
            Vd = self.P.t().matmul(self.P).matmul(self.V_s[i])
            Vd = Vd + self.V_s[i] + 1e-3 + (self.args.lamb * self.D.matmul(self.V_s[i].t())).t()
            self.V_s[i] = self.V_s[i] * Vu/ Vd
    def training(self):
        """
        Training process after pre-training.
        """
        self.A_sq = self.A.matmul(self.A.t())
        # for iteration in tqdm(range(self.args.iterations), desc="Training pass: ", leave=True):
        for iteration in range(1):
            self.setup_Q()
            self.VpVpT = self.V_s[self.p - 1].matmul(self.V_s[self.p - 1].t())
            for i in range(self.p):
                self.update_U(i)
                self.update_P(i)
                self.update_V(i)
        return self.V_s[-1]