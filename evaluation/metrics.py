import numpy as np
import networkx as nx
from sklearn.metrics import pairwise_distances
# from modules import graph_io


def optimize_scale(X, D, func):
    from scipy.optimize import minimize_scalar
    def f(a): return func(a * X, D)
    min_a = minimize_scalar(f)
    return func(min_a.x * X, D)


def dist(u, v):
    return np.sqrt(np.sum(np.square(u - v)))


class Metrics():

    def __init__(self, G: nx.Graph, pos, apsp):
        """
        G: networkx graph
        pos: 2 dimensional embedding of graph G
            can be either a dictionary or numpy array
        """
        self.G = nx.convert_node_labels_to_integers(G)
        self.N = G.number_of_nodes()

        if isinstance(pos, dict):
            X = np.zeros((G.number_of_nodes(), 2))
            for i, (v, p) in enumerate(pos.items()):
                X[i] = p
            self.X = X
        elif isinstance(pos, np.ndarray):
            self.X = pos


        if G.number_of_nodes() != self.X.shape[0]: print("Error!!")
        self.Xij = pairwise_distances(self.X)

        # self.D = graph_io.get_apsp(self.G)
        # tmp_distances = dict(nx.all_pairs_shortest_path_length(G))
        tmp_distances = apsp
        self.D = np.array([[tmp_distances[i][j] for i in tmp_distances.keys()] for j in tmp_distances.keys() ])

    def setX(self, X):
        self.X = X
        self.Xij = pairwise_distances(self.X)

    def compute_stress_norm(self):
        """
        Computes \sum_{i,j} ( (||X_i - X_j|| - D_{i,j}) / D_{i,j})^2
        """

        Xij = self.Xij
        D = self.D

        stress = np.sum(np.square((Xij - D) / np.maximum(D, 1e-15)))

        return stress / 2

    def compute_stress_raw(self):
        """
        Computes \sum_{i,j} ( (||X_i - X_j|| - D_{i,j}) / D_{i,j})^2
        """

        Xij = self.Xij
        D = self.D
        N = self.N

        stress = np.sum(np.square(Xij - D))

        return stress / 2

    def compute_stress_kruskal(self):
        from sklearn.isotonic import IsotonicRegression

        #sklearn has implemented an efficient distance matrix algorithm for us
        output_dists = pairwise_distances(self.X)

        #Extract the upper triangular of both distance matrices into 1d arrays
        #We know the diagonal is all zero, so offset by one
        xij = output_dists[ np.triu_indices( output_dists.shape[0], k=1 ) ]
        dij  = self.D[ np.triu_indices( self.D.shape[0], k=1 ) ]
        # print(xij, dij)

        #Find the indices of dij that when reordered, would sort it. Apply to both arrays
        sorted_indices = np.argsort(dij)
        dij = dij[sorted_indices]
        xij = xij[sorted_indices]

        hij = IsotonicRegression().fit(dij, xij).predict(dij)

        #
        raw_stress  = np.sum( np.square( xij - hij ) )
        norm_factor = np.sum( np.square( xij ) )
        # print(raw_stress, norm_factor)
        kruskal_stress = np.sqrt( raw_stress / norm_factor )
        return kruskal_stress

    def compute_stress_ratios(self):

        return 1.0

    def compute_stress_minopt(self):

        return 1.0

    def compute_stress_minopt(self):

        return 1.0

    def compute_stress_sheppard(self):
        Xij = self.Xij #Embedding distance
        D = self.D     #Graph/edge distance

        Xij_flat = Xij.flatten()
        D_flat = D.flatten()

        valid_indices = Xij_flat > 0 #Exclude any self comparisons (distance = 0)
        Xij_filtered = Xij_flat[valid_indices]
        D_filtered = D_flat[valid_indices]

        correlation = np.corrcoef(Xij_filtered, D_filtered)[0, 1]

        return correlation

    def compute_stress_sheppardscale(self):
        Xij = self.Xij #Embedding distance
        D = self.D     #Graph/edge distance

        max_D = D.max()
        max_X = Xij.max()

        scale_factor = max_D/max_X if max_X > max_D else max_X/max_D

        return scale_factor

    def compute_stress_unitball(self):

        X = self.X / np.max(np.linalg.norm(self.X,ord=2,axis=1))
        Xij = pairwise_distances(X)
        D = self.D

        stress = np.sum(np.square(Xij - D))

        return stress / 2

    def compute_stress_unitsquare(self):

        return 1.0

    def compute_stress_kk(self):
        #https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=b8d3bca50ccc573c5cb99f7d201e8acce6618f04

        return 1.0

    def compute_neighborhood(self, rg=2):
        """
        How well do the local neighborhoods represent the theoretical neighborhoods?
        Closer to 1 is better.
        Measure of percision: ratio of true positives to true positives+false positives
        """
        X = self.X
        d = self.D

        def get_k_embedded(X, k_t):
            dist_mat = pairwise_distances(X)
            return [np.argsort(dist_mat[i])[1:len(k_t[i])+1] for i in range(len(dist_mat))]

        k_theory = [np.where((d[i] <= rg) & (d[i] > 0))[0]
                    for i in range(len(d))]

        k_embedded = get_k_embedded(X, k_theory)

        NP = 0
        for i in range(len(X)):
            if len(k_theory[i]) <= 0:
                continue
            intersect = np.intersect1d(k_theory[i], k_embedded[i]).size
            jaccard = intersect / (2*k_theory[i].size - intersect)

            NP += jaccard

        return NP / len(X)

    def compute_ideal_edge_avg(self):
        X = self.X
        edge_lengths = np.array([dist(X[i], X[j]) for i, j in self.G.edges()])

        avg = edge_lengths.mean()

        return np.sum(np.square((edge_lengths - avg) / avg))

    def compute_ideal_edge_avg_indep(self):
        X = self.X
        edge_lengths = np.array([dist(X[i], X[j]) for i, j in self.G.edges()])

        return np.sum(np.square((edge_lengths - 1) / 1))

class MetricsData(Metrics):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        """
        X: Low dimensional embedding of matrix Y. Can be given as coordinates N x d matrix
        Y: High dimensional coordiantes of objects. Can be given as N x D matrix
            (for which a pairwise distance matrix will be computed) or the distances directly as an N x N matrix
        """

        #Check data format
        self.X = X
        if Y.shape[0] == Y.shape[1]: self.D = Y
        else: self.D = pairwise_distances(Y)

        self.N = X.shape[0]

        #Ensure compatibility with parent class
        self.name = None
        self.G = None

