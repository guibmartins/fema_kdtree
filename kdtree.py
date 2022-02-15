import numpy as np
from sklearn.neighbors import KDTree


class KDtree:

    def __init__(self, params=None):
        
        if params is None:
            params = {'leaf_size': 30, 'distance': 'minkowski'}

        self.tree = None
        self.params = params

        if params.get('leaf_size') is None:
            self.params['leaf_size'] = 30

        if params.get('distance') is None:
            self.params['distance'] = 'minkowski'

    @property
    def tree(self):
        return self._tree

    @tree.setter
    def tree(self, tree):
        self._tree = tree

    def fit(self, X: np.ndarray):
        
        self.tree = KDTree(
            X, leaf_size=self.params.get('leaf_size'),
            metric=self.params.get('distance'))

    def query(self, sample: np.ndarray, k: int, dual_tree: bool = False, sort: bool = True):

        distances, indices = self.tree.query(
            [sample], k=k, dualtree=dual_tree, sort_results=sort)
        
        return list(indices[0]), list(distances[0])
    
    def query_2d(self, sample: np.ndarray, k: int, dual_tree: bool = False, sort: bool = True):
        
        distances, indices = self.tree.query(
            sample, k=k, dualtree=dual_tree, sort_results=sort)
        
        return indices, distances
