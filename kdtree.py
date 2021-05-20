import numpy as np
from sklearn.neighbors import KDTree

class KD_Tree:

    def __init__(self, params: dict={}):
        
        self.tree = None
        self.params = params

        if params.get('n_leaf') is None:
            self.params['n_leaf'] = 30

        if params.get('distance') is None:
            self.params['distance'] = 'minkowski'

    @property
    def tree(self):
        return self._tree

    @tree.setter
    def tree(self, tree):
        self._tree = tree

    def fit(self, X: np.ndarray, save_file: bool=False):
        
        self.tree = KDTree(
            X, leaf_size=self.params.get('n_leaf'), 
            metric=self.params.get('distance'))

    def query(self, sample: np.ndarray, k: int, dual_tree: bool=False, sorted: bool=True):
        
        distances, indices = self.tree.query(
            [sample], k=k, dualtree=dual_tree, sort_results=sorted)
        
        return list(indices[0]), list(distances[0])
    
    def query_2d(self, sample: np.ndarray, k: int, dual_tree: bool=False, sorted: bool=True):
        
        distances, indices = self.tree.query(
            sample, k=k, dualtree=dual_tree, sort_results=sorted)
        
        return indices, distances