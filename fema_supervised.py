import numpy as np
from numba import njit, prange
# from numba.typed import List
# from math import sqrt
# from numpy.lib.type_check import asfarray
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import accuracy_score
from fema import FEMa
from kdtree import KDtree


class FEMaClassifier(FEMa):

    def __init__(self, z: int = 3, kdtree_params: dict = {'k': 1}):

        super(FEMaClassifier, self).__init__(z=z)
        self._kdtree = KDtree(kdtree_params)
        self.k = kdtree_params.get('k')
    
    @property
    def k(self):
        return self._k
    
    @k.setter
    def k(self, k):

        if not isinstance(k, (int, np.int32, np.int64)):
            raise TypeError('`k` should be an integer or null')

        if k < 1:
            raise ValueError('`k` should be >= 1')
        
        self._k = k

    def _shepard(self, x: np.ndarray):

        # from scipy.spatial import distance
        # diff = x - self._X
        # diff = np.where(diff == 0, 0.000001, diff) 
        # dist = np.linalg.norm(diff, axis=1)
        # dist = np.array([distance.euclidean(xk, xj) for xj in X_train])

        indices, distances = self._kdtree.query(x, k=self.k, sort=False)
        idw = np.array(distances, dtype=np.float64) ** -self._z
        
        return idw / np.sum(idw), indices

    def _predict_prob(self, X_test: np.ndarray):

        check_is_fitted(self, '_train_count')
        
        test_count = len(X_test)
        prob_per_class = np.zeros((test_count, self._class_count))
        
        for i in range(test_count):
            
            phi, idx_train = self._shepard(X_test[i])
            
            for c in range(self._class_count):

                prob_per_class[i, c] = np.dot(np.fromiter(
                    (1 if y_tr == c else 0 for y_tr in self._y[idx_train]), 
                    dtype=float), phi) 
        
        return prob_per_class

    def fit(self, X: np.ndarray, y: np.ndarray):

        self._train_count = X.shape[0]
        self._class_count = len(np.unique(y))
        self._X = X
        self._y = y

        # Initialize tree index
        self._kdtree.fit(self._X)
        # return self

    def predict(self, X: np.ndarray):

        check_is_fitted(self, '_train_count')
        return np.argmax(self._predict_prob(X), axis=1)
    
    def score(self, X: np.ndarray, y: np.ndarray):
        
        self.fit(X, y)
        preds = self.predict(X)
        return accuracy_score(y, preds)


class FEMaRegressor(FEMa):

    def __init__(self, z: int = 3, use_numba: bool = True, kdtree_params: dict = {'k': 1}):
        
        super(FEMaRegressor, self).__init__(z=z)
        self._use_numba = use_numba
        self._kdtree = KDtree(kdtree_params)
        self._k = kdtree_params.get('k')

        self._X = None
        self._y = None

    @property
    def k(self):
        return self._k
    
    @k.setter
    def k(self, k):

        if not isinstance(k, (int, np.int32)):
            raise TypeError('`k` should be an integer or null')

        if k < 1:
            raise ValueError('`k` should be >= 1')
        
        self._k = k

    def _shepard(self, x: np.ndarray):

        indices, distances = self._kdtree.query(x, k=self.k, sort=False)

        basesP = np.array(distances, dtype=np.float64) ** -self._z
        basesN = basesP / np.sum(basesP)

        return np.sum(basesN * self._y[indices])

    def fit(self, X: np.ndarray, y: np.ndarray, copy: bool = True):

        if copy:
            self._X = X.copy(order='F')
            self._y = y.copy(order='K')
        else:
            self._X = X.view()
            self._y = y.view()
        
        # Initialize tree index
        self._kdtree.fit(self._X)
    
    def predict(self, X_test: np.ndarray):

        if self._use_numba:
            
            k_idx, k_dist = self._kdtree.query_2d(X_test, k=self.k, sort=False)
            
            return self._shepard_numba(
                X_test, self._y, self._z, k_idx, k_dist)

        return np.apply_along_axis(self._shepard, axis=1, arr=X_test)

    @staticmethod
    @njit(fastmath=True, nogil=True, cache=True, parallel=True)
    def _shepard_numba(X_test: np.ndarray, y_train: np.ndarray,
                       z: int, k_indices, k_distances):
        
        y_pred = np.empty(X_test.shape[0])

        for j in prange(X_test.shape[0]):
            
            basesP = k_distances[j] ** -z
            basesN = basesP / np.sum(basesP)
            
            idx = k_indices[j]
            y_pred[j] = np.sum(basesN * y_train[idx])

        return y_pred
