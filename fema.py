import numpy as np
# import math
# from scipy.spatial import distance
from sklearn.base import BaseEstimator, ClassifierMixin
# from sklearn.utils.validation import check_is_fitted


class FEMa(BaseEstimator, ClassifierMixin):

    def __init__(self, z: int):
        self._z = z
    
    def get_params(self, deep=True):
        return {'z': self._z}

    def set_params(self, **parameters):

        for parameter, value in parameters.items():
            setattr(self, parameter, value)

        return self
    
    def _shepard(self, x: np.ndarray):
        
        raise NotImplementedError
    
    def fit(self, X: np.ndarray, y: np.ndarray):

        raise NotImplementedError

    def predict(self, X: np.ndarray):

        raise NotImplementedError

    
