import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split as split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from fema_supervised import FEMaClassifier, FEMaRegressor

scaler = StandardScaler()

X, y = load_wine(return_X_y=True)
_X_train, _X_test, y_train, y_test = split(X, y, test_size=0.3, random_state=1234)

X_train = scaler.fit_transform(_X_train)
X_test = scaler.transform(_X_test)

# k = training set size --> reproduce the original FEMa behavior

tree_prm = {
    'k': X_train.shape[0],
    'n_leaf': 20, 
    'distance': 'minkowski'
    }

z = 3

# Top-k FEMa classifier (with kd-tree)
femac = FEMaClassifier(z=z, kdtree_params=tree_prm)
femac.fit(X_train, y_train)
preds = femac.predict(X_test)

acc = accuracy_score(y_test, preds)
print(f'Accuracy score on test set for top-{femac.k} nearest neighbors (training samples): {acc: .4f}')

# Top-k FEMa regressor (with kd-tree)
femar = FEMaRegressor(z=z, kdtree_params=tree_prm, use_numba=True)
femar.fit(X_train, y_train)
preds = femar.predict(X_test)

acc = accuracy_score(y_test, np.round(preds).astype(int))
print(f'Accuracy score on test set for top-{femar.k} nearest neighbors (training samples): {acc: .4f}')
