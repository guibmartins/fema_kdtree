import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split as split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from fema_supervised import FEMaClassifier, FEMaRegressor

# Preparing the data
scaler = StandardScaler()

X, y = load_wine(return_X_y=True)
_X_train, _X_test, y_train, y_test = split(X, y, test_size=0.3, random_state=19)

X_train = scaler.fit_transform(_X_train)
X_test = scaler.transform(_X_test)


# Setting KD-tree parameters
# If not initialized by the user, parameters 'leaf_size' 
# and 'distance' are set to their default values.

# k = training set size --> reproduce the original FEMa behavior
tree_prm = {
    'k': int(X_train.shape[0]),
    'leaf_size': 20,
    'distance': 'minkowski'
}

# 'z' is used by FEMa/FEMaR during the computation of the shepard basis
z = 3

# Running FEMa/FEMaR using k = n_training_samples
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


# Running FEMa/FEMaR using k = 10
tree_prm['k'] = 10

# Top-k FEMa classifier (with kd-tree)
femac = FEMaClassifier(z=z, kdtree_params=tree_prm)
femac.fit(X_train, y_train)
preds = femac.predict(X_test)

acc = accuracy_score(y_test, preds)
print(f'Accuracy score on test set interpolating the top-{femac.k} training samples: {acc: .4f}')

# Top-k FEMa regressor (with kd-tree)
femar = FEMaRegressor(z=z, kdtree_params=tree_prm, use_numba=True)
femar.fit(X_train, y_train)
preds = femar.predict(X_test)

# Evaluating results with respect to accuracy score
acc = accuracy_score(y_test, np.round(preds).astype(int))
print(f'Accuracy score on test set interpolating the top-{femar.k} training samples: {acc: .4f}')