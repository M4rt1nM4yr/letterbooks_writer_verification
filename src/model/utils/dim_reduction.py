from sklearn.decomposition import PCA
from joblib import dump, load
import numpy as np
from sklearn import preprocessing

from src.model.utils.rpca import RegularizedPCA

class DimReduction():
    def __init__(self, reduction="pca", n_dim=1000, whiten=True, regularization=0.001):
        self.reduction = reduction
        self.n_dim = n_dim
        if reduction=="pca":
            self.model = PCA(n_components=n_dim, whiten=whiten)
        elif reduction=="rpca":
            self.model = RegularizedPCA(n_components=n_dim, whiten=whiten, regularization=regularization)

    def fit(self, x):
        if self.reduction == "pca":
            assert self.n_dim<=x.shape[0] and self.n_dim<=x.shape[1], f"PCA wasn't used. PCA target dim: {self.n_dim}; Shape m-VLAD{x.shape}"
        self.model.fit(x)

    def transform(self,x):
        x = self.model.transform(x)
        return preprocessing.normalize(x, norm="l2", axis=1)

    def save(self, path):
        dump(self.model,path)

    def load(self, path):
        self.model = load(path)

if __name__ == "__main__":
    n_samples = 100
    n_dim = 64
    reduction = "pca"
    # TODO: "rpca" doesn't produce consistent outputs after saving and loading
    x = np.random.randn(n_samples, n_dim)
    x_test = np.random.randn(n_samples, n_dim)
    dim_reduction = DimReduction(n_dim=16, reduction=reduction)
    dim_reduction.fit(x)
    encodings = dim_reduction.transform(x_test)
    dim_reduction.save("test.joblib")
    dim_reduction=None
    dim_reduction = DimReduction(n_dim=16, reduction=reduction)
    dim_reduction.load("test.joblib")
    encodings2 = dim_reduction.transform(x_test)
    print(np.sum(encodings-encodings2))