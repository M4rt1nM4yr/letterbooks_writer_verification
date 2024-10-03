import os
import numpy as np
from sklearn.cluster import KMeans
from joblib import dump, load

class KM(object):
    def __init__(self, n_components=5):
        self.n_components = n_components
        self.model = KMeans(n_clusters=n_components)

    def fit(self,x):
        self.model.fit(x)

    def save(self, path):
        dump(self.model,path)

    def load(self, path):
        self.model = load(path)
