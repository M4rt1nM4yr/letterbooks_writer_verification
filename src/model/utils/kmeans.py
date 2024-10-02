import os
import numpy as np
import sklearn
from sklearn.cluster import KMeans
from joblib import dump, load
import torch
from torch import nn

class KM(object):
    def __init__(self, n_components=5):
        self.n_components = n_components
        self.model = KMeans(n_clusters=n_components)

    def fit(self,x):
        self.model.fit(x)

    def predict_proba(self,x):
        # print(self.model.transform(x))
        sm = nn.Softmax(dim=-1)
        out = sm(torch.from_numpy(1-self.model.transform(x)).float()).numpy()
        return out

    def save(self, path):
        dump(self.model,path)

    def load(self, path):
        self.model = load(path)
