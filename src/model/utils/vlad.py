import os
from tqdm import tqdm
import numpy as np
import cv2 as cv
from sklearn.linear_model import Ridge
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize
from joblib import dump, load
from timeit import default_timer as time


def compute_assignments(descriptors, clusters):
    """
    compute assignment matrix
    parameters:
        descriptors: TxD descriptor matrix
        clusters: KxD cluster matrix
    returns: TxK assignment matrix
    """
    descriptors = np.nan_to_num(descriptors, nan=0)
    # compute nearest neighbors
    matcher = cv.BFMatcher(cv.NORM_L2)
    matches = matcher.knnMatch(descriptors.astype(np.float32),
                               clusters.astype(np.float32),
                               k=1)

    # create hard assignment
    assignment = np.zeros((len(descriptors), len(clusters)))
    for e, m in enumerate(matches):
        if len(m) > 0:
            assignment[e, m[0].trainIdx] = 1

    return assignment

class VLAD():
    def __init__(self, multitude=5, n_clusters=100, powernorm=True, gmp=False, gamma=1000):
        self.powernorm = powernorm
        self.n_clusters = n_clusters
        self.gmp = gmp
        self.gamma = gamma
        self.cluster_centers = []
        self.kmeans_models = [MiniBatchKMeans(n_clusters=n_clusters, batch_size=500000) for i in range(multitude)]

    def fit(self, X):
        if isinstance(X, list):
            X = np.vstack(X)
        else:
            X = X.reshape(-1, X.shape[-1])

        print(f"computing cluster {len(self.kmeans_models)*self.n_clusters} centers for X {X.shape}")
        self.cluster_centers = []
        for kmeans_model in tqdm(self.kmeans_models):
            kmeans_model.fit(X)
            self.cluster_centers.append(kmeans_model.cluster_centers_)
        self.cluster_centers

    # TODO: fk for loops!
    def predict(self,X):
        m_encodings = []
        for cluster_centers in self.cluster_centers:
            encodings = []
            for x in X:
                encodings.append(self._vlad(x, cluster_centers))
            m_encodings.append(np.concatenate(encodings,axis=0))
        return np.concatenate(m_encodings, axis=1)

    def _vlad(self, x, cluster_centers):
        assignment = compute_assignments(x, cluster_centers)
        T, D = x.shape
        f_enc = np.zeros((D * cluster_centers.shape[0]), dtype=np.float32)
        for k in range(cluster_centers.shape[0]):
            # it's faster to select only those descriptors that have
            # this cluster as nearest neighbor and then compute the
            # difference to the cluster center than computing the differences
            # first and then select

            # get only descriptors that are possible for this cluster
            nn = x[assignment[:, k] > 0]
            # it can happen that we don't have any descriptors associated for
            # this cluster
            if len(nn) == 0:
                f_enc[k * D:(k + 1) * D] = 0
            else:
                # compute residuals
                res = nn - cluster_centers[k]

                # e) generalized max pooling
                if self.gmp:
                    clf = Ridge(alpha=self.gamma,
                                fit_intercept=False,
                                solver='sparse_cg',
                                max_iter=500)  # conjugate gradient
                    clf.fit(res, np.ones((len(nn))))
                    f_enc[k * D:(k + 1) * D] = clf.coef_
                else:
                    enc = np.sum(res, axis=0)
                    f_enc[k * D:(k + 1) * D] = enc

        # c) power normalization
        if self.powernorm:
            f_enc = np.sign(f_enc) * np.sqrt(np.abs(f_enc))
        # l2 normalization
        f_enc = normalize(f_enc.reshape(1, -1), norm='l2')

        return f_enc

    def save(self,path):
        save_dict = {"cluster_centers": self.cluster_centers,
                     "gmp": self.gmp,
                     "gamma": self.gamma,
                     "powernorm": self.powernorm,
                     }
        dump(save_dict,path)

    def load(self, path):
        saved_dict = load(path)
        self.cluster_centers = saved_dict["cluster_centers"]
        self.gmp = saved_dict["gmp"]
        self.gamma = saved_dict["gamma"]
        self.powernorm = saved_dict["powernorm"]

if __name__ == "__main__":
    X_train = np.random.randn(10,200,64)
    X_train2 = [np.random.randn(200,64), np.random.randn(2000,64), np.random.randn(3000,64)]

    X_test = np.random.randn(1,200,64)
    mvlad = VLAD(multitude=5, n_clusters=100, gmp=False, powernorm=True)
    mvlad.fit(X_train2)
    encodings = mvlad.predict(X_test)
    mvlad.save("vlad.joblib")
    mvlad = None
    mvlad = VLAD(multitude=2, n_clusters=100, gmp=False, powernorm=True)
    mvlad.load("vlad.joblib")
    start = time()
    encodings2 = mvlad.predict(X_test)
    end = time()
    print(end-start)
    # print(np.sum(encodings-encodings2))