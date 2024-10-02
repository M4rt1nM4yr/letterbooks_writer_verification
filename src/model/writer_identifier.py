import os
import sys
sys.path.insert(0,os.getcwd())
from time import time
import multiprocessing

# print(os.environ)
import joblib
from tqdm import tqdm
import numpy as np
import random

from src.model.utils.descriptor import Descriptor
from src.model.utils.dim_reduction import DimReduction
from src.model.utils.vlad import VLAD

from src.utils.pylogger import get_pylogger

log = get_pylogger(__name__)

def worker_fit(args):
    x, descriptor_function, max_per_item = args
    return descriptor_function(x)[:max_per_item]

def worker_transform(args):
    x, descriptor_function = args
    return descriptor_function(x)["encodings"]


class WriterIdentifier(object):
    def __init__(
            self,
            descriptor_binarizer="local_otsu",
            descriptor_hellinger_norm=True,
            descriptor_sampler="sift",
            vlad_multitude=10,
            vlad_n_clusters=100,
            vlad_gmp=False,
            vlad_powernorm=True,
            reduction_dim=256,
            reduction_method="rpca",
    ):
        self.descriptor = Descriptor(
            binarizer=descriptor_binarizer,
            sampler=descriptor_sampler,
            hellinger_norm=descriptor_hellinger_norm)
        self.vlad = VLAD(multitude=vlad_multitude, n_clusters=vlad_n_clusters, gmp=vlad_gmp, powernorm=vlad_powernorm)
        self.dim_reduction_features = DimReduction(n_dim=64, reduction=reduction_method)
        self.dim_reduction_encoding = DimReduction(n_dim=reduction_dim, reduction=reduction_method)

    def fit(self, X, max_descriptors=500_000, max_per_item=2048, fast_feature_extraction=True):
        start_local = time()
        random.shuffle(X)
        n_features = 0
        features = []
        if fast_feature_extraction:
            task_data = [(x, self.descriptor, max_per_item) for x in X]
            with multiprocessing.Pool(multiprocessing.cpu_count()-1) as pool:
                # imap returns an iterator
                for f in tqdm(pool.imap(worker_fit, task_data), total=len(X)):
                    features.append(f)
                    n_features += len(f)
                    if n_features > max_descriptors:
                        break
        else:
            for x in tqdm(X):
                f = self.descriptor(x)[:max_per_item]
                features.append(f)
                n_features+=len(f)
                del(f)
                if n_features>max_descriptors:
                    break
        del(X)
        features_fit = np.vstack(features)
        log.debug("Descriptors used:", len(features_fit))
        print(f"local {time()-start_local}")
        log.debug("reducing feature dim")
        start_reduce1 = time()
        self.dim_reduction_features.fit(features_fit)
        features_reduced = [self.dim_reduction_features.transform(f) for f in features]
        print(f"reduce1 {time()-start_reduce1}")
        del(features)
        log.debug("Computing vlad encoding")
        start_vlad = time()
        self.vlad.fit(features_reduced)
        encodings = self.vlad.predict(features_reduced)
        print(f"vlad {time()-start_vlad}")
        del(features_reduced)
        start_reduce2 = time()
        self.dim_reduction_encoding.fit(encodings)
        log.debug("Reducing feature dim of encoding")
        encodings_reduced = self.dim_reduction_encoding.transform(encodings)
        print(f"reduce2 {time()-start_reduce2}")
        return encodings_reduced
        # return {"encodings":encodings_reduced, "probs":probs}

    def transform(self, X):
        start_transform = time()
        encodings, probs = list(), list()
        if not isinstance(X, list):
            X = [X]
        for x in tqdm(X):
            output = self.transform_item(x)
            encodings.append(output["encodings"])
        print(f"end transform {time()-start_transform}")
        return np.vstack(encodings)
        # return {"encodings": np.vstack(encodings), "probs": np.vstack(probs)}

    def transform_fast(self, X):
        start_transform = time()
        if not isinstance(X, list):
            X = [X]
        # Using multiprocessing pool
        task_data = [(x, self.transform_item) for x in X]
        with multiprocessing.Pool(multiprocessing.cpu_count()-1) as pool:
            encodings = list(tqdm(pool.imap(worker_transform, task_data), total=len(X)))
        print(f"end transform {time() - start_transform}")
        return np.vstack(encodings)

    def transform_item(self,x):
        features = self.descriptor(x)
        features = self.dim_reduction_features.transform(features)
        encodings = self.vlad.predict([features])
        encodings_reduced = self.dim_reduction_encoding.transform(encodings)
        return {"encodings": encodings_reduced}

    def load(self, path):
        assert os.path.exists(path)
        self.descriptor = Descriptor(**joblib.load(os.path.join(path,"descriptor.pkl")))
        self.vlad.load(os.path.join(path,"vlad.pkl"))
        self.dim_reduction_features.load(os.path.join(path,"dim_red_feat.pkl"))
        self.dim_reduction_encoding.load(os.path.join(path,"dim_red_enc.pkl"))

    def save(self, path):
        os.makedirs(path,exist_ok=True)
        descriptor_save_dict = {"binarizer":self.descriptor.binarizer,
                                "descriptor": self.descriptor.descriptor,
                                "hellinger_norm": self.descriptor.hellinger_norm,
                                "sampler": self.descriptor.sampler}
        joblib.dump(descriptor_save_dict,os.path.join(path,"descriptor.pkl"))
        self.vlad.save(os.path.join(path,"vlad.pkl"))
        self.dim_reduction_features.save(os.path.join(path,"dim_red_feat.pkl"))
        self.dim_reduction_encoding.save(os.path.join(path,"dim_red_enc.pkl"))
