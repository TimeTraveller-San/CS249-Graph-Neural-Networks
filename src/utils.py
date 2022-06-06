import sklearn
import numpy as np

def preprocess(v):
    v = np.array(v)
    return sklearn.preprocessing.normalize(v.reshape(-1,1), axis=0, norm="l2").ravel()


def fetch_mask(p, mask_vecs, mask_db):
    p = preprocess(p)
    idx = mask_vecs.dot(p).argmax()
    return mask_db[idx][0], idx