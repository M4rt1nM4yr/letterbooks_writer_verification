import sklearn
import numpy as np
import sklearn.preprocessing


def distances(encs):
    """
    compute pairwise distances

    parameters:
        desc = cPickle.load(ff) #, encoding='latin1')
        encs:  TxK*D encoding matrix
    returns: TxT distance matrix
    """
    # compute cosine distance = dot product between l2-normalized
    # descriptors
    dists = 1.0 - encs.dot(encs.T)
    # mask out distance with itself
    np.fill_diagonal(dists, np.finfo(dists.dtype).max)
    return dists

def evaluate(encs, labels, topk=False,print_results=False):
    """
    evaluate encodings assuming using associated labels
    parameters:
        encs: TxK*D encoding matrix
        labels: array/list of T labels
    """

    encs = sklearn.preprocessing.normalize(encs, norm="l2", axis=1)

    dist_matrix = distances(encs)

    if topk:
        computeStats(name="fk", dist_matrix=dist_matrix, labels_probe=labels)
        print("backup")

    # sort each row of the distance matrix
    indices = dist_matrix.argsort()
    classification_list = list()
    closest_match = np.array(labels)[indices[:, 0]]
    classification_list = np.array(labels) == closest_match

    n_encs = len(encs)

    mAP = []
    correct = 0
    for r in range(n_encs):
        precisions = []
        rel = 0
        for k in range(n_encs - 1):
            if labels[indices[r, k]] == labels[r]:
                rel += 1
                precisions.append(rel / float(k + 1))
                if k == 0:
                    correct += 1
        ## This is just a hack for labels that appear only once
        #if len(precisions) !=0:
        avg_precision = np.mean(precisions)
        mAP.append(avg_precision)
    mAP = np.mean(mAP)
    if print_results:
        print('Top-1 accuracy: {} - mAP: {}\n'.format(float(correct) / n_encs, mAP))
    return {
        "top1": float(correct) / n_encs,
        "mAP": mAP,
        "classification_results": classification_list,
    }


def evaluate_dist(dist_matrix, labels_query, labels_pool, topk=False, print_results=True):
    indices = dist_matrix.argsort()
    print(dist_matrix.shape, len(labels_query), len(labels_pool))
    mAP = []
    correct = 0
    for idx_q in range(len(labels_query)):
        precisions = []
        rel = 0
        for idx_p in range(len(labels_pool)):
            if labels_pool[indices[idx_q, idx_p]] == labels_query[idx_q]:
                rel += 1
                precisions.append(rel / float(idx_p + 1))
                if idx_p == 0:
                    correct += 1
        avg_precision = np.mean(precisions)
        mAP.append(avg_precision)
    mAP = np.mean(mAP)
    if print_results:
        print('Top-1 accuracy: {} - mAP: {}\n'.format(float(correct) / len(labels_query), mAP))
    if topk:
        computeStats(name="fk", dist_matrix=dist_matrix, labels_probe=labels_query, labels_gallery=labels_pool)
    return float(correct) / len(labels_query), mAP


def computeStats(name, dist_matrix, labels_probe,
                 labels_gallery=None,
                 parallel=False, distance=True, nprocs=4,
                 eval_method='cosine', groups=None,):
    n_probe, n_gallery = dist_matrix.shape
    # often enough we make a leave-one-out-cross-validation
    # here we don't have a separation probe / gallery
    if labels_gallery is None:
        n_gallery -= 1
        labels_gallery = labels_probe
        # assert not needed or?
        assert(dist_matrix.shape[0] == dist_matrix.shape[1])
    assert(dist_matrix.shape[0] == len(labels_probe))
    assert(dist_matrix.shape[1] == len(labels_gallery))
    ind_probe = len(set(labels_probe))
    ind_gall = len(set(labels_gallery))
    labels_gallery = np.array(labels_gallery)
    labels_probe = np.array(labels_probe)
    print('number of probes: {}, individuals: {}'.format(n_probe, ind_probe))
    print('number of gallery: {}, individuals: {}'.format(n_gallery, ind_probe))
    indices = dist_matrix.argsort()

    if not distance:
        indices = indices[:, ::-1]

    # compute relevance list
    def loop_descr(r):
        rel_list = np.zeros((1, n_gallery))
        not_correct = []
        rel_cnt = 0
        for k in range(0, n_gallery):
            # if in the same group, then let's ignore it totally
            # -> not only ignore it but also don't forward the rank, i.e.
            # don't update the rel_list
            if groups is not None and groups[indices[r, k]] == groups[r]:
                continue
            else:
                if labels_gallery[indices[r, k]] == labels_probe[r]:
                    rel_list[0, rel_cnt] = 1
                elif k == 0:
                    not_correct.append((r, indices[r, k]))
                rel_cnt += 1
        return rel_list, not_correct

    # if parallel:
    #     all_rel, top1_fail = zip(*pc.parmap(loop_descr, range(n_probe), nprocs=nprocs))
    # else:
    all_rel, top1_fail = zip(*map(loop_descr, range(n_probe)))
        # make all computations with the rel-matrix
    rel_conc = np.concatenate(all_rel, 0)
    # are there any zero rows?
    z_rows = np.sum(rel_conc, 1)
    n_real2 = np.count_nonzero(z_rows)
    if n_real2 != rel_conc.shape[0]:
        print('WARNING: not for each query exist also a label in the gallery'
              '({} / {})!'.format(n_real2, rel_conc.shape[0]))
    rel_mat = rel_conc[z_rows > 0]
    print('rel_mat.shape:', rel_mat.shape)
    prec_mat = np.zeros(rel_mat.shape)
    # TODO: this is still quite slow for large matrices
    # -> parallelize?
    soft2 = np.zeros(50)
    hard2 = np.zeros(4)
    for i in range(n_gallery):
        # TODO this could be done more efficiently
        # if we take the previous sum! -> cumsum
        rel_sum = np.sum(rel_mat[:, :i + 1], 1)
        # note: we could do this after the for loop (with correct shaped
        # i+1 vector: arange(1,n_gallery+1) and a rel_sum_mat
        # -> create rel_sum_mat
        prec_mat[:, i] = rel_sum / (i + 1)
        # could be partially speed up if outside w. rel_sum_mat
        if i < 20:
            soft2[i] = np.count_nonzero(rel_sum > 0) / float(n_real2)
        if i < 4:
            hh = rel_sum[np.isclose(rel_sum, (i + 1))]
            #            print 'i: {} len(hh): {}'.format(i, len(hh))
            hard2[i] = len(hh) / float(n_real2)
    print('correct: {} / {}'.format(np.sum(rel_mat[:, 0]), n_real2))
    print('top-k soft:', soft2[:10])
    print('top-k hard:', hard2)
    # Average precisions
    ap = []
    for i in range(n_real2):
        ap.append(np.mean(prec_mat[i][rel_mat[i] == 1]))
    # mAP
    map1 = np.mean(ap)
    print('mAP mean(ap):', map1)
    # precision@x scores
    p2 = np.sum(prec_mat[:, 1]) / n_real2
    p3 = np.sum(prec_mat[:, 2]) / n_real2
    p4 = np.sum(prec_mat[:, 3]) / n_real2
    print('mean P@2,P@3,P@4:', p2, p3, p4)


if __name__ == "__main__":
    a = np.random.randn(5,8)
    b = np.random.randn(5,8)
    labels = ["0052-1-0", "0052-1-1", "0052-1-2", "0052-1-3", "0052-1-4"]
    out = evaluate(np.concatenate([a,b],axis=0), labels+labels, print_results=True, topk=True)
    print(np.sum(out["classification_results"]))