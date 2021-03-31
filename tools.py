import numpy as np
import scipy.sparse as sp
import torch
import torch_geometric

from sklearn.metrics import roc_auc_score, average_precision_score

from typing import Tuple

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sp.coo_matrix(sparse_mx)

    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape

    return coords, values, shape

def get_scores(adj_orig: np.ndarray, adj_pred: torch.tensor, edges_pos:sp.coo_matrix, edges_neg:sp.coo_matrix) -> Tuple[float, float]:
    preds, preds_neg, pos, neg = [[] for i in range(0, 4)]
    edges = zip(edges_pos, edges_neg)

    for [p1, p2], [n1, n2] in edges:
        preds.append(adj_pred[p1, p2])
        preds_neg.append(adj_pred[n1, n2])
        pos.append(adj_orig[p1, p2])
        neg.append(adj_orig[n1, n2])
    
    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([pos, neg])
    roc = roc_auc_score(labels_all, preds_all)
    ap = average_precision_score(labels_all, preds_all)

    return roc, ap
    
def train_test_masks(np_adj: np.ndarray):
    adj_triu = sp.triu(np_adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(np_adj)[0]

    num_test = int(np.floor(edges.shape[0] * 0.1))
    num_val = int(np.floor(edges.shape[0] * 0.2))

    all_edge_idx = list(range(edges.shape[0]))
    # TODO: shuffle with seed to allow repeatability
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    val_edges = edges[val_edge_idx]
    test_edges = edges[test_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    def isamember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, np_adj.shape[0])
        idx_j = np.random.randint(0, np_adj.shape[0])
        if idx_i == idx_j:
            continue
        # NOTE: check if they have an edge
        if isamember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if isamember([idx_i, idx_j], np.array(test_edges_false)):
                continue
            if isamember([idx_j, idx_i], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, np_adj.shape[0])
        idx_j = np.random.randint(0, np_adj.shape[0])
        if idx_i == idx_j:
            continue
        if isamember([idx_i, idx_j], train_edges):
            continue
        if isamember([idx_j, idx_i], train_edges):
            continue
        if isamember([idx_i, idx_j], val_edges):
            continue
        if isamember([idx_j, idx_i], val_edges):
            continue
        if val_edges_false:
            if isamember([idx_i, idx_j], np.array(test_edges_false)):
                continue
            if isamember([idx_j, idx_i], np.array(test_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    # NOTE: Ensure no overlap in data division
    assert ~isamember(test_edges_false, edges_all)
    assert ~isamember(val_edges_false, edges_all)
    assert ~isamember(test_edges, train_edges)
    assert ~isamember(val_edges, train_edges)
    assert ~isamember(val_edges, test_edges)

    # TODO: add to data object as binary mask
    data = np.ones(train_edges.shape[0])

    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=np_adj.shape)
    adj_train = adj_train + adj_train.T # Recover symmetery
    adj_train = torch.tensor(adj_train.toarray()).float()

    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false