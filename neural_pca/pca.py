import numpy as np

def compute_pca(X, path=None):
    """
    X - DxN np.ndarray, datapoints
    """
    # center data
    X_mean = np.mean(X, axis=1)
    X_centered = X - X_mean[:, np.newaxis]

    # covariance matrix
    cov = np.cov(X_centered)

    # sort by eigenvalues
    eig_vals, eig_vecs = np.linalg.eigh(cov)
    sort_idcs = np.argsort(eig_vals)[::-1]

    if path is not None:
        np.save(eig_vecs, f'{path}/eigenvectors.npy')
        np.save(eig_vals, f'{path}/eigenvalues.npy')
        np.save(X_mean, f'{path}/pca_mean.npy')
    
    return eig_vals[sort_idcs], eig_vecs[:, sort_idcs], X_mean

