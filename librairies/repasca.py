"""repasca.py

Python translation of the MATLAB rep_asca function (REP-ASCA repeatability reduction).

Functions:
  rep_asca(X, design, X_rep, d_rep, klimit)

Helpers included: conjtodis (conjonctif -> disjonctif), pca_old (PCA fallback)

This implementation uses the local `asca` module if available to recompute ASCA
after orthogonalization. If `asca` is missing, it raises an informative error.
"""
from typing import Any, Dict, List, Tuple
import numpy as np
from numpy.linalg import svd
from scipy.linalg import pinv


def conjtodis(cl: np.ndarray, lbcl: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    """Convert conjunctive coding to disjunctive (one-hot) representation.

    Returns (cld, clname) where cld is (n_samples, n_classes) 0/1 matrix and
    clname are the unique labels used.
    """
    cl = np.asarray(cl)
    if cl.ndim == 1:
        cl = cl.reshape(-1, 1)

    if lbcl is None:
        # unique rows
        # For consistency with MATLAB unique on rows
        # we convert each row to a tuple for uniqueness
        tuples = [tuple(r) for r in cl]
        unique = sorted(set(tuples))
        nbcl = len(unique)
        index_map = {u: i for i, u in enumerate(unique)}
        n = np.array([index_map[t] for t in tuples]) + 1
        clname = np.array([list(u) for u in unique], dtype=object)
    else:
        lb = np.asarray(lbcl)
        if lb.ndim == 1:
            lb = lb.reshape(-1, 1)
        all_rows = np.vstack([lb, cl])
        tuples = [tuple(r) for r in all_rows]
        unique = []
        seen = set()
        for t in tuples:
            if t not in seen:
                seen.add(t)
                unique.append(t)
        nbcl = lb.shape[0]
        # mapping for all_rows
        index_map = {u: i for i, u in enumerate(unique)}
        n_all = np.array([index_map[t] for t in tuples]) + 1
        n = n_all[nbcl:]
        clname = np.array([list(u) for u in unique], dtype=object)

    nbech = cl.shape[0]
    cld = np.zeros((nbech, nbcl), dtype=int)
    for i in range(nbcl):
        idx = np.where(n == (i + 1))[0]
        if idx.size > 0:
            cld[idx, i] = 1

    return cld, clname


def pca_old(X: np.ndarray, center: bool = True, scale: bool = False, klimit: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """Simple PCA fallback returning (scores, loadings).

    This function returns (scores, loadings) similar to MATLAB outputs used in the original code.
    - X : (n_samples, n_features)
    - center : center the data
    - klimit : number of components to return
    """
    X = np.asarray(X, dtype=float)
    if center:
        Xc = X - X.mean(axis=0)
    else:
        Xc = X.copy()
    # simple SVD
    U, s, Vt = svd(Xc, full_matrices=False)
    # scores = U * s
    scores = U * s
    loadings = Vt.T
    k = min(klimit, loadings.shape[1])
    return scores[:, :k], loadings[:, :k]


def rep_asca(X: np.ndarray, design: np.ndarray, X_rep: np.ndarray, d_rep: np.ndarray, klimit: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """Main function ported from MATLAB rep_asca.

    Returns:
      explained_var : array shape (klimit+1, n_factors+1) with explained variances
      L_err : loadings associated with repeatability error (n_features x n_components)
    """
    # Build disjunctive matrix from repeated sample ids
    Ds, _ = conjtodis(d_rep)
    Ds = np.asarray(Ds, dtype=float)

    # Compute between-sample effect (BS) and within-sample residual (WS)
    # BS = Ds * inv(Ds'*Ds) * Ds' * X_rep  -> using pinv
    # In Python: Bs = Ds @ inv(Ds.T @ Ds) @ Ds.T @ X_rep
    inv_term = pinv(Ds.T @ Ds)
    BS = Ds @ inv_term @ Ds.T @ X_rep
    WS = X_rep - BS

    # PCA on WS to obtain error loadings
    _, L_err = pca_old(WS, center=True, scale=False, klimit=klimit)

    # Now compute explained variances for increasing numbers of components removed
    # First, compute baseline ASCA explained variances (no removal)
    try:
        import asca as asca_module
    except Exception as e:
        raise ImportError('Local asca module is required to run rep_asca; import failed: {}'.format(e))

    opts = asca_module.asca_light('options')
    opts['nperm'] = 1
    results_asca = asca_module.asca_light(X, design, opts)
    list_factor = results_asca['TermLabels'][1:]

    # prepare containers
    n_factors = len(list_factor)
    explained_var = np.zeros((L_err.shape[1] + 1, n_factors + 1))
    list_factor_name = []

    # baseline (row 0)
    for j, lab in enumerate(list_factor):
        key = 'X' + ''.join(lab.split())
        explained_var[0, j] = results_asca.get(key, {}).get('EffectExplVar', 0.0)
        list_factor_name.append(key)
    explained_var[0, n_factors] = results_asca['XRes']['EffectExplVar']
    list_factor_name.append('XRes')

    # for i components removed
    for i in range(L_err.shape[1]):
        k_W = L_err[:, : (i + 1)].T  # shape (i+1, n_features)
        X_bar = X - (X @ k_W.T) @ k_W
        results_asca = asca_module.asca_light(X_bar, design, opts)
        for j, lab in enumerate(list_factor):
            key = 'X' + ''.join(lab.split())
            explained_var[i + 1, j] = results_asca.get(key, {}).get('EffectExplVar', 0.0)
            # ensure factor name list updated
            if len(list_factor_name) <= j:
                list_factor_name.append(key)
        explained_var[i + 1, n_factors] = results_asca['XRes']['EffectExplVar']
        if len(list_factor_name) <= n_factors:
            list_factor_name.append('XRes')

    return explained_var, L_err
def rep_asca_applied(X,d, loadings_error, k,nperm = 1) -> Dict[str, Any]:
    """Apply REP-ASCA reduction with specified number of components from loadings_error.

    Returns:
      model : dict with ASCA results after removing k components from loadings_error
    """
    try:
        import asca as asca_module
    except Exception as e:
        raise ImportError('Local asca module is required to run rep_asca_applied; import failed: {}'.format(e))

    if k > loadings_error.shape[1]:
        raise ValueError(f"Requested k={k} exceeds available components in loadings_error with shape {loadings_error.shape}")

    k_W = loadings_error[:, :k].T  # shape (k, n_features)
    X_bar = X - (X @ k_W.T) @ k_W

    opts = asca_module.asca('options')
    opts['nperm'] = nperm
    model = asca_module.asca(X_bar, d, opts)
    return model

if __name__ == '__main__':
    print('repasca module loaded. Use rep_asca(X, design, X_rep, d_rep, klimit)')
    klimit = 10
    
    import scipy.io as sio
    import os
    import sys
    # Cell 3: Load MATLAB .mat data
    mat_path = os.path.join(r'D:\IA\REPTOPY', 'data.mat')
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"data.mat not found at {mat_path}. Please place your data.mat there.")

    mat = sio.loadmat(mat_path)
    print('Variables in data.mat:', sorted([k for k in mat.keys() if not k.startswith('__')]))

    # Map expected variables with fallbacks
    X = mat.get('X', None)
    if X is not None:
        X = np.asarray(X, dtype=float)

    d = mat.get('d', None)
    X_rep = mat.get('X_rep', None)
    d_rep = mat.get('d_rep', None)
    palette = mat.get('palette', None)
    lambda_ = mat.get('lambda', None)

    print('X shape:', None if X is None else X.shape)
    print('X_rep shape:', None if X_rep is None else np.asarray(X_rep).shape)
    print('d shape:', None if d is None else np.asarray(d).shape)
    print('d_rep shape:', None if d_rep is None else np.asarray(d_rep).shape)

    # Run asca and ascasca
    if X is not None and d is not None:

        a, b  = rep_asca(X, d, X_rep, d_rep, klimit=klimit)
        print('Explained variances (cum) shape:', a.shape)
        print('Loadings_error shape:', b.shape)
        import matplotlib.pyplot as plt

        # Suppose 'a' is the explained_var matrix (not cumulative)
        plt.figure(figsize=(8, 5))
        labels = ['A', 'B', 'AxB', 'R']
        linestyles = ['-', '--', '-.', ':']

        for i in range(a.shape[1]):
            plt.plot(range(a.shape[0]), a[:, i], linestyle=linestyles[i % len(linestyles)], label=labels[i])

        plt.xlabel('Number of components removed')
        plt.ylabel('Explained variance (%)')
        plt.title('Explained variance by factor')
        plt.legend()
        plt.tight_layout()
        plt.savefig('explained_variance_plot.png')
        plt.close()
        print("Plot saved as explained_variance_plot.png")
        print(a)
        print('---')
        
        model = rep_asca_applied(X, d, b, k=2)
        
        scores = model['XB']['SCA']['Model']['ScoreswithRes']
        plt.figure(figsize=(6, 5))
        plt.scatter(scores[:, 0], scores[:, 1], c='b', alpha=0.7)
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.title('SCA Scores (first two dimensions)')
        plt.tight_layout()
        plt.savefig('sca_scores_scatter.png')
        plt.close()
        print("Scatter plot saved as sca_scores_scatter.png")
        
        