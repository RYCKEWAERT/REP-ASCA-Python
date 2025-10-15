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
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
from scipy.linalg import pinv, svd
from scipy.sparse.linalg import svds
import os


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


def asca(*args) -> Dict[str, Any]:
    """Main entry point mirroring the MATLAB function.

    Usage:
      model = asca('options')
      model = asca(X, desmatrix)
      model = asca(X, desmatrix, opt)

    Returns a nested dictionary `model` containing results.
    """
    # default options
    default_opt = {
        'preproc': 'none',
        'reducedmodel': 'standard',
        'permtest': 'on',
        'permfacts': 'all',
        'nperm': 10000,
        'bootstrap': 'off',
        'bootmatrix': 'original',
        'bootsave': 'confint',
        'nboot': 1000,
        'plots': 'on',
        'confl': 0.95,
    }

    if len(args) == 1 and args[0] == 'options':
        return default_opt

    if len(args) == 2:
        X = np.asarray(args[0], dtype=float)
        desmatrix = args[1]
        opt = default_opt.copy()
    elif len(args) == 3:
        X = np.asarray(args[0], dtype=float)
        desmatrix = args[1]
        opt = args[2].copy()
        # fill missing options from defaults
        for k, v in default_opt.items():
            opt.setdefault(k, v)
    else:
        raise ValueError('Invalid arguments. Use asca("options") or asca(X, desmatrix[, opt])')

    ntot = X.shape[0]

    # preprocessing
    if opt['preproc'] == 'none':
        Xp = X.copy()
    elif opt['preproc'] == 'auto':
        Xp = X / np.std(X, axis=0, ddof=0)
    elif opt['preproc'] == 'pareto':
        Xp = X / np.sqrt(np.std(X, axis=0, ddof=0))
    else:
        raise ValueError(f"Unknown preproc option: {opt['preproc']}")

    model: Dict[str, Any] = {}
    model['Xdata'] = {
        'PreprData': Xp,
        'TotalSSQ': float(np.sum(Xp ** 2))
    }
    model['Design'] = desmatrix

    dmain = gendmat(desmatrix)
    dmatrices, desterms, deslabels, desorder = createdesign(dmain)

    Xd = Xp.copy()

    ssqtot = None
    for i in range(len(dmatrices)):
        Dm = dmatrices[i]
        # pseudo-inverse projection
        Xeff = Dm @ pinv(Dm) @ Xd
        ssqEff = float(np.sum(Xeff ** 2))
        Xd = Xd - Xeff
        l = deslabels[i].strip()
        model_key = f'X{l}'
        model.setdefault(model_key, {})
        model[model_key]['EffectMatrix'] = Xeff
        model[model_key]['EffectSSQ'] = ssqEff

        if i == 0:
            ssqtot = float(np.sum(Xd ** 2))
            model['Xdata']['CenteredData'] = Xd
            model['Xdata']['CenteredSSQ'] = ssqtot
        else:
            expVar = 100.0 * (ssqEff / ssqtot) if ssqtot != 0 else 0.0
            model[model_key]['EffectExplVar'] = expVar

        model[model_key]['DesignMatrix'] = Dm
        model[model_key]['DesignTerms'] = desterms[i]
        model[model_key]['EffectLabel'] = deslabels[i].strip()
        model[model_key]['TermOrder'] = int(desorder[i])

    model['XRes'] = {
        'EffectMatrix': Xd,
        'EffectSSQ': float(np.sum(Xd ** 2)),
        'EffectExplVar': 100.0 * (float(np.sum(Xd ** 2)) / ssqtot) if ssqtot != 0 else 0.0,
        'EffectLabel': 'Res',
        'TermOrder': int(max(desorder) + 1)
    }

    # Reduced matrices
    for i in range(1, len(dmatrices)):
        l = deslabels[i].strip()
        model_key = f'X{l}'
        if isinstance(opt['reducedmodel'], str) and opt['reducedmodel'] == 'standard':
            # remove factors except the current one
            remfact = desterms[0:i] + desterms[i+1:]
        else:
            remfact = opt['reducedmodel'][i]
        Xx = model['Xdata']['CenteredData'].copy()
        for rf in remfact:
            # rf contains indices of factors; in MATLAB they used letters A,B,C -> 1,2,3
            # Here desterms stores arrays of indices; to map to labels we use 1-based indexing
            if isinstance(rf, (list, tuple, np.ndarray)):
                for r in rf:
                    m = chr(64 + int(r))
                    Xx = Xx - model[f'X{m}']['EffectMatrix']
            else:
                m = chr(64 + int(rf))
                Xx = Xx - model[f'X{m}']['EffectMatrix']
        model[model_key]['ReducedMatrix'] = Xx

    model['TermLabels'] = deslabels
    model['Options'] = opt

    # permutation tests
    # model = ascaptest(model)
    
    model = ascaptest_fast(model)

    # SCA models
    model = ascasca(model)
    
    

    # bootstrap
    model = ascaboot(model)

    return model


def asca_light(*args) -> Dict[str, Any]:
    """Main entry point mirroring the MATLAB function.

    Usage:
      model = asca('options')
      model = asca(X, desmatrix)
      model = asca(X, desmatrix, opt)

    Returns a nested dictionary `model` containing results.
    """
    # default options
    default_opt = {
        'preproc': 'none',
        'reducedmodel': 'standard',
        'permtest': 'on',
        'permfacts': 'all',
        'nperm': 10000,
        'bootstrap': 'off',
        'bootmatrix': 'original',
        'bootsave': 'confint',
        'nboot': 1000,
        'plots': 'on',
        'confl': 0.95,
    }

    if len(args) == 1 and args[0] == 'options':
        return default_opt

    if len(args) == 2:
        X = np.asarray(args[0], dtype=float)
        desmatrix = args[1]
        opt = default_opt.copy()
    elif len(args) == 3:
        X = np.asarray(args[0], dtype=float)
        desmatrix = args[1]
        opt = args[2].copy()
        # fill missing options from defaults
        for k, v in default_opt.items():
            opt.setdefault(k, v)
    else:
        raise ValueError('Invalid arguments. Use asca("options") or asca(X, desmatrix[, opt])')

    ntot = X.shape[0]

    # preprocessing
    if opt['preproc'] == 'none':
        Xp = X.copy()
    elif opt['preproc'] == 'auto':
        Xp = X / np.std(X, axis=0, ddof=0)
    elif opt['preproc'] == 'pareto':
        Xp = X / np.sqrt(np.std(X, axis=0, ddof=0))
    else:
        raise ValueError(f"Unknown preproc option: {opt['preproc']}")

    model: Dict[str, Any] = {}
    model['Xdata'] = {
        'PreprData': Xp,
        'TotalSSQ': float(np.sum(Xp ** 2))
    }
    model['Design'] = desmatrix

    dmain = gendmat(desmatrix)
    dmatrices, desterms, deslabels, desorder = createdesign(dmain)

    Xd = Xp.copy()

    ssqtot = None
    for i in range(len(dmatrices)):
        Dm = dmatrices[i]
        # pseudo-inverse projection
        Xeff = Dm @ pinv(Dm) @ Xd
        ssqEff = float(np.sum(Xeff ** 2))
        Xd = Xd - Xeff
        l = deslabels[i].strip()
        model_key = f'X{l}'
        model.setdefault(model_key, {})
        model[model_key]['EffectMatrix'] = Xeff
        model[model_key]['EffectSSQ'] = ssqEff

        if i == 0:
            ssqtot = float(np.sum(Xd ** 2))
            model['Xdata']['CenteredData'] = Xd
            model['Xdata']['CenteredSSQ'] = ssqtot
        else:
            expVar = 100.0 * (ssqEff / ssqtot) if ssqtot != 0 else 0.0
            model[model_key]['EffectExplVar'] = expVar

        model[model_key]['DesignMatrix'] = Dm
        model[model_key]['DesignTerms'] = desterms[i]
        model[model_key]['EffectLabel'] = deslabels[i].strip()
        model[model_key]['TermOrder'] = int(desorder[i])

    model['XRes'] = {
        'EffectMatrix': Xd,
        'EffectSSQ': float(np.sum(Xd ** 2)),
        'EffectExplVar': 100.0 * (float(np.sum(Xd ** 2)) / ssqtot) if ssqtot != 0 else 0.0,
        'EffectLabel': 'Res',
        'TermOrder': int(max(desorder) + 1)
    }

    # Reduced matrices
    for i in range(1, len(dmatrices)):
        l = deslabels[i].strip()
        model_key = f'X{l}'
        if isinstance(opt['reducedmodel'], str) and opt['reducedmodel'] == 'standard':
            # remove factors except the current one
            remfact = desterms[0:i] + desterms[i+1:]
        else:
            remfact = opt['reducedmodel'][i]
        Xx = model['Xdata']['CenteredData'].copy()
        for rf in remfact:
            # rf contains indices of factors; in MATLAB they used letters A,B,C -> 1,2,3
            # Here desterms stores arrays of indices; to map to labels we use 1-based indexing
            if isinstance(rf, (list, tuple, np.ndarray)):
                for r in rf:
                    m = chr(64 + int(r))
                    Xx = Xx - model[f'X{m}']['EffectMatrix']
            else:
                m = chr(64 + int(rf))
                Xx = Xx - model[f'X{m}']['EffectMatrix']
        model[model_key]['ReducedMatrix'] = Xx

    model['TermLabels'] = deslabels
    model['Options'] = opt



    return model


def createdesign(designmat: Sequence[np.ndarray]) -> Tuple[List[np.ndarray], List[List[int]], List[str], np.ndarray]:
    """Create design (effect) matrices for all terms from factor design matrices.

    For each term (combination of factors) we build a matrix whose columns are
    all element-wise products of one column from each factor's design matrix
    (or a column of ones when the factor is not present). This mirrors the
    construction of interaction columns in the original MATLAB code but is
    more explicit and robust.
    """
    from itertools import product

    nfact = len(designmat)
    ns = designmat[0].shape[0]

    # build full factorial indicator matrix: rows enumerate which factors are present
    indmat = fullfact([2] * nfact)
    nmat = indmat.shape[0]

    dmatrices: List[np.ndarray] = []
    desterms: List[List[int]] = []
    deslabels: List[str] = []
    desorder = np.zeros(nmat, dtype=int)

    for i in range(nmat):
        # For this term, prepare per-factor column matrices: either the factor's
        # design matrix (ns x nc) or a column of ones (ns x 1)
        factor_cols: List[np.ndarray] = []
        for j in range(nfact):
            if indmat[i, j] == 1:
                Aj = np.asarray(designmat[j])
                # ensure 2D (ns x cols)
                if Aj.ndim == 1:
                    Aj = Aj.reshape(-1, 1)
                factor_cols.append(Aj)
            else:
                factor_cols.append(np.ones((ns, 1), dtype=float))

        # Now form all combinations of one column per factor and multiply element-wise
        cols = []
        # list of column counts
        col_counts = [fc.shape[1] for fc in factor_cols]
        for idxs in product(*[range(c) for c in col_counts]):
            col = np.ones(ns, dtype=float)
            for fidx, cidx in enumerate(idxs):
                col = col * factor_cols[fidx][:, cidx]
            cols.append(col.reshape(ns, 1))

        if len(cols) == 0:
            dm = np.ones((ns, 1), dtype=float)
        else:
            dm = np.hstack(cols)

        dmatrices.append(dm)
        terms = list(np.where(indmat[i, :] == 1)[0] + 1)
        desterms.append(terms)
        deslabels.append(''.join([chr(64 + idx) for idx in terms]) if len(terms) > 0 else '')
        desorder[i] = int(len(terms))

    # Sort terms by interaction order (number of factors) then by label for stability
    sort_idx = np.lexsort((np.array(deslabels, dtype='U'), desorder))
    dmatrices = [dmatrices[k] for k in sort_idx]
    desterms = [desterms[k] for k in sort_idx]
    deslabels = [deslabels[k] for k in sort_idx]
    desorder = desorder[sort_idx]

    # First term should be the mean
    if len(deslabels) > 0:
        deslabels[0] = 'Mean'

    return dmatrices, desterms, deslabels, desorder


def gendmat(designmat: np.ndarray) -> List[np.ndarray]:
    ns = designmat.shape[0]
    nfact = designmat.shape[1]
    dmain: List[np.ndarray] = []
    for i in range(nfact):
        lev = np.unique(designmat[:, i])
        nl = lev.size
        dmat = np.zeros((ns, nl - 1))
        for j in range(nl - 1):
            dmat[designmat[:, i] == lev[j], j] = 1
        mask = designmat[:, i] == lev[-1]
        if dmat.size > 0:
            dmat[mask, :] = -1
        dmain.append(dmat)
    return dmain


def fullfact(levels: Sequence[int]) -> np.ndarray:
    # replicates MATLAB fullfact for 2-level factors
    grids = [np.arange(1, lv + 1) for lv in levels]
    mesh = np.array(np.meshgrid(*grids)).T.reshape(-1, len(levels))
    return mesh


def ascaptest(ascamodel: Dict[str, Any]) -> Dict[str, Any]:
    pmodel = ascamodel
    dlab = ascamodel['TermLabels']

    signfacts: List[str] = []

    if ascamodel['Options']['permtest'] == 'on':
        sc = 0
    else:
        sc = 0

    for i in range(1, len(dlab)):
        l = dlab[i].strip()
        model_key = f'X{l}'
        if ascamodel['Options']['permtest'] == 'on':
            if isinstance(ascamodel['Options']['permfacts'], str) and ascamodel['Options']['permfacts'] == 'all':
                Xr = pmodel[model_key]['ReducedMatrix']
                Dr = pmodel[model_key]['DesignMatrix']
                ssqp = ptest(Xr, Dr, ascamodel['Options']['nperm'])
                seff = pmodel[model_key]['EffectSSQ']
                p = np.sum(ssqp >= seff) / ascamodel['Options']['nperm']
                if p <= 0.05:
                    signfacts.append(l)

                pmodel[model_key].setdefault('EffectSignif', {})
                pmodel[model_key]['EffectSignif']['NullDistr'] = ssqp
                pmodel[model_key]['EffectSignif']['p'] = float(p)
            else:
                if l in ascamodel['Options']['permfacts']:
                    Xr = pmodel[model_key]['ReducedMatrix']
                    Dr = pmodel[model_key]['DesignMatrix']
                    ssqp = ptest(Xr, Dr, ascamodel['Options']['nperm'])
                    seff = pmodel[model_key]['EffectSSQ']
                    p = np.sum(ssqp >= seff) / ascamodel['Options']['nperm']
                    if p <= 0.05:
                        signfacts.append(l)

                    pmodel[model_key].setdefault('EffectSignif', {})
                    pmodel[model_key]['EffectSignif']['NullDistr'] = ssqp
                    pmodel[model_key]['EffectSignif']['p'] = float(p)
                else:
                    pmodel[model_key].setdefault('EffectSignif', {})
                    pmodel[model_key]['EffectSignif']['NullDistr'] = np.array([])
                    pmodel[model_key]['EffectSignif']['p'] = None
        else:
            pmodel[model_key].setdefault('EffectSignif', {})
            pmodel[model_key]['EffectSignif']['NullDistr'] = np.array([])
            pmodel[model_key]['EffectSignif']['p'] = None

    pmodel['SignificantTerms'] = signfacts
    return pmodel


def _compute_ssq_for_permutation(P: np.ndarray, X: np.ndarray, hh: np.ndarray) -> float:
    """Compute SSQ for a single permutation hh using precomputed projection P.

    Uses the identity P_h = H P H^T and computes Xpp = P_h @ X efficiently via
    temporary permutation: Xpp = (P @ X[hh_inv, :])[hh, :]
    where hh_inv = argsort(hh).
    """
    # compute inverse permutation
    hh_inv = np.argsort(hh)
    temp = P @ X[hh_inv, :]
    Xpp = temp[hh, :]
    return float(np.sum(Xpp ** 2))


def ptest_fast(X: np.ndarray, D: np.ndarray, nperm: int, n_jobs: Optional[int] = 1, chunk_size: int = 1000) -> np.ndarray:
    """Faster permutation test.

    Replaces repeated computation of pinv(D(hh,:)) by precomputing P = D @ pinv(D)
    and using permutation of P (via row/col reindexing) implicitly. Supports
    processing permutations in chunks and optional parallel execution.

    Parameters
    - X: (ns x p) data
    - D: (ns x q) design matrix
    - nperm: number of permutations
    - n_jobs: number of worker processes (1 = serial). If None or 1, runs serial.
    - chunk_size: number of permutations handled per chunk to limit memory.
    """
    import concurrent.futures

    ns = X.shape[0]
    P = D @ pinv(D)

    ssqp = np.zeros(nperm, dtype=float)

    def gen_permutation_chunk(start, size):
        rng = np.random.default_rng()
        for _ in range(size):
            yield rng.permutation(ns)

    idx = 0
    while idx < nperm:
        this_chunk = min(chunk_size, nperm - idx)
        perms = [np.random.permutation(ns) for _ in range(this_chunk)]

        if n_jobs is None or n_jobs == 1:
            # serial
            for k, hh in enumerate(perms):
                ssqp[idx + k] = _compute_ssq_for_permutation(P, X, hh)
        else:
            # parallel
            with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as exe:
                futures = [exe.submit(_compute_ssq_for_permutation, P, X, hh) for hh in perms]
                for k, fut in enumerate(concurrent.futures.as_completed(futures)):
                    # as_completed yields in arbitrary order; place results accordingly
                    # we can't know which index corresponds, so get result and find which hh
                    # to avoid complexity, iterate in submission order with map instead
                    pass
            # simpler parallel map using executor.map to preserve order
            with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as exe:
                for k, val in enumerate(exe.map(_compute_ssq_for_permutation, [P]*this_chunk, [X]*this_chunk, perms)):
                    ssqp[idx + k] = val

        idx += this_chunk

    return ssqp


def ascaptest_fast(ascamodel: Dict[str, Any], n_jobs: Optional[int] = 1, chunk_size: int = 1000) -> Dict[str, Any]:
    """Accelerated permutation testing wrapper.

    It mirrors `ascaptest` but uses `ptest_fast` which precomputes the projection
    matrix and optionally parallelizes permutations.
    """
    pmodel = ascamodel
    dlab = ascamodel['TermLabels']

    signfacts: List[str] = []

    for i in range(1, len(dlab)):
        l = dlab[i].strip()
        model_key = f'X{l}'
        if ascamodel['Options']['permtest'] == 'on':
            if isinstance(ascamodel['Options']['permfacts'], str) and ascamodel['Options']['permfacts'] == 'all':
                Xr = pmodel[model_key]['ReducedMatrix']
                Dr = pmodel[model_key]['DesignMatrix']
                ssqp = ptest_fast(Xr, Dr, ascamodel['Options']['nperm'], n_jobs=n_jobs, chunk_size=chunk_size)
                seff = pmodel[model_key]['EffectSSQ']
                p = np.sum(ssqp >= seff) / ascamodel['Options']['nperm']
                if p <= 0.05:
                    signfacts.append(l)

                pmodel[model_key].setdefault('EffectSignif', {})
                pmodel[model_key]['EffectSignif']['NullDistr'] = ssqp
                pmodel[model_key]['EffectSignif']['p'] = float(p)
            else:
                if l in ascamodel['Options']['permfacts']:
                    Xr = pmodel[model_key]['ReducedMatrix']
                    Dr = pmodel[model_key]['DesignMatrix']
                    ssqp = ptest_fast(Xr, Dr, ascamodel['Options']['nperm'], n_jobs=n_jobs, chunk_size=chunk_size)
                    seff = pmodel[model_key]['EffectSSQ']
                    p = np.sum(ssqp >= seff) / ascamodel['Options']['nperm']
                    if p <= 0.05:
                        signfacts.append(l)

                    pmodel[model_key].setdefault('EffectSignif', {})
                    pmodel[model_key]['EffectSignif']['NullDistr'] = ssqp
                    pmodel[model_key]['EffectSignif']['p'] = float(p)
                else:
                    pmodel[model_key].setdefault('EffectSignif', {})
                    pmodel[model_key]['EffectSignif']['NullDistr'] = np.array([])
                    pmodel[model_key]['EffectSignif']['p'] = None
        else:
            pmodel[model_key].setdefault('EffectSignif', {})
            pmodel[model_key]['EffectSignif']['NullDistr'] = np.array([])
            pmodel[model_key]['EffectSignif']['p'] = None

    pmodel['SignificantTerms'] = signfacts
    return pmodel


def ascaptest_fast(ascamodel: Dict[str, Any]) -> Dict[str, Any]:
    pmodel = ascamodel
    dlab = ascamodel['TermLabels']

    signfacts: List[str] = []

    if ascamodel['Options']['permtest'] == 'on':
        sc = 0
    else:
        sc = 0

    for i in range(1, len(dlab)):
        l = dlab[i].strip()
        model_key = f'X{l}'
        if ascamodel['Options']['permtest'] == 'on':
            if isinstance(ascamodel['Options']['permfacts'], str) and ascamodel['Options']['permfacts'] == 'all':
                Xr = pmodel[model_key]['ReducedMatrix']
                Dr = pmodel[model_key]['DesignMatrix']
                ssqp = ptest(Xr, Dr, ascamodel['Options']['nperm'])
                seff = pmodel[model_key]['EffectSSQ']
                p = np.sum(ssqp >= seff) / ascamodel['Options']['nperm']
                if p <= 0.05:
                    signfacts.append(l)

                pmodel[model_key].setdefault('EffectSignif', {})
                pmodel[model_key]['EffectSignif']['NullDistr'] = ssqp
                pmodel[model_key]['EffectSignif']['p'] = float(p)
            else:
                if l in ascamodel['Options']['permfacts']:
                    Xr = pmodel[model_key]['ReducedMatrix']
                    Dr = pmodel[model_key]['DesignMatrix']
                    ssqp = ptest(Xr, Dr, ascamodel['Options']['nperm'])
                    seff = pmodel[model_key]['EffectSSQ']
                    p = np.sum(ssqp >= seff) / ascamodel['Options']['nperm']
                    if p <= 0.05:
                        signfacts.append(l)

                    pmodel[model_key].setdefault('EffectSignif', {})
                    pmodel[model_key]['EffectSignif']['NullDistr'] = ssqp
                    pmodel[model_key]['EffectSignif']['p'] = float(p)
                else:
                    pmodel[model_key].setdefault('EffectSignif', {})
                    pmodel[model_key]['EffectSignif']['NullDistr'] = np.array([])
                    pmodel[model_key]['EffectSignif']['p'] = None
        else:
            pmodel[model_key].setdefault('EffectSignif', {})
            pmodel[model_key]['EffectSignif']['NullDistr'] = np.array([])
            pmodel[model_key]['EffectSignif']['p'] = None

    pmodel['SignificantTerms'] = signfacts
    return pmodel

def ptest(X: np.ndarray, D: np.ndarray, nperm: int) -> np.ndarray:
    ns = X.shape[0]
    ssqp = np.zeros(nperm)
    for i in range(nperm):
        hh = np.random.permutation(ns)
        Xpp = D[hh, :] @ pinv(D[hh, :]) @ X
        ssqp[i] = np.sum(Xpp ** 2)
    return ssqp


def ascasca(ascamodel: Dict[str, Any]) -> Dict[str, Any]:
    smodel = ascamodel
    dlab = ascamodel['TermLabels']

    for i in range(1, len(dlab)):
        l = dlab[i].strip()
        key = f'X{l}'
        Xr = ascamodel[key]['EffectMatrix']
        ssqr = ascamodel[key]['EffectSSQ']
        R = np.linalg.matrix_rank(Xr)
        if R == 0:
            continue
        # use svd or svds depending on size
        try:
            # svds returns (u, s, vt) where vt has shape (k, n_variables)
            u, s, vt = svds(Xr, k=R)
            # svds returns singular values in ascending order; reorder to descending
            idx = np.argsort(-s)
            u = u[:, idx]
            s = s[idx]
            # build loadings P with shape (n_variables, k)
            P = vt.T[:, idx]
        except Exception:
            u, s_full, vh = svd(Xr, full_matrices=False)
            s = s_full[:R]
            u = u[:, :R]
            P = vh.T[:, :R]

        t = u * s
        taug = (Xr + ascamodel['XRes']['EffectMatrix']) @ P
        varex = 100.0 * (s ** 2) / ssqr if ssqr != 0 else np.zeros_like(s)

        smodel[key].setdefault('SCA', {})
        smodel[key]['SCA']['Model'] = {
            'Scores': t,
            'ScoreswithRes': taug,
            'Loadings': P,
            'ExplVar': varex,
        }
    return smodel


def ascaboot(ascamodel: Dict[str, Any]) -> Dict[str, Any]:
    bmodel = ascamodel
    dlab = ascamodel['TermLabels']
    Xd_centered = ascamodel['Xdata']['CenteredData']

    for i in range(1, len(dlab)):
        l = dlab[i].strip()
        key = f'X{l}'
        Pb = None
        Pbcrit = None
        svars = None
        if ascamodel['Options']['bootstrap'] == 'all':
            if ascamodel['Options']['bootmatrix'] == 'original':
                Xd = Xd_centered
            elif ascamodel['Options']['bootmatrix'] == 'reduced':
                Xd = ascamodel[key]['ReducedMatrix']
            Dd = ascamodel[key]['DesignMatrix']
            Pd = ascamodel[key]['SCA']['Model']['Loadings']
            Pb, Pbcrit, svars = bootload(Xd, Dd, Pd, ascamodel['Options']['confl'], ascamodel['Options']['nboot'])
        elif ascamodel['Options']['bootstrap'] == 'signif':
            if l in ascamodel.get('SignificantTerms', []):
                if ascamodel['Options']['bootmatrix'] == 'original':
                    Xd = Xd_centered
                elif ascamodel['Options']['bootmatrix'] == 'reduced':
                    Xd = ascamodel[key]['ReducedMatrix']
                Dd = ascamodel[key]['DesignMatrix']
                Pd = ascamodel[key]['SCA']['Model']['Loadings']
                Pb, Pbcrit, svars = bootload(Xd, Dd, Pd, ascamodel['Options']['confl'], ascamodel['Options']['nboot'])
            else:
                Pb = np.array([])
                Pbcrit = np.array([])
                svars = []
        elif ascamodel['Options']['bootstrap'] == 'off':
            Pb = np.array([])
            Pbcrit = np.array([])
            svars = []

        bmodel[key].setdefault('SCA', {})
        bmodel[key].setdefault('SCA', {}).setdefault('Bootstrap', {})
        if ascamodel['Options']['bootsave'] == 'all':
            bmodel[key]['SCA']['Bootstrap']['Loadings'] = Pb
            bmodel[key]['SCA']['Bootstrap']['ConfIntervals'] = Pbcrit
            bmodel[key]['SCA']['Bootstrap']['SignificantVariables'] = svars
        elif ascamodel['Options']['bootsave'] == 'confint':
            bmodel[key]['SCA']['Bootstrap']['Loadings'] = np.array([])
            bmodel[key]['SCA']['Bootstrap']['ConfIntervals'] = Pbcrit
            bmodel[key]['SCA']['Bootstrap']['SignificantVariables'] = svars
        elif ascamodel['Options']['bootsave'] == 'signvars':
            bmodel[key]['SCA']['Bootstrap']['Loadings'] = np.array([])
            bmodel[key]['SCA']['Bootstrap']['ConfIntervals'] = np.array([])
            bmodel[key]['SCA']['Bootstrap']['SignificantVariables'] = svars

    return bmodel


def bootload(Xd: np.ndarray, Dd: np.ndarray, Pd: np.ndarray, confl: float, nboot: int) -> Tuple[np.ndarray, np.ndarray, List[List[int]]]:
    sl = (confl + 1.0) / 2.0
    ll = 1.0 - sl
    svars: List[List[int]] = [None] * Pd.shape[1]

    Pb = np.zeros((nboot, ) + Pd.shape)

    ns = Xd.shape[0]
    bootp = np.zeros(ns, dtype=int)
    # unique rows of Dd
    lev = np.unique(Dd, axis=0)

    for i in range(nboot):
        for j in range(lev.shape[0]):
            # indices matching level j
            matches = np.all(Dd == lev[j, :], axis=1)
            xx = np.where(matches)[0]
            if xx.size == 0:
                continue
            yy = np.random.randint(0, xx.size, size=xx.size)
            bootp[xx] = xx[yy]
        Xboot = Xd[bootp, :]
        Xdp = Dd @ pinv(Dd) @ Xboot
        # get singular values of Xdp
        try:
            # svds returns (u, s, vt); want vb as loadings (n_variables x k)
            _, _, vt = svds(Xdp, k=Pd.shape[1])
            vb = vt.T
        except Exception:
            _, s_full, vh = svd(Xdp, full_matrices=False)
            vb = vh.T[:, :Pd.shape[1]]
        _, Pb[i, :, :] = orth_proc(Pd, vb)

    Pb = np.sort(Pb, axis=0)
    lower_idx = int(np.ceil(ll * nboot)) - 1
    upper_idx = int(np.ceil(sl * nboot)) - 1
    lower_idx = max(0, lower_idx)
    upper_idx = min(nboot - 1, upper_idx)
    Pbcrit = Pb[[lower_idx, upper_idx], :, :]

    for i in range(Pd.shape[1]):
        prod = Pbcrit[0, :, i] * Pbcrit[1, :, i]
        svars[i] = list(np.where(np.sign(prod) == 1)[0])

    return Pb, Pbcrit, svars


def orth_proc(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Orthogonal Procrustes rotation projecting matrix y onto subspace spanned by x
    u, _, v = svd(y.T @ x, full_matrices=False)
    r = u @ v
    yrot = y @ r
    return r, yrot


if __name__ == "__main__":

    import scipy.io as sio
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
        options = asca('options')
        options['nperm'] = 1  # reduce for speed in testing
        model = asca(X, d, options)
        print("ASCA model keys:", list(model.keys()))
        # Show SCA results for each effect
        for key in model:
            if key.startswith("X") and "SCA" in model[key]:
                print(f"{key} SCA Model keys:", list(model[key]["SCA"]["Model"].keys()))
    else:
        print("X or d not found in data.mat")

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
    # try:
    #     import librairies.asca as asca_module
    # except Exception as e:
    #     raise ImportError('Local asca module is required to run rep_asca; import failed: {}'.format(e))

    opts = asca_light('options')
    opts['nperm'] = 1
    results_asca = asca_light(X, design, opts)
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
        results_asca = asca_light(X_bar, design, opts)
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
    # try:
    #     import asca as asca_module
    # except Exception as e:
    #     raise ImportError('Local asca module is required to run rep_asca_applied; import failed: {}'.format(e))

    if k > loadings_error.shape[1]:
        raise ValueError(f"Requested k={k} exceeds available components in loadings_error with shape {loadings_error.shape}")

    k_W = loadings_error[:, :k].T  # shape (k, n_features)
    X_bar = X - (X @ k_W.T) @ k_W

    opts = asca('options')
    opts['nperm'] = nperm
    model = asca(X_bar, d, opts)
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
        
        