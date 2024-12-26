import numpy as np
from scipy.linalg import pinv
from scipy.sparse.linalg import svds

def rep_asca(X, design, X_rep=None, d_rep=None, klimit=None):
    if X_rep is None or d_rep is None or klimit is None:
        return asca_final(X, design)

    BS = d_rep @ pinv(d_rep.T @ d_rep) @ d_rep.T @ X_rep
    WS = X_rep - BS
    _, _, L_err = svds(WS, k=klimit)

    results_asca = asca_min(X, design)
    list_factor = results_asca['TermLabels'][1:]
    explained_var = np.zeros((klimit + 1, len(list_factor) + 1))
    list_factor_name = []

    for j, factor in enumerate(list_factor):
        explained_var[0, j] = results_asca[f'X{factor}']['EffectExplVar']
        list_factor_name.append(f'X{factor}')

    explained_var[0, len(list_factor)] = results_asca['XRes']['EffectExplVar']
    list_factor_name.append('XRes')

    for i in range(1, klimit + 1):
        k_W = L_err[:i, :]
        X_bar = X - X @ k_W.T @ k_W
        results_asca = asca_min(X_bar, design)
        list_factor = results_asca['TermLabels'][1:]

        for j, factor in enumerate(list_factor):
            explained_var[i, j] = results_asca[f'X{factor}']['EffectExplVar']
            list_factor_name[j] = f'X{factor}'

        explained_var[i, len(list_factor)] = results_asca['XRes']['EffectExplVar']
        list_factor_name[len(list_factor)] = 'XRes'

    model = {
        'explained_var': explained_var,
        'list_factor_name': list_factor_name,
        'L_err': L_err
    }

    return model

def asca_final(X, design):
    dmatr = designmat_fct(design)
    dmatr_factor, terms, labels, order = design_creator(dmatr)

    Xcomputed = X.copy()
    model = {}

    for i, dmat in enumerate(dmatr_factor):
        X_i = dmat @ pinv(dmat) @ Xcomputed
        ssq_i = np.sum(X_i ** 2)
        Xcomputed -= X_i
        l = labels[i].strip()
        model[f'X{l}'] = {
            'EffectMatrix': X_i,
            'EffectSSQ': ssq_i,
            'DesignMatrix': dmat,
            'DesignTerms': terms[i],
            'EffectLabel': l,
            'TermOrder': order[i]
        }

        if i == 0:
            ssqtot = np.sum(Xcomputed ** 2)
            model['Xdata'] = {
                'CenteredData': Xcomputed,
                'CenteredSSQ': ssqtot
            }
        else:
            expVar = 100 * (ssq_i / ssqtot)
            model[f'X{l}']['EffectExplVar'] = expVar

    for i in range(1, len(dmatr_factor)):
        l = labels[i].strip()
        X_i = model['Xdata']['CenteredData'].copy()
        for j in range(len(l)):
            X_i -= model[f'X{l}']['EffectMatrix']
        model[f'X{l}']['ReducedMatrix'] = X_i

    model['XRes'] = {
        'EffectMatrix': Xcomputed,
        'EffectSSQ': np.sum(Xcomputed ** 2),
        'EffectExplVar': 100 * (np.sum(Xcomputed ** 2) / ssqtot),
        'EffectLabel': 'Res',
        'TermOrder': max(order) + 1
    }
    model['TermLabels'] = labels

    model = permutation_test(model)
    model = scastep(model)

    return model

def permutation_test(model):
    nperm = 500
    newmodel = model.copy()
    labels = model['TermLabels']
    signfacts = []
    sc = 0

    for i in range(1, len(labels)):
        l = labels[i].strip()
        Xr = model[f'X{l}']['ReducedMatrix']
        Dr = model[f'X{l}']['DesignMatrix']
        ssqp = ptest(Xr, Dr, nperm)
        seff = model[f'X{l}']['EffectSSQ']
        p = len(np.where(ssqp >= seff)[0]) / nperm
        if p <= 0.05:
            sc += 1
            signfacts.append(l)
        newmodel[f'X{l}']['EffectSignif'] = {
            'NullDistr': ssqp,
            'p': p
        }

    newmodel['SignificantTerms'] = signfacts
    return newmodel

def design_creator(design):
    nfactor = len(design)
    nsize = design[0].shape[0]

    indmat = np.array(np.meshgrid(*[[0, 1]] * nfactor)).T.reshape(-1, nfactor)
    nmat = indmat.shape[0]
    dmatr = []
    terms = []
    labels = []
    order = np.zeros(nmat)

    for i in range(nmat):
        Dm = np.ones((nsize, 1))
        for j in range(nfactor):
            effmat = design[j] if indmat[i, j] == 1 else np.ones((nsize, 1))
            Dm = np.kron(Dm, effmat)
            Dm = Dm[:nsize, :]
        dmatr.append(Dm)
        terms.append(np.where(indmat[i, :] == 1)[0])
        labels.append(''.join(chr(65 + idx) for idx in np.where(indmat[i, :] == 1)[0]))
        order[i] = len(terms[-1])

    labels, newindex = zip(*sorted(zip(labels, range(nmat)), key=lambda x: (len(x[0]), x[0])))
    terms = [terms[i] for i in newindex]
    dmatr = [dmatr[i] for i in newindex]
    order =  [order[i] for i in newindex]
    
    # if order.ndim == 1 and len(newindex) == 1:
    # order = order[newindex]
    # else:
    #     # Handle other cases or raise an error
    #     raise IndexError("Invalid indexing for order array")
    labels = list(labels)
    labels[0] = 'Mean'

    return dmatr, terms, labels, order

def ptest(X, D, nperm):
    ns = X.shape[0]
    ssqp = np.zeros(nperm)

    for i in range(nperm):
        hh = np.random.permutation(ns)
        Xpp = D[hh, :] @ pinv(D[hh, :]) @ X
        ssqp[i] = np.sum(Xpp ** 2)

    return ssqp

def designmat_fct(design):
    nsize, nfactor = design.shape
    dmatr = []

    for i in range(nfactor):
        lev = np.unique(design[:, i])
        nl = len(lev)
        dmat = np.zeros((nsize, nl - 1))
        for j in range(nl - 1):
            dmat[design[:, i] == lev[j], j] = 1
        dmat[design[:, i] == lev[nl - 1], :] = -1
        dmatr.append(dmat)

    return dmatr

def scastep(ascamodel):
    smodel = ascamodel.copy()
    dlab = ascamodel['TermLabels']

    for i in range(1, len(dlab)):
        l = dlab[i].strip()
        Xr = ascamodel[f'X{l}']['EffectMatrix']
        ssqr = ascamodel[f'X{l}']['EffectSSQ']
        R = np.linalg.matrix_rank(Xr)
        u, s, P = svds(Xr, k=R)
        T = u @ np.diag(s)
        explainedVar = 100 * (s ** 2) / ssqr
        smodel[f'X{l}']['SCA'] = {
            'Model': {
                'Scores': T,
                'Loadings': P,
                'ExplVar': explainedVar
            }
        }

    return smodel

def asca_min(X, design):
    dmatr = designmat_fct(design)
    dmatr_factor, terms, labels, order = design_creator(dmatr)

    Xcomputed = X.copy()
    model = {}

    for i, dmat in enumerate(dmatr_factor):
        X_i = dmat @ pinv(dmat) @ Xcomputed
        ssq_i = np.sum(X_i ** 2)
        Xcomputed -= X_i
        l = labels[i].strip()
        model[f'X{l}'] = {
            'EffectMatrix': X_i,
            'EffectSSQ': ssq_i,
            'DesignMatrix': dmat,
            'DesignTerms': terms[i],
            'EffectLabel': l,
            'TermOrder': order[i]
        }

        if i == 0:
            ssqtot = np.sum(Xcomputed ** 2)
            model['Xdata'] = {
                'CenteredData': Xcomputed,
                'CenteredSSQ': ssqtot
            }
        else:
            expVar = 100 * (ssq_i / ssqtot)
            model[f'X{l}']['EffectExplVar'] = expVar

    model['XRes'] = {
        'EffectMatrix': Xcomputed,
        'EffectSSQ': np.sum(Xcomputed ** 2),
        'EffectExplVar': 100 * (np.sum(Xcomputed ** 2) / ssqtot),
        'EffectLabel': 'Res',
        'TermOrder': max(order) + 1
    }
    model['TermLabels'] = labels

    return model