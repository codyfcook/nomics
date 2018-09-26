'''
Author: Cody Cook (borrowing heavily from code by Josef Perktold)
'''

import numpy as np
from scipy import sparse
import scipy.sparse.linalg as sparsela
import pandas as pd
from statsmodels.regression.linear_model import OLS, WLS

'''
This script allows you to run absorbed fixed effects. Specify which columns to absorb, which 
should be categorical variables ranging from 0-N (with no skips). They do not need to be blown 
up into dummies -- this will happen automatically (and much more efficiently, as it uses sparse matrices)

You won't get any coeffients for absorbed cols. You can back them out afterwards using the OLSAbsorb._get_fixed_effects
method, which backs them out from the residuals.

Example code
------------
y_col = ['log_implied_hourly_total']
dense_cols = ['intercept', 'is_male']
absorb_cols = ['week', 'day_of_week']

model = OLSAbsorb(d[y_col], d[dense_cols], np.array(d[absorb_cols]))
results = model.fit()

print.results.get_robustcov_results(cov_type='cluster', **{'groups':d['driver_uuid'].values}).summary()

'''

class PartialingSparse(object):
    '''
    Class to condition an partial out a sparse matrix

    Inputs 
    ----------
    x_conditioning : sparse matrix
    method : 'lu' or 'lsqr'
        sparse method to solve least squares problem. 'lu' stores
        the LU factorization and should be faster in repeated calls.
        'lsqr' does not store intermediate results.
    '''

    def __init__(self, x_conditioning, method='lu'):
        self.xcond = x = x_conditioning
        self.method = method.lower()
        if self.method == 'lu':
            self.xtx_solve = sparsela.factorized(x.T.dot(x))
        else:
            if self.method not in ['lsqr']:
                raise ValueError('method can only be lu or lsqr')


    def partial_params(self, x):
        '''
        Find least squares parameters

        Inputs 
        ----------
        x : ndarray, 1D or 2D
        
        Returns
        -------
        params : ndarray
            least squares parameters
        '''
        if self.method == 'lu':
            try:
                p = self.xtx_solve( self.xcond.T.dot(x))
            except SystemError:
                p =  np.column_stack([self.xtx_solve(xc) for
                                      xc in self.xcond.T.dot(x).T])
            return p
        else:
            return sparsela.lsqr(self.xcond, x)


    def partial_sparse(self, x):
        '''
        Calculate projection of x on x_conditioning
        
        Inputs 
        ----------
        x : ndarray, 1D or 2D
        
        Returns
        -------
        xhat : ndarray
            projection of x on xcond
        resid : ndarray
            orthogonal projection on null space
        '''
        xhat = self.xcond.dot(self.partial_params(x))
        resid = x - xhat
        return xhat, resid

def dummy_sparse(groups, dtype=np.float64):
    '''
    Create a sparse indicator from a group array with integer labels

    Inputs 
    ----------
    groups: ndarray, int, 1d (nobs,)
        an array of group indicators for each observation. Group levels are assumed
        to be defined as consecutive integers, i.e. range(n_groups) where
        n_groups is the number of group levels.
    dtype : numpy dtype
        dtype of sparse matrix.
        We need float64 for some applications with sparse linear algebra. For other
        uses we can use an integer type which will use less memory.
    Returns
    -------
    indi : ndarray, int8, 2d (nobs, n_groups)
        an indicator array with one row per observation, that has 1 in the
        column of the group level for that observation
    '''

    indptr = np.arange(len(groups)+1)
    data = np.ones(len(groups), dtype=dtype)
    indi = sparse.csr_matrix((data, groups, indptr))

    return indi

def cat2dummy_sparse(xcat, use_pandas=False):
    """
    Categorical to sparse dummy, use pandas for quick implementation
    xcat needs to be ndarray for now
    """
    # prepare np array for column iteration
    if xcat.ndim == 1:
        xcat_t = [xcat]
    else:
        xcat_t = xcat.T

    ds = [dummy_sparse(xc) for xc in xcat_t]
    xsp = sparse.hstack(ds, format='csr')[:, 1:]   # full rank

    return xsp


def _group_demean_iterative(exog_dense, groups, add_mean=True, max_iter=10, atol=1e-8):
    '''
    Demean an array for two-way fixed effects
    
    Inputs  
    ----------
    exog_dense : 2d ndarray
        data with observations in rows and variables in columns.
        This array will currently not be modified.
    groups : 2d ndarray, int
        groups labels specified as consecutive integers starting at zero
    add_mean : bool
        If true (default), then the total variable means are added back into
        the group demeand exog_dense
    max_iter : int
        maximum number of iterations
    atol : float
        tolerance for convergence. Convergence is achieved if the
        maximum absolute change (np.ptp) is smaller than atol.
    Returns
    -------
    ex_dm_w : ndarray
        group demeaned exog_dense array in wide format
    ex_dm : ndarray
        group demeaned exog_dense array in long format
    it : int
        number of iterations used. If convergence has not been
        achieved then it will be equal to max_iter - 1
    '''

    k_cat = tuple((groups.max(0) + 1).tolist())
    xm = np.empty(exog_dense.shape[1:] + k_cat)
    xm.fill(np.nan)
    xm[:, groups[:, 0], groups[:, 1]] = exog_dense.T
    keep = ~np.isnan(xm[0]).ravel()
    finished = False
    for it in range(max_iter):
        for axis in range(1, xm.ndim):
            group_mean = np.nanmean(xm, axis=axis, keepdims=True)
            xm -= group_mean
            if np.ptp(group_mean) < atol:
                finished = True
                break
        if finished:
            break

    xd = xm.reshape(exog_dense.shape[-1], -1).T[keep]
    if add_mean:
        xmean = exog_dense.mean(0)
        xd += xmean
        xm += xmean[:, None, None]
    return xm, xd, it

class OLSAbsorb(WLS):
    '''
    OLS model that absorbs categorical explanatory variables
    
    Inputs 
    ----------
    exog_absorb : ndarray, 1D or 2D
        categorical, factor variables that will be absorbed
    absorb_method : string, 'lu' or 'lmgres'
        method used in projection for absorbing the factor variables.
        Currently the options use either sparse LU decomposition or sparse
        `lmgres`
    Notes
    -----
    constant: the current parameterization produces a constant when the mean
    categorical effect is set to zero
    Warning: currently not all inherited methods for OLS are correct.
    Parameters and inference are correct and correspond to the full model for
    OLS that includes all factor variables as dummies with zero mean factor
    effects.
    '''

    def __init__(self, endog, exog, exog_absorb, absorb_method='lu', **kwds):
        absorb = cat2dummy_sparse(exog_absorb)
        self.projector = PartialingSparse(absorb, method=absorb_method)
        super(OLSAbsorb, self).__init__(endog, exog, **kwds)

        self.k_absorb = absorb.shape[1]
        self.df_resid = self.df_resid - self.k_absorb + 1
        self.df_model = self.df_model + self.k_absorb - 1

        self.absorb = absorb

    def whiten(self, y):
        # add the mean back in to get a constant
        # this does not reproduce the constant if fixed effect use reference encoding
        # It produces the constant if the mean fixed effect is zero
        y_mean = y.mean(0)
        return self.projector.partial_sparse(y)[1] + y_mean


    def _get_fixed_effects(self, resid):
        '''
        temporary: recover fixed effects from regression using residuals
        Warning: This uses dense fixed effects dummies at the moment
        We add a constant to correct for constant in absorbed regression.
        This will depend on the encoding of the absorb fixed effects.
        
        Inputs 
        ---------
        resid : ndarray
            residuals of absorbed OLS regression
        '''

        exog_dummies = np.column_stack((np.ones(len(self.endog)),
                                        self.absorb.toarray()[:, :-1]))

        return OLS(resid, exog_dummies).fit()