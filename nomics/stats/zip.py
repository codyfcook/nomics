'''
Code from https://gist.github.com/jseabold/6617976
'''

from future import __division__
import numpy as np
from scipy import special, stats
from statsmodels.base.model import GenericLikelihoodModel

def vuong_test(p1, p2):
    r"""
    Vuong-test for non-nested models.
    Parameters
    ----------
    p1 : array-like
        f1(Y=y_i | x_i)
    p2 : array-like
        f2(Y=y_i | x_i)
    Notes
    -----
    This is hard-coded for testing Poisson vs. Zero-inflated. E.g.,
    it does not account for
    Let f_j(y_i|x_i) denote the predicted probability that random variable Y
    equals y_i under the assumption that the distribution is f_j(y_i|x_i) for
    j = 1,2. Let
    .. math::
       m_i = log(\frac{f_1(y_i|x_i)}{f_2(y_i|x_i)})
    The test statistic from Vuong to test the hypothesis of Model 1 vs.
    Model 2 is
    .. math::
       v = \frac{\sqrt{n}(1/n \sum_{i=1}^{n}m_i)}{\sqrt{1/n \sum_{i=1}^{n}(m_i - \bar{m})^2}}
    This statistic has a limiting standard normal distribution. Values of
    v greater than ~2, indicate that model 1 is preferred. Values of V
    less than ~-2 indicate the model 2 is preferred. Values of |V| < ~2 are
    inconclusive.
    References
    ----------
    Greene, W. Econometric Analysis.
    Vuong, Q.H. 1989 "Likelihood ratio tests for model selection and
        non-nested hypotheses." Econometrica. 57: 307-333.
    """
    m = np.log(p1) - np.log(p2)
    n = len(m)
    v = n ** .5 * m.mean() / m.std()
    return v, stats.norm.sf(np.abs(v))


def poisson_pmf(x, mu):
    return mu ** x / special.gamma(x + 1) * np.exp(-mu)

def trunc_poisson_pmf(x, mu):
    # 10x faster than poisson.pmf(x, mu)/(1 - poisson.cdf(0, mu))
    return mu ** x * np.exp(-mu) / (special.gamma(x + 1) * (1 - np.exp(-mu)))

def trunc_poisson_logpmf(x, mu):
    # 10x faster than poisson.logpmf(x, mu) - np.log1(-poisson.cdf(0, mu))
    if not np.all(x > 0):
        raise ValueError("x must be positive counts")
    if not mu > 0:
        raise ValueError("mu must be a postive real")
    x = np.asarray(x)
    pdf = x * np.log(mu) - mu - special.gammaln(x + 1) - np.log1p(-np.exp(-mu))
    return pdf

class ZIPoisson(GenericLikelihoodModel):
    def __init__(self, y, X, inflateX=None, missing='none'):
        #TODO: inflateX needs to go through the pandas data rigamarole
        if inflateX is None:
            inflateX = np.ones((len(y), 1))
        else:
            inflateX = np.asarray(inflateX)
        super(ZIPoisson, self).__init__(y, X, inflateX=inflateX,
                                        missing=missing)
        self.link = None # chosen during fit
        self.k_exog = X.shape[1]
        self.k_inflate = inflateX.shape[1]

    def loglikeobs(self, params):
        y = self.endog
        zero_idx = np.nonzero(y == 0)[0]
        nonzero_idx = np.nonzero(y)[0]
        X = self.exog[nonzero_idx]
        inflateX = self.inflateX

        # separate params into zero and non-zero
        params_X = params[:self.k_exog]
        params_inflate = params[self.k_exog:]

        # linear predictions
        XB_nonzero = np.dot(X, params_X) # + offset_nonzero
        inflateXB = np.dot(inflateX, params_inflate) # + offset_zero

        # probability of zero event
        link = self.link or stats.logistic.cdf # in case not set in fit yet
        p = link(inflateXB)

        # mu is lambda in the literature, mu in stats.poisson
        # log(lambda_i) = XB_i
        mu = np.exp(np.dot(self.exog, params_X))

        # cf. Lambert (1992) Zero-Inflated Poisson Regression...
        p_zero = p[zero_idx]
        p_nonzero = p[nonzero_idx]
        y_nonzero = y[nonzero_idx]
        llf_zero = np.log(p_zero + (1 - p_zero)*np.exp(-mu[zero_idx]))
        llf_nonzero = (np.log(1 - p_nonzero) - mu[nonzero_idx] +
                          XB_nonzero*y_nonzero -
                          special.gammaln(y_nonzero + 1))
        # not the most memory-friendly way to do this but certainly the
        # laziest
        llf_obs = np.zeros(len(y))
        llf_obs[zero_idx] = llf_zero
        llf_obs[nonzero_idx] = llf_nonzero
        return llf_obs

    def loglike(self, params):
        """
        Parameters
        ----------
        params : ndarray
            Params is expected to contain the parameters for both the
            exogenous variables and the parameters for modeling the excess
            zeros in that order.
        Notes
        -----
        The dependent variable Y is assumed to be distributed such that
        Y_i ~ 0 with probability p_i and Y_i ~ Poisson(:math:`\mu_i`) with
        probability 1 - p_i so that Y_i = 0 with probability
        p_i + (1 - p_i)*exp(-\mu_i) and Y_i = k with probability
        (1 - p_i)*exp(:math:`-\mu_i`)*:math:`mu_i`**k/k! for k = 1, 2, ...
        where log(:math:`mu`) = dot(X, model_params)
        and
        logit(p) = log(p/(1-p)) = dot(inflateX, inflate_params)
        or
        probit(p) = norm.ppf(p) = dot(inflateX, inplate_params)
        depending on the link function specified in the fit method.
        References
        ----------::
           Lambert, D. (1992) "Zero-Inflated Poisson Regression, With an
               Application to Defects in Manufacturing." Technometrics,
               34, 1.
        """
        # setup
        y = self.endog
        zero_idx = np.nonzero(y == 0)[0]
        nonzero_idx = np.nonzero(y)[0]
        X = self.exog[nonzero_idx]
        inflateX = self.inflateX

        # separate params into zero and non-zero
        params_X = params[:self.k_exog]
        params_inflate = params[self.k_exog:]

        # linear predictions
        XB_nonzero = np.dot(X, params_X) # + offset_nonzero
        inflateXB = np.dot(inflateX, params_inflate) # + offset_zero

        # probability of zero event
        link = self.link or stats.logistic.cdf # in case not set in fit yet
        p = link(inflateXB)

        # mu is lambda in the literature, mu in stats.poisson
        # log(lambda_i) = XB_i
        mu = np.exp(np.dot(self.exog, params_X))

        # cf. Lambert (1992) Zero-Inflated Poisson Regression...
        p_zero = p[zero_idx]
        p_nonzero = p[nonzero_idx]
        y_nonzero = y[nonzero_idx]
        llf_zero = sum(np.log(p_zero + (1 - p_zero)*np.exp(-mu[zero_idx])))
        llf_nonzero = sum(np.log(1 - p_nonzero) - mu[nonzero_idx] +
                          XB_nonzero*y_nonzero -
                          special.gammaln(y_nonzero + 1))
        return llf_zero + llf_nonzero

    def fit(self, start_params=None, link="logit", method='bfgs', maxiter=100,
            full_output=True, disp=1, fargs=(), callback=None,
            retall=False, **kwargs):
        """
        Parameters
        ----------
        link : str {"logit", "probit"}
            The link function for modeling the excess zeros.
        """
        link = link.lower()
        if link == "logit":
            link = stats.logistic.cdf
        elif link == "probit":
            link = stats.norm.cdf
        else: #pragma: no cover
            raise ValueError("link %s not understood")
        self.link = link

        if start_params is None:
            start_params = np.zeros(self.exog.shape[1] +
                                    self.inflateX.shape[1]) + .1

        zip_fit = super(ZIPoisson, self).fit(start_params=start_params,
                    method=method, maxiter=maxiter, full_output=full_output,
                    disp=disp, callback=callback, **kwargs)
        #TODO: make a results class and wrapper
        # zip_fit = ZIPResults(self, zip_fit)
        # return ZIPResultsWrapper(zip_fit)
        zip_fit.model_params = zip_fit.params[:self.exog.shape[1]]
        zip_fit.inflate_params = zip_fit.params[self.exog.shape[1]:]
        return zip_fit

def test_params():
    from statsmodels.tools.tools import webuse
    dta = webuse("fish")
    y = dta["count"]
    dta["const"] = 1.
    X = dta[["const", "persons", "livebait"]]
    inflateX = dta[["const", "child", "camper"]]
    mod = ZIPoisson(y, X, inflateX)
    res = mod.fit()

    # from stata
    params = np.array([-2.1784716320442, .80688527006274,   1.7572893795292,
                       -.49228715725108, 1.6025705051289, -1.0156983048851])
    np.testing.assert_almost_equal(res.params, params, 4)
    np.testing.assert_almost_equal(res.llf, -850.7014181090408, 4)