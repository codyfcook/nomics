import numpy as np
import pandas as pd

""" Tools for getting ATT """


def mean_func(df):
    return df.mean()


def median_func(df):
    return df.median()


def get_att(response, matches, agg_func=mean_func, percentage=True, verbose=False):
    """
    Computes ATT using difference in means
    The data passed in should already have unmatched individuals and duplicates removed.

    :param response: Series where index is identifiers, values are response measurements
    :param matches: Dict where keys are treated identifiers, values are lists of identifiers of the (untreated) matched
    :param agg_func: Use mean_func for average treatment effect; median_func for median treatment func; etc.
    :param percentage: Boolean, whether or not to convert ATT to percentage
    :param verbose: Boolean, whether or not to return distribution of effect on treated and untreated on top of ATT
    """

    response0 = []
    response1 = []

    for k in matches.keys():
        response1.append(response.ix[k])
        response0.append((response.ix[matches[k]]).mean())   # Take mean response of matched

    response1 = pd.Series(response1)
    response0 = pd.Series(response0)

    att = agg_func(response1) - agg_func(response0)
    if percentage:
        att = att/agg_func(response0)

    if verbose:
        return att, response1, response0
    else:
        return att


""" Tools for measuring balance """


def balance_pvalues(matches, covariates, nan_policy='omit', plot=False):
    """
    Computes p-value of difference in means in covariates between treated and untreated

    :param matches: Dict where keys are treated identifiers, values are untreated identifiers
    :param covariates: Dataframe of covariates indexed by identifiers
    :param nan_policy: When calculating p-values, how to treat NaNs
    :param plot: Boolean, whether or not to draw a heatmap of the pvalues
    """
    from scipy.stats import ttest_rel

    treated = covariates.ix[matches.keys()]
    untreated = covariates.ix[matches.values()]

    pvalues = pd.Series({
        var: ttest_rel(treated[var], untreated[var], nan_policy=nan_policy)[1]
        for var in covariates
    })

    if plot:
        import seaborn
        pvalues.name = 'p-value'
        df = np.round(pd.DataFrame(pvalues), 4)
        seaborn.heatmap(df, annot=True)

    return pvalues


def plot_covariate_ratio_of_means(matches, covariates, pvalues=None,
                                  title=None, ylim=(0.8, 1.2)):
    """
    Plots the ratio of the means of covariates of treated to untreated.
    Optionallly label bars with p-values of diff in means.

    :param matches: Dict where keys are treated identifiers, values are untreated identifiers
    :param covariates: Dataframe of covariates indexed by identifiers
    :param pvalues: (optional) output of balance_pvalues()
    :param title: Title for plot
    :param ylim: y-axis range for plot
    """
    treated = covariates.ix[matches.keys()]
    untreated = covariates.ix[matches.values()]

    diff = {'treated': treated.mean(), 'untreated': untreated.mean()}

    ax = (diff['treated'] / diff['untreated']).plot(kind='bar', title=title)
    ax.set_ylim(ylim)
    if pvalues is not None:
        i = 0
        for p in ax.patches:
            ax.annotate(pvalues.iloc[i],
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 10), textcoords='offset points')
            i += 1

    return diff
