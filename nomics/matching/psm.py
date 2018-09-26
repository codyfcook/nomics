import numpy as np
import statsmodels.api as sm


""" Tools for propensity score matching, using KNN or caliper """


def calculate_propensity_score(formula, data, verbose=False):
    """
    Calculate propensity scores

    :param formula: String of the form 'Treatment ~ covariate1 + covariate2 + ...', where these are column names in data
    :param data: Matrix-like object with columns corresponding to terms in the formula
    :param verbose: Boolean, whether or not to print logit summary
    """

    glm_binom = sm.formula.glm(formula=formula, data=data, family=sm.families.Binomial())
    res = glm_binom.fit()
    if verbose:
        print res.summary()
    return res.fittedvalues


def propensity_score_match(treated, propensity,
                           k=1,
                           caliper=None, caliper_method='none',
                           replace=True,
                           match_restrictions=None):
    """
    Implements one-to-many matching on propensity scores.

    :param treated: Series where index is identifiers, values are True/False
    :param propensity: Series where index is identifiers, values are propensity scores 
        (i.e. output of calculate_propensity_score())
    :param k: Integer. Number of matches
    :param caliper: Float. Specifies maximum distance, in units determined by caliper_method
    :param caliper_method: String: "stdev" to caliper by number of standard deviations in propensity score; "
        difference" to caliper by raw difference in propensity score; or "none" to have no caliper
    :param replace: Boolean for whether individuals from the untreated group should be allowed to match multiple 
        individuals in the treated group.
    :param match_restrictions: None OR dict where keys are identifiers and values are lists of identifiers. If not none,
        then when matching a given identifier, will only look in that set of identifiers. {Values must include the key 
        itself!}{nope not any more}

    :returns: Series containing the individuals in the control group matched to the treatment group.
    Note that with caliper matching, not every treated individual may have a match within calipers.
        In that case we match it to its single nearest neighbor.
    """

    # Validate inputs
    if caliper_method not in ('stdev', 'difference', 'none'):
        raise ValueError("Caliper method be 'stdev' or 'difference' or 'none'")
    if (caliper_method == 'difference') and (not (0 <= caliper < 1)):
        raise ValueError("Caliper with 'difference' caliper_method must be between 0 and 1")
    if (caliper_method == 'none') and (caliper is not None):
        raise ValueError("Must specificy caliper method")
    if (propensity < 0).any() or (propensity > 1).any():
        raise ValueError("Propensity scores must be between 0 and 1")
    if len(treated) != len(propensity):
        if treated.loc[propensity.index].isnull().sum():
            raise ValueError("All members of 'treated' must have propensity scores in 'propensity'")
    if treated.dtype != 'bool':
        raise ValueError("Variable 'treated' should be boolean")
    if match_restrictions:
        # Check that for random identifier, that it is included in its set of restrictions
        test = np.nan
        while test is np.nan:
            i = np.random.randint(len(match_restrictions.keys()))
            test = match_restrictions.keys()[i]
        # if test not in match_restrictions[test]:
        #     raise ValueError("For match_restrictions, values must include the key itself!")

    # Transform caliper when caliper_method is 'stdev'
    if caliper_method == 'stdev':
        caliper = caliper*propensity.std()

    # Make sure treated is coded as 0 and 1
    n1, n2 = treated[treated == True].index, treated[treated != True].index
    g1, g2 = propensity.loc[n1], propensity.loc[n2]
    # Check if treatment groups got flipped - the smaller should correspond to n1 and g1
    if len(n1) > len(n2):
        n1, n2, g1, g2 = n2, n1, g2, g1

    # Randomly permute the smaller group to get order for matching; matters if matching without replacement
    morder = np.random.permutation(n1)
    matches = {}

    for m in morder:
        # Calculate PSM distance
        dist = abs(g1[m] - g2)
        # If match_restrictions, drop all items which are not in the set of valid possible matches
        if match_restrictions:
            dist = dist.loc[match_restrictions[m]].dropna()

        # Restrict to only potential matches that fall within caliper
        if caliper_method != 'none':
            dist = dist[dist <= caliper]

        # Sort:
        dist = dist.sort_values()

        # Take k nearest matches
        matches[m] = dist.iloc[:k].index.values

        # If not drawing with replacement, drop the matches from the universe
        if not replace:
            g2 = g2.drop(matches[m])

    return matches
