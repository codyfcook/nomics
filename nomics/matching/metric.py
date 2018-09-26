import numpy as np
import pandas as pd
from IPython.display import clear_output


""" Helper functions """


def dict_to_df(dic):
    return pd.DataFrame(dict([(k, pd.Series(v))
                              for k, v in dic.iteritems()]))


def get_map_from_df(df, column1, column2, dropna=False):
    """
    (Dataframe) --> (Dict mapping df[column1] to df[column2])

    :param df: Dataframe
    :param column1: Column of df which will be the domain
    :param column2: Column of df which will be the range
    :param dropna: boolean indicating if you want to drop items from the map if they have no corresponding element
    :return: 
    """
    if dropna:
        return {k: g[column2].dropna().tolist() for k, g in df.groupby(column1)}
    else:
        return {k: g[column2].tolist() for k, g in df.groupby(column1)}


""" Manipulation functions """


def bin_characteristic(dfr, characteristic, num_bins=2):
    """
    For a DataFrame dfr with entities in the index and characteristics of each entity in the column,
    return the same dataframe with additional boolean columns binning entities based on characteristic 

    :param dfr: DataFrame, indexed by entity where columns are characteristics of each item
    :param characteristic: column name to be binned
    :param num_bins: number of bins to create
    :return: inputted DataFrame, with additional columns indicating boolean which bin the entity belongs to
    """
    assert characteristic in dfr
    dfr = dfr.copy()
    for i in range(num_bins):
        bottom_percentile = i / float(num_bins)
        top_percentile = (i + 1) / float(num_bins)
        dfr[characteristic + '_' + str(i + 1)] = (
            (dfr[characteristic] > dfr[characteristic].quantile(bottom_percentile))
            & (dfr[characteristic] <= dfr[characteristic].quantile(top_percentile))
        )
    return dfr


def clean_data(data,
               exclude=None, bin_characteristics=set(), num_bins=set(), manual_exclude=set()):
    """
    Clean up the data by dropping some entities, and discretizing some variables

    :param data: Dataframe where rows are cities and columns are city characteristics
    :param exclude: List of variable names or None. If list of variables, then drop all cities for which these variables
        are True (these variables should be boolean)
    :param bin_characteristics: Set of characteristics to convert from continuous variables to discrete variables by
        binning by percentile
    :param num_bins: Dict keyed by characteristics that are to be binned, with values for how many bins to create
    :param manual_exclude: List of cities to drop
    :return: 
    """

    df = data.copy()

    # Validation
    if exclude:
        for var in exclude:
            assert (df[var].dtype == bool
                    or
                    set(df[var].unique()).issubset({0, 1}))
    if bin_characteristics:
        for var in bin_characteristics:
            assert var in df.columns
            assert var in num_bins
    #if manual_exclude:
    #    for city in manual_exclude:
    #        assert city in df.index

    # Exclude cities which are 'True' for characteristics in exclude
    if exclude:
        for var in exclude:
            df = df[~df[var].astype(bool)].copy()

    # Exclude cities which are in manual_exclude_cities
    df = df.ix[[c for c in df.index if c not in manual_exclude]]

    # Create columns for binned characteristics
    for char in bin_characteristics:
        df = bin_characteristic(df, char, num_bins[char])

    return df


def partition_universe(data, exact_match_on):
    """
    Partition the universe of cities into cells which exact match on characteristics in exact_match_on

    :param data: Dataframe where rows are cities and columns are city characteristics
    :param exact_match_on: Set of variables which you want cities to match *exactly* on (should generally be discrete)
    :return: Dictionary of cities, where values are a list of all other cities in that city's partition
    """
    super_df = data.reset_index().merge(data.reset_index(), on=exact_match_on, how='left')
    partitions = get_map_from_df(super_df, 'index_x', 'index_y', dropna=True)

    return partitions


""" Tools for calculating distance and matching (KNN; greedy matching; or optimal matching"""


def get_dist(df, metric='mahalanobis'):
    """
    Calculates the distance between each identifier using the specified metric
    
    :param df: DataFrame of characteristics, indexed by identifiers. Characteristics must be continuous variables
    :param metric: Metric used to measure distance between items, e..g 'euclidean', 'mahalanobis'
    :return: DataFrame of distances between identifiers
    """
    # TODO: handle non-continuous variables
    # TODO: allow weighting of different characteristics

    if metric == 'euclidean':
        # For Euclidean distance matching, we first need to z-score the variables to make them comparable
        df = (df - df.mean())/df.std()

    import sklearn.metrics.pairwise as pw
    dist = pd.DataFrame(pw.pairwise_distances(df, metric=metric), index=df.index, columns=df.index)
    dist = np.round(dist, 4)  # clear out numerical bugs
    dist = dist.replace(0.0, np.nan)  # when finding the nearest neighbor, don't want to find itself

    return dist


def knn(dist, k=1, restrictions=None):
    """
    Matches each item to nearest k neighbor(s), based on distance metric over criteria in df.
    Each characteristic is equal-weighted.

    :param dist: DataFrame of characteristics, indexed by item. 
    :param k: Number of matches
    :param restrictions: (optional) Dict of items to list of items, if you want to restrict items
        to only match within a set of other possible items.
    :return: Series containing the individuals in the control group matched to the treatment group.
    """
    # TODO: configure with replacement vs. not

    # If no restrictions, allow any item to be paired with any other item
    if not restrictions:
        restrictions = {item: dist.index for item in dist}

    # For each item, consider all items within its restrictions; sort by distance; and take the k nearest neighbors
    matches = dict_to_df({col: dist[col].ix[restrictions[col]].sort_values()[:k].index
                          for col in dist}).T
    # Collect the distance between each item and its match(es)
    # matches_dist = {col: dist[col].ix[restrictions[col]].sort_values()[:k].dropna().tolist()
    #                 for col in dist}
    # matches_w_dist = pd.Panel({
    #     'matches': matches,
    #     'dist': matches_dist
    # })

    return matches


def greedy_match(dist, partitions=None):
    """
    Naive greedy matching algorithm: start from the pair with the shortest distance and pair them; then iterate through
        all other pairwise distances, going from shortest to longest distance, matching them if neither of the cities
        is already matched.
    
    :param dist: DataFrame of pairwise distances
    :param partitions: (optional) Dict of items to list of items, if you want to restrict items
        to only match within a set of other possible items.
    
    
    More detailed description of logic:
    
    The algorithm here is this:
        1. Rank the cities by their distance with their best match (within its partition)
        2. Store the best match pair (i.e. shortest distance). Delete them from the ranking.
        3. Look at the next city. 
            If its matched city is not already paired up, store the match.
            Elif its matched city is already matched, rerank the remaining cities based on their distance with their
            match, but now using this city's distance with its next best match
        4. Iterate until all cities matched.
    Note that if a partition has an odd number of cities, obviously the worst-matched city will not get a match.

    ***There is surely a more elegant way to do this, this was quick and dirty***
    """
    if not partitions:
        partitions = {item: dist.index for item in dist}

    dist = np.round(dist, 4)  # clear out numerical bugs
    dist = dist.replace(0.0, np.nan)

    final_match = {}
    dists = {col: dist[col].ix[partitions[col]].sort_values() for col in dist}
    rank = pd.Series({c: dists[c].values[0] for c in dists}).sort_values()

    while len(final_match) < len(dist):
        c = rank.index[0]
        if dists[c].index[0] not in final_match.values():
            # print c, dists[c].index[0]
            final_match[c] = dists[c].index[0]
            del rank[c]
        else:
            # print 'nope'
            rank.ix[c] = dists[c].iloc[1]
            rank = rank.sort_values()
            dists[c].drop(dists[c].index[0], inplace=True)

    return pd.Series(final_match)


def all_pairs(lst):
    """
    Return all combinations of pairs of items of lst where order within the pair and order of pairs does not matter.

    Note that when the list has an odd number of items, one of the pairs will be a singleton.
    """
    if not lst:
        yield [tuple()]
    elif len(lst) == 1:
        yield [tuple(lst)]
    elif len(lst) == 2:
        yield [tuple(lst)]
    else:
        if len(lst) % 2:
            for i in (None, True):
                if i not in lst:
                    lst = list(lst) + [i]
                    PAD = i
                    break
            else:
                while chr(i) in lst:
                    i += 1
                PAD = chr(i)
                lst = list(lst) + [PAD]
        else:
            PAD = False
        a = lst[0]
        for i in range(1, len(lst)):
            pair = (a, lst[i])
            for rest in all_pairs(lst[1:i] + lst[i+1:]):
                rv = [pair] + rest
                if PAD is not False:
                    for i, t in enumerate(rv):
                        if PAD in t:
                            rv[i] = (t[0],)
                            break
                yield rv


def ell_one(x):
    return sum(x)


def ell_two(x):
    return np.sqrt(sum([y ** 2 for y in x]))


def get_optimal_matches(dist, partitions, func=ell_one):
    """
    Implements optimal matching algorithm
    
    :param dist: DataFrame of pairwise distances
    :param partitions: (optional) Dict of items to list of items, if you want to restrict items
        to only match within a set of other possible items.
    :param func: Loss function when optimizing
        
    """
    # TODO: make this more elegant/efficient!

    if not partitions:
        partitions = {item: dist.index for item in dist}

    # Get unique cells; this is clunky
    unique_partitions = []
    for lst in partitions.values():
        if lst not in unique_partitions:
            unique_partitions.append(lst)

    final_match = []

    # Loop over cells
    for partition in unique_partitions:
        # Arbitrarily choose first item in the cell and use it to "name" the cell
        print "Getting optimal matches for cell " + partition[0] + " of size " + str(len(partition[0]))

        # Get permutations of possible pairs (inefficient to not use combinations, srynotsry)
        # Dropping 'leftover' city in partitions with odd numbers
        permutations = [[y for y in x if len(y) > 1] for x in all_pairs(partition)]
        permutations = {i: permutations[i] for i in range(len(permutations))}

        # Get func(distances) for each possible set of permutations
        tmp_dists = {}
        for i in permutations:
            tmp_dists[i] = func([dist.loc[permutations[i][j]] for j in range(len(permutations[i]))])
        best = pd.Series(tmp_dists).idxmin()

        final_match.extend(permutations[best])
    clear_output()

    # Convert to DataFrame
    x = pd.DataFrame(final_match).set_index(0)[1]
    y = pd.DataFrame(final_match).set_index(1)[0]
    result = pd.concat([x, y], axis=0)

    return result


def match(data, metric, matching_algorithm,
          k=1, loss_func=ell_one,
          bin_characteristics=set(), num_bins=set(),
          exclude=None, manual_exclude=None,
          exact_match_on=None, dist_match_on=None):
    """
    Matches units using matching_algorithm, based on distance metric over criteria in data.

    :param data: DataFrame. Indexed by cities, with characteristics across columns
    :param metric: Distance metric to use for calculating distance. e.g. mahalanobis or euclidean
    :param matching_algorithm: Matching algorithm to use: knn, greedy, or optimal
    
    :param k: (Optional) k in KNN, if KNN is used
    :param loss_func: (Optional) Loss function to use, if optimal matching is used

    :param exclude: (Optional) list of characteristics. These characteristics must be boolean. If a city is True
        for any of these characteristics, it will be dropped.
    :param manual_exclude: (Optional) list of cities to drop.

    :param bin_characteristics: (Optional) list of characteristics to transform into bins
    :param num_bins: (Optional) Required if bin_characteristics is not none. Dictionary where keys are characteristics
        to be binned, with values for the number of bins to be created
    :param exact_match_on: (Optional) list of characteristics to match exactly on (should be categorical variables)
    :param dist_match_on: (Optional) list of characteristics to match via the distance metric
    :return: 
    """

    assert matching_algorithm in ['knn', 'greedy', 'optimal']

    # Clean data
    df = clean_data(data, exclude=exclude, bin_characteristics=bin_characteristics, num_bins=num_bins,
                    manual_exclude=manual_exclude)

    # Partition the universe of cities into cells which exact match on characteristics in exact_match_on
    partitions = partition_universe(df, exact_match_on)

    # Calculate distance between all cities
    if dist_match_on:
        df = df[dist_match_on]
    dist = get_dist(df, metric=metric)

    # Run matching algorithm
    if matching_algorithm == 'knn':
        result = knn(dist, k=k, restrictions=partitions)
    elif matching_algorithm == 'greedy':
        result = greedy_match(dist, partitions)
    elif matching_algorithm == 'optimal':
        result = get_optimal_matches(dist, partitions, func=loss_func)

    return result, dist, partitions, df
