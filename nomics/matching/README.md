# Matching

Code to run (1) propensity score matching (2) distance matching (KNN; greedy matching; optimal matching) and (3) basic analytics on matched pairs (calculate ATT; some measures of balance)

To install it see [nomics repo readme](https://code.uberinternal.com/diffusion/UBNOM/browse/master/README.md).

## psm.py

Functions for performing propensity score matching. Estimate propensity score using a logit model and match across treatment/control based on propensity score (using KNN; to-do: caliper).

## evaluate.py

Functions to get ATT and measure the balance of covariates across matches.

## metric.py

Functions for performing distance matching based on some metric of distance. These functions are particularly useful for city matching to create "sister city" pairs.

There are currently two options for measuring distance: Mahalanobis distance or Euclidean distance. For Euclidean distance, variables are first standardized (z-scored) to make them comparable.

There are three options for matching based on distance: KNN, greedy matching, and optimal matching. Currently, KNN matches can only be made with replacement. Greedy matching and optimal matching currently can be made without replacement.

#### Calculating distance

The walkthrough below will use the example of matching cities, but any set of entities with characteristics can be matched.

1. Pass in **dataframe of cities with characteristics**
  * Includes the functionality to take characteristics which are continuous variables and discretize them by binning into (e.g.) quartiles

2. For any given city, **exact match** on a set of characteristics (e.g. has_pool; quartile of CP)

3. For any given city, **calculate distance** to all other cities within the set of exactly matched cities
  * Can use Mahalanobis or Euclidean distance

Now you have a collection of cities, with each city have a list of distances to all other cities in its partition.


#### Matching


**Option 1: k nearest neighbors**

KNN matching uses the k nearest neighbors based on distance. Currently, these matches can only be made with replacement.

**Option 2: greedy match**

1. Pick a partition
2. Match the two cities that are the very closest
3. Examine the next closest match across all cities in the partition
  1. If neither of the cities is already matched, match them
  2. If one of them is already matched, move on
4. Iterate until all cities are matched

* Advantage: Fast
* Disadvantage: When system is considered as a whole, may result in poor matches overall
* Can be done with or without replacement

**Option 3: optimal matching (Rosenbaum 2002)**

1. Pick a partition
2. Consider the permutations of all possible pairs at once
3. Compute some loss function over the set of distances (e.g. sum of squared)
 
