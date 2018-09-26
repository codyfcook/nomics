# Nomics

Containing general utilities/functions often useful for economic analyses at Uber. More detailed documentation is found in the comments of the code itself

To install it run

  git clone gitolite@code.uber.internal:ubernomics/nomics
  cd nomics && pip install -e . && cd ..

## Utilities

There are various simple functions that may be of use. Often these functions just simplify the notation to reduce Googling/typing/typos

Utilities also imports from `basil_utils.py`, so running `from nomics.utilities import *` will get you the functions below plus a bunch of Basil's.

Sample of a few of the utility functions:

1. `groups_apply_parallel(groups, func, cpu_count=4)`
    Run applies in parallel. Make sure not to overwhelm the CPUs.
2. `format_list_for_qr(list, quote=True)` formats a list of items so that it can be used in a query (e.g. as 'name in ({list})')
3. `dfs_in_mem()` shows you the dataframes currently being held in memory
4. `mem_usage(obj)` prints how much memory the object is using


## Statistics

##### Absorber

The absorber allows you to run absorbed FE regressions. See Stata's areg command for more details on what this means.

To run it, do:

```
from nomics.stats.absorber import OLSAbsorb
y = d['y_col']
dense = d['dense_cols']
absorb = np.array(d['absorb_cols'])
model = OLSAbsorb(y, dense, absorb)
results = model.fit()
print results.summary()
```

The absorbed cols should be integers from 1 to N denoting the group for which the effect size (coefficient) will be absorbed. You can turn any old column in to a series such integeres by running:

```
d[new_col] = d[old_col].astype('category').cat.codes.astype(int)
```

#### ZIP

Code to run a Zero Inflated Poisson.

```
from nomics.stats.zip import ZIPoisson
y = d['y_cols']
x = d['x_cols']
# The columns you want included in the logit part of a ZIP to predict 0/1 in the y col
inflateX = ['inflate_cols']

mod = ZIPoisson(y,x,inflateX).fit()
print mod.summary()
```

Also has code for a Vuong test.


#### Experiments

Has two functions to check whether treatment groups are balanced.

1. `check_for_balance_with_regs(cat_vars, cont_vars, group_var, data)` For continuous variables, get's the p-value from an F-test of var on treatment to see if treatment can predict variable. For categorical vars uses a chi squared test to see if treatment more likely to appear in some categories than others
2. `pairwise_t_test(data, cont_vars, group_var)`: For each variable, compare pairwise across groups and do a t-test to see if different


## Matching

Code to run (1) propensity score matching (2) distance matching (KNN; greedy matching; optimal matching) and (3) basic analytics on matched pairs (calculate ATT; some measures of balance)

See [matching readme](https://code.uberinternal.com/diffusion/UBNOM/browse/master/nomics/matching/README.md) for further details.
