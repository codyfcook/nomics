'''
Author: Cody Cook 
'''

def check_for_balance_with_regs(cat_vars, cont_vars, group_var, data):
    '''
    For continuous vars: get's the pvalue of the F test of var ~ treatment, to see if treatment can predict variable
    For categorical vars: chi squared test to see if treatment is more likely to appear in some categories than others
    '''
   
    # Check that vars are in data
    assert (len(cont_vars)+len(cat_vars))==(len([x for x in cont_vars if x in data.columns])+len([x for x in cat_vars if x in data.columns]))
    assert group_var in data.columns

    # For each of the continuous variables, check the p value of the F-test in a regression to see if the treatment predicts the variable
    rv = {}
    for cont_var in cont_vars:
        mod_formula = '%s ~ C(%s)' % (cont_var, group_var)
        mod = smf.ols(mod_formula, data=data).fit()
        rv[cont_var] = mod.f_pvalue

    # For each of the categorical variables, check to see if the treatment is more likely to appear in some categories than others using a chi-sq test
    for cat_var in cat_vars:
        t = pd.crosstab(data[group_var],data[cat_var]).reset_index().iloc[:,1:]
        chi2, p, dof, ex = stats.chi2_contingency(t)
        rv[cat_var] = p
   
    # Return a data frame
    rv = pd.Series(rv).reset_index()
    rv.columns = ['variable', 'p_val']

    return rv

def pairwise_t_test(data, cont_vars, group_var):
    '''
    For each variable (need to be continue, or zero-one), compares pairwise the means across all groups. Returns the average, min, and max pvalue (t-test) 
    '''
    def _get_pvals(data, groups, variable):
        pvals = []
        for g1 in groups: 
            for g2 in [x for x in groups if x!=g1]: 
                pvals = pvals + [stats.ttest_ind(data[data.treatment==g1][variable], data[data.treatment==g2][variable]).pvalue]
        return pvals 

    groups = data[group_var].unique()
    rv = pd.DataFrame(columns=['variable', 'mean_pval', 'min_pval', 'max_pval'])
    for v in cont_vars:
        pvals = _get_pvals(data, groups, v)
        temp = pd.DataFrame([[v, np.mean(pvals), np.min(pvals), np.max(pvals)]], columns=['variable', 'mean_pval', 'min_pval', 'max_pval'])
        rv = pd.concat([rv, temp])
        del temp
    
    return rv 

