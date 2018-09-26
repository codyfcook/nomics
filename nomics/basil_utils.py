import pandas as pd
import numpy as np
from math import isnan
from numpy import nan
import datetime as dt
from itertools import chain
import hashlib


def dta_to_csv(filename, path, save_path='/Users/basil.halperin/temp'):
    data = pd.read_stata(path + filename + '.dta')
    data.to_csv(save_path + filename + '.csv')
    print path + filename + ".dta converted to CSV at " + save_path + filename + ".csv"


def invert_dict(dic):
    inv = {}
    for k, v in dic.iteritems():
        inv[v] = inv.get(v, [])
        inv[v].append(k)
    return inv


def invert_dict_one_to_one(dic):
    return {v: k for k, v in dic.iteritems()}


def invert_dict_of_lists(dic):
    return {z: k for k, v in dic.iteritems() for z in v}


def flatten_list(lst, remove_nan=False):
    flat = [item for sublist in lst for item in sublist]
    if remove_nan:
        flat = [x for x in flat if str(x) != 'nan']
    return flat


def get_onetoone_map_from_df(df, column1, column2):
    return dict(zip(df[column1].values, df[column2].values))


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


def dict_to_df(dic):
    return pd.DataFrame(dict([(k, pd.Series(v))
                              for k, v in dic.iteritems()]))


def to_index(df):
    return (1+df).cumprod()


def get_max_drawdown(rets, min_periods):
    ti = to_index(rets)
    ratio = ti/ti.expanding(min_periods=min_periods).max()
    max_drawdown = ratio.min() - 1
    max_drawdown_date = ratio.idxmin()
    return max_drawdown, max_drawdown_date


def add_dicts(orig, extra):
    dest = dict(chain(orig.items(), extra.items()))
    return dest


def list_to_sql_lower(lst):
    return "('" + "', '".join(str(x).lower() for x in lst) + "')"


def t_stat_of_mean(beta, beta_0=0):
    beta = pd.DataFrame(beta)
    t = {}
    for col in beta.columns:
        tmp = beta[col].dropna()
        n = len(tmp)
        se = tmp.std()/np.sqrt(n)
        t[col] = (tmp.mean() - beta_0[col])/se
    return pd.Series(t)


def slice_dict(dic, keys):
    return {k: v for k, v in dic.iteritems() if k in keys}


def unix_to_datetime(x, fmt='%Y-%m-%d %H:%M:%S'):
    if isnan(float(x)):
        return nan
    else:
        return dt.datetime.fromtimestamp(
            int(x)/1000
        ).strftime(fmt)


def dummify(df, columns, float_dummies=True):
    dummified = df.copy()
    dummy_names = {}
    for col in columns:
        dummy_names[col] = []
        for x in dummified[col].unique():
            col_name = col + '__' + str(x)[0:4].replace('.', '_')
            dummified[col_name] = (dummified[col] == x)
            dummy_names[col].append(col_name)
        dummy_names[col] = sorted(dummy_names[col])
        if float_dummies:
            dummified[dummy_names[col]] *= 1.0

    return dummified, dummy_names


def to_int(df, cols=None):
    """
    Converts the values of relevant columns of a dataframe into integers by unique values
    Useful for running regressions with absorbed FE

    Inputs 
    ----------
    df : dataframe
    cols : list, or None
        subset of columns of df; or None to use all columns
    Returns
    -------
    df : dataframe
        dataframe with values in columns cols converted to integers by unique values
    """
    if cols is None:
        cols = df.columns

    for col in cols:
        df[col] = df[col].astype('category')
    df[cols] = df[cols].apply(lambda x: x.cat.codes)

    return df


def bin_characteristic(dfr, characteristic, num_bins):
    """
    For a DataFrame dfr with items in the index and characteristics of each item in the column,
    return the same dataframe with additional boolean columns binning items based on characteristic 

    :param dfr: DataFrame, indexed by item where columns are characteristics of each item
    :param characteristic: column name to be binned
    :return: inputted DataFrame, with additional columns indicating boolean which bin the item belongs to
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


def convert_presto_to_hive(qry):
    # Check for multi-line comments"
    if "/*" in qry or "*/" in qry:
        raise ValueError("Hive queries can't have multi-line comments!")

    # date_parse
    # TODO

    # Direct mapping
    direct_mapping = {
            'date_diff': 'datediff',
            'week_of_year': 'weekofyear',
            'day_of_week': 'dayofweek',
            'varchar': 'string',
        }
    for k, v in direct_mapping.iteritems():
        qry.replace(k, v)

    return qry


def bin_variables(df, varz, var_name_append='_bucketed',
                  qcut=True, qbins=4, qlabels=None, qduplicates='drop',
                  cut=False, **cut_kwargs
                  ):
    if qcut:
        for var in varz:
            df[var + var_name_append] = pd.qcut(df[var], qbins, labels=qlabels, duplicates=qduplicates)

    elif cut:
        for var in varz:
            df[var + var_name_append] = pd.cut(df[var], **cut_kwargs)

    return df


def bin_fml(fml, varz):
    for var in varz:
        replace_with = "C(" + var + ")"
        fml = fml.replace(var, replace_with)
    return fml


def get_pretty_reg_table(reg, renames, fe, order, num_digits=NUM_DIGITS):
    """
    Transforms a statsmodel regression Result object into a pretty column with coefficients; standard errors below
    each coefficient in parantheses; asterixes for significance; X's indicating fixed effects included; and num. obs.

    reg : statsmodels regression Result object
    renames : dictionary where keys are parameter names in reg and values are desired pretty renames
    fe : dictionary where keys are parameter names in reg that are FE and values are pretty renames
        For example, if your regression includes "C(city)", you would have {'city': 'City'}
    order : desired ordering of coefficients
    num_digits : integer, number of digits to round to for coefficient estimates
    """
    # Get betas, SE, t-values, nobs
    betas = reg.params.rename(renames)
    se = reg.bse.rename(renames)
    pvalues = reg.pvalues.rename(renames)
    nobs = str(int(reg.nobs))

    # Drop FE variables
    for fe_var in fe:
        vars_to_keep = [c for c in betas.index if fe_var not in c]
        betas = betas.loc[vars_to_keep]
        se = se.loc[vars_to_keep]
        pvalues = pvalues.loc[vars_to_keep]

    # Get stars
    one_star = pvalues <= 0.05
    two_stars = pvalues <= 0.01
    three_stars = pvalues <= 0.001

    # Add stars
    betas = np.round(betas, num_digits).astype(str)
    betas.loc[one_star] += '*'
    betas.loc[two_stars] += '*'
    betas.loc[three_stars] += '*'

    # Prep SEs to put below betas, using an arbitrary string
    se = np.round(se, num_digits + 2).astype(str)
    arbitrary_string = '          _'
    se = se.rename({
        k: k + arbitrary_string for k in se.index
    })
    se = '(' + se + ')'

    # If order is None, just use the default
    order = order or reg.params.index

    # Put SEs below betas
    output = pd.concat([betas, se], axis=0).sort_index()
    # Reorder as desired
    zipped = zip(order, [x + arbitrary_string for x in order])
    order = [item for sublist in zipped for item in sublist]
    output = output.loc[order]
    # Remove arbitrary string
    output = output.rename({
        k: '' for k in output.index
        if arbitrary_string in k
    })

    # Add FE at bottom
    output[' '] = ''
    for fe_var in fe:
        output.loc[fe[fe_var]] = 'X'

    # Add nobs at bottom
    output['  '] = ''
    output['No. observations'] = nobs

    return output


def get_treatment_group(user_uuid, experiment_name, treatment_map):
    """
    Get Morpheus treatment from user UUID and experiment name.
    WARNING: when I have tested this, it hasn't matched Morpheus, _shruggie_
    
    :param user_uuid: User UUID. Can also be device UUID if that is the unit of treatment. 
    :param experiment_name: Morpheus experiment name.
    :param treatment_map: Map from integer (00 - 99) to treatment name.
        Should obviously be based on Morpheus treatment distribution.
    :return: 
    """
    if experiment_name is not None:
        hexadecimal = hashlib.md5(user_uuid + 'experiments.' + experiment_name).hexdigest()
        decimal = int(hexadecimal, 16)
        mod100 = decimal % 100
        return treatment_map[mod100]
    else:
        return None


def df_to_latex(df, title, footnote='', default_stars=True, to_replace_additional=None,
                save=True, save_name='latex_tables/latex_table.txt'):
    """
    We're going to use pandas' default .to_latex() method, but then modify it to make it pretty

    :param df: dataframe
    :param title: title of table
    :param footnote: description at bottom of table, if any, e.g. "\textit{Note:} The dependent variable is..."
    :param default_stars: boolean, if True automatically append '*** p$<$0.01, ** p$<$0.05, * p$<$0.1' to footnote
    :param to_replace_additional: customize latex further using string replacement
    :param save: boolean
    :param save_name: file path in which to save latex as a txt file
    :return: 
    """
    latex_base = df.to_latex()

    # First, grab the header that we need to replace
    ncol = len(df.columns)
    l_times_n_table_col = 'l' * (ncol + 1)
    to_replace_top = '\\begin{tabular}' + '{{{0}}}'.format(l_times_n_table_col)

    # This is what we're going to replace it with
    replace_with_top = '\\begin{table}[htbp] \\centering\n'
    replace_with_top += '''\\caption{{{0}}}'''.format(title)
    replace_with_top += '''
\\label{contemp_base}
\\begin{adjustbox}{max width=\\textwidth, max totalheight=\\textheight}
\\begin{threeparttable}
\\begin{tabular}'''
    replace_with_top += '{{{0}}}\n'.format('l' + 'c' * ncol)

    # Now, what we're going to replace at the end
    if default_stars:
        footnote += '\n*** p$<$0.01, ** p$<$0.05, * p$<$0.1'

    to_replace_bottom = '\\end{tabular}'
    replace_with_bottom = '''
\end{tabular}
\\begin{tablenotes}\n'''
    replace_with_bottom += '\\item \\footnotesize{{footnote}}'.format(footnote)
    replace_with_bottom += '''
\\end{tablenotes}
\\end{threeparttable}
\\end{adjustbox}
\\end{table}
'''

    # Collect it together
    to_replace = {
        to_replace_top: replace_with_top,
        to_replace_bottom: replace_with_bottom,
    }
    if to_replace_additional:
        to_replace = dict(to_replace, **to_replace_additional)

    # Replace
    latex = latex_base
    for k, v in to_replace.iteritems():
        latex = latex.replace(k, v)

    # Save
    if save:
        with open(save_name, 'w') as f:
            f.write(latex)

    return latex
