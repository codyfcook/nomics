import pandas as pd
from scipy import stats
import inspect
from datetime import datetime, date
import pytz
import sys
from multiprocessing import Pool, cpu_count
from basil_utils import * 


## DATA MANIPULATION ##
def groups_apply_parallel(groups, func, cpu_count=4):
    '''
    Function to do applies in parallel. 
    '''
    pool = Pool(cpu_count)
    ret_list = pool.map(func, [group for name, group in groups])
    return pd.DataFrame.from_dict(map(dict, ret_list))

def search_cols(d, col):
    '''
    Search the cols of dataframe d and return cols with 'col' in the title
    '''
    return list([x for x in d.columns if col in x])

## UBER-SPECIFIC ##
def format_list_for_qr(cc, quote=True):
    '''
    Function that takes a list and returns a string that can be inserted right into queries 
    (e.g., in SQL can do 'where name in ({formatted_list})')
    '''
    if isinstance(cc, basestring):
        cc = [cc]
    if quote:
        return ','.join(("'{}'".format(x) for x in cc))
    else:
        return ','.join(("{}".format(x) for x in cc))


## MEMORY / SYSTEM CHECKS 
def dfs_in_mem():
    '''
    Returns a list of Pandas dataframes in memory
    '''
    return [var for var in dir() if isinstance(eval(var), pd.core.frame.DataFrame)]

def mem_usage(obj): 
    '''
    Prints how much memory the object is taking up 
    '''
    print "Mem usage: 0.4f gb" % (sys.getsizeof(obj)/1000000000)

