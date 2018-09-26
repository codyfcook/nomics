import pandas as pd
from scipy import stats
import inspect
from datetime import datetime, date
import pytz
from queryrunner_client import Client 

def rolling_average_hours(driver_filter, start_time, end_time, aggregation_level = 'weeks', window=4, qr=None, hdfs_port=None):
    '''
    This function returns a rolling average hours and completed trip for the drivers in the driver filter. It's warehouse, so can't do a bajillion drivers. 
    '''
    # If they didn't pass a client object, attempt to create one. I say attempt, because this will fail in some environments (like AWS boxes, where getuser() returns 'uber'
    if qr is None:
        qr = Client(user_email = getuser()+'@uber.com', hdfs_port=hdfs_port)
   
    params = {'aggregation_level': aggregation_level, 'start_time':start_time, 'end_time':end_time, 'window':window-1}
    q = '''
        with drivers as (
            {driver_filter}
        ),
        period_hrs as (
            select
                ahds.driver_uuid
                ,date_trunc('{aggregation_level}', start_timestamp_local) as {aggregation_level} 
                ,sum(minutes_worked)/60.0 as rolling_avg_hrs 
                ,sum(completed_trips) as rolling_avg_trips 
            from agg_hourly_driver_supply ahds 
            join drivers d 
                on d.driver_uuid = ahds.driver_uuid 
            where 1=1 
                and ahds.start_timestamp_local between '{start_time}' and '{end_time}' 
            group by 1,2 
        ) 
        select 
            driver_uuid 
            ,week 
            ,avg(hrs_worked) over (partition by driver_uuid order by week asc rows between {window} preceding and current row) as rolling_avg_hrs
            ,avg(completed_trips) over (partition by driver_uuid order by week asc rows between {window} preceding and current row) as rolling_avg_trips 
        from period_hrs 
    '''.format(**params)
    
    e = qr.execute('warehouse', q)
    d = pd.DataFrame(e.load_data())

    return d



