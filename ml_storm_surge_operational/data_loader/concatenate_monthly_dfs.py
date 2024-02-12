#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Concatenate operational files generated with prepare_df.py while running jobs in 
parallel.

There is one file for each month starting on Jan 2018 and ending in Mar 2021.
"""
import pandas as pd
import pickle
from multiprocessing import Pool

from storm_surge_oper.data_loader.prepare_df_operational import main_arome, main_obs_tide, main_stormsurge_corrected  
        
dir_dict = (
    '/lustre/storeB/project/IT/geout/machine-ocean/workspace'
    + '/paulinast/operational/data/preprocessed_input'
)

datetime_start_hourly = [
    '20180101',
    '20180201',
    '20180301',
    '20180401',
    '20180501',
    '20180601',
    '20180701',
    '20180801',
    '20180901',
    '20181001',
    '20181101',
    '20181201',
    
    '20190101',
    '20190201',
    '20190301',
    '20190401',
    '20190501',
    '20190601',
    '20190701',
    '20190801',
    '20190901',
    '20191001',
    '20191101',
    '20191201',
    
    '20200101',
    '20200201',
    '20200301',
    '20200401',
    '20200501',
    '20200601',
    '20200701',
    '20200801',
    '20200901',
    '20201001',
    '20201101',
    '20201201',
    
    '20210101',
    '20210201',
    '20210301'
    ]

datetime_end_hourly = [
    '20180131',
    '20180228',
    '20180331',
    '20180430',
    '20180531',
    '20180630',
    '20180731',
    '20180831',
    '20180930',
    '20181031',
    '20181130',
    '20181231',
    
    '20190131',
    '20190228',
    '20190331',
    '20190430',
    '20190531',
    '20190630',
    '20190731',
    '20190831',
    '20190930',
    '20191031',
    '20191130',
    '20191231',
    
    '20200131',
    '20200229',
    '20200331',
    '20200430',
    '20200531',
    '20200630',
    '20200731',
    '20200831',
    '20200930',
    '20201031',
    '20201130',
    '20201231',
    
    '20210131',
    '20210228',
    '20210331'
    ]
# Run processes in parallel 
# Call either main or main2
n_processes = len(datetime_start_hourly)
with Pool(n_processes) as pool:
    results = pool.map(main_stormsurge_corrected, zip(datetime_start_hourly, datetime_end_hourly))
    
# Concatenate monthly data
df_list_12_hours = []
##df_list_hourly = []
for date in datetime_start_hourly:
    print(date)
    #file_name = '/dict_df_all_stations_only_arome_12_hours' + date + '.pickle'
    #file_name = '/dict_df_all_stations_kyststasjoner_12_hours' + date + '.pickle'
    #file_name = '/dict_df_all_stations_obs_tide_kyststasjoner_12_hours' + date + '.pickle'
    file_name = '/dict_df_all_stations_stormsurge_corrected_kyststasjoner_past_forecasts_24_12_hours_' + date + '.pickle'
    with open(dir_dict + file_name, 'rb') as handle:
        b = pickle.load(handle)
        
    df_list_12_hours.append(b['df_12_hours'])
    #df_list_hourly.append(b['df_hourly_t0'])
    
df_list_12_hours_all_months = pd.concat(df_list_12_hours, axis=0)
#df_list_hourly_all_months = pd.concat(df_list_hourly, axis=0)

# Store df with all the data in a dict
dict_all_months = {
    'df_12_hours' : df_list_12_hours_all_months, 
    #'df_hourly' : df_list_hourly_all_months
    }

# Save dict with all the data
#Sdata_dir = '/home/paulinast/meps_store_a'

#fname_dict =  (
#    '/dict_df_all_stations_only_arome_all_months_12_hours_' 
#    + 'new_no_forecast_generated_in_the_past.pickle'
#)
# fname_dict =  '/dict_df_all_stations_kyststasjoner_all_months_12_hours_new_no_forecast_generated_in_the_past.pickle'
# fname_dict =  '/dict_df_all_stations_obs_tide_kyststasjoner_all_months_12_hours_new_no_forecast_generated_in_the_past.pickle'
#fname_dict =  (
#    '/dict_df_all_stations_obs_tide_kyststasjoner_all_months_12_hours.pickle'
#) # For the paper - old version

fname_dict =  (
    'dict_df_all_stations_obs_tide_kyststasjoner_all_months_12_hours'
    + '_past24hours.pickle'
)

fname_dict =  (
    'dict_df_all_stations_stormsurge_corrected_kyststasjoner' 
    + '_past_forecasts_24_12_hours_v2.pickle'
)

path_dict = dir_dict + '/' + fname_dict
print('path_dict:', path_dict)
with open(path_dict, 'wb') as handle:
    pickle.dump(dict_all_months, handle, protocol=pickle.HIGHEST_PROTOCOL)

