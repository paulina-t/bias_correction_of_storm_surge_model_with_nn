import os
import time
import datetime as dt
import pickle 
import lzma

from ml_models.data_loader.prepare_df import PrepareDataFrames 
import ml_models.utils.helpers as hlp

print("Put the interpreter in UTC, to make sure no TZ issues...")
os.environ["TZ"] = "UTC"
time.tzset()


def save_prepared_dataframe_2001_2020():

    stations = hlp.get_norwegian_station_names()
    stations.remove('NO_MSU')  # This station is not in the kyststasjoner file  
    
    data_dir = (
            '/lustre/storeB/project/IT/geout/'
            +'machine-ocean/workspace/paulinast/'
            )
    
    stations = hlp.get_norwegian_station_names()
    stations.remove('NO_MSU')  # This station is not in the kyststasjoner file  
    
    variables = [
        'obs', 'tide_py', 'roms', '(obs - tide_py)',
        'msl', 'u10', 'v10', 'wind_speed', 'wind_dir',
        'stormsurge_corrected', 'stormsurge_corrected - (obs - tide_py)',  'bias',
        '(roms - biasMET) - (obs - tide_py)', '(roms - biasMET)'
        ]
    
    era5_forcing = False   # TODO: Remove this argument and use forcings
    horizon = 120
    
    print('Correcting operational data...')
    # Test on Mar 2020-2021
    datetime_start_hourly=dt.datetime(2001, 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc)
    datetime_end_hourly=dt.datetime(2021, 3, 31, 0, 0, 0, tzinfo=dt.timezone.utc)

    prep = PrepareDataFrames(
        variables = variables,
        stations = stations, 
        ensemble_members = [-1],
        datetime_start_hourly = datetime_start_hourly,
        datetime_end_hourly = datetime_end_hourly,
        horizon = horizon + 1, 
        use_station_data = True,
        run_on_ppi = True,
        new_station_loc=True,
        era5_forcing=era5_forcing,
        add_aux_variables=False,
        data_from_kartverket=False
        )
    df = prep.prepare_features_labels_df()
    
    df_dict = {
        'df' : df, 
        'variables' : variables,
        'stations' : stations, 
        'ensemble_members' : [-1],
        'datetime_start_hourly' : datetime_start_hourly,
        'datetime_end_hourly' : datetime_end_hourly,
        'horizon' : horizon + 1, 
        'use_station_data' : True,
        'run_on_ppi' : True,
        'new_station_loc' : True,
        'era5_forcing' : era5_forcing,
        'add_aux_variables' : False,
        'data_from_kartverket' : False
        }
    
    print("df_dict['df']: ", df_dict['df'])
    
    with lzma.open(data_dir + '/df_2001_2020.pickle', "wb") as f:
        pickle.dump(df_dict, f, protocol=4)  # pickle.HIGHEST_PROTOCOL) https://github.com/lucianopaz/compress_pickle/issues/23


if __name__ == "__main__":
    
    save_prepared_dataframe_2001_2020()
    