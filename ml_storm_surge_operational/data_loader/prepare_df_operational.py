"""Prepare a single DataFrame with raw operational data.

Prepare a DataFrame that contains all the features and labels needed to run the 
ML models at the specified locations and for the specified time range. For this, 
we need to load storm surge data from the operational files "kyststasjoner" 
files and meteorological forecasts from MEPS/AROME. We transform each subset of
data into DataFrames, and lastly we join the DataFrames obtained for each group
of variables.

Notice that this class generates only operational data, i.e., no hindcast data. 
These forecast data are stored on Met Norway's Post-Processing Infrastructure 
(PPI).

Notice that the data generated are not post-processed (e.g. gap filling, 
normalizing, etc.), for that we use preprocessor_operational.py.
"""
import numpy as np
import pandas as pd
import datetime as dt
from pathlib import Path
import os
import pickle


import ml_storm_surge_operational.utils.helpers as hlp
import ml_storm_surge_operational.utils.assertions as asrt
from ml_storm_surge_operational.data_loader.read_kyststasjoner import ReadKyststasjoner
from ml_storm_surge_operational.data_loader.read_kyststasjoner_hourly import (
    ReadKyststasjoner 
    as ReadKyststasjonerHourly
)
from ml_storm_surge_operational.data_loader.read_arome import ReadArome
import time

os.environ["TZ"] = "UTC"
time.tzset()

class PrepareDataFrames():
    """Prepare a single DataFrame with raw observations and data from Nordic4-SS
    
    Prepare a DataFrame that contains all the features and labels needed to run 
    the ML models at the specified locations and for the specified time range.

    1) We group variable names of data that can be loaded using the same 
    method.
    2) We determine which variables are derived from others and create 2 
    DataFrames, one containing all the auxiliary variables, and one containing 
    only the variables we are interested in. An example of derived variables is
    (obs - tide_py). In this examples, the auxiliary variables are obs and 
    tide_py.

    Attributes
    ----------   
    stations : list of strings
        List of station IDs, for example ['NO_OSC', 'NO_BGO']

    horizon : int
        Horizon of the predictions.

    self.window_width_past : int
        Hours of past observations and tides.
        
    datetime_start_hourly : datetime
        Date indicating when the period of interest begins. The final period 
        depends on data availability. All dates without observations will be  # TODO: check if it is true
        removed. Default is 2018.01.01 00:00.
        
    datetime_end_hourly : datetime
        Date indicating when the period of interest ends. The final period 
        depends on data availability. All dates without observations will be 
        removed. Default is 2021.03.31 00:00. 

    use_existing_files : bool, optional
        Whether to load data from some pre-generated files saved on PPI. Setting 
        this parameter to True will save computational time. These pre-generated
        files can be generated with the methods `main_arome`, `main_obs_tide`, 
        and `main_stormsurge_corrected`, for the set of parameters that the user 
        needs. 

    past_forecast : int, optional
        When the oldest forecasts are generated, measured as number of hours 
        before the analysis time of the correction model. It must, in general, 
        be a multiple of 12. The default is 0, meaning no past forecasts are 
        added to the final DataFrame.

    filter_times_in_hourly_df : bool, optional
    """
    
    def __init__(
            self,
            variables,
            stations,
            horizon:int,  # max
            window_width_past:int=48, # use the same for obs and tide
            datetime_start_hourly=dt.datetime(
                2018, 1, 1, 0, 0, 0, 
                tzinfo=dt.timezone.utc
                ),
            datetime_end_hourly=dt.datetime(
                2021, 3, 31, 0, 0, 0, 
                tzinfo=dt.timezone.utc
                ),
            use_existing_files:bool=True,
            past_forecast:int=0,
            ):

        self.variables = variables
        self.stations = stations
        self.horizon = horizon
        self.window_width_past = window_width_past
        self.datetime_start_hourly = datetime_start_hourly
        self.datetime_end_hourly = datetime_end_hourly
        self.use_existing_files = use_existing_files
        self.past_forecast = past_forecast
        
        asrt.assert_dt_start_less_than_dt_end(
            self.datetime_start_hourly, 
            self.datetime_end_hourly
            )

        # Group variables that are retrieved form the same dataset using the 
        # same method from this class. These groups of variables will then be 
        # used to generate with the same method DataFrames to be concatenated in 
        # the final DataFrame returned with this class.
        self.obs_tide_vars = [
            'obs', 
            'tide_py', 
            '(obs - tide_py)'
            ]  
        self.operational_vars = [
            'stormsurge', 
            'stormsurge_corrected', 
            #'bias'
            ]
        self.combined_stormsurge_obs_var = [
            'stormsurge - (obs - tide_py)',
            'stormsurge_corrected - (obs - tide_py)'
            ]
        self.station_forcing_vars = [
            'msl', 
            'u10', 
            'v10', 
            'swh', 
            'mwd'
            ] 
        self.wind_vars = [
            'wind_speed', 
            'wind_dir'
        ]
        self.wind_arome_vars = [
            'wind_speed_arome', 
            'wind_dir_arome'
        ]

        # Get a list of the auxiliary variables needed
        self.create_aux_variables_list()
        
        # Get a list of the permanent stations
        self.operational_stations()
        
    def create_aux_variables_list(self):
        """
        Make a list of the auxiliary variables needed for the computation of the 
        derived variables.

        Returns
        -------
        None.

        """
        self.aux_variables = []

        if '(obs - tide_py)' in self.variables:
            self.aux_variables.append('obs')
            self.aux_variables.append('tide_py')
        if 'stormsurge_corrected - (obs - tide_py)' in self.variables:
            self.aux_variables.append('obs')
            self.aux_variables.append('tide_py')
            self.aux_variables.append('stormsurge_corrected')
            self.aux_variables.append('(obs - tide_py)')
        if 'wind_speed' in self.variables or 'wind_dir' in self.variables:
            self.aux_variables.append('u10')
            self.aux_variables.append('v10')
        if 'stormsurge - (obs - tide_py)' in self.variables:
            self.aux_variables.append('obs')
            self.aux_variables.append('tide_py')
            self.aux_variables.append('stormsurge')
            self.aux_variables.append('(obs - tide_py)')            
        if '(obs - tide_py)' in self.aux_variables:
            self.aux_variables.append('obs')
            self.aux_variables.append('tide_py')
        if ('wind_speed' or 'wind_dir') in self.variables:
            if self.station_forcing_vars:
                self.aux_variables.append('u10')
                self.aux_variables.append('v10')
            else:
                self.aux_variables.append('x_wind_10m')
                self.aux_variables.append('y_wind_10m')
        
    def operational_stations(self):
        """Provides only the Norwegian stations in self.stations"""
        # TODO: This is unnecessary if we only work with Norwegian data.
        # We could instead check that all the stations actually are Norwegian 
        # and return an error message or warning if they are not.
        norwegian_stations = hlp.get_norwegian_station_names()
        
        oper_stations = list(set(self.stations) & set(norwegian_stations))
        try:
            oper_stations.remove('NO_MSU') # This station is not in all the files
        except:
            pass

        self.oper_stations = oper_stations
        
    def prepare_features_labels_df(self):
        """Prepare the final DataFrame.
        
        Prepare the final DataFrame and the auxiliary DataFrame for the 
        specified variables and stations in the class arguments.

        Returns
        -------
        self.df_features_and_labels : DataFrame
            DataFrame with features and labels.

        """
        self.df_aux = self.prepare_df(aux=True)
        self.df_features_and_labels = self.prepare_df(aux=False)  
        return self.df_features_and_labels  # TODO: call this method from init and do not return anything

        
    def prepare_df(self, aux=False):
        """Generate a DataFrame for a set of variables, either the auxiliary df
        or the final df.
        
        This is a wrapper function that concatenates DataFrames generated with 
        data from different sources using this class. The DataFrames generated 
        for each group of variables are stored in a list and later concatenated 
        in a single DataFrame. 
        
        For each group of variables we:

        1) intersect the list of variables in the group with all the variables 
        requested. 
        If there are any variables in the intersection:
        2) we generate a DataFrame for that group of variables.

        Returns
        -------
        df : DataFrame

        """
        if aux:
            print('Preparing auxiliary DataFrame...')
            list_all_variables = list(set(self.aux_variables))
        else: 
            list_all_variables = list(set(self.variables))
            print('Preparing Datarame with features and labels...')

        # Create empty list of DataFrames to fill in with the DataFrames 
        # generated of each group of variables
        dfs = []

        # ----------------------------------------------------------------------
        # 1) Make a DataFrame for each group of variables and append to list.
        # ----------------------------------------------------------------------
        
        # ------ a) Observations and tide ------
        # Check which obs_tide variables are requested
        df_obs_tide_vars= list(
            set.intersection(
                set(self.obs_tide_vars), 
                set(list_all_variables)
                )
            )
        # Generate DataFrame for the requested obs_tide variables and add it to 
        # the list of DataFrames.
        if df_obs_tide_vars:
            dfs.append(self.generate_obs_tide_df(df_obs_tide_vars)) 
            print('Added obs and tide data.' )
            # Generate DataFrame with (obs - tide)
            df_obs_tide = self.add_obs_tide(variables=list_all_variables)
            # Add obs_tide DataFrame to the list of DataFrames
            if not df_obs_tide.empty:
                dfs.append(df_obs_tide)
                print('Added (obs-tide) data.')

        # ------ b) Forcing data at station locations (MEPS) ------
        df_station_forcing_vars= list(
            set.intersection(
                set(self.station_forcing_vars), 
                set(list_all_variables)
                )
            )
        if df_station_forcing_vars:
            dfs.append(
                self.generate_station_forcing_df(
                    variables=df_station_forcing_vars
                    )
                )
            print('Added meteorological data.')
        
        # ------ c) Operational storm surge data (Nordic4-SS) ------      
        df_operational_vars = list(
            set.intersection(
                set(self.operational_vars), 
                set(list_all_variables)
                )
            ) 
        if df_operational_vars:
            df = self.generate_operational_df(variables=df_operational_vars)
            dfs.append(df)
            print('Added operational data')
            
        # ------ d) Combined operational storm surge data and obs ------
        df_combined_stormsurge_obs_vars = list(
                set.intersection(
                    set(self.combined_stormsurge_obs_var), 
                    set(list_all_variables)
                    )
                )
        if df_combined_stormsurge_obs_vars:
            df_stormsurge_obs_tide = self.add_roms_obs_tide(
                variables=list_all_variables
                )
            if not df_stormsurge_obs_tide.empty:
                dfs.append(df_stormsurge_obs_tide)
                print('Added stormsurge - (obs-tide) data.')

        df_wind_vars = list(
                set.intersection(
                    set(self.wind_vars), 
                    set(list_all_variables)
                    )
                )

        if df_wind_vars:
            dfs.append(self.add_wind_speed_and_direction_vars())
            print('Added wind data: ')
            print(dfs[-1])

        # ----------------------------------------------------------------------
        # 1) Concatenate DataFrames in the list.
        # ----------------------------------------------------------------------

        if len(dfs) > 1: 
            df = dfs[0]
            dfs.pop(0)
            while len(dfs) > 0:
                try:
                    df = df.join(dfs[0])
                except:
                    print('Could not join DataFrames')
                dfs.pop(0)
        elif len(dfs) ==1:
            df = dfs[0]
        else:
            df = pd.DataFrame()

        # Remove duplicated rows. Find out why they are duplicated, because this
        # dows not happen in the older versions.
        df = df[~df.index.duplicated(keep='last')]
             
        return df

    def add_wind_speed_and_direction_vars(self):
        """Compute wind speed and/or wind direction.
        
        For each station, compute the wind speeds and/directions if the 
        variables 'wind_speed' or 'wind_dir' are specified as feature or label 
        variables.

        Parameters
        ----------
        df : DataFrame
            DataFrame containing at least the variables 'u10' and 'v10' if 
            using forcing at station location or 'x_wind_dir' and 'y_wind_dir'
            if using forcing over a mesh.

        Returns
        -------
        df : DataFrame
            DataFrame with wind speed and/or wind direction columns.

        """
        if True:#self.use_station_data:
            u_var = 'u10'
            v_var = 'v10'
        else:
            u_var = 'x_wind_10m'
            v_var = 'y_wind_10m'
                
        df = pd.DataFrame()
        if 'wind_speed' in self.variables:
            print('Adding wind_speed columns...')
            #u_var = 'u10'
            #v_var = 'v10'
            for station in self.stations:
                df['wind_speed_' + station] = hlp.wind_speed(
                    self.df_aux[u_var + '_' + station],
                    self.df_aux[v_var + '_' + station])
        
        if 'wind_dir' in self.variables:
            print('Adding wind_dir columns...')
            #u_var = 'u10'
            #v_var = 'v10'
            for station in self.stations:
                df['wind_dir_' + station] = hlp.wind_dir(
                    self.df_aux[u_var + '_' + station],
                    self.df_aux[v_var + '_' + station])            
                
        return df
    
    def add_obs_tide(self, variables):
        """Compute the difference between observations and tides.
        
        Compute the difference between observations and tides at each location 
        and for each lead time. Then, add the column 
        '(obs - tide_py)_t_<leadtime>_<stationID>' to the DataFrame.
        
        Parameters
        ----------            
        variables : list of str
            list with variables to check. If '(obs - tide_py)' is in the list, 
            add the column to the DataFrame.

        Returns
        -------
        df : DataFrame
            DataFrame with the difference between observations and tides.

        """
        df = pd.DataFrame()
        if '(obs - tide_py)' in variables:
            print('Adding (obs - tide_py) columns...')
            for station in self.stations:
                for h in range(self.horizon):
                    if h==0:
                        col_name = '(obs - tide_py)_' + station
                        obs_col_name = 'obs_' + station
                        tide_col_name = 'tide_py_' + station
                    else:
                        col_name = '(obs - tide_py)_t' + str(h) + '_' + station
                        obs_col_name = 'obs_t' + str(h) + '_'  + station
                        tide_col_name = 'tide_py_t' + str(h) + '_' + station
                    df[col_name] = (
                        self.df_obs_tide[obs_col_name]
                        - self.df_obs_tide[tide_col_name]
                        )
        return df
    
    def add_roms_obs_tide(self, variables):        
        """Compute the difference between Nordic4-SS and observed storm surges.
        
        Compute the difference between the surges modeled with Nordic4-SS and 
        the observed storm surges at each location and for each lead time. Then, 
        add the columns 'roms - (obs - tide_py)_t_<leadtime>_<stationID>' to the 
        DataFrame.
        
        Parameters
        ----------            
        variables : list of str
            list with variables to check. If 'roms - (obs - tide_py)' is in the 
            list, add the column to the DataFrame.

        Returns
        -------
        df : DataFrame
            DataFrame with the difference between the Nordic4-SS and the 
            observed storm surgers.

        """
        df = pd.DataFrame()
        roms_var = ''
        roms_obs_tide_vars = [
            'stormsurge - (obs - tide_py)', 
            'stormsurge_corrected - (obs - tide_py)'
            ]
        vars_to_iterate_over = list(set(variables) & set(roms_obs_tide_vars))
        
        if vars_to_iterate_over:
            for var in vars_to_iterate_over:                
                # --- Operational ---
                if var == 'stormsurge - (obs - tide_py)':
                    roms_var = 'stormsurge'
                elif var == 'stormsurge_corrected - (obs - tide_py)':
                    roms_var = 'stormsurge_corrected'
                df_roms = self.df_operational

                print('Adding ' + roms_var + ' - (obs - tide_py) columns...')
                for station in self.stations:
                    for h in range(self.horizon):
                        if h==0:
                            col_name = (
                                roms_var 
                                + ' - (obs - tide_py)_' 
                                + station
                            )
                            df[col_name] = (
                                df_roms[roms_var + '_' + station]
                                - (
                                    self.df_obs_tide['obs_' + station]
                                    - self.df_obs_tide['tide_py_' + station]
                                    )
                                )
                        else:  # h!=0
                            col_name = (
                                roms_var 
                                + ' - (obs - tide_py)_t' 
                                + str(h) 
                                + '_'  
                                + station
                            )

                            roms_col_name = (
                                roms_var 
                                + '_t' 
                                + str(h) 
                                + '_'  
                                + station
                            )

                            obs_col_name = (
                                'obs_t' 
                                + str(h) 
                                + '_'  
                                + station
                            )

                            tide_col_name =(
                                'tide_py_t' 
                                + str(h) 
                                + '_'  
                                + station
                            )

                            df[col_name] = (
                                df_roms[roms_col_name]
                                - (
                                    self.df_obs_tide[obs_col_name]
                                    - self.df_obs_tide[tide_col_name]
                                    )
                                )       

        return df
    
    def generate_obs_tide_df(self, variables):
        """Generate a DataFrame with observations and tides.
        
        Generate a DataFrame with past observations and past and future tide 
        estimations at each location.
        
        Parameters
        ----------            
        variables : list of str
            list with variables to check. If 'obs' or 'tide_py' are in the 
            list, add the column to the DataFrame.

        Returns
        -------
        df : DataFrame
            DataFrame with observations and/or tides.
        """
        # Data directory where some pre-generated dfs are stored in dicts
        data_dir = (
            '/lustre/storeB/project/IT/geout/machine-ocean/workspace'
            + '/paulinast/operational/data/preprocessed_input'
        )

        file_name =  (
            'dict_df_all_stations_obs_tide_kyststasjoner_all_months_12_hours'
            + '_past24hours.pickle'
        )

        if  (self.use_existing_files  
             and Path(data_dir + '/' + file_name).is_file()):
            
            # Reading all the kyststasjoner files is time consuming. Therefore,
            # we give the option to load data pre-generated with this class and 
            # concatenate_monthly_dfs.py to same computational cost. Notice that
            # this files has not necessarily been generated for the parameters 
            # needed. In that case, set self.use_existing_files to False.
            
            print('Opening already generated kyststasjoner file.')
            with open(data_dir + '/' + file_name, 'rb') as handle:
                station_file = pickle.load(handle)

            # Get the 12-hourly data at all stations for all the lead times.
            station_file = (
                station_file['df_12_hours']  # 'df_hourly_t0'
                .loc[self.datetime_start_hourly: self.datetime_end_hourly]
                )
            
            col_names = []                            
            for station in self.stations:
                    for var in variables:
                        for h in range(-self.window_width_past, self.horizon):
                            if h == 0 :
                                h_str = ''
                            else:
                                h_str = '_t' + str(h)
                            col_name  = var + h_str +  '_' + station
                            col_names.append(col_name) 

            col_names = list(set(col_names) & set(list(station_file.columns)))         
            self.df_obs_tide = station_file[col_names]

        else:
            # Generate obs and tide DataFrames for times before and after 
            # analysis time. 

            df_obs_tide_past= (
                self.generate_obs_tide_kyststasjoner_df_past(variables)
            )

            df_obs_tide_future = (
                self.generate_obs_tide_kyststasjoner_df_future(variables)
            )

            self.df_obs_tide =  pd.concat(
                        [df_obs_tide_past, df_obs_tide_future], 
                        axis=1
                        )
            
            m1 = (pd.to_datetime(self.df_obs_tide.index).hour == 0)
            m2 = (pd.to_datetime(self.df_obs_tide.index).hour == 12)
            self.df_obs_tide = self.df_obs_tide[m1 | m2]

        return self.df_obs_tide
    

    def generate_obs_tide_kyststasjoner_df_past(self, variables):

        # Creating a DataFrame that starts window_width_past number of hours 
        # before. Then, change the index so that the hours coincide with 
        # datetime_start_hourly and datetime_end_hourly, and concatenate it with 
        # the DataFrame created for the future. t0 should only be included in 
        # the DataFrame for the future, not for the past. 

        print(
            'Generating obs and tide_py DataFrame form operational data for the'
            + ' past...'
            )
        data_dir = '/lustre/storeB/project/fou/hi/stormsurge_eps/2dEPS_archive'  

        # We need to generate a DataFrame starting at 00 or 12 hours before 
        # self.window_width_past. Then, we can subset it to have the same number 
        # of hours as the period between self.datetime_start_hourly and 
        # self.datetime_end_hourly, but starting at 
        # (self.datetime_start_hourly - self.window_width_past)
        if np.mod(self.window_width_past, 12) == 0:
            n_hours_to_prev_12 = 0
        else:
            n_hours_to_prev_12 = int(12 - np.mod(self.window_width_past, 12))
        dt_start = (
            self.datetime_start_hourly 
            - dt.timedelta(hours=int(self.window_width_past + n_hours_to_prev_12))
        )

        dt_end = (
            self.datetime_end_hourly 
        )
        # ------ Open data in operational files ------  

        # Just to get the right times
        rk = ReadKyststasjonerHourly(
            dt_start=self.datetime_start_hourly, 
            dt_end=self.datetime_end_hourly,
            data_dir=data_dir
            ) 
        times = rk.make_time_idx()

        # To open files #window_width_past hours before t0 (lag the variables)
        rk = ReadKyststasjonerHourly(
            dt_start=dt_start, 
            dt_end=dt_end,
            data_dir=data_dir
            ) 
        
        path_list = rk.make_path_list()
        
        first_iter = True
        for station in self.oper_stations:
            if 'obs' in variables:
                # Load observations and store in df
                data_array = rk.make_data_arrays(
                    'observed_at_chartdatum', 
                    station, 
                    path_list, 
                    self.window_width_past
                    )

                data_array = (
                    data_array[n_hours_to_prev_12:-self.window_width_past]
                )
                col_names = rk.make_hz_col_names_past(
                    'observed_at_chartdatum', 
                    station, 
                    self.window_width_past
                    )
                df = rk.array_to_df(
                    data_array, 
                    times, 
                    col_names
                    )
                if first_iter:
                    oper_df  = df
                else: 
                    oper_df =  pd.concat([oper_df, df], axis=1)
                first_iter = False
            if 'tide_py' in variables:
                # Load tide data and store in df
                data_array = rk.make_data_arrays(
                    'tide_at_chartdatum', 
                    station, 
                    path_list, 
                    self.window_width_past
                    )
                data_array = (
                    data_array[n_hours_to_prev_12:-self.window_width_past]
                )
                col_names = rk.make_hz_col_names_past(
                    'tide_at_chartdatum', 
                    station, 
                    self.window_width_past
                    )
                df = rk.array_to_df(
                    data_array, 
                    times, 
                    col_names
                    )
                if first_iter:
                    oper_df  = df
                else: 
                    oper_df =  pd.concat(
                        [oper_df, df], 
                        axis=1
                        )
                first_iter = False                      
            
        return oper_df


    def generate_obs_tide_kyststasjoner_df_future(self, variables):

        print(
            'Generating obs and tide_py DataFrame form operational data for the'
            + ' future...'
            )
        data_dir = '/lustre/storeB/project/fou/hi/stormsurge_eps/2dEPS_archive'      

        # ------ Open data in operational files ------   
        rk = ReadKyststasjonerHourly(
            dt_start=self.datetime_start_hourly, 
            dt_end=self.datetime_end_hourly,
            data_dir=data_dir
            ) 
        
        path_list = rk.make_path_list()
        times = rk.make_time_idx()
        
        first_iter = True
        for station in self.oper_stations:
            if 'obs' in variables:
                # Load observations and store in df
                data_array = rk.make_data_arrays(
                    'observed_at_chartdatum', 
                    station, 
                    path_list, 
                    self.horizon
                    )
                col_names = rk.make_hz_col_names(
                    'observed_at_chartdatum', 
                    station, 
                    self.horizon
                    )
                df = rk.array_to_df(
                    data_array, 
                    times, 
                    col_names
                    )
                if first_iter:
                    oper_df  = df
                else: 
                    oper_df =  pd.concat([oper_df, df], axis=1)
                first_iter = False
            if 'tide_py' in variables:
                # Load tide data and store in df
                data_array = rk.make_data_arrays(
                    'tide_at_chartdatum', 
                    station, 
                    path_list, 
                    self.horizon
                    )
                col_names = rk.make_hz_col_names(
                    'tide_at_chartdatum', 
                    station, 
                    self.horizon
                    )
                df = rk.array_to_df(
                    data_array, 
                    times, 
                    col_names
                    )
                if first_iter:
                    oper_df  = df
                else: 
                    oper_df =  pd.concat(
                        [oper_df, df], 
                        axis=1)

                first_iter = False                      
            
        return oper_df
    

    def generate_obs_tide_kyststasjoner_df_old(self, variables):
        # Deprecated
        # This is hourly data.
        print('Generating obs and tide_py DataFrame form operational data...')
        data_dir = '/lustre/storeB/project/fou/hi/stormsurge_eps/2dEPS_archive'      

        # ------ Open data in operational files ------   
        rk = ReadKyststasjonerHourly(
            dt_start=self.datetime_start_hourly, 
            dt_end=self.datetime_end_hourly,
            data_dir=data_dir
            ) 
        
        path_list = rk.make_path_list()
        times = rk.make_time_idx()
        
        first_iter = True
        for station in self.oper_stations:
            if 'obs' in variables:
                # Load observations and store in df
                data_array = rk.make_data_arrays(
                    'observed_at_chartdatum', 
                    station, 
                    path_list, 
                    1
                    )
                col_names = rk.make_hz_col_names(
                    'observed_at_chartdatum', 
                    station, 
                    1
                    )
                df = rk.array_to_df(
                    data_array, 
                    times, 
                    col_names
                    )
                if first_iter:
                    oper_df  = df
                else: 
                    oper_df =  pd.concat([oper_df, df], axis=1)
                first_iter = False
            if 'tide_py' in variables:
                # Load tide data and store in df
                data_array = rk.make_data_arrays(
                    'tide_at_chartdatum', 
                    station, 
                    path_list, 
                    1
                    )
                col_names = rk.make_hz_col_names(
                    'tide_at_chartdatum', 
                    station, 
                    1
                    )
                df = rk.array_to_df(
                    data_array, 
                    times, 
                    col_names
                    )
                if first_iter:
                    oper_df  = df
                else: 
                    oper_df =  pd.concat([oper_df, df], axis=1)
                first_iter = False                      
            
        return oper_df
    
   
    def generate_station_forcing_df(self, variables):
        """DataFrame with MEPS data at each station.
        
        Generate a DataFrame containing MEPS data for all stations for the
        weather forecast variables requested. The methods omits the variables 
        that are not provided in the original data files.

        Parameters
        ----------
        variables: list of str
            Station forcing variables to be included in the DataFrame.
            
        Returns
        -------
        df_station_forcing : DataFrame
            DataFrame containing weather forecasts for all the stations and lead 
            times.
        """
        
        print('Generating station forcing df...')
        
        # # Data directory where some pre-generated dfs are stored in dicts

        # TODO: Re-run and update paths

        # data_dir = (
        #     '/lustre/storeB/project/IT/geout/machine-ocean/workspace'
        #     + '/paulinast/operational/data/preprocessed_input'
        # )
        # # In forecast mode, we use AROME data 
        # #file_name = 'dict_df_5_stations_only_arome_all_months_12_hours_new_no_forecast_generated_in_the_past.pickle'
        # #file_name = 'dict_df_5_stations_only_arome_and_waves_all_months_12_hours_new_no_forecast_generated_in_the_past.pickle'
        
        # file_name = (
        #     '/dict_df_all_stations_only_arome_all_months_12_hours_'
        #     + 'new_no_forecast_generated_in_the_past.pickle'
        #     )


        # Because the old MEPS archive has been moved to storeA, we have not 
        # pre-generated a new file for this version of the code. We are using 
        # the same version as in the paper.
        data_dir = (
            '/lustre/storeB/project/IT/geout/machine-ocean/workspace'
            + '/paulinast/storm_surge_results/data_preprocess_input/5_stations' 
            + '/monthly/'
        )
        
        file_name =  (
            'dict_df_all_stations_only_arome_all_months_12_hours_'
            + 'new_no_forecast_generated_in_the_past.pickle'
            )

        if  (self.use_existing_files  
             and Path(data_dir + '/' + file_name).is_file()):
            print('Using already generated AROME file.')
            # Open file that has already been generated as it takes a long time
            with open(data_dir + '/' + file_name, 'rb') as handle:
                station_file = pickle.load(handle)
            station_file = (
                # Get hourly data from dict
                station_file['df_hourly']
                .loc[self.datetime_start_hourly: self.datetime_end_hourly]
                )
            
            # Generate column names for the data we need
            col_names = []
            for station in self.stations:
                for var in variables:
                    for h in range(self.horizon):
                        if h == 0 :
                            col_name = var + '_' + station
                        else:
                            col_name = var + '_t' + str(h) + '_' + station
                        col_names.append(col_name) 
            # Intersect the column names of the data requested with the columns
            # in the original df            
            col_names = list(set(col_names) & set(list(station_file.columns))) 
            # Subset df        
            df_station_forcing = station_file[col_names]
        else:
            # Generate files 
            # I use to run this for shorter periods and concatenate the files
            # See concatenate_monthly_dfs.py
            df_station_forcing = self.generate_arome_df(variables)
            print('generate_arome_df should be updated with new paths')
        return df_station_forcing

    
    def generate_arome_df(self, variables):
        # Generate a DataFrame with weather forecasts.
        print('Generate AROME df...')

        # Define the actual times of the final DataFrame
        ra_small = ReadArome(
            self.datetime_start_hourly, 
            self.datetime_end_hourly
            )   
        times_small = ra_small.make_time_idx()    
        
        # Use the following piece of code if you do not need past forecasts:
        path_list = ra_small.make_path_list_ppi()
        first_iter = True
        for station in self.oper_stations:
            for var in variables:
                # Generate big data array that includes past forecasts
                small_data_array = ra_small.make_data_arrays(
                    [var], 
                    [station], 
                    path_list, 
                    self.horizon
                    )

                col_names = ra_small.make_hz_col_names(
                    [var], 
                    [station], 
                    self.horizon, 
                    fhr=0
                    )
                # Store data in df
                df  = ra_small.array_to_df(
                    small_data_array, 
                    times_small, 
                    col_names
                    )
                if first_iter:
                    df_arome = df
                    first_iter = False
                else:
                    df_arome = pd.concat([df_arome, df], axis=1)
        return df_arome
    
    def generate_operational_df(self, variables):
        """Generate a DataFrame with storm surge data.
        
        Generate a DataFrame with storm surge data at each location and for all 
        lead times.
        
        Parameters
        ----------            
        variables : list of str
            list with variables to check. If 'stormsurge' or 
            'stormsurge_corrected' are in the list, add the column to the 
            DataFrame.

        Returns
        -------
        df : DataFrame
            DataFrame with observations and/or tides.
        """

        # TODO: Merge with generation of obs and tide data
        
        # Data directory where some pre-generated dfs are stored in dicts
        data_dir = (
            '/lustre/storeB/project/IT/geout/machine-ocean/workspace'
            + '/paulinast/operational/data/preprocessed_input'
            )
        
        file_name = (
            'dict_df_all_stations_stormsurge_corrected_kyststasjoner'
            + '_past_forecasts_24_12_hours_v2.pickle'
            )
        
        if  (self.use_existing_files  
             and Path(data_dir + '/' + file_name).is_file()): 
            print('Using already generated kyststasjoner file.')
            # Open file that has already been generated as it runs faster
            with open(data_dir + '/' + file_name, 'rb') as handle:
                station_file = pickle.load(handle)
            station_file = (
                station_file['df_12_hours']
                .loc[self.datetime_start_hourly: self.datetime_end_hourly]
                )
            
            col_names = []                            
            for station in self.stations:
                    for var in variables:
                        for fhr in range(-self.past_forecast, 1, 12):
                            if fhr == 0:
                                past_forecast_str = ''
                            else:
                                past_forecast_str = '_' + str(fhr)
                            for h in range(fhr, self.horizon): 
                                if h == 0 :
                                    h_str = ''
                                else:
                                    h_str = '_t' + str(h)
                                col_name  = (
                                    var 
                                    + h_str 
                                    + past_forecast_str 
                                    + '_' + station
                                )
                                col_names.append(col_name) 

            col_names = list(set(col_names) & set(list(station_file.columns)))         
            self.df_operational = station_file[col_names]
        else:
            # Generate files 
            # I use to run this for shorter periods and concatenate the files
            self.df_operational = self.generate_kyststasjoner_df(variables)

        return self.df_operational
    
        
    def generate_kyststasjoner_df(self, variables):
        """Generate a DataFrame with storm surge data.
        
        Generate a DataFrame with storm surge data at each location for all the 
        lead times. The methods skips the variables that are not provided in the 
        original data file.
        
        Parameters
        ----------
        variables : list of str
            Operational variables to include in the DataFrame, these can be 
            'stormsurge' or 'stormsurge_corrected'.
            
        Returns
        -------
        df_crrt : DataFrame
            DataFrame with storm surge data.
        """ 
        print('Generating operational DataFrame...')
        
        # Open file generated with read_kyststasjoner.py
        data_dir = '/lustre/storeB/project/fou/hi/stormsurge_eps/2dEPS_archive'      
        first_iter = True
        m = -1 # or ''?

        # Define the actual times of the final DataFrame
        rk_small = ReadKyststasjoner(
            dt_start=self.datetime_start_hourly, 
            dt_end=self.datetime_end_hourly,
            data_dir=data_dir
            ) 
        times_small = rk_small.make_time_idx()

        # Define start and end of big DataFrame that contains past forecasts to
        # subset
        start_hour = (
            self.datetime_start_hourly 
            - dt.timedelta(hours=self.past_forecast)
        )
        end_hour = self.datetime_end_hourly
        #if self.datetime_end_hourly.hour == 0:
        #    end_hour = self.datetime_end_hourly + dt.timedelta(hours=12)

        rk = ReadKyststasjoner(
            dt_start=start_hour, 
            dt_end=end_hour,
            data_dir=data_dir
            ) 
        
        path_list = rk.make_path_list()

        for station in self.oper_stations:
            for var in variables:
                if var in ['stormsurge', 'stormsurge_corrected']:
                    # Generate big data array that includes past forecasts
                    # This way we extract the data from the kyststasjoner 
                    # files only once, and speed up the process
                    big_data_array = rk.make_data_arrays(
                        var, 
                        station, 
                        path_list, 
                        self.horizon + self.past_forecast
                        )
                    count=0
                    for fhr in range(-self.past_forecast, 12, 12):
                        if fhr == 0:
                            small_data_array = (
                            big_data_array[count: , :self.horizon - fhr]
                            )
                        else:
                            end_row = int(-self.past_forecast/12 + count)
                            small_data_array = (
                                big_data_array[count:end_row , :self.horizon - fhr]
                                )
                        
                        col_names = rk.make_hz_col_names(
                            var, 
                            station, 
                            self.horizon, 
                            fhr=fhr, 
                            )
                        df  = rk.array_to_df(
                            small_data_array, 
                            times_small, 
                            col_names
                            )
                        if first_iter:
                            df_crrt = df
                            first_iter = False
                        else:
                            df_crrt = pd.concat([df_crrt, df], axis=1)
                        count = count + 1
        if 'df_crrt' in locals():         
            return df_crrt
        
        
def main_arome(args):
    """ 
    Run this once to pregenerate MEPS file to be used later.
    """
    datetime_start_hourly, datetime_end_hourly = args
    datetime_start_hourly = dt.datetime.strptime(
        datetime_start_hourly, 
        '%Y%m%d'
        )
    datetime_end_hourly = dt.datetime.strptime(
        datetime_end_hourly, 
        '%Y%m%d'
        )
    datetime_start_hourly = datetime_start_hourly.replace(
        tzinfo=dt.timezone.utc
        )
    
    datetime_end_hourly = datetime_end_hourly.replace(
        tzinfo=dt.timezone.utc
        )
    
    prep = PrepareDataFrames(
    variables = ['msl', 'u10', 'v10'],
    stations = hlp.get_norwegian_station_names(), 
    horizon=60+1,
    window_width_past=0,
    datetime_start_hourly = datetime_start_hourly,
    datetime_end_hourly = datetime_end_hourly,
    use_existing_files=False,
    past_forecast=0
    )        

    df = prep.prepare_features_labels_df()

    my_dict = { 
        'df_12_hours' : df, 
        }

    path_to_dict = (
        '/lustre/storeB/project/IT/geout/machine-ocean/workspace'
        + '/paulinast/operational/data/preprocessed_input'
        )
    
    with open(
            path_to_dict 
            + '/dict_df_all_stations_only_arome_12_hours' 
            + datetime_start_hourly.strftime('%Y%m%d') 
            + '.pickle', 
            'wb'
            ) as handle:
        pickle.dump(my_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        
def main_obs_tide(args):
    """
    Run this once to pregenerate file with observations to be used later.
    """
    datetime_start_hourly, datetime_end_hourly = args
    datetime_start_hourly = dt.datetime.strptime(
        datetime_start_hourly, 
        '%Y%m%d'
        )
    datetime_end_hourly = dt.datetime.strptime(
        datetime_end_hourly, 
        '%Y%m%d'
        )
    datetime_start_hourly = datetime_start_hourly.replace(
        tzinfo=dt.timezone.utc
        )
    datetime_end_hourly = datetime_end_hourly.replace(
        tzinfo=dt.timezone.utc
        )
    
    #print('datetime_start_hourly: ', datetime_start_hourly)
    
    prep = PrepareDataFrames(
        variables = ['obs', 'tide_py'],
        stations = hlp.get_norwegian_station_names(),
        horizon=60 + 1,
        window_width_past=24,
        datetime_start_hourly = datetime_start_hourly,
        datetime_end_hourly = datetime_end_hourly,
        use_existing_files=False,
        past_forecast=0
        )        

    df = prep.prepare_features_labels_df()

    my_dict = { 
        'df_12_hours' : df, 
        }
    
    path_to_dict = (
        '/lustre/storeB/project/IT/geout/machine-ocean/workspace'
        + '/paulinast/operational/data/preprocessed_input'
        )
    
    with open(
            path_to_dict 
            + '/dict_df_all_stations_obs_tide_kyststasjoner_12_hours'
            + datetime_start_hourly.strftime('%Y%m%d') 
            + '.pickle', 
            'wb'
            ) as handle:
        pickle.dump(my_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

def main_stormsurge_corrected(args):
    """
    Run this once to pregenerate file with operational storm surge data to be 
    used later.
    """
    datetime_start_hourly, datetime_end_hourly = args
    datetime_start_hourly = dt.datetime.strptime(
        datetime_start_hourly, 
        '%Y%m%d'
        )
    datetime_end_hourly = dt.datetime.strptime(
        datetime_end_hourly, 
        '%Y%m%d'
        )
    datetime_start_hourly = datetime_start_hourly.replace(
        tzinfo=dt.timezone.utc
        )
    datetime_end_hourly = datetime_end_hourly.replace(
        tzinfo=dt.timezone.utc
        )
    
    #print('datetime_start_hourly: ', datetime_start_hourly)
    
    prep = PrepareDataFrames(
        variables = ['stormsurge_corrected'],
        stations = hlp.get_norwegian_station_names(),
        horizon = 60 + 1, #1,  # horizon in prepare_df is an int +1!!! Lags are computed in self.generate_feature_array
        window_width_past=0,
        use_existing_files=False,
        datetime_start_hourly = datetime_start_hourly,
        datetime_end_hourly = datetime_end_hourly,
        past_forecast=24
        )        

    df = prep.prepare_features_labels_df()

    my_dict = { 
        'df_12_hours' : df, 
        }
    
    path_to_dict = (
        '/lustre/storeB/project/IT/geout/machine-ocean/workspace'
        + '/paulinast/operational/data/preprocessed_input'
        )
    
    with open(
            path_to_dict 
            + '/dict_df_all_stations_stormsurge_corrected_kyststasjoner'
            + '_past_forecasts_24_12_hours_'
            + datetime_start_hourly.strftime('%Y%m%d') 
            + '.pickle', 
            'wb'
            ) as handle:
        pickle.dump(my_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    
if __name__ == "__main__":

    # **************************************************************************
    # Modify this part to run the experiments of interest
    # **************************************************************************
    
    # To run from concatenate_monthly_dfs.py: 
    #args = argparse.parse_arguments()
    #main(args)



    """
    label_stations=['NO_MAY']#['NO_BGO']
    label_var = 'stormsurge_corrected - (obs - tide_py)'

    # Short test
    feature_stations=['NO_MAY']#['NO_BGO', 'NO_AES']
    feature_vars = ['obs', 'tide_py', 'stormsurge_corrected', 'v10']
    #label_var = 'stormsurge_corrected'
    window_width_past = 4
    horizon = 6
    datetime_start_hourly=dt.datetime(
        2021, 1, 1, 0, 0, 0, 
        tzinfo=dt.timezone.utc
        )
    datetime_end_hourly=dt.datetime(
        2021, 1, 15, 12, 0, 0, 
        tzinfo=dt.timezone.utc
        )
    datetime_start_hourly=dt.datetime(
        2020, 7, 1, 0, 0, 0, 
        tzinfo=dt.timezone.utc
        )
    datetime_end_hourly=dt.datetime(
        2020, 7, 31, 12, 0, 0, 
        tzinfo=dt.timezone.utc
        )

    prep = PrepareDataFrames(
            variables=feature_vars + [label_var],
            stations=feature_stations + label_stations,
            horizon=horizon,  # max
            window_width_past=window_width_past, # use the same for obs and tide
            datetime_start_hourly=datetime_start_hourly,
            datetime_end_hourly=datetime_end_hourly,
            use_existing_files=False,
            past_forecast=0
            )
    
    prep.prepare_features_labels_df()
    """

    feature_vars = ['obs', 'tide_py', 'stormsurge_corrected']#, 'u10', 'v10'] 
    label_var = 'stormsurge_corrected - (obs - tide_py)'

    window_width_past = [24, 24, 0]#, 0, 0]
    window_width_future = [-1, 60, 60]#, 60, 60]  # -1 does not include time t
    horizon = 60  

    feature_stations = [
            'NO_OSC', 
            'NO_AES', 
            'NO_BGO', 
            'NO_VIK', 
            'NO_TRG'
            ]

    station = 'NO_OSC'

    datetime_start_hourly = datetime_split=dt.datetime(2020, 3, 31, 12, 0, 0, tzinfo=dt.timezone.utc)

    prep = PrepareDataFrames(
        variables=feature_vars + [label_var],
        stations=feature_stations + [station],
        horizon=horizon,  # max
        window_width_past=24, # use the same for obs and tide
        datetime_start_hourly=datetime_start_hourly,
        datetime_end_hourly=datetime_start_hourly,
        use_existing_files=False,
        past_forecast=24
        )
    
    prep.prepare_features_labels_df()
