"""Prepare DataFrame from ROMS data and observations.

Prepare a DataFrame that contains all the variables in feature_variables and
lable_variables at the specified locations and for the specified time range.
"""
import numpy as np
import pandas as pd
import datetime as dt
from pathlib import Path
import xarray as xr
import os
import copy
import pickle
import argparse


import ml_models.utils.helpers as hlp
import ml_models.data_loader.src_helper_load_observations_ROMS_stations as loader  # <- not working anymore on server
import ml_models.utils.assertions as asrt
from ml_models.data_loader.read_kyststasjoner import ReadKyststasjoner
from ml_models.data_loader.read_kyststasjoner_hourly import ReadKyststasjoner as ReadKyststasjonerHourly
from ml_models.data_loader.read_arome import ReadArome
import time
import pytz

os.environ["TZ"] = "UTC"
time.tzset()

class PrepareDataFrames():
    """Prepare DataFrame from ROMS data and observations.
    
    Generates feature and label arrays from station data and ROMS data.
    Selects common period without missing observations.  
    
    Attributes
    ----------   
    stations : list of strings
        List of stations, example ['NO_OSC', 'SW_2111', 'NL_Westkapelle']
        
    datetime_start_hourly : datetime
        Date indicating when the period of interest begins. The final period 
        depends selected depends on data availability. All dates without 
        observations will be removed. Default is 2002.01.01 00:00.
        
    datetime_start_hourly : datetime
        Date indicating when the period of interest begins. The final period 
        depends selected depends on data availability. All dates without 
        observations will be removed. Default is 2019.12.01 00:00.

    use_station_data : bool, optional
        True if station data is used to generate feature or label arays. 
        Otherwise, mesh data is used. The default is True.
        
    run_on_ppi : bool, optional
        Determines the paths to use. True if the code will run on PPI. The
        default is True.
        
    new_station_loc : bool, optional
        If true, selects grid point for the hindcast that do not necessary 
        match the station's location, but avoid resonance effect. This usually
        happens in the fjords. The default is false.
    
    era5_forcing : bool, optional
        If true, open the hindcast file forced with era5. The default is false.
        
    add_aux_variables : bool, optional
        If true, add all the auxiliary variables to the dataframe. The 
        auxiliary variables are the ones from which some of the variables 
        derive. For instance, the variables needed to compute '(obs - tide_py)' 
        are 'obs' and 'tide_py'.
        
        
    List of possible stations:
        stations = [
                'NO_OSC', 
                'SW_2111', 
                  #'DK_6501', 
                  #'DK_4201',
                  #'DK_5203',
                'NL_Westkapelle', 
                  #'UK_Lowestoft', 
                  #'UK_Newhaven',
                  #'UK_Wick'
                  #'UK_Aberdeen'
                  #'DE_Cuxhaven'
                  ]
     
    """
    
    def __init__(
            self,
            variables,
            stations,
            horizon:int,  # max
            datetime_start_hourly=dt.datetime(2002, 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc),
            datetime_end_hourly=dt.datetime(2019, 12, 1, 0, 0, 0, tzinfo=dt.timezone.utc),
            ensemble_members = [-1],
            use_station_data:bool=True,
            run_on_ppi:bool=True,
            new_station_loc=False,
            era5_forcing=False,
            add_aux_variables=False,
            data_from_kartverket=False,
            forecast_mode=False,
            hourly=True,  # Otherwise make 12-hourly data at 00 and 12 hours.
            use_existing_files=True,
            past_forecast=0,
            filter_times_in_hourly_df=True
            ):

        self.variables = variables
        self.stations = stations
        self.datetime_start_hourly = datetime_start_hourly
        self.datetime_end_hourly = datetime_end_hourly
        self.ensemble_members = ensemble_members
        self.horizon = horizon
        self.use_station_data = use_station_data
        self.run_on_ppi = run_on_ppi
        self.new_station_loc = new_station_loc
        self.era5_forcing = era5_forcing
        self.add_aux_variables = add_aux_variables 
        self.data_from_kartverket = data_from_kartverket
        self.forecast_mode = forecast_mode
        self.hourly = hourly
        self.use_existing_files = use_existing_files
        self.past_forecast = past_forecast
        self.filter_times_in_hourly_df = filter_times_in_hourly_df
        
        asrt.assert_dt_start_less_than_dt_end(
            self.datetime_start_hourly, 
            self.datetime_end_hourly
            )
        
        #asrt.assert_variables_exist(self.variables)

        # Group variables that are retrieved form the same dataset
        self.obs_tide_vars = ['obs', 'tide_py', '(obs - tide_py)']  
        self.hindcast_vars = ['roms']  
        self.combined_roms_obs_vars = ['roms - (obs - tide_py)']
        self.combined_stormsurge_obs_var = ['stormsurge - (obs - tide_py)', 'stormsurge_corrected - (obs - tide_py)']
        self.station_forcing_vars = ['msl', 'u10', 'v10', 'swh', 'mwd'] 
        self.mesh_forcing_vars = ['pz0', 'x_wind_10m', 'y_wind_10m', 'wdir', 'swh', 'tp'] 
        self.arome_vars = ['msl_arome', 'u10_arome', 'v10_arome']
        self.operational_vars = ['stormsurge', 'stormsurge_corrected', 'bias']
        self.corrected_vars = ['(roms - biasMET)', '(roms - biasMET) - (obs - tide_py)']
        self.wind_vars = ['wind_speed', 'wind_dir']
        self.wind_arome_vars = ['wind_speed_arome', 'wind_dir_arome']

        self.create_aux_variables_list()
        
        self.operational_stations()
        
    def create_aux_variables_list(self):
        """
        Add auxiliary variables for the computation of derived variables.

        Returns
        -------
        None.

        """
        # 
        self.aux_variables = []

        if '(obs - tide_py)' in self.variables:
            self.aux_variables.append('obs')
            self.aux_variables.append('tide_py')
        if 'roms - (obs - tide_py)' in self.variables:
            self.aux_variables.append('obs')
            self.aux_variables.append('tide_py')
            self.aux_variables.append('roms')
            self.aux_variables.append('(obs - tide_py)')
        if 'wind_speed' in self.variables or 'wind_dir' in self.variables:
            #if self.forecast_mode:
            #    self.aux_variables.append('u10_arome')
            #    self.aux_variables.append('v10_arome')
            #else:
            self.aux_variables.append('u10')
            self.aux_variables.append('v10')
        if 'wind_speed_mesh' in self.variables or 'wind_dir_mesh' in self.variables:
            self.aux_variables.append('x_wind_10m')
            self.aux_variables.append('y_wind_10m')
        if 'stormsurge_corrected - (obs - tide_py)' in self.variables:
            self.aux_variables.append('obs')
            self.aux_variables.append('tide_py')
            self.aux_variables.append('stormsurge_corrected')
            self.aux_variables.append('(obs - tide_py)')
        if 'stormsurge - (obs - tide_py)' in self.variables:
            self.aux_variables.append('obs')
            self.aux_variables.append('tide_py')
            self.aux_variables.append('stormsurge')
            self.aux_variables.append('(obs - tide_py)')            
        if '(roms - biasMET) - (obs - tide_py)' in self.variables:
            self.aux_variables.append('obs')
            self.aux_variables.append('tide_py')
            self.aux_variables.append('roms')
            self.aux_variables.append('(obs - tide_py)')
            self.aux_variables.append('(roms - biasMET)')
        if ('wind_speed' or 'wind_dir') in self.variables:
            if self.station_forcing_vars:
                self.aux_variables.append('u10')
                self.aux_variables.append('v10')
            else:
                self.aux_variables.append('x_wind_10m')
                self.aux_variables.append('y_wind_10m')
        
            
        if '(obs - tide_py)' in self.aux_variables:
            self.aux_variables.append('obs')
            self.aux_variables.append('tide_py')

        # TODO: Test that stormsurge corrected is only for norWegian stations
        
    def operational_stations(self):
        norwegian_stations = hlp.get_norwegian_station_names()
        
        oper_stations = list(set(self.stations) & set(norwegian_stations))
        try:
            oper_stations.remove('NO_MSU')
        except:
            pass

        self.oper_stations = oper_stations
        
    def prepare_features_labels_df(self):
        """Prepare the final dataframe.
        
        Prepare the DataFrame for the specified variables and stations in the 
        class arguments.

        Returns
        -------
        None.

        """
        self.df_aux = self.prepare_df(aux=True)
        self.df_features_and_labels = self.prepare_df(aux=False)
# =============================================================================
#         asrt.assert_columns_in_prepared_dataframe(
#             self.df_features_and_labels, 
#             self.stations,
#             self.variables
#             )
# =============================================================================
        
        if self.add_aux_variables:
            common_columns = np.intersect1d(
                self.df_features_and_labels.columns, 
                self.df_aux.columns
                )
            
            # TODO: Fix - not working!! #transforms datetime index to int
            if len(common_columns) > 0:
                # Concatenate both dataframes on common columns
                self.df_features_and_labels = pd.merge(
                    self.df_features_and_labels,
                    self.df_aux,
                    #on=common_columns,
                    how='outer'
                    )
            else:
                # Concatenate             
                self.df_features_and_labels = pd.concat(
                    [
                        self.df_features_and_labels, 
                        self.df_aux, 
                        ],
                    axis=1
                    )
    
        return self.df_features_and_labels  # TODO: call this method from init and dont return anything

        
    def prepare_df(self, aux=False):
        """Generate a DataFrame for a set of variables.
           
        Generate a DataFrame either with the variables provided as class 
        arguments or the auxiliary variables neede to compute the former. 
        This is a wrapperfunction that needs to be called in order to generate 
        the final DataFrame. 
        
        The method generates first separated DataFrames for each group of 
        variables. These DataFrames are then concatenate to provide a single
        DataFrame.
        
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
        
        print('list_all_variables: ', list_all_variables)
        # Create empty list to fill in with DataFrames
        dfs = []
        
        # ------ a) Observations and tide ------
        df_obs_tide_vars= list(
            set.intersection(
                set(self.obs_tide_vars), 
                set(list_all_variables)
                )
            )
        
        if df_obs_tide_vars:
            if self.data_from_kartverket:
                dfs.append(self.generate_obs_tide_df_kartverket(df_obs_tide_vars)) 
                print('Added obs and tide data from Kartverket: ')
                print(dfs[-1])
            else:
                dfs.append(self.generate_obs_tide_df(df_obs_tide_vars)) 
                print('Added obs and tide data:' )
                print(dfs[-1])
            
            df_obs_tide = self.add_obs_tide(variables=list_all_variables)
            if not df_obs_tide.empty:
                dfs.append(df_obs_tide)
                print('Added (obs-tide) data')
                print(dfs[-1])
            
            print('dfs: ', dfs)

            # TODO: Check that if obs - tide_py in aux variables, then obs and tide_py also is.
            
        # ------ b) Hindcast data (ROMS) ------
        df_hindcast_vars= list(
            set.intersection(
                set(self.hindcast_vars), 
                set(list_all_variables)
                )
            )
        
        if df_hindcast_vars:
            dfs.append(self.generate_hindcast_df())  
            print('Added hindcast data: ')
            print(dfs[-1])
        
        # ------ c) Forcing data at station locations ------
        if self.use_station_data:
            df_station_forcing_vars= list(
                set.intersection(
                    set(self.station_forcing_vars), 
                    set(list_all_variables)
                    )
                )
            
            if df_station_forcing_vars:
                #if self.forecast_mode:
                #    df_station_forcing_vars = list(
                #        set(df_station_forcing_vars) 
                #        & set(['msl', 'u10', 'v10'])
                #        )
                #    print('Adding AROME data...')
                #    dfs.append(
                #        self.generate_arome_df(
                #            variables=df_station_forcing_vars
                #            )
                #        )
                #else:
                print('Added meteorological data...')
                dfs.append(
                    self.generate_station_forcing_df(
                        variables=df_station_forcing_vars
                        )
                    )
                print('Added station forcing data: ')
                print(dfs[-1])
        
        # ------ d) Forcing data in a mesh ------
        else:
            df_mesh_forcing_vars = list(
                set.intersection(
                    set(self.mesh_forcing_vars), 
                    set(list_all_variables)
                    )
                )
            
            if df_mesh_forcing_vars:
                dfs.append(
                    self.generate_mesh_forcing_df(
                        variables=df_mesh_forcing_vars
                        )
                    )
                print('Added mesh forcing data: ')
                print(dfs[-1])
                
        # ------ f) Operational data (ROMS) ------      
        df_operational_vars = list(
            set.intersection(
                set(self.operational_vars), 
                set(list_all_variables)
                )
            )
        
        if df_operational_vars:
            print('df_operational_vars: ', df_operational_vars)
            df = self.generate_operational_df(variables=df_operational_vars)
            print('df: ', df)
            dfs.append(df)
            print('Added operational data: ')
            print(dfs[-1])
            
            print('self.df_operational:', self.df_operational)
        
        
        # ------ g) Combined hindcast (ROMS) and obs ------
        df_combined_roms_obs_vars = list(
                set.intersection(
                    set(self.combined_roms_obs_vars), 
                    set(list_all_variables)
                    )
                )
        
        if df_combined_roms_obs_vars:
            df_roms_obs_tide = self.add_roms_obs_tide(variables=list_all_variables)
            if not df_roms_obs_tide.empty:
                dfs.append(df_roms_obs_tide)
                print('Added roms - (obs - tide) data: ')
                print(dfs[-1])

            
        # ------ g) Combined operational (ROMS) and obs ------
        df_combined_stormsurge_obs_vars = list(
                set.intersection(
                    set(self.combined_stormsurge_obs_var), 
                    set(list_all_variables)
                    )
                )
        
        if df_combined_stormsurge_obs_vars:
            df_stormsurge_obs_tide = self.add_roms_obs_tide(variables=list_all_variables)
            if not df_stormsurge_obs_tide.empty:
                dfs.append(df_stormsurge_obs_tide)
                print('Added stormsurge - (obs-tide) data: ')
                print(dfs[-1])
        
        # ------ h) Corrected hindcast (ROMS) with bias_MET ------
        df_corrected_vars = list(
                set.intersection(
                    set(self.corrected_vars), 
                    set(list_all_variables)
                    )
                )
        
        if df_corrected_vars:
            dfs.append(self.correction(variables=list_all_variables))
            print('Added (hindcast - biasMET) data: ')
            print(dfs[-1])
        
        
        # ------ i) Wind speed and wind direction ------
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

        
        # TODO: add (roms - (obs - tide))
        

        print('Print each element in dfs list: ')
        i=0
        for d in dfs:
            print(i)
            print(d)
            i = i+1
        
        if len(dfs) > 0:
            if dfs[0] is None:
                print("dfs[0] is None.")
                dfs.pop(0)
            
        if len(dfs) > 1: 
            df = dfs[0]
            dfs.pop(0)
            print('df: ', df)
            while len(dfs) > 0:
                try:
                    df = df.join(dfs[0])
                except:
                    print('Could not join DataFrames')
                    print('df: ', df)
                dfs.pop(0)
                    
            #df = dfs[0].join(dfs[1:])
        elif len(dfs) ==1:
            df = dfs[0]
        else:
            df = pd.DataFrame()
        
        
# =============================================================================
#         asrt.assert_columns_in_prepared_dataframe(
#             df, 
#             self.stations, 
#             list_all_variables
#             )
#         
# =============================================================================

        print('df: ', df)
        print('df.index: ', df.index)
        #print('df.index[0]: ', df.index[0])
        if not self.hourly:
            if not df.empty:
                if not aux:
                    if self.filter_times_in_hourly_df:
                        self.df_hourly_t0 = df.drop(df.filter(like='_t',axis=1).columns,axis=1)  
                    else:
                        self.df_hourly_t0 = df
                m1 = (pd.to_datetime(df.index).hour == 0)
                m2 = (pd.to_datetime(df.index).hour == 12)
                df = df[m1 | m2]
            
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
        if self.use_station_data:
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
        """Compute the difference between the observations and tides.
        
        For each station, compute the difference between the observations the 
        tide effect computed with the pytide package. Then add the column 
        '(obs - tide_py)' to the DataFrame.
        
        Parameters
        ----------            
        variables : list of str
            list with variables to check. If ('obs - tide_py') is in the list, 
            add the column to the DataFrame.

        Returns
        -------
        df : DataFrame
            DataFrame with the difference between the ROMS hindcast and the 
            observations.

        """
        df = pd.DataFrame()
        print('self.df_obs_tide in add_os_tide: ', self.df_obs_tide)
        if '(obs - tide_py)' in variables:
            print('Adding (obs - tide_py) columns...')
            for station in self.stations:
                #try:
                    for h in range(self.horizon):
                        if h==0:
                            df['(obs - tide_py)_' + station] = (
                                self.df_obs_tide['obs_' + station]
                                - self.df_obs_tide['tide_py_' + station]
                                )
                        else:
                           df['(obs - tide_py)_t' + str(h) + '_' + station] = (
                                self.df_obs_tide['obs_t' + str(h) + '_'  + station]
                                - self.df_obs_tide['tide_py_t' + str(h) + '_' + station]
                                ) 
                #except:
                #    print('Could not add obs - tide_py for station: ', station, ' .Data not available.')
                #    self.stations.remove(station)
                #    print('Removed station ', station, ' from self.stations.')
 
        return df
    
    def add_roms_obs_tide(self, variables):
        
        # TODO: Add without lagging. Use the true horizon!!!
        
        """Compute the difference between the ROMS model and observations.
        
        For each station, compute the difference between ROMS hindcast and the
        observations compensating by the tide effect computed with the pytide
        package. Then add the columns 'roms - (obs - tide_py)' to the DataFrame.
        
        Parameters
        ----------            
        variables : list of str
            list with variables to check. If ('obs - tide_py') is in the list, 
            add the column to the DataFrame.

        Returns
        -------
        df : DataFrame
            DataFrame with the difference between the ROMS hindcast and the 
            observations.

        """
        df = pd.DataFrame()
        roms_var = ''
        
        roms_obs_tide_vars = [
            'roms - (obs - tide_py)', 
            'stormsurge - (obs - tide_py)', 
            'stormsurge_corrected - (obs - tide_py)'
            ]
        
        vars_to_iterate_over = list(set(variables) & set(roms_obs_tide_vars))
        
        if vars_to_iterate_over:
            for var in vars_to_iterate_over:
                
                # --- hindcast ---
                if var == 'roms - (obs - tide_py)':
                    df_roms = self.df_roms
                    roms_var = 'roms'  
                    for station in self.stations:
                        col_name = roms_var + ' - (obs - tide_py)_' + station
                        df[col_name] = (
                            df_roms[roms_var + '_' + station]
                            - (
                                self.df_obs_tide['obs_' + station]
                                - self.df_obs_tide['tide_py_' + station]
                                )
                            )
                else:
                
                    # --- Operational ---
                    if var == 'stormsurge - (obs - tide_py)':
                        roms_var = 'stormsurge'
                        df_roms = self.df_operational
                    elif var == 'stormsurge_corrected - (obs - tide_py)':
                        roms_var = 'stormsurge_corrected'
                        df_roms = self.df_operational
                    
                    print('df_roms: ', df_roms)
                    for station in self.stations:
                        for m in self.ensemble_members:
                            if (m>=0) and (m<51):
                                m_str = '_m' + str(m)
                            elif m<-1:
                                m_str = '_m' + str(52 + m)
                            else: # m is the last one
                             m_str = ''
                            for h in range(self.horizon):
                                if h==0:
                                    col_name = roms_var + m_str + ' - (obs - tide_py)_' + station
                                    df[col_name] = (
                                        df_roms[roms_var + m_str + '_' + station]
                                        - (
                                            self.df_obs_tide['obs_' + station]
                                            - self.df_obs_tide['tide_py_' + station]
                                            )
                                        )
                                else:
                                    col_name = roms_var + ' - (obs - tide_py)_t' + str(h) + '_'  + station
                                    df[col_name] = (
                                        df_roms[roms_var + '_t' + str(h) + '_'  + station]
                                        - (
                                            self.df_obs_tide['obs_t' + str(h) + '_'  + station]
                                            - self.df_obs_tide['tide_py_t' + str(h) + '_'  + station]
                                            )
                                        )                   
        return df
    
  
    def get_list_mesh_points(self):
        """Get location of points in mesh.
               
        Return a sorted list of latitude and longitudes corresponding to the 
        points in the mesh file generated by Jean Rabault.

        Returns
        -------
        list_of_mesh_points : list
            List of all the points (latitude and longitude) in the mesh file.

        """
        # TODO: Write this in helpers
        # list of lats and lons
        # Taken from Jean's script
        list_of_mesh_lats = [52 + 0.5 * x for x in range(0, 18, 1)]
        list_of_mesh_lons = [x + 0.5 for x in range(-3, 13, 1)]
        
        list_of_mesh_points = []
        for crrt_lat in list_of_mesh_lats:
            for crrt_lon in list_of_mesh_lons:
                list_of_mesh_points.append((crrt_lat, crrt_lon))
        
        return list_of_mesh_points
    
    def subset_kartverket_data(self, dataset, stationid, data_timestamp_full, variables):
        """
        Subset dataset with data from Kartverket and make a DataFrame.

        Parameters
        ----------
        dataset : DataSet
            DataSet with data from Kartverket.
        stationid : str
            Station ID without land code (only Norwegian stations).
        data_timestamp_full: 

        Returns
        -------
        df : DataFrame
            DataFrame with tide data from Kartverket for a specific station.

        """
        # Subset by time
        tol=1e-6
        
        crrt_ind = np.where(dataset.stationid.values == stationid)[0][0]  # Number - position in array     
        
        # POSIX timestamp
        timestamp_start = dataset.timestamp_start[crrt_ind].data
        if timestamp_start < self.datetime_start_hourly.timestamp():
            timestamp_start = self.datetime_start_hourly.timestamp() 
            
        timestamp_end = dataset.timestamp_end[crrt_ind].data
        if timestamp_end > self.datetime_end_hourly.timestamp():
            timestamp_end = self.datetime_end_hourly.timestamp()

        first_index = np.argmax(data_timestamp_full >= timestamp_start - tol)
        last_index = np.argmax(data_timestamp_full >= timestamp_end - tol) + 1 
        
        time = [
            dt.datetime.fromtimestamp(crrt_timestamp, pytz.utc) 
            for crrt_timestamp 
            in (data_timestamp_full[first_index:last_index])
            ]
        
        tide_crrt = dataset.prediction.sel({
            'station' : crrt_ind, 
            'time' : slice(first_index, last_index)
            })
        tide_crrt = tide_crrt.where(tide_crrt != 1.0e37)
        obs_crrt = dataset.observation.sel({
            'station' : crrt_ind, 
            'time' : slice(first_index, last_index)
            })
        obs_crrt = obs_crrt.where(obs_crrt != 1.0e37)
        
        # Make df with hourly data in meters
        df = pd.DataFrame(
            {
                'tide_py_NO_' + stationid : tide_crrt[::6]/100,
                'obs_NO_' + stationid : obs_crrt[::6]/100
            },
            index = time[::6]
            ) # TODO: rename tide_py in all methods
        
        # Subset by variables
        obs_vars = []
        if 'obs' in variables:
            obs_vars.append('obs_NO_' + stationid)
        if 'tide_py' in variables:
            obs_vars.append('tide_py_NO_' + stationid)
            
        df = df[obs_vars]        
        return df
        
    
    def generate_obs_tide_df_kartverket(self, variables):
        print('Generating obs and tide_py DataFrame with data from Kartverket...')
        
        # ------ Get tide and observations from operational files ------
        oper_df = self.generate_obs_tide_df_operational(variables)
        print('oper_df: ', oper_df)
        
        data_dir = (
            '/lustre/storeB/project/IT/geout/machine-ocean'
            + '/prepared_datasets/storm_surge/kartkverket'
            )
        file_name = '/full_size_data_kartverket_storm_surge.nc4'
        
        kv_dataset = xr.open_dataset(data_dir + file_name)
        data_timestamp_full = kv_dataset.timestamps.values
        
        first_station = True
        print('self.oper_stations: ', self.oper_stations)
        for station in self.oper_stations:
            print(station)
            stationid = station.split('_')[-1]
            station_df = self.subset_kartverket_data(
                kv_dataset, 
                stationid, 
                data_timestamp_full, 
                variables
                )
            try:
                #pd.to_datetime(station_df.index).tz_localize('Etc/UCT')
                station_df.index = station_df.index.tz_localize('Etc/UCT')
            except:
                print('Could not tz-localize station_df.')
            #print('station_df.index: ', station_df.index)
            if not station_df.empty:
                if first_station:
                    #print('First iter')
                    df = station_df
                    first_station = False
                    #print('df.index: ', df.index)
                else:
                    try:
                        df.index = df.index.tz_localize('Etc/UCT')
                    except:
                        print('Could not tz-localize df.')
                    df = pd.concat([df, station_df], axis=1)
                    #print('df.index: ', df.index)
                print('station_df: ', station_df)
                print('df: ', df)
            
        # Add missing rows at the end of the dataframe 
        # Last date in data from kartverket is 2020-02-11
        if df.index[-1] < self.datetime_end_hourly:
            print('Adding empty rows at the end of the dataframe...')
            dti = pd.date_range(self.datetime_start_hourly, self.datetime_end_hourly)
            #dti = dti.tz_localize("UTC")
            df = df.reindex(dti)
            print('df new index: ', df)
                
        print('Columns in df from Kartverket.')
        for c in df.columns:
            print(c)
                       
        # ------ Fill in Jean's file with operational data ------
        if len(self.oper_stations) > 0:
            for station in self.oper_stations:
                try:
                    if 'obs' in variables:
                        df['obs_' + station] = (
                            df['obs_' + station]
                            .fillna(oper_df['obs_' + station].astype('float32')
                                    )
                            )
                    if 'tide_py' in variables:
                        df['tide_py_' + station] = (
                            df['tide_py_' + station]
                            .fillna(oper_df['tide_py_' + station].astype('float32')
                                    )
                            ) 
                except:
                    print('Could not add obs and tide data for station ', station)
                    self.oper_stations.remove(station)
                    self.stations.remove(station)
                    print('Removed station ', station, ' from self.oper_stations and self.stations')
                    
        self.df_obs_tide = df
        del df
        del oper_df
        
        # ------ Add horizon variables ------
        for station in self.stations: #self.oper_stations:
            print('station: ', station, ' horizon ', self.horizon)
            if self.horizon > 0:
                if 'obs' in variables:
                    for h in range(1, self.horizon):
                        if h == 1:
                            self.df_obs_tide['obs_t' + str(h) + '_' + station] = self.df_obs_tide['obs_' + station].shift(-1)
                            print("h=1 self.df_obs_tide['obs_t' + str(h) + '_' + station]: ", self.df_obs_tide['obs_t' + str(h) + '_' + station])
                        else:
                            self.df_obs_tide['obs_t' + str(h) + '_' + station] = self.df_obs_tide['obs_t' + str(h-1) + '_' + station].shift(-1)
                            
                if 'tide_py' in variables:
                    for h in range(1, self.horizon):
                        if h == 1:
                            self.df_obs_tide['tide_py_t' + str(h) + '_' + station] = self.df_obs_tide['tide_py_' + station].shift(-1)
                        else:
                            self.df_obs_tide['tide_py_t' + str(h) + '_' + station] = self.df_obs_tide['tide_py_t' + str(h-1) + '_' + station].shift(-1)

            
        return self.df_obs_tide
    
    def generate_hindcast_df(self):
        """Generate ROMS hindcast DataFrame.
        
        Generates a DataFrame with the ROMS hindcast output from all the 
        feature_stations in the list, and the times.
        
        Parameters
        ----------
        stations : list
            List of stations.
            
        Returns
        -------       
        df : DataFrame
            DataFrame containing ROMS hindcast data.
        """
        print('Generating ROMS hindcast DataFrame...')
         
        # TODO: get paths and other variables from config file
        # TODO: check that roms and obs have the same lenght. Since I'm attaching 
        #       data from the operational files to the observations, after 2019,
        #       The observations df might have more rows than the roms df.
# 
        start = self.datetime_start_hourly
        if self.run_on_ppi:
            if self.new_station_loc:
                if self.era5_forcing:
                    path_to_ROMS_file = Path((
                    '/lustre/storeB/project/IT/geout/machine-ocean/' 
                    + 'prepared_datasets/storm_surge/ROMS_hindcast_at_stations/'
                    + 'roms_hindcast_storm_surge_data_era5_new_loc.nc'
                    ))
                    print('Preparing NORA3ERA5 hindcast data...')
                    
                    # TODO: set 2001 as the earliest date and fill in with nans
                else:
                    path_to_ROMS_file = Path((
                    '/lustre/storeB/project/IT/geout/machine-ocean/' 
                    + 'prepared_datasets/storm_surge/ROMS_hindcast_at_stations/'
                    + 'roms_hindcast_storm_surge_data_new_loc.nc'
                    ))
                    print('Preparing ERA5 hindcast data...')
                    first_hindcast_date = dt.datetime(
                        2001, 1, 1, 0, 0, 0,   ##### NOR 2000??? /lustre/storeB/project/fou/om/StormRisk/RunsNordic4/Run2000_2019
                        tzinfo=dt.timezone.utc
                        )
                    if self.datetime_start_hourly < first_hindcast_date:
                        start = first_hindcast_date
                   
            else:
                path_to_ROMS_file = Path((
                    '/lustre/storeB/project/IT/geout/machine-ocean/' 
                    + 'prepared_datasets/storm_surge/ROMS_hindcast_at_stations/'
                    + 'roms_hindcast_storm_surge_data.nc'
                    ))
                first_hindcast_date = dt.datetime(
                    2002, 1, 1, 0, 0, 0, 
                    tzinfo=dt.timezone.utc
                    )
                if self.datetime_start_hourly < first_hindcast_date:
                    start = first_hindcast_date
           
        else:   
            path_to_ROMS_file = Path((
                '/home/paulinast/Desktop/data/ROMS/hindcast/'
                +'roms_hindcast_storm_surge_data.nc'
                ))
            first_hindcast_date = dt.datetime(
                2002, 1, 1, 0, 0, 0, 
                tzinfo=dt.timezone.utc
                )
            if self.datetime_start_hourly < first_hindcast_date:
                start = first_hindcast_date
        
        accessor_ROMS = loader.DatasetAccessorROMS(
            path_to_ROMS_file
            ) 
        
        # Get the data with Jean's function and store it in dictionary
        data_dict = {}
        for crrt_station in self.stations: 
            roms_time, roms_data =\
                accessor_ROMS.get_roms_station_data(
                    crrt_station,
                    start - dt.timedelta(minutes=1),
                    self.datetime_end_hourly + dt.timedelta(minutes=1)
                    )
            data_dict['roms_' + crrt_station] = roms_data / 100
            
        data_dict['time'] = roms_time
            
        df_roms = pd.DataFrame.from_dict(data_dict) 
        del data_dict
        self.df_roms = df_roms.set_index('time')

        return self.df_roms
    
    
    def generate_obs_tide_df_operational(self, variables):
        
        # Data directory where some pre-generated dfs are stored in dicts
        data_dir = ('/lustre/storeB/project/IT/geout/machine-ocean/workspace'
                    + '/paulinast/storm_surge_results/data_preprocess_input'
                    + '/5_stations/monthly'
                    )
        
        #if self.forecast_mode: 
        file_name = '/dict_df_all_stations_obs_tide_kyststasjoner_all_months_12_hours_new_no_forecast_generated_in_the_past.pickle'
        if  (self.use_existing_files  and Path(data_dir + '/' + file_name).is_file()):
            print('Using already generated kyststasjoner file.')
            # Open file that has already been generated as it takes a long time
            with open(data_dir + '/' + file_name, 'rb') as handle:
                station_file = pickle.load(handle)
            station_file = (
                #station_file['df_arome_5_stations']
                station_file['df_hourly']  # 'df_hourly_t0'
                .loc[self.datetime_start_hourly: self.datetime_end_hourly]
                )
            
            col_names = []
            for station in self.stations:
                for var in variables:
                    col_name = var +  '_' + station
                    col_names.append(col_name)   
            col_names = list(set(col_names) & set(list(station_file.columns)))         
            oper_df = station_file[col_names]
        else:
            # Generate files 
            # I use to run this for shorter periods and concatenate the files
            oper_df = self.generate_obs_tide_kyststasjoner_df(variables)
            
        if 'oper_df' in locals():
            return oper_df
        
        return
    
    def generate_obs_tide_kyststasjoner_df(self, variables):
        
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
            print('station: ', station)
            if 'obs' in variables:
                data_array = rk.make_data_arrays('observed_at_chartdatum', station, path_list, 1)
                col_names = rk.make_hz_col_names('observed_at_chartdatum', station, 1)
                df = rk.array_to_df(data_array, times, col_names)
                if first_iter:
                    oper_df  = df
                else: 
                    oper_df =  pd.concat([oper_df, df], axis=1)
                first_iter = False
            if 'tide_py' in variables:
                data_array = rk.make_data_arrays('tide_at_chartdatum', station, path_list, 1)
                col_names = rk.make_hz_col_names('tide_at_chartdatum', station, 1)
                df = rk.array_to_df(data_array, times, col_names)
                if first_iter:
                    oper_df  = df
                else: 
                    oper_df =  pd.concat([oper_df, df], axis=1)
                first_iter = False
            
        return oper_df
                
    
        
    
    def generate_obs_tide_df(self, variables):
        """Generate observation and tide DataFrame.
        
        Generate a DataFrame with the observations and the tide data
        from all the feature_stations in the list, and the times.
        
        Parameters
        ----------
        variables : list str
            Variables to include in the DataFrame: 'obs', 'tide_py', or both.            
            
        Returns
        -------       
        df : DataFrame
            DataFrame containing observations and tide.
        """
        print('Generating obs and tide_py DataFrame...')
        
        # ------ Get tide and observations from operational files ------
        oper_df = self.generate_obs_tide_df_operational(variables)
        print('oper_df: ', oper_df)
         
        # TODO: get paths and other variables from config file
        if self.run_on_ppi:            
            path_to_obs_file = Path((
                '/lustre/storeB/project/IT/geout/machine-ocean/prepared_datasets/'
                + 'storm_surge/aggregated_water_level_data/' 
                #+ 'aggregated_water_level_observations_with_pytide_prediction_dataset.nc'
                + 'aggregated_water_level_observations_with_pytide_prediction_dataset_2020_12.nc'
                ))
        else:   
            path_to_obs_file = Path((
                '/home/paulinast/Desktop/data/aggregated_water_level_data/'
                + 'aggregated_water_level_observations_with_pytide_prediction_dataset.nc'
                ))
            
        accessor_obs = loader.DatasetAccessorObservations(
            path_to_obs_file
            )
        
        # Get the data with Jean's function and store it in dictionary
        crrt_dict = {}
        
        # ------ Dictionary with data from Jean's file ------
        # ------ All stations ------

        for station in self.stations: #self.feature_stations: #, "SW_2111"]: 
            obs_times, obs_data, tide = accessor_obs.get_observation_tide(
                station,
                self.datetime_start_hourly - dt.timedelta(minutes=2),
                self.datetime_end_hourly + dt.timedelta(minutes=2)
                )
            # use only observations at minutes 00 ie start of each hour
            list_indexes_hourly_obs = list(range(0, len(obs_times), 6))
            
            # if some invalid observation data, replace with NaN
            naned_obs_data = np.array(obs_data)
            naned_obs_data[np.abs(naned_obs_data) > 1.0e6] = np.nan
            if 'obs' in variables:
                crrt_dict['obs_' + station] = naned_obs_data[list_indexes_hourly_obs] / 100
            if 'tide_py' in variables:
                crrt_dict['tide_py_' + station] = np.array(tide)[list_indexes_hourly_obs] / 100
            obs_times = np.array(obs_times)[list_indexes_hourly_obs]
            
        crrt_dict['time'] = obs_times       
        crrt_df = pd.DataFrame.from_dict(crrt_dict)  
        del crrt_dict
        crrt_df = crrt_df.set_index('time')
        
        # Add rows at the end 
        df_range = pd.date_range(
            self.datetime_start_hourly,
            self.datetime_end_hourly, 
            freq='H'
            )
        
        crrt_df = crrt_df.reindex(df_range)
        print('crrt_df: ', crrt_df)
    
        
        # TODO: Do it the other way around, first compute horizons and then 
        # fill in missing data so that there aren't missings at the end
        # ------ Fill in Jean's file with operational data ------
        if len(self.oper_stations) > 0:
            for station in self.oper_stations:
                if 'obs' in variables:
                    crrt_df['obs_' + station] = (
                        crrt_df['obs_' + station]
                        .fillna(oper_df['obs_' + station].astype('float32')
                                )
                        )
                if 'tide_py' in variables:
                    crrt_df['tide_py_' + station] = (
                        crrt_df['tide_py_' + station]
                        .fillna(oper_df['tide_py_' + station].astype('float32')
                                )
                        )  
        self.df_obs_tide = copy.deepcopy(crrt_df)
        del crrt_df
        del oper_df
        
        #print("self.df_obs_tide['obs_' + station]: ", self.df_obs_tide['obs_' + station])

        # ------ Add horizon variables ------
        for station in self.stations:#self.oper_stations:
            print('station: ', station, ' horizon: ', self.horizon)
            if self.horizon > 0:
                if 'obs' in variables:
                    for h in range(1, self.horizon):
                        self.df_obs_tide['obs_t' + str(h) + '_' + station] = self.df_obs_tide['obs_' + station].shift(-h)
                        #if h == 1:  
                        #    print("h=1 'obs_t' + str(h) + '_' + station", 'obs_t' + str(h) + '_' + station)
                        #    self.df_obs_tide['obs_t' + str(h) + '_' + station] = self.df_obs_tide['obs_' + station].shift(-1)
                        #    print("self.df_obs_tide['obs_t' + str(h) + '_' + station]: ", self.df_obs_tide['obs_t' + str(h) + '_' + station])
                        #else:
                        #    self.df_obs_tide['obs_t' + str(h) + '_' + station] = self.df_obs_tide['obs_t' + str(h-1) + '_' + station].shift(-1)
                            
                if 'tide_py' in variables:
                    for h in range(1, self.horizon):
                        self.df_obs_tide['tide_py_t' + str(h) + '_' + station] = self.df_obs_tide['tide_py_' + station].shift(-h)
                        #if h == 1:
                        #    self.df_obs_tide['tide_py_t' + str(h) + '_' + station] = self.df_obs_tide['tide_py_' + station].shift(-1)
                        #else:
                        #    self.df_obs_tide['tide_py_t' + str(h) + '_' + station] = self.df_obs_tide['tide_py_t' + str(h-1) + '_' + station].shift(-1)
                        
        return self.df_obs_tide
    
    def correction(self, variables):
        """Correct the hindcast data.
        
        Correct the hindcast data using the bias/correction as computed for
        MET's operational model (Nils' correction). Add the new columns to the
        dataframe.
    
        Parameters
        ----------
        df : DataFrame
            DataFrame containing the ROMS (hindcast), tide predictions, and 
            observations for each station in station_list.
        
        variables : list of str
            List containing tha variables to correct (among others).
            
        Returns
        -------
        df : DataFrame
            Dataframe with the corrected ROMS data.

        """       
        df = pd.DataFrame()    
        for station in self.stations:
            offset = np.zeros([self.df_roms.shape[0], 120]) 
            # The data is already in self.df_aux
            roms = self.df_roms['roms_' + station].sort_index()
            
            # TODO:
                # OBS!! There are some duplicated indeces
                # This is a temporary solution!!!!
            obs = self.df_obs_tide['obs_' + station].sort_index()
            print('Duplicated indices in df_obs_tide: ', obs.index.duplicated)
            obs = obs[~obs.index.duplicated(keep='first')]
            #obs.reset_index(inplace=True, drop=True)

            tide = self.df_obs_tide['tide_py_' + station].sort_index()
            tide = tide[~tide.index.duplicated(keep='first')]
            #tide.reset_index(inplace=True, drop=True)
            
            # Since the roms df might be shorter than the obs df (because roms
            # is only available until Dec 2019), we subset the observation.
            
            last_roms_idx = roms.index[-1]
            first_roms_idx = roms.index[0]

            obs = obs.loc[first_roms_idx:last_roms_idx]
            tide = tide.loc[first_roms_idx:last_roms_idx]
            
            for i in range(120):
                try:
                    offset[:, 120-i] = (
                    roms.shift(i+1)
                    - (
                        obs.shift(i+1)
                        - tide.shift(i+1)
                        )
                    ) 
                except:
                    pass
                    
            # Weigh the last obs more:
            w = np.arange(float(offset.shape[1]))  # TODO: Check if it has to start at 1, not 0!!!!
            w = w/w.sum()

            bias = np.sum(offset * w, axis=1)
            
            if '(roms - biasMET)' in variables: 
                df['(roms - biasMET)_' + station] = (
                    roms 
                    - bias
                    )
                  
            if ('(roms - biasMET) - (obs - tide_py)') in variables:
               df['(roms - biasMET) - (obs - tide_py)_' + station] = (
                roms
                - bias
                - obs
                + tide
                ) 
            
        return df

    
    def generate_station_forcing_df(self, variables):
        """Dataframe with forcing data at each station location.
                Generate a dataframe containing station data for all stations for the
        variables in the feature_vars and label_vars lists. The methods skips 
        the variables that are not providaded in the original data files.

        Parameters
        ----------
        variables: list of str
            Station forcing variables to be included in the DataFrame.
            
        Returns
        -------
        df_station_forcing : DataFrame
            DataFrame containing data for all the stations for the varibles in 
            self.feature_vars and self.label_vars.

        """
        
        print('Generating station forcing df...')
        
        # Data directory where some pre-generated dfs are stored in dicts
        data_dir = ('/lustre/storeB/project/IT/geout/machine-ocean/workspace'
                    + '/paulinast/storm_surge_results/data_preprocess_input'
                    + '/5_stations/monthly'
                    )
        
        if self.forecast_mode: # In forecast mode, we use AROME data 
            file_name = 'dict_df_5_stations_only_arome_all_months_12_hours_new_no_forecast_generated_in_the_past.pickle'
            file_name = 'dict_df_5_stations_only_arome_and_waves_all_months_12_hours_new_no_forecast_generated_in_the_past.pickle'
            file_name = '/dict_df_all_stations_only_arome_all_months_12_hours_new_no_forecast_generated_in_the_past.pickle'
            if  (self.use_existing_files  and Path(data_dir + '/' + file_name).is_file()):
                print('Using already generated AROME file.')
                # Open file that has already been generated as it takes a long time
                with open(data_dir + '/' + file_name, 'rb') as handle:
                    station_file = pickle.load(handle)
                station_file = (
                    #station_file['df_arome_5_stations']
                    station_file['df_hourly']  # 'df_hourly_t0'
                    .loc[self.datetime_start_hourly: self.datetime_end_hourly]
                    )
                
                col_names = []
                for station in self.stations:
                    for var in variables:
                        for h in range(self.horizon):
                            if h == 0 :
                                col_name = var + '_' + station
                            else:
                                col_name = var + '_t' + str(h) + '_' + station
                            col_names.append(col_name) 
                            
                col_names = list(set(col_names) & set(list(station_file.columns)))         
                df_station_forcing = station_file[col_names]
            else:
                # Generate files 
                # I use to run this for shorter periods and concatenate the files
                df_station_forcing = self.generate_arome_df(variables)
                print('generate_arome_df should be updated with new paths')
                      
            
        else:  # not forecast_mode, we use the ERA5 data That Martin has prepared
            file_name = 'dict_df_5_stations_era5_all_months_12_hours_new_no_forecast_generated_in_the_past.pickle'
            if  (self.use_existing_files  and Path(data_dir + '/' + file_name).is_file()):
                with open(data_dir + '/' + file_name, 'rb') as handle:
                    station_file = pickle.load(handle)
                station_file = (
                    #station_file['df_arome_5_stations']
                    station_file['df_hourly']
                    .loc[self.datetime_start_hourly: self.datetime_end_hourly]
                    )
                
                # We do not iterate over the horizon dimension because we 
                # will lag the data in the preprocessor.
                col_names = []
                for station in self.stations:
                    for var in variables:
                        col_name = var + '_' + station
                        col_names.append(col_name) 
                            
                col_names = list(set(col_names) & set(list(station_file.columns)))
                            
                df_station_forcing = station_file[col_names]
            else:
                # Generate files 
            
                if self.run_on_ppi:
                   data_dir_station = '/lustre/storeB/project/IT/geout/machine-ocean/workspace/martinls' 
                else:
                    data_dir_station ='/home/paulinast/Desktop/data/wam3_nora3era5/martinls/new'
        
                station_file = xr.open_mfdataset(
                    data_dir_station 
                    + '/aggregated_era5_*.nc',
                    data_vars ='minimal',
                    parallel=False # for Centos
                    )
        
                station_file = station_file.sel(
                    time=slice(self.datetime_start_hourly, self.datetime_end_hourly)
                    )
                
                # Subset file by selecting only the feature and label stations
                station_id = []
                for name in self.stations:
                    station_id.append(np.where(station_file.stationid.values==name)[0].item())
                station_file = station_file.sel(station=station_id)
                
                count=0
                for var in variables:
                    if count==0:                             
                        df_station_forcing = station_file[var].to_pandas().transpose() 
                        df_station_forcing.columns = self.stations
                        df_station_forcing = df_station_forcing.add_prefix(var + '_')
                        count=count+1
                        
                            
                    else:
                        # Select variable and convert to DataFrame
                        df = station_file[var].to_pandas().transpose()
                        # Change column names
                        df.columns = self.stations                   
                        df = df.add_prefix(var + '_')
                        # Concatenate DataFrames
                        df_station_forcing = pd.concat(
                            [
                                df_station_forcing,
                                df
                                ],
                            axis=1
                            )                   
                        del df
                        count=count+1
                            
                df_station_forcing.index = df_station_forcing.index.tz_localize('UTC')
                
                # Add horizon data
                for station in self.stations:
                    for var in variables:
                        for h in range(1, self.horizon):
                            df_station_forcing[var +'_t' + str(h) + '_' + station] = df_station_forcing[var + '_' + station].shift(-h)
                        
        return df_station_forcing
    
    
    def find_nearest_point(self, station_name:str):
        """Find nearest mesh points to station point.
        
        Finds the nearest mesh points to the station point of interest 
        defined by station_idx.

        Parameters
        ----------
        station_name: str
            Station name (stationid), either a feature or a label station.
        
        Returns
        -------
        lat_nearest : array of float
            Latitude of the nearest mesh points to the station point.
        lon_nearest : array of float
            Longitude of the nearest mesh points to the station point.
        """
        list_of_mesh_points = self.get_list_mesh_points()
        array_of_mesh_points = np.asarray(list_of_mesh_points)
        
        lat, lon = hlp.get_station_lat_lon(station_name=station_name)
               
        deltas = array_of_mesh_points - (lat, lon)

        dist = np.abs(deltas[:, 0]) + np.abs(deltas[:, 1])
        idx_nearest = np.argmin(dist) 
        
        lat_nearest = array_of_mesh_points[idx_nearest, 0]
        lon_nearest = array_of_mesh_points[idx_nearest, 1]
        
        return idx_nearest, lat_nearest, lon_nearest
    
    
    def find_mesh_stations(self):
        """Return indices of the stations' nearest gridbox.
        
        Return the indices (points) in the mesh df of the nearest gridbox to 
        the stations of interest.

        Returns
        -------
        mesh_indices : bool
            Indices of the stations in the mesh df.
            
        """
        mesh_indices = []
        for s in self.stations:
            idx_nearest, _ , _ = self.find_nearest_point(station_name=s)
            mesh_indices.append(idx_nearest)
                
        return mesh_indices
    
    
    def generate_mesh_forcing_df(self, variables):  
        """Generate Dataframe with forcing data over a mesh.
        
        Generate a dataframe containing forcing data for the nearest grid box 
        in the mesh to all stations for the variables in the feature_vars and 
        label_vars lists. The methods skips the variables that are not 
        providaded in the original data file.

        Parameters
        ----------
        variables : list of str
            Mesh forcing variables to include in the DataFrame.
            
        Returns
        -------
        df_mesh_forcing : DataFrame
            DataFrame containing data for the nearest grid box to all the 
            stations for the varibles in self.feature_vars and self.label_vars.

        """
        # The function with the same name in preprocessor.py has not been updated.
        print('Generating mesh forcing df...')
        if self.run_on_ppi:
            data_dir_mesh = (
                '/lustre/storeB/project/IT/geout/machine-ocean/workspace/jeanr/'
                + 'interact_with_roms/lat_lon_to_ROMS_timeseries'
                )
        else:
            data_dir_mesh = '/home/paulinast/Desktop/data/wam3_nora3era5/jeanr'
        # 

        mesh_file = xr.open_dataset(
            data_dir_mesh
            + '/wam3_nora3era5_forcing.nc'
            )
        
        mesh_file = mesh_file.sel(
            time=slice(self.datetime_start_hourly, self.datetime_end_hourly)
            )
        
        i = self.find_mesh_stations() 
        
        count = 0
        for var in variables:
            if count==0:              
                df_mesh_forcing = mesh_file[var].to_pandas().transpose()
                df_mesh_forcing = df_mesh_forcing.iloc[i].transpose()
                df_mesh_forcing.columns = self.stations
                df_mesh_forcing = df_mesh_forcing.add_prefix(var + '_')
                count = count + 1
            else:
                # Select variable and convert to DataFrame
                df = mesh_file[var].to_pandas().transpose()
                df = df.iloc[i].transpose()
                # Change column names
                df.columns = self.stations
                df = df.add_prefix(var + '_')
                # Concatenate DataFrames
                df_mesh_forcing = pd.concat(
                    [
                        df_mesh_forcing,
                        df
                        ],
                    axis=1
                    )
                del df
                    
        df_mesh_forcing.index = df_mesh_forcing.index.tz_localize('UTC')
        return df_mesh_forcing
    """
    def generate_arome_df(self, variables):
        
        print('Generate AROME df...')
        
        # TODO: generate a file that can be used multiple times when  calling self.use_existing_file
        
        data_dir = '/lustre/storeB/project/fou/hi/stormsurge_eps/2dEPS_archive'      
        
        # ------ Open data in operational files ------
        ra = ReadArome(self.datetime_start_hourly, self.datetime_end_hourly)       
        
        path_list = ra.make_url_list()
        times = ra.make_time_idx()
        
        self.oper_stations
        first_iter = True
        #for station in self.oper_stations:
            #print('Station: ', station)
            #data_array = ra.make_data_arrays(variables, [station], path_list, self.horizon)
            #col_names = ra.make_hz_col_names(variables, [station], self.horizon)
        data_array = ra.make_data_arrays(variables, self.oper_stations, path_list, self.horizon)
        col_names = ra.make_hz_col_names(variables, self.oper_stations, self.horizon)
        df_station = ra.array_to_df(data_array, times, col_names)          
        
        if first_iter:
            df = df_station 
        else:
            df = pd.concat([df_station, df], axis=1)
        first_iter = False
        
        # Already 12-hourly
        # TODO: generate warning if hourly and arome data used
        return df
    """
    
    def generate_arome_df(self, variables):
        
        print('Generate AROME df...')
        
        # TODO: generate a file that can be used multiple times when  calling self.use_existing_file
        
        data_dir = '/lustre/storeB/project/fou/hi/stormsurge_eps/2dEPS_archive'      
        
        start_hour = self.datetime_start_hourly -  dt.timedelta(self.past_forecast)
        
        # ------ Open data in operational files ------
        ra = ReadArome(start_hour, self.datetime_end_hourly)       
        
        path_list = ra.make_path_list_ppi()
        times = ra.make_time_idx()
        
        first_iter = True
        for station in self.oper_stations:
            for var in variables:
                print('station: ', station, ' var: ', var)
                # Generate big data array that includes past forecasts
                big_data_array = ra.make_data_arrays([var], [station], path_list, self.horizon + self.past_forecast)
                print('big_data_array.shape: ', big_data_array.shape)
                for fhr in range(-self.past_forecast, 12, 12):
                    small_data_array = big_data_array[self.past_forecast + fhr: fhr - 1,  : self.horizon - fhr]
                    print('small_data_array.shape: ', small_data_array.shape)
                    times_small = times[self.past_forecast + fhr: fhr - 1]
                    
                    #print('len(times_small): ', len(times_small))
                    print('fhr: ', fhr)
                    print('self.past_forecast + fhr: ', self.past_forecast + fhr)
                    print('self.horizon - fhr: ', self.horizon - fhr)
                    col_names = ra.make_hz_col_names([var], [station], self.horizon, fhr=fhr)
                    print('col_names: ', col_names)
                    print('len(col_names): ', len(col_names))
                    df  = ra.array_to_df(small_data_array, times_small, col_names)
                    if first_iter:
                        df_arome = df
                        first_iter = False
                    else:
                        df_arome = pd.concat([df_arome, df], axis=1)

        # Already 12-hourly
        # TODO: generate warning if hourly and arome data used     
        return df_arome
    
    

    
    def generate_operational_df(self, variables):
        # TODO: Fix generated file so that it contains data for all hz
        
        # Data directory where some pre-generated dfs are stored in dicts
        data_dir = ('/lustre/storeB/project/IT/geout/machine-ocean/workspace'
                    + '/paulinast/storm_surge_results/data_preprocess_input'
                    + '/5_stations/monthly'
                    )
        
        #if self.forecast_mode: 
        file_name = '/dict_df_all_stations_kyststasjoner_all_months_12_hours_new_no_forecast_generated_in_the_past.pickle'
        if  (False and self.use_existing_files  and Path(data_dir + '/' + file_name).is_file()):
            print('Using already generated kyststasjoner file.')
            # Open file that has already been generated as it takes a long time
            with open(data_dir + '/' + file_name, 'rb') as handle:
                station_file = pickle.load(handle)
            station_file = (
                #station_file['df_arome_5_stations']
                station_file['df_hourly']  # 'df_hourly_t0'
                .loc[self.datetime_start_hourly: self.datetime_end_hourly]
                )
            
            col_names = []
            for station in self.stations:
                for var in variables:
                    col_name = var + '_' + station
                    col_names.append(col_name) 
            
            """
            member_str = ''
            for m in self.ensemble_members:
                if (m>=0) and (m<51):
                    member_str = '_m' + str(m)
                elif m < -1:
                    member_str = '_m' + str(52 + m)
                print('prepare_df m: ', m)
                for station in self.stations:
                    for var in variables:
                        for fhr in range(-self.past_forecast, 0, 12):
                            past_forecast_str = '_' + str(fhr)
                            for h in range(fhr, self.horizon): 
                                if h == 0 :
                                    col_name = var + member_str +  past_forecast_str + '_' + station
                                else:
                                    col_name  = var + member_str + '_t' + str(h) + past_forecast_str + '_' + station
                                col_names.append(col_name) """
                            
            col_names = list(set(col_names) & set(list(station_file.columns)))         
            self.df_operational = station_file[col_names]
        else:
            # Generate files 
            # I use to run this for shorter periods and concatenate the files
            self.df_operational = self.generate_kyststasjoner_df(variables)
            #print('self.df_operational.head(): ', self.df_operational.head())
            
        #if 'self.df_operational' in locals():
        return self.df_operational
    
    def generate_kyststasjoner_df(self, variables):
        """Generate Dataframe with forcing data over a mesh.
        
        Generate a dataframe containing forcing data at each location for the 
        variables in the feature_vars and label_vars lists. The methods skips 
        the variables that are not providaded in the original data file.
        
        Parameters
        ----------
        variables : list of str
            Operational variables to include in the DataFrame, these can be 
            'stormsurge', 'stormsurge_corrected', or 'bias'.
            
        Returns
        -------
        df_mesh_forcing : DataFrame
            DataFrame containing forcing data from the station locations for 
            the varibles in self.feature_vars and self.label_vars.

        """ 

        """Generate Dataframe with forcing data over a mesh.
        
        Generate a dataframe containing forcing data at each location for the 
        variables in the feature_vars and label_vars lists. The methods skips 
        the variables that are not providaded in the original data file.
        
        Parameters
        ----------
        variables : list of str
            Operational variables to include in the DataFrame, these can be 
            'stormsurge', 'stormsurge_corrected', or 'bias'.
            
        Returns
        -------
        df_mesh_forcing : DataFrame
            DataFrame containing forcing data from the station locations for 
            the varibles in self.feature_vars and self.label_vars.

        """ 

        # TODO: generate a file that can be used multiple times when  calling self.use_existing_file
        
        print('Generating operational DataFrame...')
        
        # Open file generated with read_kyststasjoner.py
        data_dir = '/lustre/storeB/project/fou/hi/stormsurge_eps/2dEPS_archive'      
        
        first_m = True
        first_iter = True
        start_hour = self.datetime_start_hourly -  dt.timedelta(hours=self.past_forecast)
        end_hour = self.datetime_end_hourly
        if self.datetime_end_hourly.hour == 0:
            end_hour = self.datetime_end_hourly + dt.timedelta(hours=12)

        for m in self.ensemble_members:
            if m < 53:
                rk = ReadKyststasjoner(
                    dt_start=start_hour, 
                    dt_end=self.datetime_end_hourly,
                    ensemble_member=m,
                    data_dir=data_dir
                    ) 
                
                path_list = rk.make_path_list()
                times = rk.make_time_idx()
                #print('times: ', times)
                print('prepare_df m: ', m)
                
                for station in self.oper_stations:
                    for var in variables:
                        #try:
                            if var in ['stormsurge', 'stormsurge_corrected']:
                                print('station: ', station, ' var: ', var)
                                # Generate big data array that includes past forecasts
                                # This way we extract the data from the kyststasjoner 
                                # files only once, and speed up the process
                                big_data_array = rk.make_data_arrays(
                                    var, 
                                    station, 
                                    path_list, 
                                    self.horizon + self.past_forecast
                                    )
                                print('big_data_array.shape: ', big_data_array.shape)
                                count=0
                                for fhr in range(-self.past_forecast, 12, 12):
                                    if fhr == 0:
                                        small_data_array = (
                                        big_data_array[count: , :self.horizon - fhr]
                                        )
                                        times_small = times[2:]
                                    else:
                                        end_row = int(-self.past_forecast/12 + count)
                                        small_data_array = (
                                            big_data_array[count:end_row , :self.horizon - fhr]
                                            )
                                        times_small = times[2:]
                                    col_names = rk.make_hz_col_names(
                                        var, 
                                        station, 
                                        self.horizon, 
                                        fhr=fhr, 
                                        member=m
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
    
                            elif var in ['bias']:
                                if first_m: # Add bias variable only for the first member
                                    # Generate big data array that includes past forecasts
                                    big_data_array = rk.make_data_arrays(var, station, path_list, self.horizon)
                                    count = 0
                                    for fhr in range(-self.past_forecast, 12, 12):
                                        if fhr == 0:
                                            small_data_array = (
                                            big_data_array[count: , :self.horizon - fhr]
                                            )
                                            times_small = times[count:]
                                        else:
                                            end_row = int(-self.past_forecast/12 + count)
                                            small_data_array = (
                                                big_data_array[count:end_row , :self.horizon - fhr]
                                                )
                                            times_small = times[count:end_row]
                                        col_names = rk.make_hz_col_names(
                                            var, 
                                            station, 
                                            self.horizon, 
                                            fhr=fhr, 
                                            member=''
                                            )
                                        df  = rk.array_to_df(
                                            small_data_array, 
                                            times_small, 
                                            col_names
                                            )  
                                        count = count + 1  
                                        if first_iter:
                                            df_crrt = df
                                            irst_iter = False
                                        else:
                                            df_crrt = pd.concat([df_crrt, df], axis=1)  
                                        count = count + 1
                        #except:
                        #    print('Could not generate kyststasjoner df for station ', station, ' and variable ', var, '.')
                first_m = False
                            
        if 'df_crrt' in locals():         
            return df_crrt

    def generate_kyststasjoner_df_old(self, variables):
        """Generate Dataframe with forcing data over a mesh.
        
        Generate a dataframe containing forcing data at each location for the 
        variables in the feature_vars and label_vars lists. The methods skips 
        the variables that are not providaded in the original data file.
        
        Parameters
        ----------
        variables : list of str
            Operational variables to include in the DataFrame, these can be 
            'stormsurge', 'stormsurge_corrected', or 'bias'.
            
        Returns
        -------
        df_mesh_forcing : DataFrame
            DataFrame containing forcing data from the station locations for 
            the varibles in self.feature_vars and self.label_vars.

        """ 

        # TODO: generate a file that can be used multiple times when  calling self.use_existing_file
        
        print('Generating operational DataFrame...')
        
        # Open file generated with read_kyststasjoner.py
        data_dir = '/lustre/storeB/project/fou/hi/stormsurge_eps/2dEPS_archive'      
        
        first_m = True
        first_iter = True
        
        start_hour = self.datetime_start_hourly -  dt.timedelta(self.past_forecast)
        for m in self.ensemble_members:
            if m < 53:
                rk = ReadKyststasjoner(
                    dt_start=start_hour, 
                    dt_end=self.datetime_end_hourly,
                    ensemble_member=m,
                    data_dir=data_dir
                    ) 
                
                path_list = rk.make_path_list()
                times = rk.make_time_idx()
                #print('times: ', times)
                print('prepare_df m: ', m)
                
                for station in self.oper_stations:
                    for var in variables:
                        try:
                            if var in ['stormsurge', 'stormsurge_corrected']:
                                print('station: ', station, ' var: ', var)
                                # Generate big data array that includes past forecasts
                                big_data_array = rk.make_data_arrays(var, station, path_list, self.horizon + self.past_forecast)
                                print('big_data_array.shape: ', big_data_array.shape)
                                for fhr in range(-self.past_forecast, 12, 12):
                                    small_data_array = big_data_array[self.past_forecast + fhr: fhr - 1,  : self.horizon - fhr]
                                    print('small_data_array.shape: ', small_data_array.shape)
                                    times_small = times[self.past_forecast + fhr: fhr - 1]
                                    
                                    #print('len(times_small): ', len(times_small))
                                    #print('fhr: ', fhr)
                                    #print('self.past_forecast + fhr: ', self.past_forecast + fhr)
                                    #print('self.horizon - fhr: ', self.horizon - fhr)
                                    col_names = rk.make_hz_col_names(var, station, self.horizon, fhr=fhr, member=m)
                                    #print('col_names: ', col_names)
                                    #print('len(col_names): ', len(col_names))
                                    #print('col_names: ', col_names)
                                    df  = rk.array_to_df(small_data_array, times_small, col_names)
                                    if first_iter:
                                        df_crrt = df
                                        first_iter = False
                                    else:
                                        df_crrt = pd.concat([df_crrt, df], axis=1)
    
                            elif var in ['bias']:
                                if first_m: # Add bias variable only for the first member
                                    # Generate big data array that includes past forecasts
                                    big_data_array = rk.make_data_arrays(var, station, path_list, self.horizon)
                                    for fhr in range(-self.past_forecast, 12, 12):
                                        small_data_array = big_data_array[self.past_forecast + fhr: fhr - 1,  : self.horizon - fhr]
                                        times_small = times[self.past_forecast + fhr: fhr]
                                        col_names = rk.make_hz_col_names(var, station, self.horizon, fhr=fhr, member='')
                                        df  = rk.array_to_df(small_data_array, times_small, col_names)    
                                    if first_iter:
                                        df_crrt = df
                                        irst_iter = False
                                    else:
                                        df_crrt = pd.concat([df_crrt, df], axis=1)     
                        except:
                            print('Could not generate kyststasjoner df for station ', station, ' and variable ', var, '.')
                first_m = False
                            
        if 'df_crrt' in locals():         
            return df_crrt
        
        
def main(args):
    datetime_start_hourly, datetime_end_hourly = args
    
    datetime_start_hourly = dt.datetime.strptime(datetime_start_hourly, '%Y%m%d')
    datetime_end_hourly = dt.datetime.strptime(datetime_end_hourly, '%Y%m%d')
    datetime_start_hourly = datetime_start_hourly.replace(tzinfo=dt.timezone.utc)
    datetime_end_hourly = datetime_end_hourly.replace(tzinfo=dt.timezone.utc)
    
    print('datetime_start_hourly: ', datetime_start_hourly)
    
    prep = PrepareDataFrames(
    variables = ['msl', 'u10', 'v10'], #, 'swh', 'mwd'],
    #variables = ['stormsurge_corrected', 'stormsurge', 'bias'],
    stations = hlp.get_norwegian_station_names(), #['NO_OSC', 'NO_AES', 'NO_BGO', 'NO_HEI', 'NO_KSU'], 
    ensemble_members=[-1],
    datetime_start_hourly = datetime_start_hourly,
    datetime_end_hourly = datetime_end_hourly,
    horizon = 60 + 1, #1,  # horizon in prepare_df is an int +1!!! Lags are computed in self.generate_feature_array
    use_station_data = True,
    run_on_ppi = True,
    new_station_loc=True,
    era5_forcing=False,
    add_aux_variables=False,
    data_from_kartverket=False,
    forecast_mode=True,
    hourly=False,  # Otherwise make 12-hourly data at 00 and 12 hours.
    use_existing_files=False,
    filter_times_in_hourly_df=False
    )        
    
    df = prep.prepare_features_labels_df()
    df_hourly_t0 = prep.df_hourly_t0
    
    #my_dict = { 'df_arome_and_waves_5_stations' : df}
    my_dict = { 
        'df_12_hours' : df, 
        'df_hourly_t0' : df_hourly_t0
        }
    #my_dict = { 'df_era5_5_stations' : df}

    path_to_dict = (
        '/lustre/storeB/project/IT/geout/machine-ocean/workspace/paulinast'
        + '/storm_surge_results/data_preprocess_input/5_stations/monthly'
        )
    
    with open(
            path_to_dict 
            #+ '/dict_df_5_stations_only_arome_and_waves_12_hours' 
            + '/dict_df_all_stations_only_arome_12_hours' 
            #+ '/dict_df_5_stations_era5_12_hours'
            #+ '/dict_df_all_stations_kyststasjoner_12_hours'
            + datetime_start_hourly.strftime('%Y%m%d') 
            + '.pickle', 
            'wb'
            ) as handle:
        pickle.dump(my_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        
def main_2(args):
    
        
    datetime_start_hourly, datetime_end_hourly = args
    
    datetime_start_hourly = dt.datetime.strptime(datetime_start_hourly, '%Y%m%d')
    datetime_end_hourly = dt.datetime.strptime(datetime_end_hourly, '%Y%m%d')
    datetime_start_hourly = datetime_start_hourly.replace(tzinfo=dt.timezone.utc)
    datetime_end_hourly = datetime_end_hourly.replace(tzinfo=dt.timezone.utc)
    
    print('datetime_start_hourly: ', datetime_start_hourly)
    
    prep = PrepareDataFrames(
    variables = ['obs', 'tide_py'],
    stations = hlp.get_norwegian_station_names(), #['NO_OSC', 'NO_AES', 'NO_BGO', 'NO_HEI', 'NO_KSU'], 
    ensemble_members=[-1],
    datetime_start_hourly = datetime_start_hourly,
    datetime_end_hourly = datetime_end_hourly,
    horizon = 60 + 1, #1,  # horizon in prepare_df is an int +1!!! Lags are computed in self.generate_feature_array
    use_station_data = True,
    run_on_ppi = True,
    new_station_loc=True,
    era5_forcing=False,
    add_aux_variables=False,
    data_from_kartverket=False,
    forecast_mode=True,
    hourly=False,  # Otherwise make 12-hourly data at 00 and 12 hours.
    use_existing_files=False
    )        
    
    df = prep.generate_obs_tide_df_operational(['obs', 'tide_py'])
    
    #my_dict = { 'df_arome_and_waves_5_stations' : df}
    my_dict = { 
        'df_hourly_t0' : df, 
        }
    #my_dict = { 'df_era5_5_stations' : df}

    path_to_dict = (
        '/lustre/storeB/project/IT/geout/machine-ocean/workspace/paulinast'
        + '/storm_surge_results/data_preprocess_input/5_stations/monthly'
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
    datetime_start_hourly = dt.datetime.strptime(datetime_start_hourly, '%Y%m%d')
    datetime_end_hourly = dt.datetime.strptime(datetime_end_hourly, '%Y%m%d')
    datetime_start_hourly = datetime_start_hourly.replace(tzinfo=dt.timezone.utc)
    datetime_end_hourly = datetime_end_hourly.replace(tzinfo=dt.timezone.utc)
    
    print('datetime_start_hourly: ', datetime_start_hourly)

    prep = PrepareDataFrames(
        variables = ['stormsurge_corrected'],
        stations = hlp.get_norwegian_station_names(), #['NO_OSC', 'NO_AES', 'NO_BGO', 'NO_HEI', 'NO_KSU'], 
        ensemble_members=[-1],
        datetime_start_hourly = datetime_start_hourly,
        datetime_end_hourly = datetime_end_hourly,
        horizon = 60 + 1, #1,  # horizon in prepare_df is an int +1!!! Lags are computed in self.generate_feature_array
        use_station_data = True,
        run_on_ppi = True,
        new_station_loc=True,
        era5_forcing=False,
        add_aux_variables=False,
        data_from_kartverket=False,
        forecast_mode=True,
        hourly=False,  # Otherwise make 12-hourly data at 00 and 12 hours.
        use_existing_files=False,
        past_forecast=24
    )       

    df = prep.prepare_features_labels_df()
    df_hourly_t0 = prep.df_hourly_t0

    my_dict = { 
        'df_12_hours' : df, 
        'df_hourly_t0' : df_hourly_t0
        }
    
    path_to_dict = (
        '/lustre/storeB/project/IT/geout/machine-ocean/workspace/paulinast'
        + '/storm_surge_results/data_preprocess_input/5_stations/monthly'
        )
    
    with open(
            path_to_dict 
            + '/dict_df_all_stations_stormsurge_corrected_kyststasjoner_past_forecasts_24_12_hours_'
            + datetime_start_hourly.strftime('%Y%m%d') 
            + '.pickle', 
            'wb'
            ) as handle:
        pickle.dump(my_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
if __name__ == "__main__":
    
    # To run from another script
    args = argparse.parse_arguments()
    main(args)
    
    
    """
    variables = [
        'obs', 'tide_py', 'roms', 'msl', 'u10', 'v10', 
        'stormsurge_corrected', 'bias', 
        'wind_speed', 'wind_dir', 'roms - (obs - tide_py)'
        ]
    
    stations=['NO_OSC'] #, 'SW_2111', 'NL_Westkapelle']
    datetime_start_hourly=dt.datetime(2002, 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc)
    datetime_end_hourly=dt.datetime(2019, 12, 1, 0, 0, 0, tzinfo=dt.timezone.utc)
   
    prep = PrepareDataFrames(
        variables = variables,
        stations = stations,
        datetime_start_hourly = datetime_start_hourly,
        datetime_end_hourly = datetime_end_hourly,
        use_station_data = True,
        new_station_loc=True,
        run_on_ppi = True,
        era5_forcing=False,
        add_aux_variables=False
        )
            
    df = prep.prepare_features_labels_df()
    
    print('df.columns: ', df.columns)
    print(df.head())
    
    """
    
    
    
    
    """
    # Use this to generate monthly files to be concatenated later
    
    datetime_start_hourly = dt.datetime(2018, 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc)
    datetime_end_hourly = dt.datetime(2018, 1, 31, 12, 0, 0, tzinfo=dt.timezone.utc)

    prep = PrepareDataFrames(
        variables = ['msl', 'u10', 'v10'],
        stations = ['NO_OSC', 'NO_AES', 'NO_BGO', 'NO_HEI', 'NO_KSU'],
        ensemble_members=[-1],
        #datetime_start_hourly = dt.datetime(2018, 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc)
        #datetime_end_hourly = dt.datetime(2021, 3, 31, 12, 0, 0, tzinfo=dt.timezone.utc)
        datetime_start_hourly = datetime_start_hourly,
        datetime_end_hourly = datetime_end_hourly,
        
        horizon = 60 + 1, #1,  # horizon in prepare_df is an int +1!!! Lags are computed in self.generate_feature_array
        use_station_data = True,
        run_on_ppi = True,
        new_station_loc=True,
        era5_forcing=False,
        add_aux_variables=False,
        data_from_kartverket=False,
        forecast_mode=True
        )        
        
    df = prep.prepare_features_labels_df()
    
    my_dict = { 'df_arome_5_stations' : df}

    path_to_dict = '/lustre/storeB/project/IT/geout/machine-ocean/workspace/paulinast/storm_surge_results/data_preprocess_input/5_stations/monthly'
    
    with open(path_to_dict + '/dict_df_5_stations_only_arome_12_hours' + datetime_start_hourly.strftime('%Y%m%d') + '.pickle', 'wb') as handle:
        pickle.dump(my_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
 
    """
    
    
    """
    datetime_start_hourly = dt.datetime(2018, 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc)
    datetime_end_hourly = dt.datetime(2021, 3, 31, 12, 0, 0, tzinfo=dt.timezone.utc)

    prep = PrepareDataFrames(
        variables = ['stormsurge_corrected', 'obs', 'tide_py', 'stormsurge_corrected - (obs - tide_py)' ],
        stations = ['NO_OSC'],
        ensemble_members=[-1],
        #datetime_start_hourly = dt.datetime(2018, 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc)
        #datetime_end_hourly = dt.datetime(2021, 3, 31, 12, 0, 0, tzinfo=dt.timezone.utc)
        datetime_start_hourly = datetime_start_hourly,
        datetime_end_hourly = datetime_end_hourly,
        
        horizon = 60 + 1, #1,  # horizon in prepare_df is an int +1!!! Lags are computed in self.generate_feature_array
        use_station_data = True,
        run_on_ppi = True,
        new_station_loc=True,
        era5_forcing=False,
        add_aux_variables=False,
        data_from_kartverket=False,
        forecast_mode=True,
        hourly=False
        )        
        
    df = prep.prepare_features_labels_df()
    
    my_dict = { 'df_kyststasjoner_osc' : df}

    path_to_dict = '/lustre/storeB/project/IT/geout/machine-ocean/workspace/paulinast/storm_surge_results/data_preprocess_input/5_stations/monthly'
    
    with open(path_to_dict + '/dict_df_osc_kyststasjoner_12_hours.pickle', 'wb') as handle:
        pickle.dump(my_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)    
    
    """
    """
    prep = PrepareDataFrames(
        variables = ['msl', 'u10', 'v10'],
        stations = ['NO_OSC', 'NO_AES'],
        ensemble_members=[-1],
        #datetime_start_hourly = dt.datetime(2018, 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc)
        #datetime_end_hourly = dt.datetime(2021, 3, 31, 12, 0, 0, tzinfo=dt.timezone.utc)
        datetime_start_hourly = dt.datetime(2018, 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc),
        datetime_end_hourly = dt.datetime(2020, 3,  31, 12, 0, 0, tzinfo=dt.timezone.utc),
        
        horizon = 50 + 1, #1,  # horizon in prepare_df is an int +1!!! Lags are computed in self.generate_feature_array
        use_station_data = True,
        run_on_ppi = True,
        new_station_loc=True,
        era5_forcing=False,
        add_aux_variables=False,
        data_from_kartverket=False,
        forecast_mode=True
        )        
        
    df = prep.prepare_features_labels_df()
    
    my_dict = { 'df_arome_2_stations' : df}

    path_to_dict = '/lustre/storeB/project/IT/geout/machine-ocean/workspace/paulinast/storm_surge_results/data_preprocess_input/5_stations'
    
    with open(path_to_dict + '/dict_df_2_stations_train.pickle', 'wb') as handle:
        pickle.dump(my_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    """