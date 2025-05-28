from datetime import timedelta, datetime, timezone
import numpy as np
import xarray as xr
import pandas as pd

import ml_storm_surge_operational.utils.helpers as hlp

class ReadForcings():
    
    def __init__(self, 
                 dt_start=datetime(2017, 4, 26, 12, 0, 0, tzinfo=timezone.utc), 
                 dt_end=datetime(2021, 3, 31, 12, 0, 0, tzinfo=timezone.utc)
                 ):
        
        self.dt_start = dt_start  
        self.dt_end = dt_end   
        
    def decode_station_names(self, data_file):
        """
        TODO: Move to utils, also used in main_predict
        Generate a dictionary that maps station names and IDs.

        Parameters
        ----------
        data_file : xarray.Dataset
            Dataset containing kyststasjoner data.

        Returns
        -------
        stations_id : dict
            The keys of the dictionary are the station names and the values are 
            the IDs in the data_file. These ID are not the same in the all files.

        """
        stations_id={}
        n_stations = len(data_file.station)
        for i in range(n_stations):
            name = data_file.station_name[0:3,i].values.astype(str).tolist()
            name = "".join(name)  # Concatenate letters in list 
            name = 'NO_' + name.upper()
            stations_id[name] = i
        return stations_id

    def get_station_id(self, data_file, station_name):
        """
        TODO: Move to utils - also used in main_predict
        Get station id of a particular station in a particular file.

        Parameters
        ----------
        data_file : xarray.Dataset
            Dataset containing kyststasjoner data.
        station_name : str
            Station name (short) including land code. Only norwegian stations
            are valid.

        Returns
        -------
        station_id : int
            station_id

        """
        stations_id = self.decode_station_names(data_file)
        station_id = stations_id[station_name]
        return station_id
    
    def make_path_list_ppi(self, valid_dates=[]):
        """
        Generates a list of date of the archive files.
        
        Parameters
        ----------
        valid_dates : listof datetimes
            If provided, checks that the date of the kystasjoner files to is
            in valid_dates. The deafault is false.

        Returns
        -------
        path_list : list of str
            List with the dates of the storm surge paths.

        """
        
        # TODO: Raise an error if dt_start is not 00 or 12
        
        print('Generating path list...')

        n_hours = ((self.dt_end -  self.dt_start).days)*24  
        
        if self.dt_start.hour > self.dt_end.hour:
            n_hours = n_hours + 24

        if self.dt_start.hour == 6:
            #print('dt_start.hour == 6')
            n_hours = n_hours - 6
        if self.dt_start.hour == 12:
            #print('dt_start.hou r == 12')
            n_hours = n_hours - 12
        if self.dt_start.hour == 18:
            #print('dt_start.hour == 18')
            n_hours = n_hours - 18
        if self.dt_end.hour == 6:
            n_hours = n_hours + 6
            #print('dt_end.hour == 6')
        if self.dt_end.hour == 12:
            n_hours = n_hours + 12
            #print('dt_end.hour == 12')
        if self.dt_end.hour == 18:
            n_hours = n_hours + 18
            #print('dt_end.hour == 18')
        
        # Start six hours before the storm surge forecast
        dt_start_arome = self.dt_start - timedelta(hours=6)
        if not valid_dates:
            valid_dates = [dt_start_arome + timedelta(hours=i) for i in range(n_hours + 1)]

        # Iterate over all days in the period and generate a string every
        # 12 hours.
        
        # We want to start generating forcings 6 hours before the start of the 
        # ROMS predictions so that pressure and wind data are available for the 
        # storm surge predictions.
        data_dir = '/lustre/storeB/project/metproduction/products/util/vannstand/'
        file_root_name = 'force_ml_kyststasjoner_norge.nc'
        path_list = []  
        for hr in range(0, n_hours+1, 12):
            date_dt = (dt_start_arome + timedelta(hours=hr))
            date_str = date_dt.strftime('%Y%m%d%H')
            # valid_dates are the dates of the pp features without nans
            if date_dt in valid_dates: 
                path_list.append(
                    data_dir 
                    + file_root_name 
                    + date_str
                    )
        return path_list
    
    def fill_in_variable(self, data_file_i, var, i, n_columns_to_fill_in, count, station_nr):
        
        # Since we consider a weather forecast that is generated 6 hours before 
        # the storm surge forecast, we need to start filling in data 6 hours 
        # after the start to match the times.

        #print("data_file_i['Pair']: ", data_file_i['Pair'])
        forecast_start_time = 6
        #print('n_columns_to_fill_in', n_columns_to_fill_in)
        #print('station_nr', station_nr)
        
        pair = data_file_i['Pair'].isel(
                    {
                    'station':station_nr, 
                    'time':slice(
                        forecast_start_time, 
                        n_columns_to_fill_in + forecast_start_time
                        ),  # * 6 because we have 10 min data in the new files
                    'ensemble_member': -1, # No support for this yet #self.ensemble_member, 
                    'dummy':0
                    }
                    ).values.squeeze()
        #print('pair', pair)    
        #try:
        if var == 'msl':
            self.data[i, count * (n_columns_to_fill_in): (count + 1) *( n_columns_to_fill_in)] = ( 
                # Indexing with xarray is slow
                # TODO: Store selection in array or df and iterate/index it.
                data_file_i['Pair']
                .isel(
                    {
                    'station':station_nr, 
                    'time':slice(
                        forecast_start_time, 
                        n_columns_to_fill_in + forecast_start_time,
                        ),  # * 6 because we have 10 min data in the new files
                    'ensemble_member': -1, # No support for this yet #self.ensemble_member, 
                    'dummy':0
                    }
                    ).values.squeeze()
                )
                
        if var == 'u10':  # TODO: convert to wind speed and wind direction
            self.data[i, count * (n_columns_to_fill_in): (count + 1) * (n_columns_to_fill_in)] = ( 
                data_file_i['Uwind']
                .isel(
                    {
                    'station':station_nr, 
                    'time':slice(
                        forecast_start_time, 
                        n_columns_to_fill_in + forecast_start_time,
                        ),  # * 6 because we have 10 min data in the new files
                    'ensemble_member': -1, # No support for this yet #self.ensemble_member, 
                    'dummy':0
                    }
                    ).values.squeeze()
                )
        if var == 'v10':
            self.data[i, count * (n_columns_to_fill_in) : (count + 1) *( n_columns_to_fill_in)] = ( 
                data_file_i['Vwind']
                .isel(
                    {
                    'station':station_nr, 
                    'time':slice(
                        forecast_start_time, 
                        n_columns_to_fill_in + forecast_start_time,
                        ),  # * 6 because we have 10 min data in the new files
                    'ensemble_member': -1, # No support for this yet #self.ensemble_member, 
                    'dummy':0
                    }
                    ).values.squeeze()
                )
        #except:  
        #    print('Variable ' + var + ' is not available.')
    
    def make_data_arrays(self, variables, station_names, path_list, hz):
        print('Generating AROME arrays...')
        
        forecast_start_time = 6
        # Define parameters
        l = len(path_list)
        print('l: ', l)
        print('variables: ', variables)
        print('station_names: ', station_names)
        nvars_x_nstations = len(variables) * len(station_names)
        
        self.data = np.full([l, (hz) * nvars_x_nstations], fill_value=np.NaN)
        
        n_columns_to_fill_in = hz
        if n_columns_to_fill_in > 61:  # We have only 67 preditions
            n_columns_to_fill_in = 61
        
        #variables_to_drop = self.vars_to_drop()
        
        # Construct arrays of shape (n_samples, n_outputs)
        for i in range(l): # Iterate over files
            print('i: ', i)
            #try:
            data_file_i = xr.open_dataset(path_list[i])
            station_nrs = []
            for station_name in station_names:
                station_nrs.append(self.get_station_id(data_file_i, station_name))
            count = 0
            #for station in station_names:
            for station_nr in station_nrs:
                print('station_nr: ', station_nr)
                for var in variables:
                    print('var: ', var)
                    self.fill_in_variable(
                        data_file_i, 
                        var, 
                        i, 
                        n_columns_to_fill_in, 
                        count, 
                        station_nr
                        )
                    count = count + 1     
                    
        #print('self.data: ', self.data)
        #print('self.data.shape: ', self.data.shape)
            #except:
            #    # Keep nans in the files
            #    print('File ', path_list[i], ' is not available.' )                
        return self.data
    
    def array_to_df(self, array, index, col_names):
        df = pd.DataFrame(data=array, index=index, columns=col_names)
        return df
    
    def make_hz_col_names(self, variables, stations, hz,  fhr):
        col_names = [] 
        if hz > 61:
            hz = 61
        past_forecast_str = ''
        if fhr != 0:
            past_forecast_str = '_' + str(fhr)
        for station in stations:
            for var in variables:
                for h in range(fhr, hz):  # We start storing data from lead time 6, but we want to call this 0
                    if h == 0:
                        col_name_h = var + past_forecast_str +'_' + station
                        col_names.append(col_name_h)
                    else:
                        col_name_h = (
                            var 
                            + '_t' 
                            + str(h) 
                            + past_forecast_str 
                            + '_' 
                            + station
                        )
                        col_names.append(col_name_h)                
        return col_names
    
    def make_time_idx(self):
        delta = timedelta(hours=12)
        date = self.dt_start
        
        date_list = []
        
        while date <= self.dt_end:
            date_list.append(date)
            date += delta
        return date_list  
   

if __name__ == "__main__":  
    # Make 12hr - DataFrame with data from 5 stations and save dict
    import pickle
    dt_start = datetime(2018, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    dt_end = datetime(2019, 3, 31, 12, 0, 0, tzinfo=timezone.utc)
    #dt_end = datetime(2021, 3, 31, 12, 0, 0, tzinfo=timezone.utc)


    ra =ReadArome(dt_start, dt_end)    
        
    path_list = ra.make_path_list_ppi()
    times = ra.make_time_idx()
    variables = ['u10'] #['msl', 'u10', 'v10']
    
    first_iter = True

    station_names = ['NO_OSL'] # ['NO_OSC', 'NO_AES', 'NO_BGO', 'NO_HEI', 'NO_KSU']
    data_array = ra.make_data_arrays(variables, station_names, path_list, 60)
    col_names = ra.make_hz_col_names(variables, station_names, 60, fhr=0)
    df_station = ra.array_to_df(data_array, times, col_names)      

    #arome_dict = {'df_arome_5_stations' : df_station}    
    arome_dict = {'df_arome_OSL' : df_station}   
    
    data_dir = '/lustre/storeB/project/IT/geout/machine-ocean/workspace/paulinast/storm_surge_results/data_preprocess_input/5_stations/monthly'
    #file_name = 'dict_df_5_stations_only_arome_all_months_12_hours.pickle'
    file_name = 'dict_df_OSL_only_arome_all_months_12_hours.pickle'
    with open(data_dir + '/' + file_name, 'wb') as handle:
        pickle.dump(arome_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)