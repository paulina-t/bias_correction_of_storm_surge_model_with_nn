from datetime import timedelta, datetime, timezone
import numpy as np
import xarray as xr
import pandas as pd

import ml_models.utils.helpers as hlp

class ReadArome():
    
    def __init__(self, 
                 dt_start=datetime(2017, 4, 26, 12, 0, 0, tzinfo=timezone.utc), 
                 dt_end=datetime(2021, 3, 31, 12, 0, 0, tzinfo=timezone.utc)
                 ):
        
        self.dt_start = dt_start  
        self.dt_end = dt_end        
        
    def make_url_list_thredds(self, valid_dates=[]):
        """
        Generates a list of date of the archive files.
        
        Parameters
        ----------
        valid_dates : list of datetimes
            If provided, checks that the date of the kyststasjoner files to is
            in valid_dates. The default is false.

        Returns
        -------
        path_list : list of str
            List with the dates of the storm surge paths.

        """
        
        # TODO: Add error if dt_start is not 00 or 12
        
        print('Generating URL list...')
        thredds_url = 'https://thredds.met.no'

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
        
        archive = '/thredds/dodsC/meps25epsarchive'
        url_list = []  
        # Iterate over all days in the period and generate a string every
        # 12 hours.
        
        # We want to start generating AROME 6 hours before the start of the 
        # ROMS predictions so that pressure and wind data are available for the 
        # storm surge predictions.

        #print('dt_start_arome: ', dt_start_arome)
        #print('n_hours: ', n_hours)
        for hr in range(0, n_hours+1, 12):
            date_dt = (dt_start_arome + timedelta(hours=hr))
            date_str = ( 
                date_dt.strftime('%Y%m%d') 
                + 'T' 
                + date_dt.strftime('%H') 
                + 'Z'
                )
            # valid_dates are the dates of the pp features without nans
            if date_dt in valid_dates: 
                if date_dt <= datetime(2020, 2, 4, 0, 0, 0, tzinfo=timezone.utc):
                    file_url = (
                        thredds_url
                        + archive
                        + '/' + date_dt.strftime('%Y')
                        + '/' + date_dt.strftime('%m')
                        + '/' + date_dt.strftime('%d')
                        + '/meps_mbr0_pp_2_5km_'
                        + date_str
                        + '.nc'
                        )
                else:
                    file_url = (
                        thredds_url
                        + archive
                        + '/' + date_dt.strftime('%Y')
                        + '/' + date_dt.strftime('%m')
                        + '/' + date_dt.strftime('%d')
                        + '/meps_det_2_5km_'
                        + date_str
                        + '.nc'
                        )

                url_list.append(file_url)

        return url_list
    
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

        path_list = []  
        # Iterate over all days in the period and generate a string every
        # 12 hours.
        
        # We want to start generating AROME 6 hours before the start of the 
        # ROMS predictions so that pressure and wind data are available for the 
        # storm surge predictions.

        for hr in range(0, n_hours+1, 12):
            date_dt = (dt_start_arome + timedelta(hours=hr))
            date_str = ( 
                date_dt.strftime('%Y%m%d') 
                + 'T' 
                + date_dt.strftime('%H') 
                + 'Z'
                )
            # valid_dates are the dates of the pp features without nans
            if date_dt in valid_dates: 
                if date_dt <= datetime(2019, 12, 31, 0, 0, 0, tzinfo=timezone.utc):
                    data_dir = (
                        '/lustre/storeA/immutable/archive/projects'
                        + '/metproduction/meps_old'
                    )
                else:
                    data_dir = (
                        '/lustre/storeB/immutable/archive/projects'
                        + '/metproduction/MEPS'
                    )
                if date_dt <= datetime(2020, 2, 4, 0, 0, 0, tzinfo=timezone.utc):
                    file_path = (
                        data_dir
                        + '/' + date_dt.strftime('%Y')
                        + '/' + date_dt.strftime('%m')
                        + '/' + date_dt.strftime('%d')
                        + '/meps_mbr0_pp_2_5km_'
                        + date_str
                        + '.nc'
                        )
                else:
                    file_path = (
                        data_dir
                        + '/' + date_dt.strftime('%Y')
                        + '/' + date_dt.strftime('%m')
                        + '/' + date_dt.strftime('%d')
                        + '/meps_det_2_5km_'
                        + date_str
                        + '.nc'
                        )
                path_list.append(file_path)

        return path_list
    
    def vars_to_drop(self):
        variables = [
                        'surface_air_pressure', 
                        'relative_humidity_2m', 
                        'precipitation_amount_acc', 
                        'fog_area_fraction', 
                        'precipitation_amount',
                        'precipitation_amount_high_estimate',
                        'precipitation_amount_low_estimate',
                        'precipitation_amount_middle_estimate',
                        'precipitation_amount_prob_low',
                        'cloud_area_fraction',
                        'high_type_cloud_area_fraction',
                        'low_type_cloud_area_fraction',
                        'medium_type_cloud_area_fraction',
                        'helicopter_triggered_index',
                        'thunderstorm_index_combined',
                        'air_temperature_2m'
                        ]
        return variables
    
    def fill_in_variable(self, data_file_i, var, i, n_columns_to_fill_in, count, x, y):
        
        # Since we consider an AROME forecast that is generated 6 hours before 
        # the storm surge forecast, we need to start filling in data 6 hours 
        # after the start to match the times.
        forecast_start_time = 6
        try:
            if var == 'msl':
                self.data[i, count * (n_columns_to_fill_in): (count + 1) *( n_columns_to_fill_in)] = ( 
                    # Indexing with xarray is slow
                    # TODO: Store selection in array or df and iterate/index it.
                    data_file_i['air_pressure_at_sea_level']
                    .isel(
                        {
                        'time' : slice(forecast_start_time, n_columns_to_fill_in + forecast_start_time), 
                        'height_above_msl': 0, 
                        'y': y,  
                        'x': x
                        } 
                        ).values.squeeze()
                    )
            if var == 'u10':  # TODO: convert to wind speed and wind direction
                
                self.data[i, count * (n_columns_to_fill_in): (count + 1) * (n_columns_to_fill_in)] = ( 
                    data_file_i['x_wind_10m']
                    .isel(
                        {
                        'time' : slice(
                            forecast_start_time, 
                            n_columns_to_fill_in + forecast_start_time
                            ), 
                        'y': y,  
                        'x': x
                        } 
                        ).values.squeeze()
                    )
            if var == 'v10':
                self.data[i, count * (n_columns_to_fill_in) : (count + 1) *( n_columns_to_fill_in)] = ( 
                    data_file_i['y_wind_10m']
                    .isel(
                        {
                        'time' : slice(
                            forecast_start_time, 
                            n_columns_to_fill_in + forecast_start_time
                            ), 
                        'y': y, 
                        'x': x
                        } 
                        ).values.squeeze()
                    )
        except:  
            print('Variable ' + var + ' is not available.')
    
    def make_data_arrays(self, variables, station_names, path_list, hz):
        print('Generating AROME arrays...')
        
        forecast_start_time = 6
        # Define parameters
        l = len(path_list)
        nvars_x_nstations = len(variables) * len(station_names)
        
        self.data = np.full([l, (hz) * nvars_x_nstations], fill_value=np.NaN)
        
        n_columns_to_fill_in = hz
        if n_columns_to_fill_in > 61:  # We have only 67 preditions
            n_columns_to_fill_in = 61
        
        variables_to_drop = self.vars_to_drop()

        # Construct arrays of shape (n_samples, n_outputs)
        for i in range(l): # Iterate over files
            try:
                data_file_i = xr.open_mfdataset(
                    [path_list[i]], 
                    parallel=True, 
                    concat_dim="time", 
                    combine="nested",
                    data_vars='minimal', 
                    coords='minimal', 
                    compat='override', 
                    drop_variables=variables_to_drop
                    )
                
                count = 0
                for station in station_names:
                    lat, lon = hlp.get_station_lat_lon(station)
                    # Get position of grid box for a pair of lat, long
                    x, y = hlp.arome_latlon2xy(
                        grd=data_file_i, 
                        lat_val=lat, 
                        lon_val=lon
                        )   
                    
                    for var in variables:
                        self.fill_in_variable(
                            data_file_i, 
                            var, 
                            i, 
                            n_columns_to_fill_in, 
                            count, 
                            x, 
                            y
                            )
                        count = count + 1             
            except:
                # Keep nans in the files
                print('File ', path_list[i], ' is not available.' )                
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