import numpy as np
import xarray as xr
from datetime import datetime, timedelta
import os
import pandas as pd
import pickle
import copy

from pathlib import Path


def verboseprint1(verbose, *args):
        # Print each argument separately so caller doesn't need to
        # stuff everything to be printed into a single string
        if verbose > 0:
            for arg in args:
               print(arg)          
       
def verboseprint2(verbose, *args):
    # Print each argument separately so caller doesn't need to
    # stuff everything to be printed into a single string
    if verbose > 1:
        for arg in args:
           print(arg)
               
           
def split_df(df, datetime_split):
    """Split provided DataFrame in train and test data.

    Parameters
    ----------
    df : DataFrame
        DataFrame to split.
    datetime_split : datetime
        Datetime that defines the split between train and test. The test 
        data begins the next available time after the datetime_split.

    Returns
    -------
    train : DataFramse
        Train data.
    test : DataFrame
        Test data.

    """
    #index = df.index.get_loc(
    #    datetime_split, 
    #    method='backfill'
    #    )
    #train  = df[:index]
    #test = df[index:]   
    
    train = df.loc[df.index < datetime_split]
    test = df.loc[df.index >= datetime_split]
    return train, test
       
def dump_df_with_pkl(df, path):
    """Dump DataFrame.
    
    Parameters
    ----------
    df : DataFramse
        DataFrame to dump.
    path : str
        Full path of the dataframe to dump.

    Returns
    -------
    None.

    """
    with open(path, 'wb') as handle:
        pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
def load_and_subset_df_with_pkl(path, columns=[]):
    """Load DataDrame and subset columns.    

    Parameters
    ----------
    path : str
        Full path to the DataFrame to load.
    columns : list of str, optional
        The columns to subset. If the list is empty, all columns will be 
        retrieved. The default is [].

    Returns
    -------
    df_subset : TYPE
        DESCRIPTION.

    """
    with open(path, 'rb') as handle:
        df_subset = pickle.load(handle)
    
    if columns:
        df_subset = df_subset[columns]
        
    return df_subset

def make_dir(path):
    """
    Recursively creates the directory and does not raise an exception if the 
    directory already exists.

    Parameters
    ----------
    path : str
        Path to the directory to be created.

    Returns
    -------
    None.

    """
    if not os.path.exists(path):
        Path(path).mkdir(parents=True, exist_ok=True)
    


def get_station_lat_lon(station_name:str):
        """
        Get station latitude and longitude from station_metadata.csv. 
        station_metadata.csv was generated from the WAM3 NORA3-ERA5 station
        file using station_metadata_to_csv.py.

        Parameters
        ----------
        station_name : str
            Station name (stationid), either a feature or a label station.

        Returns
        -------
        lat : float
            Latitude of the station.
        lon : float
            Longitude of the station

        """
        fpath = os.path.join(os.path.dirname(__file__), '..', 'data', 'station_metadata.csv')
        station_metadata = pd.read_csv(fpath)

        condition = station_metadata['stationid']==station_name
        #print("Condition: ", condition)
        #print("Condition.sum(): ", condition.sum())
        #index = condition.index[condition == True][0] 
        #print("Index: ", index)
        lat = station_metadata['latitude_station'][condition].item()  # Does not work in newer version
        lon = station_metadata['longitude_station'][condition].item()
        return lat, lon
    
    
def get_station_nr_from_id(station_name):
    """
    Gets the station number in the NetCDF files to the provided station name.

    Parameters
    ----------
    station_name : str
        Station id including land code.

    Returns
    -------
    station_nr : int
        The correspoding station number in the NetCDF files.

    """
    
    # Open csv with station nrs. and find the row that matches the station_name
    dirname = os.path.dirname(__file__)
    path_stationid_obs = os.path.join(dirname, '../data/station_ids_obs_roms_files.csv')
# =============================================================================
#     path_stationid_obs = os.path.join( 
#         os.getcwd(), 
#         '..', 
#         'data', 
#         'station_ids_obs_roms_files.csv' 
#         )
# =============================================================================
    stationid_obs = pd.read_csv(path_stationid_obs)
    row_in_stationid_obs = stationid_obs[stationid_obs['station_name']==station_name]  
    station_nr = row_in_stationid_obs['station_nr'].values[0] 
    
    return station_nr


def get_norwegian_station_names():
    """
    Get list with the Norwegian Station IDs.
    
    Returns
    -------
    norwegian_stations : list of str
        List containing the IDs of the Norwegian staions.

    """
    norwegian_stations = [
            'NO_AES', 'NO_ANX', 'NO_BGO', 
            'NO_BOO', 'NO_HAR', 'NO_HEI', 'NO_HFT', 'NO_HRO', 'NO_HVG', 'NO_KAB', 
            'NO_KSU', 'NO_MAY', 'NO_MSU', 'NO_NVK', 'NO_NYA', 'NO_OSC', 'NO_OSL', 
            'NO_RVK', 'NO_SVG', 'NO_TOS', 'NO_TRD', 'NO_TRG', 'NO_VAW', 'NO_VIK'
            ]
    return norwegian_stations

def get_all_station_names():
    """
    Get list with all the station IDs.

    Returns
    -------
    all_stations : list of str
        List containing the IDs of all the staions.

    """
    all_stations = [
        'DE_AlteWeser',     'DE_Eider',         'DE_Husum',     
        'DE_Wittduen',      'DE_Buesum',        'DE_Helgoland',     
        'DE_Norderney',     'DE_Cuxhaven',      'DE_Hoernum',       
        'DE_Wangerooge',    'NL_Amelander Westgat platform', 
        'NL_Aukfield platform',                 'NL_Brouwershavensche Gat 08', 
        'NL_Brouwershavensche Gat, punt 02',    'NL_Cadzand',   
        'NL_Delfzijl',      'NL_Den Helder',    'NL_Engelsmanplaat noord',      
        'NL_Euro platform', 'NL_F3 platform',   'NL_Haringvliet 10',            
        'NL_Haringvlietmond',                   'NL_Harlingen',     
        'NL_Holwerd',       'NL_Horsborngat',   'NL_Huibertgat',    
        'NL_IJmuiden stroommeetpaal',           'NL_K13a platform', 
        'NL_K14 platform',  'NL_L9 platform',   'NL_Lauwersoog', 
        'NL_Lichteiland Goeree',                'NL_Maasmond stroommeetpaal', 
        'NL_Nes',           'NL_Noordwijk meetpost', 
        'NL_North Cormorant',                   'NL_Oosterschelde 11', 
        'NL_Oosterschelde 13',                  'NL_Oosterschelde 14', 
        'NL_Oosterschelde 15',                  'NL_Oudeschild',
        'NL_Platform A12',                      'NL_Platform D15-A', 
        'NL_Platform F16-A',                    'NL_Platform Hoorn Q1-A', 
        'NL_Platform J6',                       'NL_Schiermonnikoog', 
        'NL_Terschelling Noordzee',             'NL_Texel Noordzee', 
        'NL_Vlakte van de Raan',                'NL_Vlieland haven', 'NL_West-Terschelling', 
        'NL_Westkapelle',                       'NL_Wierumergronden', 
        'WL_draugen',       'WL_ekofisk',       'WL_heidrun', 
        'WL_heimdal',       'WL_sleipner',      'WL_veslefrikk', 
        'DK_4203',          'DK_4201',          'DK_4303', 
        'DK_5103',          'DK_5104',          'DK_5203', 
        'DK_7101',          'DK_7102',          'DK_6401', 
        'DK_6501',          'NO_AES',           'NO_ANX', 
        'NO_BGO',           'NO_BOO',           'NO_HAR', 
        'NO_HEI',           'NO_HFT',           'NO_HRO', 
        'NO_HVG',           'NO_KAB',           'NO_KSU', 
        'NO_MAY',           'NO_MSU',           'NO_NVK',   
        'NO_NYA',           'NO_OSC',           'NO_OSL', 
        'NO_RVK',           'NO_SVG',           'NO_TOS', 
        'NO_TRD',           'NO_TRG',           'NO_VAW', 
        'NO_VIK',           'SW_2130',          'SW_2109', 
        'SW_2111',          'SW_35104',         'SW_35144', 
        'UK_Sheerness',     'UK_Aberdeen',      'UK_Whitby', 
        'UK_Lerwick',       'UK_Newhaven',      'UK_Lowestoft',     
        'UK_Wick'
        ]
    
    return all_stations

def copy_stationid_var():
    """
    Copy variable station_id from the observation NetCDF file prepared by Jean.
    This variable will be used in other files that will later be concatenated.

    Returns
    -------
    stationid_var : xarray DataArray
        Variable station_id

    """
    path_to_obs_file = (
            '/lustre/storeB/project/IT/geout/machine-ocean'
            +'/prepared_datasets/storm_surge/aggregated_water_level_data'
            + '/aggregated_water_level_observations_with_pytide_prediction_dataset.nc'
            )
        
    obs_file = xr.open_dataset(path_to_obs_file)
    stationid_var = obs_file.stationid.copy(deep=True)
    return stationid_var


def update_attr_history(nc_file, message:str):
    """
    Update the history attribute with the message passed.

    Parameters
    ----------
    nc_file : NetCDF
        NetCDF file that has been modified.
    message : str
        String describing the modification of the NetCDF file.

    Returns
    -------
    nc_file : NetCDF
        NetCDF file with updated history attribute.

    """
    new_log = str(datetime.utcnow()) + ' Python ' + message + '.'
   
    if 'history' in list(nc_file.attrs):
        old_log = nc_file.history
    else:
        old_log = ''
        
    nc_file.history = old_log + new_log
    
    return nc_file

def wind_speed(u, v):        
    """
    Computes the wind speed from the zonal and meridional wind components.

    Parameters
    ----------
    u : np.aarray
        Zonal wind.
    v : np.array
        Meridional wind

    Returns
    -------
    ws : np.array
        Wind speed.

    """
    ws = np.sqrt(u**2 + v**2)
    return ws
    
def wind_dir(u, v):
    """
    Computes the wind direction from the zonal and meridional wind components.

    Parameters
    ----------
    u : np.aarray
        Zonal wind.
    v : np.array
        Meridional wind

    Returns
    -------
    wd : np.array
        Wind direction

    """
    wd = np.degrees(np.arctan2(-v, -u))
    return wd

def crop_strings(string:str):
    """
    Removes white spaces and parenthesis in strings. Useful when defining
    paths based on already defined strings.

    Parameters
    ----------
    string : str

    Returns
    -------
    str
    String without white spaces and parenthesis.

    """
    return string.replace('(', '').replace(')', '').replace(' ', '') 

def rmse_by_ranges(y_true, y_pred, by_ranges=True, vector_rmse=False):
    """Computation of the Root Mean Square Error (RMSE).
    
    The method computes the RMSE, either for the whole series or for 'low', 
    'middle', and 'high' ranges separately. The method can also compute the 
    RMSE columnwise if specified by the argument vector_rmse.
    

    Parameters
    ----------
    y_true : array
        Vector or matrix with the ground truth.
    y_pred : array
        Vector or matrix with the predictions.
    by_ranges : bool, optional
        If true, divides the data in ranges according to the ground truth. The
        'low' range corresponds to values lower than one standard deviation, 
        the 'high' range corresponds to values higher than two standard 
        deviations and the 'middle' range corresponds to the observations in 
        the middle range. In this case the method returns a dict with all the
        RMSE values instead of an array.
    vector_rmse : bool, optional
        If true, returns a vector with all the RMSE values. The default is 
        False.

    Returns
    -------
    rmse : array, or dict of arrays
        Returns the RMSE. If by_ranges is true, it returns a dict with the RMSE
        computed for each range.
    """
    
    if by_ranges: # -> returns dictionary
        s = np.std(y_true)
        ranges = ['low', 'middle', 'high']  # Ranges of observed SSH
        
        for r in ranges:
    
            # Create new variables
            y_true_r = copy.deepcopy(y_true)
            y_pred_r = copy.deepcopy(y_pred)
            
            # Put Nans where the observed ssh is not in the range of 
            # consideration
            if r == 'low':
                y_true_r[y_true_r > s] = np.nan
                y_pred_r[y_true_r > s] = np.nan
            elif r == 'middle':
                y_true_r[y_true_r <= s] = np.nan
                y_true_r[y_true_r > 2*s] = np.nan
                y_pred_r[y_true_r <= s] = np.nan
                y_pred_r[y_true_r > 2*s] = np.nan
            elif r == 'high':
                y_true_r[y_true_r <= 2*s] = np.nan
                y_pred_r[y_true_r <= 2*s] = np.nan
                
            # Compute the RMSE for each range.
            rmse = {}     
            if vector_rmse:
                rmse[r] = np.sqrt(np.nanmean((y_true_r - y_pred_r)**2, axis=0))
            else:
                rmse[r] = np.sqrt(np.nanmean((y_true - y_pred)**2))
                
    else:  # not by ranges -> returns numpy array
        if vector_rmse:
            rmse[r] = np.sqrt(np.nanmean((y_true_r - y_pred_r)**2, axis=0))
        else:
            rmse[r] = np.sqrt(np.nanmean((y_true - y_pred)**2))
            
    return rmse
                
        


def rmse(y_true, y_pred):
    """
    Computes the Root Mean Squared Error for the entire period.
y_true_low = copy.deepcopy(y_true)
        y_true_middle = copy.deepcopy(y_true)
        y_true_high = copy.deepcopy(y_true)
    Parameters
    ----------
    y_true : array
    y_pred : array

    Returns
    -------
    scalar

    """
    
    return np.sqrt(np.nanmean((y_true - y_pred)**2))

def rmse_vector(y_true, y_pred):
    """
    Computes the Root Mean Squared Error for the entire period computed along 
    axis=0.

    Parameters
    ----------
    y_true : array
    y_pred : array

    Returns
    -------
    scalar

    """
    
    return np.sqrt(np.nanmean((y_true - y_pred)**2, axis=0))

def bias_vector(y_true, y_pred):
    """
    Computes the bias for the entire period computed along 
    axis=0.

    Parameters
    ----------
    y_true : array
    y_pred : array

    Returns
    -------
    scalar

    """
    
    return np.nanmean((y_true - y_pred), axis=0)

def std_vector(y_true, y_pred):
    """
    Computes the bias for the entire period computed along 
    axis=0.

    Parameters
    ----------
    y_true : array
    y_pred : array

    Returns
    -------
    scalar

    """
    
    return np.nanstd((y_true - y_pred), axis=0)

def mae(y_true, y_pred):
    """
    Computes the Mean Absolute Error for the entire period.

    Parameters
    ----------
    y_true : array
    y_pred : array

    Returns
    -------
    scalar

    """
    
    return np.mean(np.abs(y_true - y_pred))


def BFS(start_i, start_j, crit=1):
    """
    Code from josteinb@met.no

    desc:
        Breadth first search function to find index of nearest
        point with crit value (default crit=1 for finding ROMS
        wet-point in mask)
    args:
        - start_i: Start index of i
        - start_j: Start index of j
        - crit: value to search for (deafult unmasked point)
    return:
        - index of point
    """
    dirs    = [(1,0), (-1,0), (0,1),(0,-1)]
    visited = set()
    q       = [(start_i, start_j)]    # init queue to start pos
    count   = 0
    mask = xr.open_dataset('~/Desktop/data/ROMS/Run2000_2019/land_sea_mask.nc')
    
    # while something in queue
    while q:
        current = q.pop(0)      # pop the first in waiting queue
        # if we have visited this before
        if current in visited:
            continue
        visited.add(current)    # Add to set of visited
        # If not in border list
        # Test if this is land, if true go to next in queue, else return idx
        if mask[current[0], current[1]] == crit:
            return current[0], current[1]
        count += 1      #updates the count
        # Loop over neighbours and add to queue
        for di, dj in dirs:
            new_i = current[0]+di
            new_j = current[1]+dj
            q.append((new_i, new_j))
            
def roms_latlon2xy(grd, lat_val, lon_val, lat_coord='lat', lon_coord='lon', 
                   roundvals=True):
    """
    Function taken from ROMStools.py.
    Author: Nils M. Kristensen
    
    https://gitlab.met.no/ocean-ice/pyromstools/-/blob/master/pyromstools/ROMStools.py
    
    Parameters
    ----------
    grd : TYPE
        DESCRIPTION.
    lat_val : TYPE
        DESCRIPTION.
    lon_val : TYPE
        DESCRIPTION.
    lat_coord : TYPE, optional
        DESCRIPTION. The default is 'lat'.
    lon_coord : TYPE, optional
        DESCRIPTION. The default is 'lon'.
    roundvals : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    x_dim : TYPE
        DESCRIPTION.
    y_dim : TYPE
        DESCRIPTION.

    """
    a = abs( grd[lat_coord]-lat_val ) + abs( grd[lon_coord]-lon_val)    
    y_dim, x_dim= np.unravel_index(a.argmin(), a.shape)   
    if roundvals:
        x_dim= np.int(np.round(x_dim))
        y_dim = np.int(np.round(y_dim))
    return x_dim, y_dim
    
    
    


def arome_latlon2xy(grd, lat_val, lon_val, lat_coord='latitude', lon_coord='longitude', 
                   roundvals=True):
    """
    

    Parameters
    ----------
    grd : TYPE
        DESCRIPTION.
    lat_val : TYPE
        DESCRIPTION.
    lon_val : TYPE
        DESCRIPTION.
    lat_coord : TYPE, optional
        DESCRIPTION. The default is 'lat'.
    lon_coord : TYPE, optional
        DESCRIPTION. The default is 'lon'.
    roundvals : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    x_dim : TYPE
        DESCRIPTION.
    y_dim : TYPE
        DESCRIPTION.

    """
    a = abs( grd[lat_coord]-lat_val ) + abs( grd[lon_coord]-lon_val)    
    y_dim, x_dim= np.unravel_index(a.argmin(), a.shape)   
    if roundvals:
        x_dim= np.int(np.round(x_dim))
        y_dim = np.int(np.round(y_dim))
    return x_dim, y_dim