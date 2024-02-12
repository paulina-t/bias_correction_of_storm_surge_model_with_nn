"""Module containing assertions for the package."""

import numpy as np
import collections
import itertools

def ras(assertion,  message=""):
    """Raise on failed assertion.

    Parameters
    ----------
    assertion : bool
        The assertion to check for.
    message : str, optional
        The message to display as the Exception
        content if assertion fails. The default is "".

    Raises
    ------
    an
        DESCRIPTION.
    Exception
        If assertion is True, do nothing. If False, raise an
        Exception with the custom message.

    Returns
    -------
    None.

    """
    if not assertion:
        raise Exception(message)


def assert_norwegian_station_id(station_id:str):
    """
    Assert that the station ID corresponds to a Norwegian station.

    Parameters
    ----------
    station_id : str
        Station ID.

    Returns
    -------
    None.

    """
    """Assert that date_in is an UTC datetime."""
    
    land_code = station_id.split('_')[0]
    ras(land_code == 'NO', "Not a Norwegian station.")


def assert_is_numeric(x):
    """
    Assert that the argument passed is a number.

    Parameters
    ----------
    x : 
        Variable to check.

    Returns
    -------
    None.

    """
    ras(isinstance(x, (int, float)), 'Has to be a number.')
    
    
def assert_columns_in_prepared_dataframe(dataframe, stations, variables):
    """
    Check that the dataframe has the expected columns.

    Parameters
    ----------
    dataframe : DataFrame
        Dataframe to control.
    stations : list of str
        List containing the stations ID of the data in the DataFrame.
    variables : list of str
        List containing the variable names of the data in the DataFrame.

    Returns
    -------
    None.

    """
# =============================================================================
#     if 'roms - (obs - tide_py)' in variables:
#         variables.append('roms')
#         variables.append('obs')
#         variables.append('tide_py')
#         variables.append('(obs - tide_py)')
#         
#     else:
#         if '(obs - tide_py)' in variables:
#             variables.append('obs')
#             variables.append('obs')
# =============================================================================
            
    # Remove duplicates
    variables = list(set(variables))
    
    
        
    permutations_station_variable = list(itertools.product(variables, stations))
    columns = ['_'.join(perm_tuple) for perm_tuple in permutations_station_variable]
    
# =============================================================================
#     not_existing_variables = list(
#         set(dataframe.columns).difference(columns)
#         )
#     str_not_existing_variables = ', '.join(not_existing_variables)
#     
#     sorted_df_cols = collections.Counter(dataframe.columns.sort_values())
#     print('sorted_df_cols: ', sorted_df_cols )
#     
#     sorted_cols = collections.Counter(np.sort(columns))
#     print('sorted_cols: ', sorted_cols )
# =============================================================================
    
    ras(
        set(columns) == set(dataframe.columns),
        'The DataFrame does not have the expected columns.'
        #sorted_df_cols == sorted_cols , 
        #str_not_existing_variables + 'The DataFrame does not have the expected columns.'
        )
    
    
def assert_dt_start_less_than_dt_end(dt_start, dt_end):
    """
    Check that datetime start is less than datetime end.

    Parameters
    ----------
    dt_start : datetime
    dt_end : datetime

    Returns
    -------
    None.

    """
    message = 'Datetime start is greater than datetime end.'
    ras(dt_start <= dt_end, message)
    
    
def assert_variables_exist(variables):
    """
    Check if the variables exist in the original files.

    Parameters
    ----------
    variables : list of str
        List of variables to check.

    Returns
    -------
    None.

    """
    all_possible_variables = [
        'obs', 'tide_py', 'roms', 
        '(obs - tide_py)', 'roms - (obs - tide_py)', 
        'msl', 'u10', 'v10', 'swh', 'mwd',
        'pz0', 'x_wind_10m', 'y_wind_10m', 'wdir', 'tp',
        'stormsurge', 'stormsurge_corrected', 'bias',
        'wind_speed', 'wind_dir',
        'stormsurge_corrected - (obs - tide_py)', 
        'stormsurge - (obs - tide_py)',
        '(roms - biasMET) - (obs - tide_py)',
        '(roms - biasMET)'
        ]
    
    not_existing_variables = list(
        set(variables).difference(all_possible_variables)
        )
    
    assertion = True
    str_not_existing_variables = ''
    if not_existing_variables:
        assertion = False
        str_not_existing_variables = ', '.join(not_existing_variables)
    
    message = (
        str_not_existing_variables 
        + ' are not in the list of possible variables.'
        )
    # TODO: show list of possible variables in a help function
    ras(assertion, message)