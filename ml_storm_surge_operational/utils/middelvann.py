"""Add mean water level.

The kyststasjoner files contain both SSH, observed and output from models, as 
well as warning levels. The SSH data is computed using the chart datum as a
reference, whereas the warning levels are computed with respect to the mean 
waterlevel.

It is, therefore ,necessary to add the mean values to the SSH data to be able to
compare them with the warning levels.
 
The mean levels are available at:
https://projects.met.no/vannstandshendelser/230319_prog-3/
"""
  
from ml_models.utils import assertions  as asrt   

def add_mean_water_level_values(station_id: str, x):
    """Add mean water level.
    
    The mean water levels are extracted from MET's site:
        https://projects.met.no/vannstandshendelser/230319_prog-3/
    
    and stored in a dictionary that maps the values with the station_id.
    
    This functions adds the mean water level to the value passed as argument 
    for the station of interest.

    Parameters
    ----------
    station_id : str
        Station id, e.g., 'NO_OSC'.
    x : numeric or array_like
        Value to be corrected in meters. It can also be an array containing 
        numbers.

    Returns
    -------
    x_corrected : numeric or array_like
        Corrected values in meters.

    """
    asrt.assert_norwegian_station_id(station_id)
    
    # This are the original values in cm
    middelvann_values = {
        'NO_VIK':51,
        'NO_OSC':66,
        'NO_OSL':66,
        'NO_HRO':50,
        'NO_TRG':45,
        'NO_SVG':65,
        'NO_BGO':91,
        'NO_MAY':115,
        'NO_AES':122,
        'NO_KSU':130,
        'NO_HEI':146,
        'NO_TRD':165,
        'NO_RVK':151,
        'NO_BOO':166,
        'NO_NVK':184,
        'NO_KAB':173,
        'NO_ANX':130,
        'NO_HAR':135,
        'NO_TOS':163,
        'NO_HFT':168,
        'NO_HVG':165,
        'NO_VAW':192,
        'NO_NYA':92
        }
    
    x_corrected = x + middelvann_values[station_id]/100
    
    return x_corrected



