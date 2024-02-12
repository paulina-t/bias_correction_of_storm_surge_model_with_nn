"""

This file serves for binding all the NetCDF files necessary to do the analysis
of the fisrt part of the author's PhD into a single NetCDF file. This part of
the PhD is strongly related to the Work Package 2 of the Machine Ocean Project.

Data has different sources and has been prepared by several employees at the 
Norwegian Meteorological Institute.


The different datasets and their corresponding sources are listed below.

Operational files
-----------------
This files are generated every 12 hours and contain 120 hours of data which 
includes SSH observations and tide estimations from Kartverker, as well as the 
output of the operational model ROMS, a correction variable computed at MET, 
and the final storm surge prediction. The only variables retrieved are the 
stormsurge, stormsurge_corrected and bias. The files are available from 
2017, but because of quality reasons the recommendation is to use data produced
after 2018. 

Contact person: Nils Melsom Kristensen

Path: '/lustre/storeB/project/fou/hi/stormsurge_eps/2dEPS_archive'

Some preprocessing has been done with the script read_kyststajoner.py


Forcing data
------------
The forcing data from the ERA5 reanalysis (u10m, v10m, msl) is stored on MET's.
The preprocessed files contain data from the neareast gridbox to the stations
in the North Sea. The files include also swh and mwd from the WAMIII.

Contact person: Martin Lilleeng Sætra

Path: '/lustre/storeB/project/IT/geout/machine-ocean/workspace/martinls' 

Path to original grid files: '/lustre/storeB/project/fou/om/StormRisk/AtmosForceNordic4'


Hindcast
--------
The ROMS model has been run as a hindcast with two different sets of forcing 
data: NORA3 (2000-2019) and ERA5 (1979-2019). 

Contact person: Ole Johan Aarnes

Path: '/lustre/storeB/project/fou/om/StormRisk/RunsNordic4'


Furthermore, a new file containing hindcast data at the stations' location has
been generated.

Contact person: Jean Rabault

Path: '/lustre/storeB/project/IT/geout/machine-ocean/prepared_datasets/storm_surge/ROMS_hindcast_at_stations/roms_hindcast_storm_surge_data.nc'


Observations and tides
----------------------
SSH observations at 106 locations in the North Sea were collected from 
different sources by Jean Rabault. A new dataset with tide data and observations
from Kartverket at each of these locations is created. It also includes 
computations of tide made with the pytide package. For this work, tides from 
Kartverket will be used at Norwegian stations, and pytide estimations at the 
rest.

Contact person: Jean Rabault

Path: '/lustre/storeB/project/IT/geout/machine-ocean/prepared_datasets/storm_surge/aggregated_water_level_data/aggregated_water_level_observations_with_pytide_prediction_dataset.nc'



/lustre/storeB/project/IT/geout/machine-ocean/prepared_datasets/storm_surge/ROMS_hindcast_at_stations/roms_hindcast_storm_surge_data.nc

Author: Paulina Tedesco
paulinatedesco@gmail.com


"""
import xarray as xr
import os
import time

print("Put the interpreter in UTC, to make sure no TZ issues...")

os.environ["TZ"] = "UTC"
time.tzset()

def ras(assertion,  message=""):
    """
    Raise on failed assertion. 

    Parameters
    ----------
    assertion :
        The assertion to check for.
    message : str, optional
        The message to display as the Exception content if the assertion fails. 
        The default is "".

    Raises
    ------
    Exception
        If assertion is True, do nothing. If False, raise an Exception with the
        custom message.

    Returns
    -------
    None.

    """
    if not assertion:
        raise Exception(message)
        
        

def open_operational_data():
    pass

def preprocess_operational_data():
    pass

def open_forcing_data():
    pass

def preprocess_forcing_data():
    pass

def open_observations_tide_data():
    pass

def preprocess_observed_tide_data():
    pass


