"""
Run operational ML model multiple times.

Not needed in an operational context.
"""

import numpy as np
from multiprocessing import Pool

from ml_models.models.operational_5_stations import main_train_model

#i = np.arange(10)
#n_processes = len(i)
#with Pool(n_processes) as pool: # four parallel jobs
#    results = pool.map(main_train_model, i)

"""    
i = np.arange(100, 200)
n_processes = len(i)
with Pool(n_processes) as pool: # four parallel jobs
    results = pool.map(main_train_model, i)
    
i = np.arange(200, 300)
n_processes = len(i)
with Pool(n_processes) as pool: # four parallel jobs
    results = pool.map(main_train_model, i)
   
i = np.arange(300, 400)
n_processes = len(i)
with Pool(n_processes) as pool: # four parallel jobs
    results = pool.map(main_train_model, i)
    
i = np.arange(400, 500)
n_processes = len(i)
with Pool(n_processes) as pool: # four parallel jobs
    results = pool.map(main_train_model, i)
    
i = np.arange(500, 600)
n_processes = len(i)
with Pool(n_processes) as pool: # four parallel jobs
    results = pool.map(main_train_model, i)
    
i = np.arange(600, 700)
n_processes = len(i)
with Pool(n_processes) as pool: # four parallel jobs
    results = pool.map(main_train_model, i)
    
i = np.arange(700, 800)
n_processes = len(i)
with Pool(n_processes) as pool: # four parallel jobs
    results = pool.map(main_train_model, i)
    
i = np.arange(800, 900)
n_processes = len(i)
with Pool(n_processes) as pool: # four parallel jobs
    results = pool.map(main_train_model, i)
    
i = np.arange(900, 1000)
n_processes = len(i)
with Pool(n_processes) as pool: # four parallel jobs
    results = pool.map(main_train_model, i)
    
"""

for i in range(965, 1000):
    main_train_model(i)