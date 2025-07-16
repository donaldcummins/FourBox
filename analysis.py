import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from four_box import *

# Load the fluxes
ds = xr.open_dataset('./data/rtmt_tas_anom/rtmt_glbmean_CESM104_abrupt4x_0001-5900_anom.nc', decode_times=False)
rtmt = ds['rtmt_glbmean'].values
ds.close()

# Load the temperatures
ds = xr.open_dataset('./data/rtmt_tas_anom/tas_glbmean_CESM104_abrupt4x_0001-5900_anom.nc', decode_times=False)
tas = ds['tas_glbmean'].values
ds.close()

# Fit four-box EBM (set alpha to regularize timescales)
y = np.column_stack((tas, rtmt))
model, res = fit_model(y, alpha=0)

# Plot the results
plt.plot(y)
plt.plot(model.forward(len(y)))
plt.show()