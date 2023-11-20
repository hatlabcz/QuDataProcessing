import numpy as np
import matplotlib.pyplot as plt
from QuDataProcessing.fitter.arb_gaussian import classify_point, peakfinder_2d, fit_arb_gaussians

'''
Some functions for extracting basic information from a single set of IQ traces that correspond to resonator states.
For example
- state location on IQ plane
- squeezing degree and angle of coherent state
'''

def blob_info(idata, qdata, num_states, sigma, plot=False):

    range = np.max([np.max(np.abs(idata)),np.max(np.abs(qdata))])

    bins = int(np.sqrt(len(idata.flatten())))
    zz, x, y = np.histogram2d(idata.flatten(), qdata.flatten(), bins=bins, range=[[-range,range],[-range,range]])

    zz = zz.copy()

    x = (x[0:-1] + x[1:]) / 2
    y = (y[0:-1] + y[1:]) / 2
    xx, yy = np.meshgrid(x, y)


    dx = x[1]-x[0]

    radius = int(sigma/dx)

    idxx, idxy, heights = peakfinder_2d(zz, np.min([radius,bins//2]), num_states)

    if plot:
        plt.figure()
        plt.pcolor(x, y, np.log(zz.transpose()))
        plt.colorbar()
        plt.scatter(x[idxx], y[idxy], color='r')

    fitted_params = fit_arb_gaussians(x, y, zz.transpose(), idxx, idxy, heights, sigma, plot=plot)

    return fitted_params


