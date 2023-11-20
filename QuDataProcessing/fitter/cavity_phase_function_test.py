from typing import Tuple, Any, Optional, Union, Dict, List

import numpy as np
import matplotlib.pyplot as plt
import lmfit
from lmfit.model import ModelResult
import h5py
from QuDataProcessing.fitter.fitter_base import Fit, FitResult
from QuDataProcessing.helpers.unit_converter import freqUnit, rounder, realImag2magPhase
from QuDataProcessing.fitter.cavity_functions import CavReflectionPhaseOnly

TWOPI = 2 * np.pi
PI = np.pi



class CavReflectionPhaseOnly_test(CavReflectionPhaseOnly):
    def pre_process(self):
        self.coordinates = np.array(self.coordinates)
        self.data = np.array(self.data)



if __name__ == '__main__':
    import json
    file = open(r"C:\Users\zctid.LAPTOP-150KME16\Desktop\\PanfluteB_chi")
    data = json.load(file)
    freq = np.arange(85,95,0.25) *1e6 + 7e9
    phase_g = data["phi_g"]
    phase_e = data["phi_e"]
    # 
    # 
    cavRef_g = CavReflectionPhaseOnly_test(freq, phase_g, conjugate=False)
    # results = cavRef_g.run(params={"Qext": lmfit.Parameter("Qext", value=1e+03), "Qint": lmfit.Parameter("Qint", value=7e3)})
    results_g = cavRef_g.run()

    cavRef_e = CavReflectionPhaseOnly_test(freq, phase_e, conjugate=False)
    results_e = cavRef_e.run()

    plt.figure()
    plt.plot(freq, results_g.lmfit_result.data)
    plt.plot(freq, results_g.lmfit_result.best_fit)
    plt.plot(freq, results_e.lmfit_result.data)
    plt.plot(freq, results_e.lmfit_result.best_fit)

    print(cavRef_g.guess(cavRef_g.coordinates, cavRef_g.data))
    print(results_g.lmfit_result.params)

    print(results_g.f0 - results_e.f0)

    plt.figure()
    freq = np.arange(85,95,0.25) *1e6 + 7e9
    f0  = 90 *1e6 + 7e9
    plt.plot(freq, cavRef_g.model(freq, 1340, 10000, f0, 1.2, 1e-7))


