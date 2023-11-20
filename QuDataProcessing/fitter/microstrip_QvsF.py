from typing import Tuple, Any, Optional, Union, Dict, List

import numpy as np
import matplotlib.pyplot as plt
import lmfit
from lmfit.model import ModelResult
from QuDataProcessing.fitter.fitter_base import Fit, FitResult
from QuDataProcessing.helpers.unit_converter import freqUnit, rounder, realImag2magPhase
from QuDataProcessing.analyzer.cut_peak import cut_peak

TWOPI = 2 * np.pi
PI = np.pi




def ustrip_Qc_normalMetal_(coordinates, Qc_coef):
    """
    normal metal condoctor loss of microstrip resonator
    for details of Qc_coeff, check Pozar 3.8
    """
    Qc = Qc_coef * np.sqrt(coordinates)
    return Qc

def ustrip_Qd_(coordinates, tand):
    """
    dielectric loss of microstrip resonator
    """
    Qd = 1/tand
    return Qd

def ustrip_Qr_(coordinates, Qr_coef):
    """
    radiation loss of microstrip resonator
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1128615
    """
    Qr = Qr_coef / coordinates**2
    return Qr


class uStrip_Q_normalMetal_result():
    def __init__(self, lmfit_result: lmfit.model.ModelResult):
        self.lmfit_result = lmfit_result
        self.params = lmfit_result.params
        self.tand = self.params["tand"].value
        self.Qc_coef = self.params["Qc_coef"].value
        try:
            self.Qr_coef = self.params["Qr_coef"].value
            self.radLoss=True
        except Exception:
            self.radLoss = False
        self.freqData = lmfit_result.userkws[lmfit_result.model.independent_vars[0]]

    def plot(self, plot_axes=None, **figArgs):
        fig_args_ = dict(figsize=(6, 5))
        fig_args_.update(figArgs)

        if plot_axes is None:
            fig, ax = plt.subplots(1,2, **fig_args_)
        else:
            ax = plot_axes

        ax[0].plot(self.freqData, self.lmfit_result.data.real, '.')
        ax[0].plot(self.freqData, self.lmfit_result.best_fit)
        
        
        # -------- bar plot ---------------------
        freq_label = np.array(np.round(self.freqData/1e9, 1), dtype=str)
        Qd_list = ustrip_Qd_(self.freqData, self.tand)
        Qc_list = ustrip_Qc_normalMetal_(self.freqData, self.Qc_coef)

        bar_width = 0.25
        r1 = np.arange(len(self.freqData))
        r2 = [x + bar_width for x in r1]
        r3 = [x + bar_width for x in r2]

        ax[1].bar(r1, 1/Qd_list, width=bar_width,label="dielectric")
        ax[1].bar(r2, 1/Qc_list, width=bar_width,label="conductor")
        if self.radLoss:
            Qr_list = ustrip_Qr_(self.freqData, self.Qr_coef)
            ax[1].bar(r3, 1/Qr_list, width=bar_width,label="radiation")

        plt.legend()
        ax[1].set_ylabel("1/Q")
        ax[1].set_yscale("log")
        ax[1].set_xticklabels(freq_label)
        ax[1].set_xticks(r1)
        plt.show()
        
        return ax

    def print(self):

        print(f'tand: {rounder(self.tand, 5)}+-{rounder(self.params["tand"].stderr, 5)}')
        print(f'Qc_coef: {rounder(self.Qc_coef, 5)}+-{rounder(self.params["Qc_coef"].stderr, 5)}')
        try:
            print(f'Qr_coef: {rounder(self.Qr_coef, 5)}+-{rounder(self.params["Qr_coef"].stderr, 5)}')
        except Exception as E:
            pass


class uStrip_Q_normalMetal(Fit):
    def __init__(self, coordinates: Union[Tuple[np.ndarray, ...], np.ndarray],
                 data: np.ndarray, radLoss=True):
        """ fit cavity reflection function
        :param conjugate: fit to conjugated cavity reflection function (for VNA data)
        """
        self.coordinates = coordinates
        self.data = data
        self.pre_process()
        if radLoss:
            self.model = self.model_Radi
        else:
            self.model = self.model_noRadi

        self.radLoss = radLoss

    def model_noRadi(self, coordinates, tand, Qc_coef) -> np.ndarray:
        Qc = ustrip_Qc_normalMetal_(coordinates, Qc_coef)
        Qd = ustrip_Qd_(coordinates, tand)
        Qtot = 1 / (1 / Qc + 1 / Qd)
        return Qtot

    def model_Radi(self, coordinates, tand, Qc_coef, Qr_coef) -> np.ndarray:
        Qc = ustrip_Qc_normalMetal_(coordinates, Qc_coef)
        Qd = ustrip_Qd_(coordinates, tand)
        Qr = ustrip_Qr_(coordinates, Qr_coef)
        Qtot = 1 / (1 / Qc + 1 / Qd + 1/Qr)
        return Qtot



    def guess(self, coordinates, data):
        freq = coordinates

        tand_guess =  1/data[0]
        Qc_coef_guess = (data[len(data)//2] - data[0]) / (np.sqrt(freq[len(data)//2]) - np.sqrt(freq[0]))

        tand = lmfit.Parameter("tand", value=tand_guess, min=tand_guess / 100, max=tand_guess * 100)
        Qc_coef = lmfit.Parameter("Qc_coef", value=Qc_coef_guess, min=0, max=Qc_coef_guess*100)

        if self.radLoss:
            Qr_coef_guess = Qc_coef_guess * np.max(freq) ** (5 / 2)
            Qr_coef = lmfit.Parameter("Qr_coef", value=Qr_coef_guess, min=Qr_coef_guess / 100, max=Qr_coef_guess * 100)
            return dict(tand=tand, Qc_coef=Qc_coef, Qr_coef=Qr_coef)

        return dict(tand=tand, Qc_coef=Qc_coef)

    def run(self, *args: Any, **kwargs: Any) -> uStrip_Q_normalMetal_result:
        lmfit_result = self.analyze(self.coordinates, self.data, *args, **kwargs)
        return uStrip_Q_normalMetal_result(lmfit_result)






def uStrip_epsilon_eff(W, d, er):
    ee = (er + 1)/2 + (er - 1)/2 / np.sqrt(1+12*d/W)
    return ee

def uStrip_tand_eff(W, d, er, tand):
    ee = uStrip_epsilon_eff(W, d, er)
    tand_eff = er*(ee - 1) / (ee * (er-1)) * tand
    return tand_eff

def uStrip_Z0(W, d, er):
    ee = uStrip_epsilon_eff(W, d, er)
    if W <= d:
        z0 = 60/np.sqrt(ee) * np.log(8*d/W + W / 4/d)
    else:
        z0 = 120 * np.pi/(np.sqrt(ee)*(W/d+1.393+0.667*np.log(W/d+1.444)))
    return z0


if __name__ == '__main__':
    pass