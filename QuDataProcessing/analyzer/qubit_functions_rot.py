from typing import Tuple, Any, Optional, Union, Dict, List

import lmfit
import numpy as np
from matplotlib import pyplot as plt
import json

from QuDataProcessing.base import Analysis, AnalysisResult
from QuDataProcessing.analyzer.rotateIQ import RotateData
from QuDataProcessing.fitter.generic_functions import Cosine, ExponentialDecay
from QuDataProcessing.fitter import qubit_functions as qf


def _hline(ground, excited):
    plt.axhline(y=excited, color='r', linestyle='--', label = 'Excited', linewidth=1)
    plt.axhline(y=ground, color='b', linestyle='--', label = 'Ground', linewidth=1)
    plt.axhline(y=(excited + ground) / 2.0, color='y', linestyle='--', linewidth=1)
    plt.legend()

class QubitBasicResult_rot(AnalysisResult):
    def __init__(self, lmfit_result, parameters: Dict[str, Union[Dict[str, Any], Any]], rot_result=None):
        super().__init__(parameters)
        self.lmfit_result=lmfit_result
        self.rot_result = rot_result if rot_result is not None else {}
    
    def print_ge_rotation(self):
        rot_dict = {k: self.rot_result[k] for k in ["g_val", "e_val", "rot_angle"]}
        print(rot_dict)
        return rot_dict

    def plot(self, plot_ax=None, **figArgs):
        if plot_ax is None:
            fix, plot_ax = plt.subplots(1,1, **figArgs)
        g_val = self.rot_result.get("g_val", None)
        e_val = self.rot_result.get("e_val", None)
        i_new = self.rot_result["i_new"]
        q_new = self.rot_result["q_new"]
        x_data = self.lmfit_result.userkws["coordinates"]
        result_str = self.params["result_str"].value
        plot_ax.set_title(result_str)
        plot_ax.plot(x_data, q_new, ".")
        plot_ax.plot(x_data, i_new, ".")        
        plot_ax.plot(x_data, self.lmfit_result.best_fit, linewidth=3, label=result_str)
        if (g_val != None) and (e_val != None):
            _hline(g_val, e_val)
        plot_ax.legend()

    def print(self):
        for k, p in self.params:
            print(f"{k}: {p.value}")

    def get_fit_value(self, param_name):
        return self.lmfit_result.params[param_name].value

class PulseSpec(Analysis):
    @staticmethod
    def analyze(x_data, iq_data, rot_result={}, dry=False, params={}, **fit_kwargs) -> QubitBasicResult_rot:
        # rotate data
        rot_angle = rot_result.get("rot_angle", "find")
        rotIQ = RotateData(x_data, iq_data)
        iq_new = rotIQ.run(rot_angle)
        #fit to Lorentzian
        decayFit  = qf.PulseSpec(x_data, iq_new.params["i_data"].value)
        fitResult = decayFit.run(dry, params, **fit_kwargs)

        rot_result.update(i_new=iq_new.params["i_data"].value,
                          q_new=iq_new.params["q_data"].value)

        return QubitBasicResult_rot(fitResult.lmfit_result, fitResult.params, rot_result)


class PiPulseTuneUp(Analysis):
    @staticmethod
    def analyze(x_data, iq_data, rot_angle:Union[float, str]="find", dry=False, params={}, **fit_kwargs):
        # rotate data
        rotIQ = RotateData(x_data, iq_data)
        iq_new = rotIQ.run(rot_angle)
        #fit to cosine
        piPulFit = qf.PiPulseTuneUp(x_data, iq_new.params["i_data"].value)
        fitResult = piPulFit.run(dry, params, **fit_kwargs)

        g_val = piPulFit.model(fitResult.params["zero_amp"].value,**fitResult.lmfit_result.params)
        e_val = piPulFit.model(fitResult.params["pi_pulse_amp"].value, **fitResult.lmfit_result.params)

        rot_result = dict(rot_angle=iq_new.params["rot_angle"].value,
                          i_new=iq_new.params["i_data"].value,
                          q_new=iq_new.params["q_data"].value,
                          g_val=g_val, e_val=e_val)

        # store_rot_info(rot_angle, e_val, g_val, pi_pulse_amp) # TODO: keep these info in a file

        return QubitBasicResult_rot(fitResult.lmfit_result, fitResult.params, rot_result)


class T1Decay(Analysis):
    @staticmethod
    def analyze(x_data, iq_data, rot_result={}, dry=False, params={}, **fit_kwargs) -> QubitBasicResult_rot:
        # rotate data
        rot_angle = rot_result.get("rot_angle", "find")
        rotIQ = RotateData(x_data, iq_data)
        iq_new = rotIQ.run(rot_angle)
        #fit to decay
        decayFit  = qf.T1Decay(x_data, iq_new.params["i_data"].value)
        fitResult = decayFit.run(dry, params, **fit_kwargs)

        rot_result.update(i_new=iq_new.params["i_data"].value,
                          q_new=iq_new.params["q_data"].value)

        return QubitBasicResult_rot(fitResult.lmfit_result, fitResult.params, rot_result)


class T2Ramsey(Analysis):
    @staticmethod
    def analyze(x_data, iq_data, rot_result={}, dry=False, params={}, **fit_kwargs):
        # rotate data
        rot_angle = rot_result.get("rot_angle", "find")
        rotIQ = RotateData(x_data, iq_data)
        iq_new = rotIQ.run(rot_angle)
        #fit to decay
        ramseyFit  = qf.T2Ramsey(x_data, iq_new.params["i_data"].value)
        fitResult = ramseyFit.run(dry, params, **fit_kwargs)

        rot_result.update(i_new=iq_new.params["i_data"].value,
                          q_new=iq_new.params["q_data"].value)

        return QubitBasicResult_rot(fitResult.lmfit_result, fitResult.params, rot_result)
    

class T2RamseyBeating(Analysis):
    @staticmethod
    def analyze(x_data, iq_data, rot_result={}, dry=False, params={}, **fit_kwargs):
        # rotate data
        rot_angle = rot_result.get("rot_angle", "find")
        rotIQ = RotateData(x_data, iq_data)
        iq_new = rotIQ.run(rot_angle)
        #fit to decay
        ramseyFit  = qf.T2RamseyBeating(x_data, iq_new.params["i_data"].value)
        fitResult = ramseyFit.run(dry, params, **fit_kwargs)

        rot_result.update(i_new=iq_new.params["i_data"].value,
                          q_new=iq_new.params["q_data"].value)
        return QubitBasicResult_rot(fitResult.lmfit_result, fitResult.params, rot_result)


if __name__=="__main__":
    fileName = "Q1"
    filePath = r"C:/Users/hatla/Downloads//"

    # fit pipulse and rotation
    with open(filePath + fileName + "_piPulse", 'r') as infile:
        dataDict = json.load(infile)
    x_data = dataDict["x_data"]
    i_data = np.array(dataDict["i_data"])
    q_data = np.array(dataDict["q_data"])

    piPul = PiPulseTuneUp(x_data, i_data+1j*q_data)
    piResult = piPul.run()
    piResult.plot()

    # fit decay
    with open(filePath + fileName+"_t1", 'r') as infile:
        dataDict = json.load(infile)
    x_data = dataDict["x_data"]
    i_data = np.array(dataDict["i_data"])
    q_data = np.array(dataDict["q_data"])
    t1Decay = T1Decay(x_data, i_data+1j*q_data)
    t1Result = t1Decay.run(piResult.rot_result)
    t1Result.plot()


    # fit ramsey
    with open(filePath + fileName+"_t2R", 'r') as infile:
        dataDict = json.load(infile)
    x_data = dataDict["x_data"]
    i_data = np.array(dataDict["i_data"])
    q_data = np.array(dataDict["q_data"])
    t2RDecay = T2Ramsey(x_data, i_data+1j*q_data)
    t2Result = t2RDecay.run(piResult.rot_result)
    t2Result.plot()