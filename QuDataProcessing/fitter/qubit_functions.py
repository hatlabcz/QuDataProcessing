from typing import Tuple, Any, Optional, Union, Dict, List

import numpy as np
import lmfit
from matplotlib import pyplot as plt
from QuDataProcessing.fitter.fitter_base import Fit, FitResult
from QuDataProcessing.fitter.generic_functions import Cosine, ExponentialDecay, ExponentialDecayWithCosine, ExponentialDecayWithCosineBeating, Lorentzian
from QuDataProcessing.base import Analysis, AnalysisResult
from QuDataProcessing.helpers.unit_converter import t2f


class QubitBasicResult(AnalysisResult):
    def __init__(self, lmfit_result, parameters: Dict[str, Union[Dict[str, Any], Any]]):
        super().__init__(parameters)
        self.lmfit_result=lmfit_result
        self.result_str = parameters.get("result_str")

    def plot(self, xlabel=None, ylabel=None, plot_ax=None, **figArgs):
        x_data = self.lmfit_result.userkws["coordinates"]
        result_str = self.params["result_str"].value
        if plot_ax is None:
            fix, plot_ax = plt.subplots(1,1, **figArgs)
        plot_ax.set_title(result_str)
        plot_ax.plot(x_data, self.lmfit_result.data, ".")
        plot_ax.plot(x_data, self.lmfit_result.best_fit, linewidth=2, label=self.result_str)
        plot_ax.set_xlabel(xlabel)
        plot_ax.set_ylabel(ylabel)
        plot_ax.legend()

    def get_fit_value(self, param_name):
        return self.lmfit_result.params[param_name].value

class PiPulseTuneUp(Fit):
    @staticmethod
    def model(coordinates, A, f, phi, of) -> np.ndarray:
        """$A \cos(2 \pi f x + \phi) + of$"""
        return Cosine.model(coordinates, A, f, phi, of)

    @staticmethod
    def guess(coordinates, data):
        guess_ = Cosine.guess(coordinates, data)
        data0 = data[np.argmin(np.abs(coordinates))]
        data_amp = guess_["A"]
        if data0 < np.mean(data): # flip sign of amplitude if zero point data is a deep
            guess_["A"] = lmfit.Parameter("A", value=-data_amp, max=0)
        else:
            guess_["A"] = lmfit.Parameter("A", value=data_amp, min=0)

        guess_["phi"] = lmfit.Parameter("phi", value=0, min=-np.pi/2, max=np.pi/2)
        return guess_

    def analyze(self, coordinates, data, dry=False, params={}, **fit_kwargs) -> QubitBasicResult:
        cosFitResult = super().analyze(coordinates, data, dry=False, params=params, **fit_kwargs)

        fit_phi = cosFitResult.params["phi"].value
        fit_f = cosFitResult.params.valuesdict()['f']

        zero_amp, pi_2_pulse_amp, pi_pulse_amp = \
            np.array(sorted([np.pi - fit_phi, np.pi / 2 - fit_phi, -fit_phi], key=lambda x: abs(x))) / 2 / np.pi / fit_f

        result_str = f'Pi pulse amp:{str(pi_pulse_amp)[:8]} DAC, Pi/2 pulse amp:{str(pi_2_pulse_amp)[:8]} DAC'
        print(result_str)

        return QubitBasicResult(cosFitResult,
                                dict(zero_amp=zero_amp,pi_pulse_amp=pi_pulse_amp, pi_2_pulse_amp=pi_2_pulse_amp,
                                     A=cosFitResult.params["A"].value, result_str=result_str))


class PulseSpec(Fit):
    @staticmethod
    def model(coordinates, A, x0, k, of) -> np.ndarray:
        """$ A /(k*(x-x0)**2+1) +of $"""
        return Lorentzian.model(coordinates, A, x0, k, of)

    @staticmethod
    def guess(coordinates, data):
        return Lorentzian.guess(coordinates, data)

    def analyze(self, coordinates, data, dry=False, params={}, freq_unit="MHz", **fit_kwargs):
        fitResult = super().analyze(coordinates, data, dry=False, params=params, **fit_kwargs)

        fit_freq = fitResult.params["x0"].value
        result_str = f'freq is {str(fit_freq)[:8]} {freq_unit}'
        print(result_str)

        return QubitBasicResult(fitResult, dict(fit_freq=fit_freq, result_str=result_str))

class T1Decay(Fit):
    @staticmethod
    def model(coordinates, A, tau, of):
        """ A * exp(-1.0 * x / tau) + of"""
        return ExponentialDecay.model(coordinates, A, tau, of)

    @staticmethod
    def guess(coordinates, data):
        return ExponentialDecay.guess(coordinates, data)

    def analyze(self, coordinates, data, dry=False, params={}, time_unit="us", **fit_kwargs):
        fitResult = super().analyze(coordinates, data, dry=False, params=params, **fit_kwargs)

        fit_tau = fitResult.params["tau"].value
        result_str = f'tau is {str(fit_tau)[:5]} {time_unit}'
        print(result_str)

        return QubitBasicResult(fitResult, dict(tau=fit_tau, result_str=result_str))
#
#
class T2Ramsey(Fit):
    @staticmethod
    def model(coordinates, A, f, phi, tau, of):
        """ A * cos(2 pi f x + phi) * exp (-x/tau) + of"""
        return ExponentialDecayWithCosine.model(coordinates, A, f, phi, tau, of)

    @staticmethod
    def guess(coordinates, data):
        return ExponentialDecayWithCosine.guess(coordinates, data)

    def analyze(self, coordinates, data, dry=False, params={}, time_unit="us", **fit_kwargs):
        fitResult = super().analyze(coordinates, data, dry=False, params=params, **fit_kwargs)

        fit_tau = fitResult.params["tau"].value
        fit_f = fitResult.params["f"].value
        result_str = f'tau is {str(fit_tau)[:5]} {time_unit}, detuning is {f"{fit_f:.6e}"} {t2f(time_unit)}'
        print(result_str)

        return QubitBasicResult(fitResult, dict(tau=fit_tau, f=fit_f, result_str=result_str))


class T2RamseyBeating(Fit):
    @staticmethod
    def model(coordinates, A, f1, phi1, B, f2, phi2, tau, of):
        """ A * cos(2 pi f x + phi) * exp (-x/tau) + of"""
        return ExponentialDecayWithCosineBeating.model(coordinates, A, f1, phi1, B, f2, phi2, tau, of)

    @staticmethod
    def guess(coordinates, data):
        return ExponentialDecayWithCosineBeating.guess(coordinates, data)

    def analyze(self, coordinates, data, dry=False, params={}, time_unit="us", **fit_kwargs):
        fitResult = super().analyze(coordinates, data, dry=False, params=params, **fit_kwargs)

        fit_tau = fitResult.params["tau"].value
        fit_f1 = fitResult.params["f1"].value
        fit_f2 = fitResult.params["f2"].value
        fit_A = fitResult.params["A"].value
        fit_B = fitResult.params["B"].value
        fit_phi1 = fitResult.params["phi1"].value
        fit_phi2 = fitResult.params["phi2"].value
        result_str = f'tau is {str(fit_tau)[:5]} {time_unit}, f1 is {f"{fit_f1:.6e}"} {t2f(time_unit)}, ' \
                     f'f2 is {f"{fit_f2:.6e}"} {t2f(time_unit)}'
        print(result_str)

        return QubitBasicResult(fitResult, dict(tau=fit_tau, f1=fit_f1, f2=fit_f2, result_str=result_str,
                                                A=fit_A, B=fit_B, phi1=fit_phi1, phi2=fit_phi2))


if __name__ == "__main__":
    x_data = np.linspace(-30000, 30000, int(1e6+1))
    y_data = -np.cos(2*np.pi/20000*x_data + 0.001) + 0.2

    piPulseFit = PiPulseTuneUp(x_data, y_data)
    # cosFitResult = cosFit.run(params={"A":lmfit.Parameter("A", 1, vary=False)}) # example of adjusting fitting parameter
    fitResult = piPulseFit.run()

    fitResult.plot()


