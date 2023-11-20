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




def ecr_f21(coordinates, Ql, fn, sn, c1, c2):
    """"
    end coupled resonator
    Kurpiers et al. EPJ Quantum Technology  (2017) 4:8
    DOI 10.1140/epjqt/s40507-017-0059-7"""
    s21 = sn/np.sqrt(1 + 4 * (coordinates/fn-1)**2 * Ql**2) + c1 + c2 * coordinates
    return s21


class ECR_S21Result():
    def __init__(self, lmfit_result: lmfit.model.ModelResult):
        self.lmfit_result = lmfit_result
        self.params = lmfit_result.params
        self.Ql = self.params["Ql"].value
        self.fn = self.params["fn"].value
        self.kappa_2pi = self.fn / self.Ql
        self.freqData = lmfit_result.userkws[lmfit_result.model.independent_vars[0]]

    def plot(self, title='s21  dB', plot_ax=None, **figArgs):
        mag_fit = 20*np.log10(self.lmfit_result.best_fit)
        mag_data = 20*np.log10(self.lmfit_result.data.real)

        fig_args_ = dict(figsize=(6, 5))
        fig_args_.update(figArgs)

        if plot_ax is None:
            fig, ax = plt.subplots(**fig_args_)
        else:
            ax = plot_ax
        ax.set_title(title)
        ax.plot(self.freqData, mag_data, '.')
        ax.plot(self.freqData, mag_fit)
        plt.show()
        
        return ax

    def print(self):

        print(f'fn: {rounder(self.fn, 5)}+-{rounder(self.params["fn"].stderr, 5)}')
        print(f'Ql: {rounder(self.Ql, 5)}+-{rounder(self.params["Ql"].stderr, 5)}')
        print(f'kappa_int/2pi (MHz): {self.fn/self.Ql/1e6}')


class ECR_S21(Fit):
    def __init__(self, coordinates: Union[Tuple[np.ndarray, ...], np.ndarray],
                 data: np.ndarray):
        """ fit cavity reflection function
        :param conjugate: fit to conjugated cavity reflection function (for VNA data)
        """
        self.coordinates = coordinates
        self.data = data
        self.pre_process()

    def model(self, coordinates, Ql, fn, sn, c1, c2) -> np.ndarray:
        """"reflection function of a harmonic oscillator"""
        s21 = ecr_f21(coordinates, Ql, fn, sn, c1, c2)
        return s21

    @staticmethod
    def guess(coordinates, data):
        freq = coordinates

        c2Guess =  (data[-1] - data[0])/(freq[-1]-freq[0])
        snGuess = data[0]
        fnGuess = freq[np.argmax(data)]
        QlGuess = fnGuess / (freq[-1]-freq[0]) * 4
        c1Guess = data[0]

        Ql = lmfit.Parameter("Ql", value=QlGuess, min=QlGuess / 100, max=QlGuess * 100)
        sn = lmfit.Parameter("sn", value=snGuess, min=0)
        fn = lmfit.Parameter("fn", value=fnGuess, min=freq[0], max=freq[-1])
        c1 = lmfit.Parameter("c1", value=c1Guess)
        c2 = lmfit.Parameter("c2", value=c2Guess)

        return dict(Ql=Ql, sn=sn, c1=c1, fn=fn, c2=c2)

    def run(self, *args: Any, **kwargs: Any) -> ECR_S21Result:
        lmfit_result = self.analyze(self.coordinates, self.data, *args, **kwargs)
        return ECR_S21Result(lmfit_result)


class ECR_S21_MultiMode(Fit):
    def __init__(self, coordinates: Union[Tuple[np.ndarray, ...], np.ndarray],
                 data: np.ndarray, nmodes):
        """ fit cavity reflection function
        :param conjugate: fit to conjugated cavity reflection function (for VNA data)
        """
        self.coordinates = coordinates
        self.data = data
        self.nmodes = nmodes
        self.pre_process()

        model = lmfit.Model(ecr_f21, prefix='p1_')
        for i in range(1, nmodes):
            model = model + lmfit.Model(ecr_f21, prefix= f"p{i+1}_")
        self.model = model


    def guess(self, coordinates, data):
        freq = coordinates

        guess_dict = {}
        for i in range(self.nmodes):
            c2Guess = (data[-1] - data[0]) / (freq[-1] - freq[0])
            snGuess = data[0]
            fnGuess = freq[np.argmax(data)]
            QlGuess = fnGuess / (freq[-1] - freq[0]) * 4
            c1Guess = data[0]

            Ql = lmfit.Parameter(f"p{i+1}_Ql", value=QlGuess, min=QlGuess / 100, max=QlGuess * 100)
            c2 = lmfit.Parameter(f"p{i+1}_c2", value=c2Guess)
            sn = lmfit.Parameter(f"p{i+1}_sn", value=snGuess, min=0)
            fn = lmfit.Parameter(f"p{i+1}_fn", value=fnGuess, min=freq[0], max=freq[-1])
            c1 = lmfit.Parameter(f"p{i+1}_c1", value=c1Guess)
            guess_dict.update({f"p{i+1}_Ql": Ql, f"p{i+1}_c2": c2,
                               f"p{i+1}_c1": c1, f"p{i+1}_sn": sn,
                               f"p{i+1}_fn": fn})
            new_data, cut_idx_l, cut_idx_r = cut_peak(data, 0.6, plot=False)
            data = new_data
            print(f"{freq[cut_idx_l]}, {freq[cut_idx_r]}")
        print(guess_dict)
        return guess_dict

    def run(self, *args: Any, **kwargs: Any):
        lmfit_result = self.analyze(self.coordinates, self.data, *args, **kwargs)
        return ECR_S21Multi_Result(lmfit_result, nmodes=self.nmodes)


class ECR_S21Multi_Result():
    def __init__(self, lmfit_result: lmfit.model.ModelResult, nmodes):
        self.lmfit_result = lmfit_result
        self.params = lmfit_result.params
        self.nmodes=nmodes
        for i in range(nmodes):
            self.__setattr__(f"p{i+1}_Ql", self.params[f"p{i+1}_Ql"].value)
            self.__setattr__(f"p{i+1}_fn", self.params[f"p{i+1}_fn"].value)
        self.freqData = lmfit_result.userkws[lmfit_result.model.independent_vars[0]]

    def plot(self, **figArgs):
        mag_fit = 20*np.log10(self.lmfit_result.best_fit)
        mag_data = 20*np.log10(self.lmfit_result.data.real)

        fig_args_ = dict(figsize=(6, 5))
        fig_args_.update(figArgs)
        plt.figure(**fig_args_)
        plt.title('s21  dB')
        plt.plot(self.freqData, mag_data, '.')
        plt.plot(self.freqData, mag_fit)
        plt.show()

    def print(self):
        for i in range(self.nmodes):
            print(f'p{i+1}_fn": {rounder(self.__getattribute__(f"p{i+1}_fn"), 5)}+-'
                  f'{rounder(self.params[f"p{i+1}_fn"].stderr, 5)}')
            print(f'p{i+1}_Ql": {rounder(self.__getattribute__(f"p{i+1}_Ql"), 5)}+-'
                  f'{rounder(self.params[f"p{i+1}_Ql"].stderr, 5)}')

if __name__ == '__main__':
    pass