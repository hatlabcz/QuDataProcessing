from typing import Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
import lmfit
from QuDataProcessing.fitter.fitter_base import Fit
from QuDataProcessing.helpers.unit_converter import rounder



def hybridize_freq(f1, f0, g):
    """
    this g here is g/2pi in the Hamiltonian
    """
    f0_ = (f0 + f1 + np.sqrt((f0 - f1) ** 2 + 4 * g ** 2)) / 2
    f1_ = (f0 + f1 - np.sqrt((f0 - f1) ** 2 + 4 * g ** 2)) / 2
    return np.array([f0_, f1_])

def lin_freq(x, f0, x0, k):
    return  k * (x-x0) + f0


class AntiCrossingResult():
    def __init__(self, lmfit_result: lmfit.model.ModelResult, x_data, model, data):
        self.lmfit_result = lmfit_result
        self.params = lmfit_result.params

        self.f0 = self.params["f0"].value
        self.x0 = self.params["x0"].value
        self.g = self.params["g"].value
        self.k = self.params["k"].value
        self.x_data = x_data
        self.model = model
        self.data = data

    def plot(self, plot_ax=None, **figArgs):
        f0_fit = self.model(self.x_data, self.f0, self.x0, self.k, self.g)[0]
        f1_fit = self.model(self.x_data, self.f0, self.x0, self.k, self.g)[1]
        f0_data = self.data[0]
        f1_data = self.data[1]

        if plot_ax is None:
            fig_args_ = dict(figsize=(8, 5))
            fig_args_.update(figArgs)
            fig, ax = plt.subplots(1, 1, **fig_args_)
        else:
            ax = plot_ax

        ax.plot(self.x_data, f0_data,  label="f0 data")
        ax.plot(self.x_data, f1_data,  label="f1 data")
        ax.plot(self.x_data, f0_fit, label="f0 fit")
        ax.plot(self.x_data, f1_fit, label="f1 fit")
        ax.legend()
        plt.show()

    def print(self):
        print(f'g/2pi : {rounder(self.g, 5)}+-{rounder(self.params["g"].stderr, 5)}')



class AntiCrossing(Fit):
    def __init__(self, coordinates: Union[Tuple[np.ndarray, ...], np.ndarray],
                 data: np.ndarray):
        """ fit anti-crossed frequencies function
        """
        self.coordinates = coordinates
        self.data = data
        self.pre_process()

    def pre_process(self):
        super().pre_process()
        if np.nanmean(self.data[0]) < np.nanmean(self.data[1]):
            self.data = np.array([self.data[1], self.data[0]])

    @staticmethod
    def model(coordinates, f0, x0, k, g) -> np.ndarray:
        f1 = lin_freq(coordinates, f0, x0, k)
        freqs = hybridize_freq(f0, f1, g)
        return freqs

    @staticmethod
    def guess(coordinates, data):
        f0_ = data[0]
        f1_ = data[1]

        f_max_idx = np.nanargmax(f0_)
        f_min_idx = np.nanargmin(f1_)
        flat_idx_0 = np.nanargmax(f0_[f_max_idx]-f0_)
        flat_idx_1 = np.nanargmax(f1_ - f1_[f_min_idx])

        f0 = (f0_[flat_idx_0] + f1_[flat_idx_1])/2
        x0 = coordinates[int((f_max_idx + f_min_idx) / 2)]

        k0_ = (f0_[f_max_idx] - f0_[flat_idx_0])/(coordinates[f_max_idx]-coordinates[flat_idx_0])
        k1_ = (f1_[f_min_idx] - f1_[flat_idx_1])/(coordinates[f_min_idx]-coordinates[flat_idx_1])
        k = k0_ + k1_

        g = (f0_[f_max_idx] - f1_[f_min_idx])/5

        f0 = lmfit.Parameter("f0", value=f0, min=np.nanmin(data), max=np.nanmax(data))
        x0 = lmfit.Parameter("x0", value=x0, min=coordinates[0], max=coordinates[-1])
        k = lmfit.Parameter("k", value=k)
        g = lmfit.Parameter("g", value=g, min=0, max=g*20)

        return dict(f0=f0, x0=x0, k=k, g=g)

    def residual(self, pars, x, data):
        parvals = pars.valuesdict()
        f0 = parvals['f0']
        x0 = parvals['x0']
        k = parvals['k']
        g = parvals['g']
        model = self.model(x, f0, x0, k, g)
        return np.nansum(np.abs(model-data), axis=0)


    def run(self, dry=False, params={}, **fit_kwargs) -> "AntiCrossingResult":
        model = lmfit.model.Model(self.model)
        _params = lmfit.Parameters()

        for pn, pv in self.guess(self.coordinates, self.data).items():
            if type(pv) == lmfit.Parameter:
                _params.add(pv)
            else:
                _params.add(pn, value=pv)
        for pn, pv in params.items():
            if isinstance(pv, lmfit.Parameter):
                _params[pn] = pv
            else:
                _params[pn] = lmfit.Parameter(pn, pv)

        if dry:
            lmfit_result = lmfit.model.ModelResult(model, params=_params, data=self.data,
                                                   fcn_kws=dict(coordinates=self.coordinates))
        else:
            lmfit_result = lmfit.minimize(self.residual, _params, args=(self.coordinates, self.data))

        return AntiCrossingResult(lmfit_result, x_data=self.coordinates, model=self.model, data=self.data)



if __name__ == "__main__":
    # plt.close("all")
    
    # creates some fake data
    biasList = np.linspace(2, 3, 501)
    f0 = 5
    x0 = 2.5
    k = -0.2
    f1 = lin_freq(biasList, f0, x0, k)
    g = 0.02
    freqs = hybridize_freq(f0, f1, g)
    plt.figure()
    plt.plot(biasList, freqs[0, :])
    plt.plot(biasList, freqs[1, :])
    plt.plot(biasList, f1, "--")
    plt.plot(biasList, [f0]*len(biasList), "--")

    # add noise and disable some data
    data = freqs + (np.random.rand(*freqs.shape)- 0.5)*0.02
    data[0][:200] = [np.nan] * 200
    data[1][-200:] = [np.nan] * 200
    plt.figure()
    plt.plot(biasList, data[0, :])
    plt.plot(biasList, data[1, :])


    # fitting
    fit = AntiCrossing(biasList, data)
    result = fit.run()
    print(result.params)
    result.plot()
    result.print()






