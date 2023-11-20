from typing import Tuple, Any, Optional, Union, Dict, List

import numpy as np
import matplotlib.pyplot as plt
import lmfit
from lmfit.model import ModelResult
from QuDataProcessing.fitter.fitter_base import Fit, FitResult
from QuDataProcessing.helpers.unit_converter import freqUnit, rounder, realImag2magPhase
from scipy.ndimage import gaussian_filter

TWOPI = 2 * np.pi
PI = np.pi


def twoD_gaussian_func(coord: tuple, amp, x0, y0, sigmaX, sigmaY, theta, offset):
    """ 2D gaussian function
        https://en.wikipedia.org/wiki/Gaussian_function
    """
    (x, y) = coord
    a = (np.cos(theta) ** 2) / (2 * sigmaX ** 2) + (np.sin(theta) ** 2) / (2 * sigmaY ** 2)
    b = -(np.sin(2 * theta)) / (4 * sigmaX ** 2) + (np.sin(2 * theta)) / (4 * sigmaY ** 2)
    c = (np.sin(theta) ** 2) / (2 * sigmaX ** 2) + (np.cos(theta) ** 2) / (2 * sigmaY ** 2)
    z = offset + amp * np.exp(- (a * ((x - x0) ** 2) + 2 * b * (x - x0) * (y - y0)
                                 + c * ((y - y0) ** 2)))
    return z


def guess_gau2D_params(x, y, z, nBlobs, maskIndex=None):
    """ find the centers of gaussian blobs by looking for peaks of the 2d histogram,
        assume each blob is seperated by maskIndex

    :param x: bin edges along the first dimention
    :param y: bin edges along the second dimention
    :param z: The bi-dimensional histogram of samples x and y
    :param nBlobs: number of gaussian blobs
    :param maskIndex: guessed size of each blob
    :return:
    """
    border = np.max(np.abs(np.array([x, y])))

    ampList = np.zeros(nBlobs)
    x0List = np.zeros(nBlobs)
    y0List = np.zeros(nBlobs)
    sigmaXList = np.zeros(nBlobs) + border / 15
    sigmaYList = np.zeros(nBlobs) + border / 15
    thetaList = np.zeros(nBlobs)
    offsetLost = np.zeros(nBlobs)

    for i in range(nBlobs):
        x1indx, y1indx = np.unravel_index(np.argmax(z, axis=None), z.shape)
        x1ini, y1ini = x[x1indx, y1indx], y[x1indx, y1indx]
        amp1 = np.max(z)
        mask1 = np.zeros((len(x), len(y)))
        mask1[-maskIndex + x1indx:maskIndex + x1indx, -maskIndex + y1indx:maskIndex + y1indx] = 1
        z = np.ma.masked_array(z, mask=mask1)

        ampList[i] = amp1
        x0List[i], y0List[i] = x1ini, y1ini

    return ampList, x0List, y0List, sigmaXList, sigmaYList, thetaList, offsetLost


class Gaussian2DResult(FitResult):
    def __init__(self, lmfit_result: lmfit.model.ModelResult, coord, data, nBlobs):
        self.lmfit_result = lmfit_result
        self.coord = coord
        self.data = data
        self.params = lmfit_result.params
        self.nBlobs = nBlobs
        for p in self.params:
            self.__setattr__(p, self.params[p].value)

        if nBlobs == 2:
            self.sigma_g = np.sqrt(self.sigmaX1 ** 2 + self.sigmaY1 ** 2)
            self.sigma_e = np.sqrt(self.sigmaX2 ** 2 + self.sigmaY2 ** 2)
            self.ImOverSigma = np.sqrt(
                (self.x2 - self.x1) ** 2 + (self.y2 - self.y1) ** 2) / self.sigma_g

        #reorder the fit result in order of amp
        resultParams = lmfit_result.params
        ampResults = np.array([resultParams[f"amp{i + 1}"] for i in range(self.nBlobs)])
        x0Results = np.array([resultParams[f"x{i + 1}"] for i in range(self.nBlobs)])
        y0Results = np.array([resultParams[f"y{i + 1}"] for i in range(self.nBlobs)])
        sigmaXResults = np.array([resultParams[f"sigmaX{i + 1}"] for i in range(self.nBlobs)])
        sigmaYResults = np.array([resultParams[f"sigmaY{i + 1}"] for i in range(self.nBlobs)])

        stateOrder = np.argsort(ampResults)[::-1]
        self.state_x_list = x0Results[stateOrder]
        self.state_y_list = y0Results[stateOrder]
        state_sigmaX_list = sigmaXResults[stateOrder]
        state_sigmaY_list = sigmaYResults[stateOrder]
        state_sigma_list = np.sqrt(state_sigmaX_list ** 2 + state_sigmaY_list ** 2)

        self.state_xy_list = np.array([self.state_x_list, self.state_y_list]).T.flatten()
        self.state_location_list = np.append(self.state_xy_list, state_sigma_list)


    def plot(self, ax=None, **figArgs):
        x, y = self.coord
        # z = gaussian_filter(self.data, [2, 2])
        z = self.data
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        ax.set_title("state fitting")
        ax.pcolormesh(x, y, z, shading="auto", cmap="hot")
        ax.set_aspect(1)
        ax.contour(x, y, self.lmfit_result.best_fit, linestyles="dotted", colors='w')
        ax.scatter(self.state_x_list, self.state_y_list, c="r", s=0.7)

        if self.nBlobs > 1:
            for i, txt in enumerate(["g", "e", "f"][:self.nBlobs]):
                ax.annotate(txt, (self.state_x_list[i], self.state_y_list[i]))

    def print(self):
        for p in self.params:
            print(f"{p}: {np.round(self.params[p].value, 4)}")
        dec = int(max(-np.log10(np.min(np.abs(self.state_location_list))), 0))
        print(f"stateLocation: {np.around(self.state_location_list, dec)}")
        return np.around(self.state_location_list, dec)


class Gaussian2D_Base(Fit):
    def __init__(self, coordinates: Union[Tuple[np.ndarray, ...], np.ndarray],
                 data: np.ndarray, nBlobs=2, maskIndex=None):
        """ fit multiple 2D gaussian blobs
        :param nBlobs:  number of gussian blobs
        :param maskIndex: guessed size of each blob
        """
        self.coordinates = coordinates
        self.data = data
        self.nBlobs = nBlobs
        self.maskIndex = maskIndex
        self.pre_process()

    def guess(self, coordinates, data):
        border = np.max(np.abs(coordinates))
        if self.maskIndex is None:
            self.maskIndex =  int(len(coordinates[0]) // 5)

        (x, y) = coordinates
        z = gaussian_filter(data, [2, 2])

        ampList, x0List, y0List, sigmaXList, sigmaYList, thetaList, offsetLost = \
            guess_gau2D_params(x, y, z, self.nBlobs, self.maskIndex)

        params = lmfit.Model(self.model).make_params()
        paramDict = dict(params)

        for i in range(self.nBlobs):
            paramDict[f"amp{i + 1}"] = lmfit.Parameter(f"amp{i + 1}", value=ampList[i], min=0,
                                                       max=ampList[0] * 2)
            paramDict[f"x{i + 1}"] = lmfit.Parameter(f"x{i + 1}", value=x0List[i], min=-border,
                                                     max=border)
            paramDict[f"y{i + 1}"] = lmfit.Parameter(f"y{i + 1}", value=y0List[i], min=-border,
                                                     max=border)
            paramDict[f"sigmaX{i + 1}"] = lmfit.Parameter(f"sigmaX{i + 1}", value=sigmaXList[i],
                                                          min=0, max=border / 2)
            paramDict[f"sigmaY{i + 1}"] = lmfit.Parameter(f"sigmaY{i + 1}", value=sigmaYList[i],
                                                          min=0, max=border / 2)
            paramDict[f"theta{i + 1}"] = lmfit.Parameter(f"theta{i + 1}", value=thetaList[i], min=0,
                                                         max=TWOPI)
            paramDict[f"offset{i + 1}"] = lmfit.Parameter(f"offset{i + 1}", value=offsetLost[i],
                                                          min=0, max=ampList[0] / 5)

        return paramDict

    def run(self, *args: Any, **kwargs: Any) -> Gaussian2DResult:
        lmfit_result = self.analyze(self.coordinates, self.data, *args, **kwargs)
        return Gaussian2DResult(lmfit_result, self.coordinates, self.data, self.nBlobs)


class Gaussian2D_1Blob(Gaussian2D_Base):
    def __init__(self, coordinates: Union[Tuple[np.ndarray, ...], np.ndarray],
                 data: np.ndarray, maskIndex=None):
        super().__init__(coordinates, data, 1, maskIndex)

    @staticmethod
    def model(coordinates, amp1, x1, y1, sigmaX1, sigmaY1, theta1, offset1):
        """"multiple 2D gaussian function"""
        z = twoD_gaussian_func(coordinates, amp1, x1, y1, sigmaX1, sigmaY1, theta1, offset1)
        return z

class Gaussian2D_2Blob(Gaussian2D_Base):
    def __init__(self, coordinates: Union[Tuple[np.ndarray, ...], np.ndarray],
                 data: np.ndarray, maskIndex=None):
        super().__init__(coordinates, data, 2, maskIndex)

    @staticmethod
    def model(coordinates,
              amp1, x1, y1, sigmaX1, sigmaY1, theta1, offset1,
              amp2, x2, y2, sigmaX2, sigmaY2, theta2, offset2):
        """"multiple 2D gaussian function"""
        z = twoD_gaussian_func(coordinates, amp1, x1, y1, sigmaX1, sigmaY1, theta1, offset1) + \
            twoD_gaussian_func(coordinates, amp2, x2, y2, sigmaX2, sigmaY2, theta2, offset2)
        return z


class Gaussian2D_3Blob(Gaussian2D_Base):
    def __init__(self, coordinates: Union[Tuple[np.ndarray, ...], np.ndarray],
                 data: np.ndarray, maskIndex=None):
        super().__init__(coordinates, data, 3, maskIndex)

    @staticmethod
    def model(coordinates,
              amp1, x1, y1, sigmaX1, sigmaY1, theta1, offset1,
              amp2, x2, y2, sigmaX2, sigmaY2, theta2, offset2,
              amp3, x3, y3, sigmaX3, sigmaY3, theta3, offset3):
        """"multiple 2D gaussian function"""
        z = twoD_gaussian_func(coordinates, amp1, x1, y1, sigmaX1, sigmaY1, theta1, offset1) + \
            twoD_gaussian_func(coordinates, amp2, x2, y2, sigmaX2, sigmaY2, theta2, offset2) + \
            twoD_gaussian_func(coordinates, amp3, x3, y3, sigmaX3, sigmaY3, theta3, offset3)
        return z


def histo2DFitting(bufi, bufq, bins=101, histRange=None, blobs=2, maskIndex=None, guessParams={}):
    if histRange is None:
        max_val = np.max(np.abs([bufi, bufq]))
        histRange = [[-max_val, max_val], [-max_val, max_val]]
    z_, x_, y_ = np.histogram2d(bufi.flatten(), bufq.flatten(), bins=bins, range=np.array(histRange))
    z_ = z_.T
    xd, yd = np.meshgrid(x_[:-1], y_[:-1])
    if blobs == 1:
        gau2DFit = Gaussian2D_1Blob((xd, yd), z_, maskIndex=maskIndex)
        fitResult = gau2DFit.run(params=guessParams)
    elif blobs == 2:
        gau2DFit = Gaussian2D_2Blob((xd, yd), z_, maskIndex=maskIndex)
        fitResult = gau2DFit.run(params=guessParams)
    elif blobs == 3:
        gau2DFit = Gaussian2D_3Blob((xd, yd), z_, maskIndex=maskIndex)
        fitResult = gau2DFit.run(params=guessParams)
    else:
        raise NotImplementedError
    return fitResult

if __name__ == '__main__':
    pass
