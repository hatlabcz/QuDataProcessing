from typing import Tuple, Any, Optional, Union, Dict, List

import numpy as np
from matplotlib import pyplot as plt
import json
from scipy.optimize import minimize_scalar

from QuDataProcessing.base import Analysis, AnalysisResult

class RotateResult(AnalysisResult):
    def plot(self,figName=None):
        x_data = self.params['x_data'].value
        plt.figure(figName)
        plt.plot(x_data, self.params['i_data'].value)
        plt.plot(x_data, self.params['q_data'].value)


class RotateData(Analysis):
    """
    rotate the iq data in rad units, return result class that contains new IQ data and rotation angle.
    """
    @staticmethod
    def analyze(x_data, iq_data, angle:Union[float,str]="find", resolution=10001):
        x_data = np.array(x_data)
        iq_data = np.array(iq_data)
        i_data, q_data = iq_data.real, iq_data.imag
        if angle == "find":
            def std_q(rot_agl_):
                i_temp, q_temp = rotate_complex(i_data, q_data, rot_agl_)
                return np.std(q_temp)
            res = minimize_scalar(std_q, bounds=[0, 360])
            rotation_angle = res.x

            # -------- old searching method -----------------------------
            # std_ = []
            # try_angle = np.linspace(0, 2 * np.pi, resolution)
            # for agl in try_angle:
            #     i_temp, q_temp = rotate_complex(i_data, q_data, agl)
            #     std_.append(np.std(q_temp))
            # rotation_angle = try_angle[np.argmin(std_)]

        elif type(angle) in [float, np.float, np.float64]:
            rotation_angle = angle
        i_new, q_new = rotate_complex(i_data, q_data, rotation_angle)
        return RotateResult(dict(x_data=x_data, i_data=i_new,q_data=q_new, rot_angle=rotation_angle))


def rotate_complex(real_part, imag_part, angle):
    """
    rotate the complex number as rad units.
    """
    iq_new = (real_part + 1j * imag_part) * np.exp(1j * np.pi * angle/180)
    return iq_new.real, iq_new.imag


if __name__=="__main__":
    fileName = "Q1_piPulse"
    filePath = r"C:/Users/hatla/Downloads//"
    with open(filePath + fileName, 'r') as infile:
        dataDict = json.load(infile)

    x_data = dataDict["x_data"]
    i_data = np.array(dataDict["i_data"])
    q_data = np.array(dataDict["q_data"])

    rotIQ = RotateData(x_data, i_data+1j*q_data)
    iq_new = rotIQ.run()
    iq_new.plot()
