"""
Created on Mon May 30 12:04:54 2022

@author: chao
"""


from typing import List, Callable, Union, Tuple, Dict
import warnings
import time

import matplotlib.pyplot as plt
import numpy as np
import h5py
from tqdm import tqdm
from QuDataProcessing.fitter.arb_gaussian import classify_point, peakfinder_2d


from QuDataProcessing.fitter.gaussian_2d import Gaussian2D_2Blob, Gaussian2D_3Blob
from QuDataProcessing.slider_plot.sliderPlot import sliderHist2d

def auto_hist_range(data_I, data_Q):
    data_max = np.max(np.abs(np.array([data_I, data_Q])))
    histRange = [[-data_max, data_max], [-data_max, data_max]]
    return histRange

class PostSelectionData_Base():
    def __init__(self, data_I: np.ndarray, data_Q: np.ndarray, selPattern: List = [1, 0],
                 histRange=None):
        """ A base class fro post selection data. doesn't specify the size of Hilbert space of the qubit.
        :param data_I:  I data
        :param data_Q:  Q data
        :param selPattern: list of 1 and 0 that represents which pulse is selection and which pulse is experiment msmt
            in one experiment sequence. For example [1, 1, 0] represents, for each three data points, the first two are
            used for selection and the third one is experiment point.
        """

        self.data_I_raw = data_I
        self.data_Q_raw = data_Q
        self.selPattern = selPattern
        if histRange is None:
            self.histRange = auto_hist_range(data_I, data_Q)
        else:
            self.histRange = histRange

        n_avg = len(data_I)
        pts_per_exp = len(data_I[0])
        msmt_per_sel = len(selPattern)
        if pts_per_exp % msmt_per_sel != 0:
            raise ValueError(
                f"selPattern is not valid. the length of selPattern {len(selPattern)} is not a factor of "
                f"points per experiment {pts_per_exp}")
        n_sweep = int(np.round(pts_per_exp // msmt_per_sel))  # e.g. len(xData)

        self.sel_data_msk = np.array(selPattern, dtype=bool)
        self.exp_data_msk = ~ self.sel_data_msk
        self.data_I = data_I.reshape((n_avg, n_sweep, msmt_per_sel))
        self.data_Q = data_Q.reshape((n_avg, n_sweep, msmt_per_sel))

        self.I_sel = self.data_I[:, :, self.sel_data_msk]  # gather selection data
        self.Q_sel = self.data_Q[:, :, self.sel_data_msk]
        self.I_exp = self.data_I[:, :, self.exp_data_msk]  # gather experiment data
        self.Q_exp = self.data_Q[:, :, self.exp_data_msk]

    def state_split_line(self, x1, y1, x2, y2, x):
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        k_ = -(x1 - x2) / (y1 - y2)
        return k_ * (x - center_x) + center_y

    def mask_state_by_circle(self, sel_idx: int, state_x: float, state_y: float, state_r: float,
                             plot: Union[bool, int] = True, state_name: str = "", plot_ax=None):
        """
        :param sel_idx: index of the data for selection, must be '1' position in selPattern
        :param state_x: x position of the state on IQ plane
        :param state_y: y position of the state on IQ plane
        :param state_r: size of the selection circle
        :param plot: if true, plot selection
        :param state_name: name of the state, will be used in the plotting title.
        :return:
        """
        if self.selPattern[sel_idx] != 1:
            raise ValueError(
                f"sel_idx must be a position with value 1 in selPattern {self.selPattern}")
        idx_ = np.where(np.where(np.array(self.selPattern) == 1)[0] == sel_idx)[0][0]
        I_sel_ = self.I_sel[:, :, idx_]
        Q_sel_ = self.Q_sel[:, :, idx_]
        mask = (I_sel_ - state_x) ** 2 + (Q_sel_ - state_y) ** 2 < (state_r) ** 2
        if plot:
            if plot_ax is None:
                fig, ax = plt.subplots(1, 1)
            else:
                ax = plot_ax
            ax.set_title(f'{state_name} state selection range')
            ax.hist2d(I_sel_.flatten(), Q_sel_.flatten(), bins=101, range=self.histRange)
            theta = np.linspace(0, 2 * np.pi, 201)
            ax.plot(state_x + state_r * np.cos(theta), state_y + state_r * np.sin(theta),
                     color='r')
            ax.set_aspect(1)

        return mask

    def sel_data(self, mask, plot=True, plot_ax=None, progress=False):
        self.I_vld = []
        self.Q_vld = []
        if progress:
            rs1  = tqdm(range(self.I_exp.shape[1]), desc="selecting data")
        else:
            rs1 = range(self.I_exp.shape[1])
        for i in rs1:
            for j in range(self.I_exp.shape[2]):
                self.I_vld.append(self.I_exp[:, i, j][mask[:, i]])
                self.Q_vld.append(self.Q_exp[:, i, j][mask[:, i]])
        if plot:
            if plot_ax is None:
                fig, ax = plt.subplots(1, 1)
            else:
                ax = plot_ax
            selNum = np.average(list(map(len, self.I_vld)))
            ax.set_title(
                'all experiment pts after selection\n' + "sel%: " + str(selNum / len(self.data_I_raw)))
            ax.hist2d(np.hstack(self.I_vld), np.hstack(self.Q_vld), bins=101, range=self.histRange)
            ax.set_aspect(1)
            print("sel%: " + str(selNum / len(self.data_I_raw)))

        selNum = np.average(list(map(len, self.I_vld)))
        self.sel_pct = selNum / len(self.data_I_raw)
        print("sel%: " + str(self.sel_pct))
        return self.I_vld, self.Q_vld

    def auto_fit(self, nBlobs=2, fitGuess={}, bins=201, stateMask=None, plotGauFitting=True, dryRun=False):
        fit_I = self.I_sel.flatten()
        fit_Q = self.Q_sel.flatten()
        self.stateMask = stateMask

        z_, x_, y_ = np.histogram2d(fit_I, fit_Q, bins=bins, range=np.array(self.histRange))
        z_ = z_.T
        xd, yd = np.meshgrid(x_[:-1], y_[:-1])

        if nBlobs == 2:
            gau2DFit = Gaussian2D_2Blob((xd, yd), z_, stateMask)
        elif nBlobs == 3:
            gau2DFit = Gaussian2D_3Blob((xd, yd), z_, stateMask)
        else:
            raise ValueError(f"gau blob number {nBlobs} not implemented")

        fitResult = gau2DFit.run(params=fitGuess, dry=dryRun)
        if plotGauFitting:
            fitResult.print()
            fitResult.plot()

        return fitResult

    def sliderPlotSelectedData(self, xData:dict=None):
        """ plot the selected data histogram in a slider plot
        :param xData: dictionary that contains the variables that are swept in the experiment.
                        e.g : {"amp", np.linspace(0,1 101)}
        """
        if xData is None:
            xData = {"exp": np.arange(len(self.I_vld))}

        x_shape = list(map(len, xData.values()))
        if x_shape == [1]:
            return
        I_vld = np.array(self.I_vld, dtype=object).reshape(*x_shape)
        Q_vld = np.array(self.Q_vld, dtype=object).reshape(*x_shape)

        self.resultSld = sliderHist2d(I_vld, Q_vld,
                                      axes_dict=xData,
                                      range=self.histRange, bins=self.histBins)


class PostSelectionData_ge(PostSelectionData_Base):
    def __init__(self, data_I: np.ndarray, data_Q: np.ndarray, selPattern: List = [1, 0],
                 geLocation: List[float] = None, plotGauFitting=True,
                 fitGuess: dict = {}, stateMask: int = None, histBins=201, histRange=None):
        """ A post selection data class that assumes a qubit has two possible states
        :param data_I:  I data
        :param data_Q:  Q data
        :param selPattern: list of 1 and 0 that represents which pulse is selection and which pulse is experiment msmt
            in one experiment sequence. For example [1, 1, 0] represents, for each three data points, the first two are
            used for selection and the third one is experiment point.
        :param geLocation:  [g_x, g_y, e_x, e_y, g_r, e_r]
        :param fitGuess: guess parameters for gaussian fitting. should be dict of lmfit Parameters. This could be
                         useful when the blobs are slowly rotating.
        """
        super().__init__(data_I, data_Q, selPattern, histRange)
        # fit for g, e gaussian if g/e state location is not provided
        if geLocation is None:
            self.stateFitResult = self.auto_fit(2, fitGuess, histBins, stateMask, plotGauFitting)
            geLocation = self.stateFitResult.state_location_list
        else:
            self.stateFitResult = self.auto_fit(2, fitGuess, histBins, stateMask, plotGauFitting, dryRun=True)
        self.geLocation = geLocation
        self.g_x, self.g_y, self.e_x, self.e_y, self.g_r, self.e_r = self.geLocation
        self.histBins = histBins

    def ge_split_line(self, x):
        return self.state_split_line(self.g_x, self.g_y, self.e_x, self.e_y, x)

    def mask_g_by_circle(self, sel_idx: int = 0, circle_size: float = 1,
                         plot: Union[bool, int] = True, plot_ax=None):
        """
        :param sel_idx: index of the data for selection, must be '1' position in selPattern
        :param circle_size: size of the selection circle, in unit of g_r (sigma of g state Gaussian blob)
        :param plot:
        :return:
        """
        mask = self.mask_state_by_circle(sel_idx, self.g_x, self.g_y, self.g_r * circle_size,
                                         plot, "g", plot_ax)
        return mask

    def mask_e_by_circle(self, sel_idx: int = 0, circle_size: float = 1,
                         plot: Union[bool, int] = True, plot_ax=None):
        """
        :param sel_idx: index of the data for selection, must be '1' position in selPattern
        :param circle_size: size of the selection circle, in unit of e_r (sigma of e state Gaussian blob)
        :param plot:
        :return:
        """
        mask = self.mask_state_by_circle(sel_idx, self.e_x, self.e_y, self.e_r * circle_size,
                                         plot, "e", plot_ax)
        return mask

    def mask_g_by_line(self, sel_idx: int = 0, line_rotate: float = 0, line_shift: float = 0,
                       plot: Union[bool, int] = True, plot_ax=None):
        """
        :param sel_idx: index of the data for selection, must be '1' position in selPattern
        :param line_rotate: rotation angle the split line in counter clockwise direction, in unit of rad. Zero angle
            is the the perpendicular bisector of g and e.
        :param line_shift: shift the split line along the e -> g direction, in unit of half ge distance
        :param plot:
        :return:
        """
        if self.selPattern[sel_idx] != 1:
            raise ValueError(
                f"sel_idx must be a position with value 1 in selPattern {self.selPattern}")
        idx_ = np.where(np.where(np.array(self.selPattern) == 1)[0] == sel_idx)[0][0]
        I_sel_ = self.I_sel[:, :, idx_]
        Q_sel_ = self.Q_sel[:, :, idx_]

        def rotate_ge_line_(x, theta):
            k_ = -(self.g_x - self.e_x) / (self.g_y - self.e_y)
            x0_ = (self.g_x + self.e_x) / 2
            y0_ = (self.g_y + self.e_y) / 2
            return (x - x0_) * np.tan(np.arctan(k_) + theta) + y0_

        rotate_split_line = lambda x: rotate_ge_line_(x, line_rotate)
        shift_split_line = lambda x: rotate_split_line(x - line_shift * 0.5 * (self.g_x - self.e_x)) \
                                     + line_shift * 0.5 * (self.g_y - self.e_y)
        if self.g_y < self.e_y:
            mask = Q_sel_ < shift_split_line(I_sel_)
        else:
            mask = Q_sel_ > shift_split_line(I_sel_)

        if plot:
            if plot_ax is None:
                fig, ax = plt.subplots(1, 1)
            else:
                ax = plot_ax
            ax.set_title('g state selection range')
            h, xedges, yedges, image = ax.hist2d(I_sel_.flatten(), Q_sel_.flatten(), bins=101,
                                                  range=self.histRange)
            ax.plot(xedges, shift_split_line(xedges), color='r')
            ax.plot([(self.g_x + self.e_x) / 2], [(self.g_y + self.e_y) / 2], "*")
            ax.set_aspect(1)
        return mask

    def cal_g_pct(self, plot=False, plot_ax=None, progress=False):
        if progress:
            ii_ = tqdm(range(len(self.I_vld)), desc="calculating g_pct")
        else:
            ii_ = range(len(self.I_vld))
        g_pct_list = []
        for i in ii_:
            I_v = self.I_vld[i]
            Q_v = self.Q_vld[i]
            n_pts = float(len(I_v))
            if self.g_y < self.e_y:
                mask = Q_v < self.ge_split_line(I_v)
            else:
                mask = Q_v > self.ge_split_line(I_v)
            try:
                g_pct_list.append(len(I_v[mask]) / n_pts)
            except ZeroDivisionError:
                warnings.warn(
                    "! no valid point, this is probably because of wrong gaussian fitting, "
                    "please double check the system and the fitting function")
                g_pct_list.append(1)
        if plot:
            if plot_ax is None:
                fig, ax = plt.subplots(1, 1)
            else:
                ax = plot_ax
            ax.set_title("g/e state region")
            h, xedges, yedges, image = ax.hist2d(np.hstack(self.I_vld), np.hstack(self.Q_vld),
                                                  bins=101,
                                                  range=self.histRange)
            ax.plot(xedges, self.ge_split_line(xedges), color='r')
            ax.plot([(self.g_x + self.e_x) / 2], [(self.g_y + self.e_y) / 2], "*")
            ax.set_aspect(1)

        return np.array(g_pct_list)

    def cal_stateForEachMsmt(self):
        stateForEachMsmt = []
        for i in range(len(self.I_vld)):
            I_v = self.I_vld[i]
            Q_v = self.Q_vld[i]
            if self.g_y < self.e_y:
                mask = Q_v < self.ge_split_line(I_v)
            else:
                mask = Q_v > self.ge_split_line(I_v)
            state = mask * 2 - 1
            stateForEachMsmt.append(state)

        return stateForEachMsmt


class PostSelectionData_gef(PostSelectionData_Base):
    def __init__(self, data_I: np.ndarray, data_Q: np.ndarray, selPattern: List = [1, 0],
                 gefLocation: List[float] = None, plotGauFitting=True,
                 fitGuess=None, stateMask: int = None, histBins=201, histRange=None):
        """ A post selection data class that assumes a qubit has three possible states
        :param data_I:  I data
        :param data_Q:  Q data
        :param selPattern: list of 1 and 0 that represents which pulse is selection and which pulse is experiment msmt
            in one experiment sequence. For example [1, 1, 0] represents, for each three data points, the first two are
            used for selection and the third one is experiment point.
        :param gefLocation:  [g_x, g_y, e_x, e_y, f_x, f_y, g_r, e_r, f_r],
        :param plotGauFitting: plot fitting result or not
        :param fitGuess:  guess parameter for gau blob fitting, should be dict of lmfit Parameters. This could be
                         useful when the blobs are slowly rotating.
        :param stateMask: guessed size of each blob, in unit of number of bins
        :param histBins: number of bins for histogram
        :param histRange: range of histogram
        """
        super().__init__(data_I, data_Q, selPattern, histRange)
        # fit for g, e, f gaussian if g/e/f state location is not provided
        if gefLocation == None:
            self.stateFitResult = self.auto_fit(3, fitGuess, histBins, stateMask, plotGauFitting)
            gefLocation = self.stateFitResult.state_location_list

        self.gefLocation = gefLocation
        self.g_x, self.g_y, self.e_x, self.e_y, self.f_x, self.f_y, self.g_r, self.e_r, self.f_r = self.gefLocation
        self.histBins = histBins

        # find the circumcenter of the three states
        d_ = 2 * (self.g_x * (self.e_y - self.f_y) + self.e_x * (self.f_y - self.g_y)
                  + self.f_x * (self.g_y - self.e_y))
        self.ext_center_x = ((self.g_x ** 2 + self.g_y ** 2) * (self.e_y - self.f_y)
                             + (self.e_x ** 2 + self.e_y ** 2) * (self.f_y - self.g_y)
                             + (self.f_x ** 2 + self.f_y ** 2) * (self.g_y - self.e_y)) / d_
        self.ext_center_y = ((self.g_x ** 2 + self.g_y ** 2) * (self.f_x - self.e_x)
                             + (self.e_x ** 2 + self.e_y ** 2) * (self.g_x - self.f_x)
                             + (self.f_x ** 2 + self.f_y ** 2) * (self.e_x - self.g_x)) / d_

        self.in_center_x = np.mean([self.g_x, self.e_x, self.f_x])
        self.in_center_y = np.mean([self.g_y, self.e_y, self.f_y])

    def ge_split_line(self, x):
        return self.state_split_line(self.g_x, self.g_y, self.e_x, self.e_y, x)

    def ef_split_line(self, x):
        return self.state_split_line(self.e_x, self.e_y, self.f_x, self.f_y, x)

    def gf_split_line(self, x):
        return self.state_split_line(self.g_x, self.g_y, self.f_x, self.f_y, x)

    def mask_g_by_circle(self, sel_idx: int = 0, circle_size: float = 1,
                         plot: Union[bool, int] = True, plot_ax=None):
        """
        :param sel_idx: index of the data for selection, must be '1' position in selPattern
        :param circle_size: size of the selection circle, in unit of g_r
        :param plot:
        :return:
        """
        mask = self.mask_state_by_circle(sel_idx, self.g_x, self.g_y, self.g_r * circle_size,
                                         plot, "g", plot_ax)
        return mask

    def mask_e_by_circle(self, sel_idx: int = 0, circle_size: float = 1,
                         plot: Union[bool, int] = True, plot_ax=None):
        """
        :param sel_idx: index of the data for selection, must be '1' position in selPattern
        :param circle_size: size of the selection circle, in unit of e_r
        :param plot:
        :return:
        """
        mask = self.mask_state_by_circle(sel_idx, self.e_x, self.e_y, self.e_r * circle_size,
                                         plot, "e", plot_ax)
        return mask

    def mask_f_by_circle(self, sel_idx: int = 0, circle_size: float = 1,
                         plot: Union[bool, int] = True, plot_ax=None):
        """
        :param sel_idx: index of the data for selection, must be '1' position in selPattern
        :param circle_size: size of the selection circle, in unit of f_r
        :param plot:
        :return:
        """
        mask = self.mask_state_by_circle(sel_idx, self.f_x, self.f_y, self.f_r * circle_size,
                                         plot, "f", plot_ax)
        return mask

    def cal_g_pct(self, plot=True, plot_ax=None, progress=False):
        if progress:
            ii_ = tqdm(range(len(self.I_vld)), desc="calculating g_pct")
        else:
            ii_ = range(len(self.I_vld))
        g_pct_list = []
        for i in range(len(self.I_vld)):
            I_v = self.I_vld[i]
            Q_v = self.Q_vld[i]
            n_pts = float(len(I_v))
            g_dist = (I_v - self.g_x) ** 2 + (Q_v - self.g_y) ** 2
            e_dist = (I_v - self.e_x) ** 2 + (Q_v - self.e_y) ** 2
            f_dist = (I_v - self.f_x) ** 2 + (Q_v - self.f_y) ** 2
            state_ = np.argmin([g_dist, e_dist, f_dist], axis=0)
            g_mask = np.where(state_ == 0)[0]
            g_pct_list.append(len(g_mask) / n_pts)

        if plot:
            if plot_ax is None:
                fig, ax = plt.subplots(1, 1)
            else:
                ax = plot_ax
            h, xedges, yedges, image = ax.hist2d(np.hstack(self.I_vld), np.hstack(self.Q_vld),
                                                  bins=101,
                                                  range=self.histRange)

            def get_line_range_(s1, s2):
                """get the x range to plot for the line that splits three states"""
                x12 = np.mean([getattr(self, f"{s1}_x"), getattr(self, f"{s2}_x")])
                y12 = np.mean([getattr(self, f"{s1}_y"), getattr(self, f"{s2}_y")])

                v1 = [self.ext_center_x - x12, self.ext_center_y - y12]
                v2 = [self.in_center_x - x12, self.in_center_y - y12]
                if (np.dot(v1, v2) > 0 and v1[0] > 0) or (np.dot(v1, v2) < 0 and v1[0] < 0):
                    return np.array([xedges[0], self.ext_center_x])
                else:
                    return np.array([self.ext_center_x, xedges[-1]])

            x_l_ge = get_line_range_("g", "e")
            x_l_ef = get_line_range_("e", "f")
            x_l_gf = get_line_range_("g", "f")

            ax.plot(x_l_ge, self.ge_split_line(x_l_ge), color='r')
            ax.plot(x_l_ef, self.ef_split_line(x_l_ef), color='g')
            ax.plot(x_l_gf, self.gf_split_line(x_l_gf), color='b')
            ax.plot([self.ext_center_x], [self.ext_center_y], "*")
        return np.array(g_pct_list)

    def cal_gef_pct(self, plot=True, plot_ax=None, progress=False):
        if progress:
            ii_ = tqdm(range(len(self.I_vld)), desc="calculating gef_pct")
        else:
            ii_ = range(len(self.I_vld))
        g_pct_list = []
        e_pct_list = []
        f_pct_list = []
        for i in range(len(self.I_vld)):
            I_v = self.I_vld[i]
            Q_v = self.Q_vld[i]
            n_pts = float(len(I_v))
            g_dist = (I_v - self.g_x) ** 2 + (Q_v - self.g_y) ** 2
            e_dist = (I_v - self.e_x) ** 2 + (Q_v - self.e_y) ** 2
            f_dist = (I_v - self.f_x) ** 2 + (Q_v - self.f_y) ** 2
            state_ = np.argmin([g_dist, e_dist, f_dist], axis=0)
            g_mask = np.where(state_ == 0)[0]
            g_pct_list.append(len(g_mask) / n_pts)
            e_mask = np.where(state_ == 1)[0]
            e_pct_list.append(len(e_mask) / n_pts)
            f_mask = np.where(state_ == 2)[0]
            f_pct_list.append(len(f_mask) / n_pts)

        if plot:
            if plot_ax is None:
                fig, ax = plt.subplots(1, 1)
            else:
                ax = plot_ax
            h, xedges, yedges, image = ax.hist2d(np.hstack(self.I_vld), np.hstack(self.Q_vld),
                                                  bins=101,
                                                  range=self.histRange)

            def get_line_range_(s1, s2):
                """get the x range to plot for the line that splits three states"""
                x12 = np.mean([getattr(self, f"{s1}_x"), getattr(self, f"{s2}_x")])
                y12 = np.mean([getattr(self, f"{s1}_y"), getattr(self, f"{s2}_y")])

                v1 = [self.ext_center_x - x12, self.ext_center_y - y12]
                v2 = [self.in_center_x - x12, self.in_center_y - y12]
                if (np.dot(v1, v2) > 0 and v1[0] > 0) or (np.dot(v1, v2) < 0 and v1[0] < 0):
                    return np.array([xedges[0], self.ext_center_x])
                else:
                    return np.array([self.ext_center_x, xedges[-1]])

            x_l_ge = get_line_range_("g", "e")
            x_l_ef = get_line_range_("e", "f")
            x_l_gf = get_line_range_("g", "f")

            ax.plot(x_l_ge, self.ge_split_line(x_l_ge), color='r')
            ax.plot(x_l_ef, self.ef_split_line(x_l_ef), color='g')
            ax.plot(x_l_gf, self.gf_split_line(x_l_gf), color='b')
            ax.plot([self.ext_center_x], [self.ext_center_y], "*")
        return np.array([np.array(g_pct_list), np.array(e_pct_list), np.array(f_pct_list)])

    def cal_stateForEachMsmt(self, gef=0):
        warnings.warn("now we consider f as e, didn't implement 3 states calculation yet")
        g_pct_list = []
        e_pct_list = []
        f_pct_list = []
        stateForEachMsmt = []

        for i in range(len(self.I_vld)):
            I_v = self.I_vld[i]
            Q_v = self.Q_vld[i]
            n_pts = float(len(I_v))
            g_dist = (I_v - self.g_x) ** 2 + (Q_v - self.g_y) ** 2
            e_dist = (I_v - self.e_x) ** 2 + (Q_v - self.e_y) ** 2
            f_dist = (I_v - self.f_x) ** 2 + (Q_v - self.f_y) ** 2
            state_ = np.argmin([g_dist, e_dist, f_dist], axis=0)
            g_mask = np.where(state_ == 0)[0]
            g_pct_list.append(len(g_mask) / n_pts)
            e_mask = np.where(state_ == 1)[0]
            e_pct_list.append(len(e_mask) / n_pts)
            f_mask = np.where(state_ == 2)[0]
            f_pct_list.append(len(f_mask) / n_pts)

            stateForSingleMsmt = state_.copy()
            stateForSingleMsmt[np.where(state_ == 0)[0]] = 1
            stateForSingleMsmt[np.where(state_ != 0)[0]] = -1
            stateForEachMsmt.append(stateForSingleMsmt)
        return stateForEachMsmt


class PostSelectionData_fast(PostSelectionData_Base):
    """ This post selection class does not use any kind of fitting and so is quite fast. It detects
    arbitary numbers of states. Notably one can use one dataset to generate the mask, while selecting for traces on another
    dataset, which is useful for detecting states with only a small number of counts.

    author: Boris Mesits 202308

    :param data_I:  I data
    :param data_Q:  Q data
    :param selPattern: list of 1 and 0 that represents which pulse is selection and which pulse is experiment msmt
        in one experiment sequence. For example [1, 1, 0] represents, for each three data points, the first two are
        used for selection and the third one is experiment point.
    :param mask_data:  3 tuple containing x, y, and histogram counts for generating the state masks. If none is
                        provided, a new histogram will be made from the inputs data_I and data_Q
    :param num_states: You get to specify how many states will be detected
    :params bins: bins of mask histogram if not exteranlly specified
    :params radius: minimum spacing of adjacent states (measured in histogram bin widths)
    :params select_radius: radius of the circle mask used to select the number of states
    """

    def __init__(self, data_I: np.array, data_Q: np.array, selPattern: List = [1, 0],
                 mask_I=None, mask_Q=None, num_states=2, bins=101, radius=1, select_radius=1, reorder=False):
        super().__init__(data_I, data_Q, selPattern)

        self.num_states = num_states
        self.radius = radius
        self.stateDict = {}
        self.select_radius = select_radius
        self.reorder=reorder

        # Only use a dedicated histogram to generate mask if specified, otherwise just use provided data to generate histogram.
        if mask_I is None or mask_Q is None:

            self.mask_I = data_I
            self.mask_Q = data_Q

        else:
            self.mask_I = mask_I
            self.mask_Q = mask_Q

        hist, x, y = np.histogram2d(self.mask_I.flatten(), self.mask_Q.flatten(), bins=(bins, bins))

        self.mask_hist = hist
        self.mask_x = x
        self.mask_y = y

        self.identify_histogram_states()

    def identify_histogram_states(self):

        idxx, idxy, heights, max_neighbors = peakfinder_2d(self.mask_hist, self.radius, self.num_states)
        idxx, idxy, heights, max_neighbors = self.peakfinder_2d(self.mask_hist, self.radius, self.num_states, add_noise=True)


        x = self.mask_x[idxx]
        y = self.mask_y[idxy]

        # a quick reordering routine, which picks g as the tallest, and then reorders based on angle from mid point
        # not very versatile, in general you may need to hand pick states or coherently prepare, no functionality for that yet
        
        if self.reorder == True:
        
            x_centered = x - np.mean(x)
            y_centered = y - np.mean(y)
    
            theta = np.angle(x_centered + 1j * y_centered)
            theta = theta - theta[0]
            theta[np.where(theta < 0)] = 2 * np.pi + theta[np.where(theta < 0)]
    
            order = np.argsort(theta)
    
            x = x[order]
            y = y[order]
            heights = heights[order]

        for i in range(0, self.num_states):
            self.stateDict[i] = {"x": x[i],
                                 "y": y[i],
                                 "height": heights[i]}

        # for now, doing a cheap guess of the gaussian width (sigma) from the minimum distance between states
        distances = np.zeros([self.num_states, self.num_states])

        for i in range(0, self.num_states):
            for j in range(0, self.num_states):
                distances[i, j] = np.sqrt((self.stateDict[i]['x'] - self.stateDict[j]['x']) ** 2 + (
                            self.stateDict[i]['y'] - self.stateDict[j]['y']) ** 2)

        self.distances = distances

        min_distance = np.sort(distances.flatten())[self.num_states]

        for i in range(0, self.num_states):
            self.stateDict[i]['r'] = min_distance / 2 * self.select_radius
            # %TODO make this better

    def classify_points(self, x_points, y_points, x_peaks, y_peaks):

        distances = np.zeros([len(x_points), len(x_peaks)])

        for i in range(0, len(x_peaks)):
            distances[:, i] = np.sqrt((x_points - x_peaks[i]) ** 2 + (y_points - y_peaks[i]) ** 2)

        states = np.argsort(distances, axis=1)[:, 0]

        return states

    def peakfinder_2d(self, zz, radius, num_peaks, add_noise=False):
        '''
        The fastest way I can think of without a just-in-time compiler. You can imagine that each point checks for
        neighboring points (radius r) and calls itself a peak if it's bigger than all its neighbors, not including
        edges. It's done with array slicing rather than explicit loop.

        Faster than looping to each point in the 2d array and comparing, but not way faster.

        :param zz: 2d data
        :param radius: Distance to check for higher neighbors
        :param num_peaks: Only take the largest of the detected peaks (largest value).
        '''

        neighbors = []

        for i in range(0, radius * 2):
            for j in range(0, radius * 2):

                if (i != radius or j != radius):
                    neighbor = zz[i:-radius * 2 + i,
                               j:-radius * 2 + j]  # not necessarily nearest neighbor if radius > 1

                    neighbors.append(neighbor)

            neighbor = zz[i:-radius * 2 + i, radius * 2:]

            neighbors.append(neighbor)

        for j in range(0, radius * 2):
            neighbor = zz[radius * 2:, j:-radius * 2 + j]

            neighbors.append(neighbor)

        neighbor = zz[radius * 2:, radius * 2:]

        neighbors.append(neighbor)

        neighbors = np.array(neighbors)

        max_neighbors = zz * 0 + np.max(zz)
        max_neighbors[radius:-radius, radius:-radius] = np.max(neighbors, axis=0)
        
        noise_amplitude = add_noise*np.max(zz)*1e-9

        idx = np.where(max_neighbors < zz + np.random.random(np.shape(zz))*noise_amplitude )  # identifies the peaks (i.e., finds their indices)
        # the noise is there as a tiebreaker if two adjacent point have the same number of counts
        # we could take the peak to be the average, but really this is not a fitting function, only a rough identifier of location

        idxx = idx[0]
        idxy = idx[1]

        heights = zz[idxx, idxy]

        order = np.flip(np.argsort(heights))

        idxx = idxx[order]
        idxy = idxy[order]
        heights = heights[order]

        # only takes the tallest peaks, according to the requested number of states

        if num_peaks != None:
            idxx = idxx[0:num_peaks]
            idxy = idxy[0:num_peaks]
            heights = heights[0:num_peaks]

        return idxx, idxy, heights, max_neighbors

    def mask_state_index_by_circle(self, stateLabel, sel_idx: int = 0, circle_size: float = 1,
                                   plot: Union[bool, int] = False, plot_ax=None):
        mask = self.mask_state_by_circle(sel_idx, self.stateDict[stateLabel]['x'], self.stateDict[stateLabel]['y'],
                                         self.stateDict[stateLabel]['r'], plot, stateLabel, plot_ax=plot_ax)
        return mask

    def cal_state_pct(self, calStateLabel, plot=True, res_plot_ax=None):
        '''
        Calculate state population probablity.
        '''

        I_peaks = []
        Q_peaks = []

        for i in range(0, self.num_states):
            I_peaks.append(self.stateDict[i]['x'])
            Q_peaks.append(self.stateDict[i]['y'])

        I_peaks = np.array(I_peaks)
        Q_peaks = np.array(Q_peaks)

        state_pct_list = []

        all_states = []

        for i in range(len(self.I_vld)):
            I_v = self.I_vld[i]
            Q_v = self.Q_vld[i]

            n_pts = len(I_v)

            states = self.classify_points(I_v, Q_v, I_peaks, Q_peaks)

            num_in_state = len(np.where(states == calStateLabel)[0])

            state_pct_list.append(num_in_state / n_pts)

            all_states.append(states)

        if plot:

            bins = 101

            if res_plot_ax == None:
                fig, ax = plt.subplots(figsize=(7, 7))
                fig.suptitle('Result after selection')
            else:

                fig = res_plot_ax.get_figure()
                ax = res_plot_ax

            hist, x_edges, y_edges = np.histogram2d(np.hstack(self.I_vld), np.hstack(self.Q_vld), bins=bins,
                                                    range=self.histRange)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ax.pcolormesh(x_edges, y_edges, 10 * np.log10(hist.T), cmap='magma')

            histRange = self.histRange

            bins = 301

            x = np.linspace(histRange[0][0], histRange[0][1], bins)
            y = np.linspace(histRange[1][0], histRange[1][1], bins)

            xx, yy = np.meshgrid(x, y)

            states_for_plotting = self.classify_points(xx.flatten(), yy.flatten(), I_peaks, Q_peaks)
            states_for_plotting = np.reshape(states_for_plotting, np.shape(xx))

            ax.contour(x, y, states_for_plotting, colors='k')

            ax.text(I_peaks[calStateLabel], Q_peaks[calStateLabel], str(calStateLabel), color='k',
                    horizontalalignment='center', verticalalignment='center')
            ax.set_aspect(1)

        return state_pct_list

def flatten_sweep_axes(data):
    """
    flatten an abitrary nd array data into a 2d array (to be used in post selection funcitons, where the data shape must
        be (nReps, nMSMTs) )
    :param data: nd array data, the first axes must be nReps, and the rest axes are sweep parameters
    :return:
    """
    original_shape = data.shape
    return original_shape, np.array(data).reshape(original_shape[0], -1)


def simpleSelection_1Qge(Idata, Qdata, geLocation=None, plot=True,
                         fitGuess={}, stateMask: int = None, histBins=201, histRange=None,
                         selCircleSize=1, xData:dict=None, progress=False):
    """simple post selection function that selects data points where the qubit is in g
        state in the first MSMT of each two MSMTs.

        :param Idata: I data, nd array, first axes should be nReps
        :param Qdata: Q data, nd array, first axes should be nReps
        :param geLocation:  [g_x, g_y, e_x, e_y, g_r, e_r]
        :param plot: plot fitting and selection data
        :param fitGuess:  guess parameter for gau blob fitting
        :param stateMask: guessed size of each blob, in unit of number of bins
        :param histBins: number of bins for histogram
        :param histRange: range of histogram
        :param selCircleSize: size of the selection circle, in unit of g_r
                            (sigma of g state Gaussian blob)
        :param xData: dictionary that contains the variables that are swept in the experiment.
                        e.g : {"amp": np.linspace(0,1 101) }

        :returns: g_pct, I_vld, Q_vld, selData

    """
    original_shape, Idata = flatten_sweep_axes(Idata) # todo: this should be moved to the PostSelectionData classes
    _, Qdata = flatten_sweep_axes(Qdata)

    print("post selection shape: ", Idata.shape, Qdata.shape)

    selData = PostSelectionData_ge(Idata, Qdata, [1, 0], geLocation, False,
                                   fitGuess, stateMask, histBins, histRange)

    fit_location = True if geLocation is None else False

    if plot:
        fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(18,5))
        if fit_location:
            selData.stateFitResult.plot(ax1)
        selMask = selData.mask_g_by_circle(sel_idx=0, circle_size=selCircleSize, plot=plot, plot_ax=ax2)
        I_vld, Q_vld = selData.sel_data(selMask, plot=False, progress=progress)
        g_pct = selData.cal_g_pct(plot=plot, plot_ax=ax3, progress=progress)
        selData.sliderPlotSelectedData(xData)


    else:
        selMask = selData.mask_g_by_circle(sel_idx=0, circle_size=selCircleSize, plot=False)
        I_vld, Q_vld = selData.sel_data(selMask, plot=False, progress=progress)
        g_pct = selData.cal_g_pct(plot=False, progress=progress)

    # reshape results back to the shape of the input sweep axes
    # todo: these should be moved to the PostSelectionData classes
    final_shape = list((*original_shape[1:-1],
                          int(original_shape[-1] * np.sum(selData.selPattern) / len(selData.selPattern))))
    g_pct = g_pct.reshape(*final_shape)
    I_vld = np.array(I_vld, dtype=object).reshape(*final_shape, -1)
    Q_vld = np.array(Q_vld, dtype=object).reshape(*final_shape, -1)

    return g_pct, I_vld, Q_vld, selData


def simpleSelection_1Qgef(Idata, Qdata, gefLocation=None, plot=True,
                         fitGuess={}, stateMask: int = None, histBins=201, histRange=None,
                         selCircleSize=1, xData:dict=None, progress=False):
    """simple post selection function that selects data points where the qubit is in g
        state in the first MSMT of each two MSMTs when the f state needs to be considered in fitting.

        :param Idata: I data, nd array, first axes should be nReps
        :param Qdata: Q data, nd array, first axes should be nReps
        :param gefLocation:  [g_x, g_y, e_x, e_y, f_x, f_y, g_r, e_r, f_r]
        :param plot: plot fitting and selection data
        :param fitGuess:  guess parameter for gau blob fitting
        :param stateMask: guessed size of each blob, in unit of number of bins
        :param histBins: number of bins for histogram
        :param histRange: range of histogram
        :param selCircleSize: size of the selection circle, in unit of g_r
                            (sigma of g state Gaussian blob)
        :param xData: dictionary that contains the variables that are swept in the experiment.
                        e.g : {"amp": np.linspace(0,1 101) }

        :returns: g_pct, I_vld, Q_vld, selData

    """
    original_shape, Idata = flatten_sweep_axes(Idata) # todo: this should be moved to the PostSelectionData classes
    _, Qdata = flatten_sweep_axes(Qdata)

    selData = PostSelectionData_gef(Idata, Qdata, [1, 0], gefLocation, False,
                                   fitGuess, stateMask, histBins, histRange)

    fit_location = True if gefLocation is None else False

    if plot:
        fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(18,5))
        if fit_location:
            selData.stateFitResult.plot(ax1)
        selMask = selData.mask_g_by_circle(sel_idx=0, circle_size=selCircleSize, plot=plot, plot_ax=ax2)
        I_vld, Q_vld = selData.sel_data(selMask, plot=False, progress=progress)
        g_pct, e_pct, f_pct = selData.cal_gef_pct(plot=plot, plot_ax=ax3, progress=progress)
        selData.sliderPlotSelectedData(xData)


    else:
        selMask = selData.mask_g_by_circle(sel_idx=0, circle_size=selCircleSize, plot=False)
        I_vld, Q_vld = selData.sel_data(selMask, plot=False, progress=progress)
        g_pct, e_pct, f_pct = selData.cal_gef_pct(plot=False, progress=progress)

    # reshape results back to the shape of the input sweep axes
    # todo: these should be moved to the PostSelectionData classes
    final_shape = list((*original_shape[1:-1],
                          int(original_shape[-1] * np.sum(selData.selPattern) / len(selData.selPattern))))
    g_pct = g_pct.reshape(*final_shape)
    I_vld = np.array(I_vld, dtype=object).reshape(*final_shape, -1)
    Q_vld = np.array(Q_vld, dtype=object).reshape(*final_shape, -1)

    return [g_pct, e_pct, f_pct], I_vld, Q_vld, selData



#
# if __name__ == "__main__":
#     directory = r'N:\Data\Tree_3Qubits\QCSWAP\Q3C3\20210111\\'
#     # directory = r'D:\Lab\Code\PostSelProcess_dev\\'
#     fileName = '10PiPulseTest'
#     f = h5py.File(directory + fileName, 'r')
#     Idata = np.real(f["rawData"])[:-1]
#     Qdata = np.imag(f["rawData"])[:-1]
#
#     t0 = time.time()
#     IQsel = PostSelectionData_gef(Idata, Qdata,
#                                   gefLocation=[-9000, -9500, -11000, -1500, -500, -700, 3000, 3000,
#                                                3000])
#
#     # mask0 = IQsel.mask_g_by_line(0, line_shift=0, plot=True)
#     mask0 = IQsel.mask_g_by_circle(0, circle_size=1, plot=True)
#     I_vld, Q_vld = IQsel.sel_data(mask0, plot=True)
#     # I_avg, Q_avg = fdp.average_data(I_vld, Q_vld, axis0_type="xData")
#     # I_rot, Q_rot = fdp.rotateData(I_avg, Q_avg, plot=0)
#     g_pct = IQsel.cal_g_pct()
#
#     xData = np.arange(10)
#     # plt.figure(figsize=(7, 7))
#     # plt.plot(xData, I_avg)
#     plt.figure(figsize=(7, 7))
#     plt.plot(xData, g_pct)
#
#     print("time: ", time.time() - t0)
