from typing import List
import numpy as np
import matplotlib.pyplot as plt
from plottr.data.datadict_storage import datadict_from_hdf5, DataDict


class DataFromQDDH5:
    def __init__(self, ddh5_path):
        self.datadict = datadict_from_hdf5(ddh5_path)
        self.avg_iq = {}
        self.buf_iq = {}
        self.axes = {}
        self.ro_chs = []
        self.reps = len(set(self.datadict["reps"]["values"]))
        self.axes_names = []
        self.datashape = []


        for k, v in self.datadict.items():
            if "avg_iq" in k:
                rch = k.replace("avg_iq_", "")
                self.avg_iq[rch] = self._reshape_original_data(v)
                self.ro_chs.append(rch)
            if "buf_iq" in k:
                rch = k.replace("buf_iq_", "")
                self.buf_iq[rch] = self._reshape_original_data(v)

        self._get_axes_values()

        print("data_shape: ", self.datashape)
        print("axes: ", self.axes_names)


    def _reshape_original_data(self, data):
        rep_idx = data["axes"].index("reps")
        data_shape = []
        if self.axes_names == []:
            self.axes_names = data["axes"]
        for ax in data["axes"]:
            try:  # assume all the sweep axes have metadata "__list__"
                ax_val = self.datadict.meta_val("list", ax)
                data_shape.append(len(ax_val))
            except KeyError:
                pass
        data_shape.insert(rep_idx, self.reps)

        data_r = np.array(data["values"]).reshape(*data_shape, -1)
        self.datashape = list(data_r.shape)

        return data_r

    def _get_axes_values(self):
        for ax in self.axes_names:
            if ax == "reps":
                self.axes[ax] = {"unit": "n", "value": np.arange(self.reps)}
            elif ax == "msmts":
                self.axes[ax] = {"unit": "n", "value": np.arange(self.datashape[-1])}
            else:  # assume all the sweep axes have metadata "__list__"
                ax_val = self.datadict.meta_val("list", ax)
                self.axes[ax] = {"unit": self.datadict[ax].get("unit"), "value": ax_val}

    def reorder_data(self, axis_order:List[str] = None, flatten_sweep=False, mute=False):
        if axis_order is None:
            an_ = self.axes_names.copy()
            an_.insert(0, an_.pop(an_.index("reps")))
            axis_order = an_

        new_idx_order = list(map(self.axes_names.index, axis_order))

        self.axes_names = axis_order
        axes_ = {k: self.axes[k] for k in axis_order}
        self.axes = axes_
        ds_ = np.array(self.datashape)[new_idx_order].tolist()
        self.datashape = ds_

        rep_idx = self.axes_names.index("reps")

        def reshape_(data):
            d = data.transpose(*new_idx_order)
            if flatten_sweep:
                d = d.reshape(*self.datashape[:rep_idx+1], -1)
            return d

        for k, v in self.avg_iq.items():
            self.avg_iq[k] = reshape_(v)
        for k, v in self.buf_iq.items():
            self.buf_iq[k] = reshape_(v)

        if not mute:
            print("data_shape: ", self.datashape)
            print("axes: ", self.axes_names)


if __name__ == "__main__":
    testpath = r"L:\Data\SNAIL_Pump_Limitation\test\2022-08-24\\lengthRabi_sweepFreq-4.ddh5"

    qdd = DataFromQDDH5(testpath)
    qdd.reorder_data(['reps', 'freq', 'legnth', 'msmts'])
    plt.figure()
    plt.pcolormesh(qdd.axes["freq"]["value"], qdd.axes["legnth"]["value"], np.real(qdd.avg_iq["ro_1"][0,:,:,0]).T)

