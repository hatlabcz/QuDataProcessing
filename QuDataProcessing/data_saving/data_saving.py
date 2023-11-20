from typing import Any, Union, Optional, Dict, Type, Collection
from pathlib import Path
import time

import yaml
import numpy as np
from tqdm import tqdm

from plottr.data import DataDict, MeshgridDataDict
from plottr.data.datadict_storage import _data_file_path, DDH5Writer, DATAFILEXT, is_meta_key, deh5ify, FileOpener, AppendMode, datadict_to_hdf5, add_cur_time_attr


class HatDDH5Writer(DDH5Writer):
    """Context manager for writing DataDict to DDH5.
    Based on the DDH5Writer from plottr, with re-implemented data_folder method, which allows user to specify the name
    of the folder, instead of using a generated ID.

    :param datadict: Initial data object. Must contain at least the structure of the
        data to be able to use :meth:`add_data` to add data.
    :param basedir: The root directory in which data is stored.
    :param foldername: name of the folder where the data will be stored. If not provided, the current date will be used
        as the foldername.
    :param filename: Filename to use. Defaults to 'data.ddh5'.
    :param groupname: Name of the top-level group in the file container. An existing
        group of that name will be deleted.
    :param name: Name of this dataset. Used in path/file creation and added as meta data.

    """

    def __init__(self, datadict: DataDict, basedir: str = '.', foldername: Optional[str] = None, filename: str = 'data',
                 groupname: str = 'data', name: Optional[str] = None):
        super().__init__(datadict, basedir, groupname, name, filename)
        self.foldername = foldername

    def data_folder(self) -> Path:
        """Return the folder
        """
        if self.foldername is None:
            path = Path(time.strftime("%Y-%m-%d"), )

        else:
            path = Path(self.foldername)
        return path

    def save_config(self, cfg: Dict):
        datafolder = str(self.filepath.parent)
        with open(str(datafolder) + f"\\\\{str(self.filename)}_cfg.yaml", 'w') as file:
            yaml.dump(cfg, file)

    def data_file_path(self) -> Path:
        """Instead of checking for duplicate folders, here we check for duplicate filenames, so that we can have
        so that we can have different files in the same date folder.

        :returns: The filepath of the data file.
        """
        data_file_path = Path(self.basedir, self.data_folder(), str(self.filename) + f".{DATAFILEXT}")
        appendix = ''
        idx = 2
        while data_file_path.exists():
            appendix = f'-{idx}'
            data_file_path = Path(self.basedir,
                                  self.data_folder(), str(self.filename) + appendix + f".{DATAFILEXT}")
            idx += 1
        self.filename = Path(str(self.filename) + appendix)
        return data_file_path

    def add_meta(self, **kwargs: Any) -> None:
        """Add/update metadata to the datafile (and the internal `DataDict`).

        """
        for k, v in kwargs.items():
            self.datadict.add_meta(k, v)

        if self.inserted_rows > 0:
            mode = AppendMode.new
        else:
            mode = AppendMode.none
        nrecords = self.datadict.nrecords()
        if nrecords is not None and nrecords > 0:
            datadict_to_hdf5(self.datadict, str(self.filepath),
                             groupname=self.groupname,
                             append_mode=mode)

            assert self.filepath is not None
            with FileOpener(self.filepath, 'a') as f:
                add_cur_time_attr(f, name='last_change')
                add_cur_time_attr(f[self.groupname], name='last_change')




class DummyWriter():
    def __init__(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *excinfo):
        pass

    def save_config(self, *args, **kwargs):
        pass

    def add_data(self, *args, **kwargs):
        pass


def datadict_from_hdf5(path: str,
                       groupname: str = 'data',
                       startidx: Union[int, None] = None,
                       stopidx: Union[int, None] = None,
                       structure_only: bool = False,
                       ignore_unequal_lengths: bool = True,
                       progress=False,
                       data_only=False) -> DataDict:
    """Load a DataDict from file. Copied from plottr.data.datadict_storage.
        Added extra features to show loading progress and load data only.

    :param path: Full filepath without the file extension.
    :param groupname: Name of hdf5 group.
    :param startidx: Start row.
    :param stopidx: End row + 1.
    :param structure_only: If `True`, don't load the data values.
    :param ignore_unequal_lengths: If `True`, don't fail when the rows have
        unequal length; will return the longest consistent DataDict possible.
    :param progress:  when true, show loading progress.
    :param data_only: when true, only data with metadata __isdata__==True
        will be loaded

    :return: Validated DataDict.
    """
    filepath = _data_file_path(path)
    if not filepath.exists():
        raise ValueError("Specified file does not exist.")

    if startidx is None:
        startidx = 0

    res = {}
    with FileOpener(filepath, 'r') as f:
        if groupname not in f:
            raise ValueError('Group does not exist.')

        grp = f[groupname]
        keys = list(grp.keys())
        lens = [grp[k].shape[0] for k in keys]

        if len(set(lens)) > 1:
            if not ignore_unequal_lengths:
                raise RuntimeError('Unequal lengths in the datasets.')

            if stopidx is None or stopidx > min(lens):
                stopidx = min(lens)
        else:
            if stopidx is None or stopidx > lens[0]:
                stopidx = lens[0]

        for attr in grp.attrs:
            if is_meta_key(attr):
                res[attr] = deh5ify(grp.attrs[attr])

        if progress:
            tqdmk = tqdm(keys, desc="loading data")
        else:
            tqdmk = keys
        for k in tqdmk:
            ds = grp[k]
            entry: Dict[str, Union[Collection[Any], np.ndarray]] = dict(values=np.array([]), )

            if data_only and (not ds.attrs.get("__isdata__")):
                # load experiment data only, ignore axes value (will get axis values from metadata)
                if 'unit' in ds.attrs:# keep only units of axes
                    entry["unit"] = deh5ify(ds.attrs['unit'])
                res[k] = entry
                continue

            if 'axes' in ds.attrs:
                entry['axes'] = deh5ify(ds.attrs['axes']).tolist()
            else:
                entry['axes'] = []

            if 'unit' in ds.attrs:
                entry['unit'] = deh5ify(ds.attrs['unit'])

            if not structure_only:
                entry['values'] = ds[startidx:stopidx]

            entry['__shape__'] = ds[:].shape

            # and now the meta data
            for attr in ds.attrs:
                if is_meta_key(attr):
                    _val = deh5ify(ds.attrs[attr])
                    entry[attr] = deh5ify(ds.attrs[attr])

            res[k] = entry

    dd = DataDict(**res)
    if not data_only:
        dd.validate()
    return dd


if __name__ == "__main__":
    # test data saving with fake data
    a = np.zeros((10, 50))
    for i in range(len(a)):
        a[i] = np.linspace(i, i + 10, 50)
    xlist = np.arange(10)
    ylist = np.arange(50) * 2

    data = {
        "a": {
            "axes": ["x", "y"],
            "unit": "a.u."
        },
        "x": {
            'axes': []
        },
        "y": {
            'axes': []
        }
    }
    dd = DataDict(**data)
    ddw = HatDDH5Writer(dd, r"L:\Data\SNAIL_Pump_Limitation\test\\", foldername=None, filename="data11")
    with ddw as d:
        for i in range(5):
            for j in range(5):
                d.add_data(
                    a=a[i, j],
                    x=xlist[i],
                    y=ylist[j]
                )
        d.save_config({"a": 2})
