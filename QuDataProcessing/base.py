from typing import Tuple, Any, Optional, Union, Dict, List
from collections import OrderedDict

import numpy as np



class Parameter:

    def __init__(self, name, value: Any = None, **kw: Any):
        self.name = name
        self.value = value
        self._attrs = {}
        for k, v in kw:
            self._attrs[k] = v

    def __getattr__(self, key):
        return self._attrs[key]


class Parameters(OrderedDict):
    """A collection of parameters"""

    def add(self, name: str, **kw: Any):
        """Add/overwrite a parameter in the collection."""
        self[name] = Parameter(name, **kw)


class AnalysisResult(object):

    def __init__(self, parameters: Dict[str, Union[Dict[str, Any], Any]]):
        self.params = Parameters()
        for k, v in parameters.items():
            if isinstance(v, dict):
                self.params.add(k, **v)
            elif isinstance(v, Parameter):
                self.params.add(k, value=v.value)
            else:
                self.params.add(k, value=v)

    def eval(self, *args: Any, **kwargs: Any) -> np.ndarray:
        """Analysis types that produce data (like filters or fits) should implement this.
        """
        raise NotImplementedError

    def plot(self, *args: Any, **kwargs: Any) -> np.ndarray:
        """Analysis types that produce data (like filters or fits) should implement this.
        """
        raise NotImplementedError


class Analysis(object):
    """Basic analysis object.

    Parameters
    ----------
    coordinates
        may be a single 1d numpy array (for a single coordinate) or a tuple
        of 1d arrays (for multiple coordinates).
    data
        a 1d array of data
    """

    def __init__(self, coordinates: Union[Tuple[np.ndarray, ...], np.ndarray],
                 data: np.ndarray):
        """Constructor of `Analysis`. """
        self.coordinates = coordinates
        self.data = data
        self.pre_process()

    def pre_process(self):
        self.coordinates = np.array(self.coordinates)
        self.data = np.array(self.data)

    def analyze(self, coordinates, data, *args: Any,
                **kwargs: Any) -> AnalysisResult:
        """Needs to be implemented by each inheriting class."""
        raise NotImplementedError

    def run(self, *args: Any, **kwargs: Any):
        return self.analyze(self.coordinates, self.data, *args, **kwargs)


# def analyze(analysis_class: Analysis, coordinates: Union[Tuple[np.ndarray, ...], np.ndarray],
#             data: np.ndarray, **kwarg: Any) -> AnalysisResult:
#     analysis = analysis_class(coordinates, data)
#     return analysis.run(**kwarg)

