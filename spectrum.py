import os.path
from typing import Dict, Optional

import numpy as np


class Spectrum:
    _parser_files = {
        "Szz": "Szz.csv",
        "a1": "a1.csv",
        "b1": "b1.csv",
        "Sxx": "Sxx.csv",
        "Syy": "Syy.csv",
        "Qxz": "Qxz.csv",
        "Qyz": "Qyz.csv",
    }
    _toDeg = 180.0 / np.pi

    def __init__(self, path, outpath):
        self._file_available = {
            "Szz": False,
            "a1": False,
            "b1": False,
        }
        self.path = path
        self.outpath = outpath
        self._data: Dict[str, Optional[np.ndarray]] = {
            "Szz": None,
            "a1": None,
            "b1": None,
            "Sxx": None,
            "Syy": None,
            "Qxy": None,
            "Qyz": None,
        }  # type: ignore
        self._none = None
        self.time = None
        for key in self._parser_files:
            # Check which output from the parser is available
            if os.path.isfile(os.path.join(self.path, self._parser_files[key])):
                self._file_available[key] = True

            else:
                self._file_available[key] = False

        # Load a header file from the spectral data from the parser to get frequencies
        self._load_header()
        # Load the data from the parser
        self._load_parser_output()

    def data(self, key) -> np.ndarray:
        """Getter that ensures a _data item exists"""
        v = self._data[key]
        if v is None:
            raise RuntimeError(f"Field '{key}' is not initialized in _data")
        return v

    @property
    def spectral_data_is_available(self):
        available = [self._file_available[key] for key in self._file_available]
        return any(available)

    @property
    def Szz(self):
        return self.data("Szz")

    @property
    def a1(self):
        return self.data("a1")

    @property
    def b1(self):
        return self.data("b1")

    def _Qyzm(self):
        return np.mean(self.data("Qyz"), axis=1)

    def _Qxzm(self):
        return np.mean(self.data("Qxz"), axis=1)

    def _Sxxm(self):
        return np.mean(self.data("Sxx"), axis=1)

    def _Syym(self):
        return np.mean(self.data("Syy"), axis=1)

    def _Szzm(self):
        return np.mean(self.data("Szz"), axis=1)

    @property
    def f(self):
        return self._frequencies

    def _load_header(self):
        for key in self._parser_files:
            if self._file_available[key]:
                with open(os.path.join(self.path, self._parser_files[key]), "r") as file:
                    line = file.readline().split(",")[8:]
                    self._frequencies = np.array([float(x) for x in line])
            break
        else:
            raise Exception(
                "No spectral files available - please make sure the script is in the same"
                " directory as the sd-card output"
            )

    def _load_parser_output(self):
        for key in self._parser_files:
            if self._file_available[key]:
                data = np.loadtxt(
                    os.path.join(self.path, self._parser_files[key]), delimiter=","
                )
                self.time = data[:, 0:8]
                self._data[key] = data[:, 8:]
                mask = np.isnan(self.data(key))
                self.data(key)[mask] = 0.0
                self._none = np.nan + np.zeros(self.data(key).shape)

        for key in self._parser_files:
            if not self._file_available[key]:
                self._data[key] = self._none

    def _moment(self, values):
        E = self.Szz * values
        jstart = 3
        return np.trapz(E[:, jstart:], self.f[jstart:], 1)

    def _weighted_moment(self, values):
        return self._moment(values) / self._moment(1.0)

    @property
    def a1m(self):
        if self._file_available["Sxx"]:
            return self._Qxzm() / np.sqrt(self._Szzm() * (self._Sxxm() + self._Syym()))
        else:
            return self._weighted_moment(self.a1)

    @property
    def b1m(self):
        if self._file_available["Sxx"]:
            return self._Qyzm() / np.sqrt(self._Szzm() * (self._Sxxm() + self._Syym()))
        else:
            return self._weighted_moment(self.b1)

    def _peak_index(self):
        return np.argmax(self.Szz, 1)

    def _direction(self, a1, b1):
        directions = 270 - np.arctan2(b1, a1) * self._toDeg

        for ii, direction in enumerate(directions):
            if direction < 0:
                direction = direction + 360

            if direction > 360:
                direction = direction - 360
            directions[ii] = direction
        return directions

    def mean_direction(self):
        return self._direction(self.a1m, self.b1m)

    def _spread(self, a1, b1):
        return np.sqrt(2 - 2 * np.sqrt(a1**2 + b1**2)) * self._toDeg

    def mean_spread(self):
        return self._spread(self.a1m, self.b1m)

    def _get_peak_value(self, variable):
        maxloc = np.argmax(self.Szz, 1)
        out = np.zeros(maxloc.shape)
        if len(variable.shape) == 2:
            for ii, index in enumerate(maxloc):
                out[ii] = variable[ii, index]
        elif len(variable.shape) == 1:
            for ii, index in enumerate(maxloc):
                out[ii] = variable[index]

        return out

    def peak_direction(self):
        a1 = self._get_peak_value(self.a1)
        b1 = self._get_peak_value(self.b1)
        return self._direction(a1, b1)

    def peak_spreading(self):
        a1 = self._get_peak_value(self.a1)
        b1 = self._get_peak_value(self.b1)
        return self._spread(a1, b1)

    def peak_frequency(self):
        return self._get_peak_value(self.f)

    def peak_period(self):
        return 1.0 / self.peak_frequency()

    def mean_period(self):
        return 1.0 / self._weighted_moment(self.f)

    def significant_wave_height(self):
        return 4.0 * np.sqrt(self._moment(1.0))

    def generate_text_file(self):
        hm0 = self.significant_wave_height()
        tm01 = self.mean_period()
        tp = self.peak_period()
        dir = self.mean_direction()
        pdir = self.peak_direction()
        dspr = self.mean_spread()
        pdspr = self.peak_spreading()

        with open(os.path.join(self.outpath, "bulkparameters.csv"), "w") as file:
            header = (
                "# year , month , day, hour ,min, sec, milisec , Significant Wave Height,"
                " Mean Period, Peak Period, Mean Direction, Peak Direction,"
                " Mean Spreading, Peak Spreading\n"
            )
            file.write(header)
            format = "%d, " * 7 + "%6.2f, " * 6 + "%6.2f \n"
            for ii in range(0, len(hm0)):
                assert self.time is not None
                string = format % (
                    self.time[ii, 0],
                    self.time[ii, 1],
                    self.time[ii, 2],
                    self.time[ii, 3],
                    self.time[ii, 4],
                    self.time[ii, 5],
                    self.time[ii, 6],
                    hm0[ii],
                    tm01[ii],
                    tp[ii],
                    dir[ii],
                    pdir[ii],
                    dspr[ii],
                    pdspr[ii],
                )
                file.write(string)
