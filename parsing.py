import os.path
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy import io, signal

from filenames import extensions
from versions import defaultIIRWeightType, defaultVersion

#
# Phase correction by applying the IIR to the reversed
# signal
#
applyPhaseCorrection = True
applyPhaseCorrectionFromVersionNumber = 2


def parseLocationFiles(
    inputFileName,
    outputFileName="displacement.CSV",
    kind="FLT",
    reportProgress=True,
    outputFileType="CSV",
    versionNumber=defaultVersion,
    IIRWeightType=defaultIIRWeightType,
):
    """
    This functions loads all the gps-location data (located at *path*) from a Spotter into
    one datastructure and saves the result as a CSV file (*outputFileName*).
    """

    fname = os.path.splitext(outputFileName)[0]
    outputFileName = fname + "." + extensions(outputFileType)

    # Load location data into a pandas dataframe object
    if reportProgress:
        print(f"Processing Spotter displacement output - {kind}")

    header = "year,month,day,hour,min,sec,msec"
    if kind == "FLT":
        # Read the data using pandas
        data = pd.read_csv(
            inputFileName,
            skiprows=1,
            header=None,
            names=["epoch_t", "outx", "outy", "outz"],
            index_col=[0],
            usecols=range(1, 5),
        )
        colunits = {"epoch_t": "s", "outx": "mm", "outy": "mm", "outz": "mm"}
        data = data.apply(pd.to_numeric, errors="coerce")  # Seems unnecessary?
        # Drop any rows with NaNs
        data = data.dropna(axis=0, how="any")
        # Convert the epoch time to a datetime, then to separate columns
        data = _filter_bogus_epochs(data)
        data.index = pd.to_datetime(data.index, unit="s")
        period_names = ["year", "month", "day", "hour", "minute", "second", "millisecond"]
        for col_i, period in enumerate(period_names[:-1]):
            data.insert(loc=col_i, column=period, value=getattr(data.index, period))
        data.index.microsecond.astype(np.uint32)
        data.insert(
            loc=col_i + 1,
            column="millisecond",
            value=data.index.microsecond.astype(np.uint32),
        )
        float_fmt = "%.5e"

        colnames = ["outx", "outy", "outz"]
        # Convert to meters
        for colname in colnames:
            data[colname] = data[colname] / 1000
            colunits[colname] = "m"

        # Apply phase correction to displacement data
        if (
            applyPhaseCorrection
            and versionNumber >= applyPhaseCorrectionFromVersionNumber
        ):
            print("- IIR phase correction using weight type: ", str(IIRWeightType))
            for colname in colnames:
                data[colname] = applyfilter(
                    data[colname], "backward", versionNumber, IIRWeightType
                )

        fmt = "%i," * 7 + "%.5e,%.5e,%.5e"
        header = header + ", x(m), y(m), z(m)"
        if outputFileType.lower() in ["csv", "gz"]:
            data.to_csv(outputFileName, float_format=float_fmt)
            return
        else:
            data = data.to_numpy()
    elif kind == "SST":
        # Read the data using pandas, and convert to numpy
        data = pd.read_csv(inputFileName, index_col=False, usecols=(0, 1))
        data = data.apply(pd.to_numeric, errors="coerce")
        data = data.values

        msk = np.isnan(data[:, 0])
        data = data[~msk, :]
        datetime = epochToDateArray(data[:, 0])
        data = data[:, 1]
        data = np.concatenate((datetime, data[:, None]), axis=1)

        fmt = "%i," * 7 + "%5.2f"
        header = header + ", T (deg. Celsius)"
    elif kind == "GPS":
        # DEBUG MODE GPS
        data = pd.read_csv(
            inputFileName, index_col=False, usecols=(1, 2, 3, 4, 5, 6, 7, 8, 9)
        )
        data = data.apply(pd.to_numeric, errors="coerce")
        data = data.values
        datetime = epochToDateArray(data[:, 0].tolist())

        data[:, 1] = data[:, 1] + data[:, 2] / 6000000.0
        data[:, 2] = data[:, 3] + data[:, 4] / 6000000.0
        data = np.concatenate((data[:, 1:3], data[:, 5:]), axis=1)
        data = np.concatenate((datetime, data), axis=1)

        fmt = "%i," * 7 + "%13.8f" * 5 + "%13.8f"
        header = (
            header
            + ",latitude (decimal degrees),longitude (decimal degrees),elevation (m)"
            + ",SOG (mm/s),COG (deg*1000),Vert Vel (mm/s)"
        )
    else:
        data = pd.read_csv(inputFileName, index_col=False, usecols=(0, 1, 2, 3, 4))
        data = data.apply(pd.to_numeric, errors="coerce")
        data = data.values
        msk = np.isnan(data[:, 0])
        data = data[~msk, :]
        datetime = epochToDateArray(data[:, 0].tolist())

        data[:, 1] = data[:, 1] + data[:, 2] / 6000000.0
        data[:, 2] = data[:, 3] + data[:, 4] / 6000000.0
        data = data[:, 1:3]
        data = np.concatenate((datetime, data), axis=1)

        fmt = "%i," * 7 + "%13.8f,%13.8f"
        header = header + ", latitude (decimal degrees),longitude (decimal degrees)"

    if outputFileType.lower() in ["csv", "gz"]:
        np.savetxt(outputFileName, data, fmt=fmt, header=header)
    elif outputFileType.lower() == "matlab":
        # To save to matlab .mat format we need scipy
        if kind == "FLT":
            io.savemat(
                outputFileName,
                {
                    "x": data[:, 7].astype(np.float32),
                    "y": data[:, 8].astype(np.float32),
                    "z": data[:, 9].astype(np.float32),
                    "time": data[:, 0:7].astype(np.int16),
                },
            )
        elif kind == "GPS":
            io.savemat(
                outputFileName,
                {
                    "Lat": data[:, 7].astype(np.float32),
                    "Lon": data[:, 8].astype(np.float32),
                    "elevation": data[:, 9].astype(np.float32),
                    "sog": data[:, 10].astype(np.float32),
                    "cog": data[:, 11].astype(np.float32),
                    "w": data[:, 12].astype(np.float32),
                    "time": data[:, 0:7].astype(np.int16),
                },
            )
        else:
            io.savemat(
                outputFileName,
                {
                    "Lat": data[:, 7].astype(np.float32),
                    "Lon": data[:, 8].astype(np.float32),
                    "time": data[:, 0:7].astype(np.int16),
                },
            )
    elif outputFileType.lower() == "numpy":
        if kind == "FLT":
            np.savez(
                outputFileName,
                x=data[:, 7].astype(np.float32),
                y=data[:, 8].astype(np.float32),
                z=data[:, 9].astype(np.float32),
                time=data[:, 0:7].astype(np.int16),
            )
        else:
            np.savez(
                outputFileName,
                lat=data[:, 7].astype(np.float32),
                lon=data[:, 8].astype(np.float32),
                time=data[:, 0:7].astype(np.int16),
            )
    return None


def _filter_bogus_epochs(data: pd.DataFrame) -> pd.DataFrame:
    """Simple filter to remove any data with timestamps greater than now."""
    dt_now = datetime.now(tz=timezone.utc)
    if (bad_dt := data.index[data.index > dt_now.timestamp()]).size > 0:
        data = data.drop(index=bad_dt)
    return data


def applyfilter(data, kind, versionNumber, IIRWeightType):
    # Apply forward/backward/filtfilt sos filter
    # Get SOS coefficients
    sos = filterSOS(versionNumber, IIRWeightType)
    if kind == "backward":
        directions = ["backward"]
        # , axis=0
    elif kind == "forward":
        directions = ["forward"]
    elif kind == "filtfilt":
        directions = ["forward", "backward"]

    res = data
    for direction in directions:
        if direction == "backward":
            res = np.flip(res, axis=0)
        res = signal.sosfilt(sos, res, axis=0)
        if direction == "backward":
            res = np.flip(res, axis=0)

    return res


def filterSOS(versionNumber, IIRWeightType):
    # second order-sections coeficients of the filter

    if versionNumber < 1:
        sos = 0.0
        return sos
    elif versionNumber in [1, 2, 3]:
        if IIRWeightType == 0:
            # Type A
            lp = {
                "a1": -1.8514229621,
                "a2": 0.8578089736,
                "b0": 0.8972684452,
                "b1": -1.7945369122,
                "b2": 0.8972684291,
            }
            hp = {
                "a1": -1.9318795385,
                "a2": 0.9385430645,
                "b0": 1.0000000000,
                "b1": -1.9999999768,
                "b2": 1.0000000180,
            }
        elif IIRWeightType == 1:
            # Type B
            lp = {
                "a1": 1.9999999964,
                "a2": 0.9999999964,
                "b0": 0.9430391609,
                "b1": -1.8860783217,
                "b2": 0.9430391609,
            }
            hp = {
                "a1": -1.8828311523,
                "a2": 0.8893254984,
                "b0": 1.0000000000,
                "b1": 2.0000000000,
                "b2": 1.0000000000,
            }
        elif IIRWeightType == 2:
            # Type C
            lp = {
                "a1": 1.1375322034,
                "a2": 0.4141775928,
                "b0": 0.6012434213,
                "b1": -1.2024868427,
                "b2": 0.6012434213,
            }
            hp = {
                "a1": -1.8827396569,
                "a2": 0.8894088696,
                "b0": 1.0000000000,
                "b1": 2.0000000000,
                "b2": 1.0000000000,
            }
    sos = [
        [lp["b0"], lp["b1"], lp["b2"], 1.000000, lp["a1"], lp["a2"]],
        [hp["b0"], hp["b1"], hp["b2"], 1.000000, hp["a1"], hp["a2"]],
    ]
    sos = np.array(sos)
    return sos


def epochToDateArray(epochtime):
    datetime = np.array([list(time.gmtime(x))[0:6] for x in epochtime])
    milis = np.array([(1000 * (x - np.floor(x))) for x in epochtime])
    return np.concatenate((datetime, milis[:, None]), axis=1)


def parseSpectralFiles(
    inputFileName,
    outputPath,
    outputFileNameDict=None,
    spectralDataSuffix="SPC",
    reportProgress=True,
    nf=128,
    df=0.009765625,
    outputSpectra=None,
    outputFileType="CSV",
    lfFilter=False,
    versionNumber=defaultVersion,
):
    # This functions loads all the Spectral data (located at *path*) from a Spotter into
    # one datastructure and saves the result as a CSV file (*outputFileName*).

    def checkKeyNames(key, errorLocation):
        # Nested function to make sure input is insensitive to capitals, irrelevant
        # permutations (Cxz vs Czx), etc
        if key.lower() == "szz":
            out = "Szz"
        elif key.lower() == "syy":
            out = "Syy"
        elif key.lower() == "sxx":
            out = "Sxx"
        elif key.lower() in ["cxz", "czx"]:
            out = "Cxz"
        elif key.lower() in ["qxz", "qzx"]:
            out = "Qxz"
        elif key.lower() in ["cyz", "czy"]:
            out = "Cyz"
        elif key.lower() in ["qyz", "qzy"]:
            out = "Qyz"
        elif key.lower() in ["cxy", "cyx"]:
            out = "Cxy"
        elif key.lower() in ["qxy", "qyx"]:
            out = "Qxy"
        elif key.lower() in ["a1", "b1", "a2", "b2"]:
            out = key.lower()
        else:
            raise Exception("unknown key: " + key + " in " + errorLocation)
        return out

    # end def

    outputFileName = {
        "Szz": "Szz.CSV",
        "Cxz": "Cxz.CSV",
        "Qxz": "Qxz.CSV",
        "Cyz": "Cyz.CSV",
        "Qyz": "Qyz.CSV",
        "Cxy": "Cxy.CSV",
        "Qxy": "Qxy.CSV",
        "Syy": "Syy.CSV",
        "Sxx": "Sxx.CSV",
        "a1": "a1.CSV",
        "b2": "b2.CSV",
        "b1": "b1.CSV",
        "a2": "a2.CSV",
    }

    # Rename output file names for given variables if a dict is given
    if outputFileNameDict is not None:
        for key in outputFileNameDict:
            keyName = checkKeyNames(key, "output file names")
            outputFileName[keyName] = outputFileNameDict[key]

    for key in outputFileName:
        fname = os.path.splitext(outputFileName[key])[0]
        outputFileName[key] = fname + "." + extensions(outputFileType)

    # The output  files given by the script; per defauly only Szz is given, but can be
    # altered by user request
    if outputSpectra is None:
        outputSpectra = ["Szz"]
    else:
        # For user requested output, make sure variables are known and have correct
        # case/permutations/etc.
        for index, key in enumerate(outputSpectra):
            keyName = checkKeyNames(key, "output list")
            outputSpectra[index] = keyName

    # Load spectral data into a pandas dataframe object
    if reportProgress:
        print("Processing Spotter spectral output")
    if versionNumber in [0, 2, 3]:
        startColumnNumber = {
            "Szz": 5,
            "Cxz": 7,
            "Qxz": 13,
            "Cyz": 8,
            "Qyz": 14,
            "Sxx": 3,
            "Syy": 4,
            "Cxy": 6,
            "Qxy": 12,
        }
        stride = 12
    else:
        # Column ordering changed from v1.4.1 onwards; extra columns due
        # to cross-correlation filter between z and w
        #
        # Column order is now:
        #
        #   type,millis,[0] t0_GPS_Epoch_Time(s), [1] tN_GPS_Epoch_Time(s),
        #   [2] ens_count, [3] Sxx_re, [4] Syy_re, [5] Szz_re,[6] Snn_re,
        #   [7] Sxy_re,[8] Szx_re,[9] Szy_re,[10] Szn_re,[11] Sxx_im,
        #   [12] Syy_im,[13] Szz_im,[14] Snn_im,[15] Sxy_im,[16] Szx_im,
        #   [17] Szy_im,[18] Szn_im
        #
        # Note that since first two columns are ignorded counting starts from 0
        # at t0_GPS_Epoch_Time(s)
        #
        startColumnNumber = {
            "Szz": 5,
            "Cxz": 8,
            "Qxz": 16,
            "Cyz": 9,
            "Qyz": 17,
            "Sxx": 3,
            "Syy": 4,
            "Cxy": 7,
            "Qxy": 15,
            "Snn": 6,
            "Czn": 10,
            "Qzn": 18,
        }
        stride = 16
    # Read csv file using Pandas - this is the only section in the code
    # still reliant on Pandas, and only there due to supposed performance
    # benifits.
    tmp = pd.read_csv(
        inputFileName,
        index_col=False,
        skiprows=[0],
        header=None,
        usecols=tuple(range(2, 5 + stride * nf)),
    )

    # Ensure the dataframe is numeric, coerce any occurences of bad data
    # (strings etc) to NaN and return a numpy numerica array
    tmp = tmp.apply(pd.to_numeric, errors="coerce").values

    datetime = epochToDateArray(tmp[:, 0])
    ensembleNum = tmp[:, 2] * 2
    data = {}
    for key in startColumnNumber:
        # Convert to variance density and change units to m2/hz (instead
        # of mm^2/hz)
        data[key] = tmp[:, startColumnNumber[key] :: stride] / (1000000.0 * df)
        # Set low frequency columns to NaN
        data[key][:, 0:3] = np.nan

    # Calculate directional moments from data (if requested). Because these are
    # derived quantities these need to be included in the dataframe a-postiori
    if any([x in ["a1", "b1", "a2", "b2"] for x in outputSpectra]):
        with np.errstate(invalid="ignore", divide="ignore"):
            # Supress divide by 0; silently produce NaN
            data["a1"] = data["Qxz"] / np.sqrt(
                (data["Szz"] * (data["Sxx"] + data["Syy"]))
            )
            data["a2"] = (data["Sxx"] - data["Syy"]) / (data["Sxx"] + data["Syy"])
            data["b1"] = data["Qyz"] / np.sqrt(
                (data["Szz"] * (data["Sxx"] + data["Syy"]))
            )
            data["b2"] = 2.0 * data["Cxy"] / (data["Sxx"] + data["Syy"])

        for key in ["a1", "b1", "a2", "b2"]:
            # If energies are zeros, numpy produces infinities
            # set to NaN as these are meaningless
            data[key][np.isinf(data[key])] = np.nan
            data[key][np.isnan(data[key])] = np.nan

    if lfFilter:
        # Filter lf-noise
        data = lowFrequencyFilter(data)

    for key in data:
        data[key] = np.concatenate((datetime, ensembleNum[:, None], data[key]), axis=1)

    # construct header for use in CSV
    header = "year,month,day,hour,min,sec,milisec,dof"
    freq = np.array(list(range(0, nf))) * df
    for f in freq:
        header = header + "," + str(f)

    # write data to requested output format
    for key in outputSpectra:
        fmt = "%i , " * 8 + ("%.3e , " * (nf - 1)) + "%.3e"

        if outputFileType.lower() == "csv":
            if outputFileType.lower() in ["csv", "gz"]:
                np.savetxt(
                    os.path.join(outputPath, outputFileName[key]),
                    data[key],
                    fmt=fmt,
                    header=header,
                )
        elif outputFileType.lower() == "matlab":
            # To save to matlab .mat format we need scipy
            mat = data[key]
            io.savemat(
                os.path.join(outputPath, outputFileName[key]),
                {
                    "spec": mat[:, 8:].astype(np.float32),
                    "time": mat[:, 0:7].astype(np.int16),
                    "frequencies": freq.astype(np.float32),
                    "dof": mat[:, 7].astype(np.int16),
                },
            )
        elif outputFileType.lower() == "numpy":
            mat = data[key]
            np.savez(
                os.path.join(outputPath, outputFileName[key]),
                spec=mat[:, 8:].astype(np.float32),
                time=mat[:, 0:7].astype(np.int16),
                frequencies=freq.astype(np.float32),
                dof=mat[:, 7].astype(np.int16),
            )
    return None


def lowFrequencyFilter(data):
    """
    function to perform the low-frequency filter
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        # Ignore division by 0 etc (this is caught below)
        Gxz = (data["Cxz"] ** 2 + data["Qxz"] ** 2) / (data["Sxx"] * data["Szz"])
        Gyz = (data["Cyz"] ** 2 + data["Qyz"] ** 2) / (data["Syy"] * data["Szz"])
    phi = 1.5 * np.pi - np.arctan2(data["b1"], data["a1"])
    G = np.sin(phi) ** 2 * Gxz + np.cos(phi) ** 2 * Gyz
    G[np.isnan(G)] = 0.0

    I = np.argmax(G, axis=1)

    names = ["Szz", "Cxz", "Qxz", "Cyz", "Qyz", "Sxx", "Syy", "Cxy", "Qxy"]
    for key in names:
        for jj in range(0, G.shape[0]):
            data[key][jj, 0 : I[jj]] = data[key][jj, 0 : I[jj]] * G[jj, 0 : I[jj]]
    return data
