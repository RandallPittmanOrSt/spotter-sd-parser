#!/usr/bin/env python3
"""
See the NOTICE file distributed with this work for additional
information regarding copyright ownership.  Sofar Ocean Technologies
licenses this file to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the
License for the specific language governing permissions and limitations
under the License.

Purpose:

    For efficiency, SPOTTER stores wave spectra, mean location and
    displacement data on the SD card across multiple files. In order to
    access the data for post processing it is convenient to first
    recombine each data type in a single file.

    This module contains functions to process SPOTTER output files containing
    spectra, mean location and displacement information, and concatenate all
    files pertaining to a specific data type (e.g. displacements) into a single
    comma delimited (CSV) file. For example, all displacement information
    (contained in ????_FLT.CSV) is combined as

       (input)                                (output)
    0010_FLT.CSV  -|
    0011_FLT.CSV   |      running script
    0012_FLT.CSV   |           ==== >       displacement.CSV
    ............   |
    000N_FLT.CSV  -|

    and similarly for spectral ( xxxx_SPC.CSV => Szz.csv) and location
    (xxxx_LOC.CSV => location.csv) files. Further, after all spectral files have
    been combined. Bulk parameters (significant wave height, peak periodm etc.) are calculated from the spectral files,
    and stored seperately in bulkparameters.csv

    NOTE: the original data files will remain unchanged.

Install:

    In order to use this script, python (version 2 or 3) needs to be installed
    on the system (download at: www.python.org). In addition, for functionality
    the script requires that the following python python modules:

        dependencies: pandas, numpy, scipy

    These modules can be installed by invoking the python package manager
    (pip) from the command line. For instance, to install pandas you would
    run the package manager from the command line as:

        pip install pandas

    and similarly for other missing dependencies.

Usage:

    To use the module, simply copy the SPOTTER files and this script into the
    same directory. Subsequently, start a command line terminal, navigate
    to the directory containing the files and run the python script from the
    command line using the python interpreter as:

        python sd_file_parser.py

    or any other python interpreter (e.g. ipython, python3 etc.).

    Requesting additional output:

        By default, the script will only produce the variance density spectrum.
        If in addition the directional moments are desired, add the command line
        switch spectra=all, i.e.:

        python spotter.py spectra='all'

        in which case files containing a1,b1,a2,b2 (in separate files) will be
        produced.

Output:

    After completion, the following files will have been created in the working
    directory:

        FILE              :: DESCRIPTION
        ------------------------------------------------------------------------
        Szz.csv           :: Variance density spectra of vertical displacement [meter * meter / Hz]
        Sxx.csv           :: Variance density spectra of eastward displacement [meter * meter / Hz]
        Syy.csv           :: Variance density spectra of northward displacement [meter * meter / Hz]
        Qxz.csv           :: Quad-spectrum between vertical and eastward displacement [meter * meter / Hz]
        Qyz.csv           :: Quad-spectrum between vertical and northward displacement [meter * meter / Hz]
        Cxy.csv           :: Co-spectrum between northward and eastward displacement [meter * meter / Hz]
        a1.csv            :: First order cosine coefficient [ - ]
        b1.csv            :: First order sine coefficient   [ - ]
        a2.csv            :: Second order cosine coefficient  [ - ]
        b2.csv            :: Second order sine coefficient  [ - ]
        location.csv      :: Average location (lowpass filtered instantaneous
                             location) in terms of latitude and longitude
                             (decimal degrees)
        displacement.csv  :: Instantaneous displacement from mean location
                             along north, east and vertical directions(in meter)
        bulkparameters    :: Bulk wave parameters (Significant wave height, peak period, etc.)

     Data is stored as comma delimited file, where each new line corresponds to
     a new datapoint in time, and the individual columns contain different data
     entries (time, latitude, longitude etc.).

     The spectra files start at the first line with a header line and each
     subsequent line contains the wave spectrum calculated at the indicated time

     HEADER:   year,month,day,hour,min,sec,milisec,dof , 0.0 , f(1) , f(2) , .... , (nf-1) * df
               2017,11   ,10 ,5   ,3  ,1  ,300     ,30 , E(0), E(1) , E(2) , .... , E(nf-1)
               2017,11   ,10 ,5   ,33 ,1  ,300     ,30 , E(0), E(1) , E(2) , .... , E(nf-1)
                |    |    |   |    |   |   |        |    |    |       |     |
               2017,12   ,20 ,0   ,6  ,1  ,300     ,30 , E(0), E(1) , E(2) , .... , E(nf-1)

    The first columns indicate the time (year, month etc.) and dof is the
    degrees of freedom (dof) used to calculate the spectra. After
    the degrees of freedom, each subsequent entry corresponds to the variance
    density at the frequency indicated by the header line (E0 is the energy in
    the mean, E1 at the first frequency f1 etc). The Spotter records
    at an equidistant spectral resolution of df=0.009765625 and there are
    nf=128 spectral entries, given by f(j) = df * j (with 0<=j<128). Frequencies are
    in Hertz, and spectral entries are given in squared meters per Hz (m^2/Hz) or
    are dimensionless (for the directional moments a1,a2,b1,b2).

    The bulk parameter (bulkparameters.csv) file starts with a header line
    and subsequent lines contain the bulk parameters calculated at the
    indicated time

    HEADER:    # year , month , day, hour ,min, sec, milisec , Significant Wave Height, Mean Period, Peak Period, Mean Direction, Peak Direction, Mean Spreading, Peak Spreading
               2017,11   ,10 ,5   ,3  ,1  ,300     ,30 , Hs , Tm01, Tp, Dir, PDir, Spr, PSpr
               2017,11   ,10 ,5   ,33 ,1  ,300     ,30 , Hs , Tm01, Tp, Dir, PDir, Spr, PSpr
                |    |    |   |    |   |   |        |     | ,   | , | , |  , |   , |  , |
               2017,12   ,20 ,0   ,6  ,1  ,300     ,30 , Hs , Tm01, Tp, Dir, PDir, Spr, PSpr

    For the definitions used to calculate the bulk parameters from the
    variance density spectra, and a short description please refer to:

    https://content.sofarocean.com/hubfs/Spotter%20product%20documentation%20page/wave-parameter-definitions.pdf


Major Updates:

    Author   | Date      | Firmware Version | Script updates
    -----------------------------------------------------------------------
    P.B.Smit | Feb, 2018 | 1.4.2            | firmware SHA verification
    P.B.Smit | May, 2018 | 1.5.1            | Included IIR phase correction
    P.B.Smit | June, 2019| 1.7.0            | Bulk parameter output
    P.B.Smit | Oct, 2019 | 1.8.0            | SST Spotter update
    various  | Dec, 2021 | 1.8.0+, 2.0.0+   | Spotter v3 update
"""

import inspect
import os
import sys
from pathlib import Path
from typing import List, Literal, Optional

from concat import cat
from filenames import PathLike
from parsing import parseLocationFiles, parseSpectralFiles
from spectrum import Spectrum
from versions import getVersions


def main(
    path: Optional[PathLike] = None,
    outpath: Optional[PathLike] = None,
    outputFileType: Literal["CSV", "matlab", "numpy", "gz"] = "CSV",
    spectra: str = "all",
    suffixes: Optional[List[str]] = None,
    parsing: Optional[List[str]] = None,
    lfFilter=False,
    bulkParameters=True,
):
    """
    Combine selected SPOTTER output files  into CSV files. This routine is called by
    __main__ and that calls in succession the separate routines to concatenate, and parse
    files.

    Inputs
    ------
    path : str
        Path to a directory containing the Spotter data SD card data files
    outpath : str
        Path to a directory in which the concatenated and processed data should be saved
    outputFileType : str
        CSV is the default. Alternatives are matlab, numpy, and gz.
    spectra : str
        To just output one spectra, specify it. Options are Szz, a1, b1, a2, b2, Sxx, Syy,
        Qxz, Qyz, Cxy. Default is 'all' for all.
    suffixes : list of str, optional
        Suffixes to parse. Default is to parse all.
    parsing : list of str, optional
        Suffixes to postprocess. Default is all.
    lfFilter : bool
        Should a low-frequency filter be done on the spectral files? Default is False.
    bulkParameters : bool
        Should the bulk spectral parameters be calculated? Default is True.

    """

    # Check the version of Files
    versions = getVersions(path)

    # The filetypes to concatenate
    if suffixes is None:
        suffixes = ["FLT", "SPC", "SYS", "LOC", "GPS", "SST", "SMD", "BARO"]

    if parsing is None:
        parsing = ["FLT", "SPC", "LOC", "SST"]

    outFiles = {
        "FLT": "displacement",
        "SPC": "spectra",
        "SYS": "system",
        "LOC": "location",
        "GPS": "gps",
        "SST": "sst",
        "SMD": "smartmooring_data",
        "BARO": "barometer",
    }
    # If no path given, assume current directory
    path = Path(path).absolute() if path else Path().absolute()
    # If no outpath given, assume same as path
    outpath = Path(outpath).absolute() if outpath else path

    # Which spectra to process
    if spectra == "all":
        outputSpectra = ["Szz", "a1", "b1", "a2", "b2", "Sxx", "Syy", "Qxz", "Qyz", "Cxy"]
    else:
        outputSpectra = [spectra]
    if len(versions) == 1:
        # if there is only a single version- we allow all files to be parsed. this is a
        # clutch to account for the fact that sys files are not garantueed to be written.
        # In general assuming everything is the same version seems safe- allowing to parse
        # multiple different versions is perhaps something we want to stop supporting as
        # it adds a lot of fragile logic.
        versions[0]["fileNumbers"] = None

    for index, version in enumerate(versions):
        outd = Path(outpath, str(index)) if len(versions) > 1 else outpath
        outd.mkdir(parents=True, exist_ok=True)

        for suffix in suffixes:
            file_path = outd / f"{outFiles[suffix]}.csv"
            # For each filetype, concatenate files to intermediate CSV files...
            print(f"Concatenating all {suffix} files:")
            if not (
                cat(
                    path=path,
                    outputFileType="CSV",
                    Suffix=suffix,
                    output_file_path=file_path,
                    versionFileList=version["fileNumbers"],
                    compatibility_version=version["number"],
                )
            ):
                continue

            # ... once concatenated, process files further (if appropriate)
            if suffix in parsing:
                if suffix in ["FLT", "LOC", "GPS", "SST"]:
                    # parse the mean location/displacement files; this step transforms
                    # unix epoch to date string.
                    parseLocationFiles(
                        input_file_path=file_path,
                        kind=suffix,
                        output_file_path=file_path,
                        outputFileType=outputFileType,
                        versionNumber=version["number"],
                        IIRWeightType=version["IIRWeightType"],
                    )
                elif suffix in ["SPC"]:
                    # parse the mean location/displacement files; this step extract
                    # relevant spectra (Szz, Sxx etc.) from the bulk spectral file
                    parseSpectralFiles(
                        inputFileName=file_path,
                        outputPath=outd,
                        outputFileType=outputFileType,
                        outputSpectra=outputSpectra,
                        lfFilter=lfFilter,
                        versionNumber=version["number"],
                    )
                    os.remove(file_path)
            # parsing
        # suffix
        # Generate bulk parameter file
        if bulkParameters:
            spectrum = Spectrum(spectra_dir=outd, out_dir=outd)
            if spectrum.spectral_data_is_available:
                spectrum.generate_text_file()


def validCommandLineArgument(arg: str):
    out = arg.split("=")

    if not (len(out) == 2):
        print(f"ERROR: Unknown commandline argument: {arg}")
        sys.exit(1)
    key, val = out

    # normalize arg names to the capitalization required by main()
    argnames = list(inspect.signature(main).parameters)
    for argname in argnames:
        if key.lower() == argname.lower():
            key = argname
            break
    else:
        print(f"ERROR: unknown commandline argument {key}")
        sys.exit(1)
    if key in ["suffixes", "parsing"]:
        # Make into a list
        val = val.replace("[", "").replace("]", "")
        val = [val]
    elif key in ["lfFilter", "bulkParameters"]:
        val = val.lower() == "true" or val.lower() == "yes"
    return (key, val)


if __name__ == "__main__":
    # execute only if run as a script
    narg = len(sys.argv[1:])
    if narg > 0:
        # parse and check command line arguments
        arguments = dict()
        for argument in sys.argv[1:]:
            key, val = validCommandLineArgument(argument)
            arguments[key] = val
    else:
        arguments = dict()
    main(**arguments)
