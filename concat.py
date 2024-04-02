import gzip
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from filenames import PathLike, extensions, getFileNames
from versions import defaultVersion


def floatable(v):
    """Check if v can be converted to a float"""
    try:
        v = float(v)
    except ValueError:
        return False
    return True


def cat(
    path: Path,
    output_file_path: Path = Path("displacement.CSV"),
    Suffix="FLT",
    reportProgress=True,
    outputFileType="CSV",
    versionFileList=None,
    compatibility_version=defaultVersion,
):
    """
    This functions concatenates raw csv files with a header. Only for the first file it
    retains the header. Note that for SPEC files and SST files special processing is done.
    Specifically, for SST files we map the millis timebase onto the epochtime base using a
    relation estimated from the FLT files.
    """

    # Get a list of location filenames and the absolute path
    path, fileNames = getFileNames(
        path=path, suffix=Suffix, message=f"_{Suffix}", versionFileList=versionFileList
    )
    if len(fileNames) == 0:
        return False

    # convert extension to the proper extension of this output file type
    output_file_path = (
        output_file_path.parent / f"{output_file_path.name}.{extensions(outputFileType)}"
    )

    with Outfile(output_file_path, outputFileType) as outfile:
        for index, filename in enumerate(fileNames):
            if reportProgress:
                print(f"- {filename} (File {index + 1} of {len(fileNames)})")

            if Suffix == "SPC":
                ip = 0
                prevLine = ""
                mode = modeDetection(path, filename)

                with open(path / filename) as infile:
                    try:  #
                        # lines = infile.readlines()
                        ii = 0
                        line = infile.readline()
                        if index == 0:
                            outfile.write(line)

                        for line in infile:
                            # Read the ensemble counter
                            if (
                                line[0:8] == "SPEC_AVG"
                                or line[0:5] == "SPECA"
                                or line[0:8] == "SPECA_CC"
                            ):
                                a, b = find_nth(line, ",", 5)
                                ii = int(line[b + 1 : a])

                                if mode == "production":
                                    outfile.write(line)
                                    ip = ii
                                else:
                                    # Debug spectral file, contains all averages, only
                                    # need last.
                                    if line[0:8] == "SPECA_CC":
                                        outfile.write(line)
                                        ip = 0
                                    elif ii < ip:
                                        outfile.write(prevLine)
                                        ip = 0
                                    else:
                                        ip = ii
                                        prevLine = line
                    except Exception:
                        log_corrupt_file(path / filename)

            else:
                # Suffix is not 'SPC'
                fqfn = path / filename
                with open(fqfn) as infile:
                    try:
                        lines = infile.readlines()
                    except Exception:
                        log_corrupt_file(fqfn)
                    else:
                        if not lines:
                            continue
                        header_line = lines.pop(0)
                        # If SST file, map millis onto epochs
                        if Suffix == "SST":
                            if compatibility_version < 3:
                                lines = process_sst_lines(lines, fqfn)
                        if Suffix == "SMD":
                            # remove SMD lines with epoch 0
                            lines = process_SMD_lines(lines)
                        elif fqfn.suffix.lower() == ".csv":
                            # remove any other lines that don't start with a number
                            lines = [
                                line for line in lines if floatable(line.split(",")[0])
                            ]

                        # if this is the first file of this type, keep the header
                        if index == 0:
                            lines.insert(0, header_line)
                        # Strip dos newline char
                        lines = [line.replace("\r", "") for line in lines]

                        outfile.writelines(lines)
    return True


class Outfile:
    """A wrapper for both regular text file and GZip file I/O"""

    def __init__(self, outputFileName: PathLike, outputFileType: str):
        self.path: PathLike = outputFileName
        self.gzip = outputFileType.lower() == "gz"

    def open(self):
        self.file = gzip.open(self.path, "wb") if self.gzip else open(self.path, "w")

    def close(self):
        self.file.close()

    def write(self, text: str):
        if isinstance(self.file, gzip.GzipFile):
            self.file.write(text.encode("utf-8"))
        else:
            self.file.write(text)

    def writelines(self, lines: List[str]):
        for line in lines:
            self.write(line)

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def get_epoch_to_milis_relation(sst_file: Path):
    # This function gets the relation between milis and epoch from the
    # FLT file. This assumes FLT exist, otherwise we get an error

    flt_file = sst_file.parent / sst_file.name.replace("SST", "FLT")
    data = pd.read_csv(flt_file, index_col=False, usecols=(0, 1))
    data = data.apply(pd.to_numeric, errors="coerce")
    data = data.values
    msk = np.isnan(data[:, 0])
    for ii in range(0, data.ndim):
        msk = np.isfinite(data[:, ii])
        data = data[msk, :]
    data = data[msk, :]
    millis = data[:, 0]
    epochs = data[:, 1]

    ii = np.argmax(millis)

    if ii < 10:
        raise Exception("Roll-over in millis")

    def millis_to_epoch(milis_in):
        return int(
            epochs[0]
            + (milis_in - millis[0]) * (epochs[ii] - epochs[0]) / (millis[ii] - millis[0])
        )

    return millis_to_epoch


def process_sst_lines(lines, infile: Path):
    # Get the function that maps milis to epochs
    #
    # max int used for roll-over; Spotter millis clock resets after reaching the max
    # ints.
    max = 4294967295

    # Get the millis to epoch mapping from the FLT file
    millis_to_epoch = get_epoch_to_milis_relation(infile)

    # Do a line by line processing, replacing millis with epochtime from
    # the mapping
    outlines = []  # Store the output lines
    previousvalue = 0
    for line in lines:
        if "millis" in line:
            # header
            outlines.append(line)
        else:
            data = line.strip().split(",")
            # last line can be empty, check if there are two entries (as expected)
            if len(data) == 2:
                # Look at the delta, millis should be monotonically increasing
                # unless we hit roll-over
                delta = int(data[0]) - previousvalue

                # If the delta is negative, wrap the value
                if delta < -4294000000:
                    delta = delta + max

                # New value from (potentially) corrected delta
                value = previousvalue + delta

                # Convert to epochtime from mapping
                epoch = millis_to_epoch(value)
                outlines.append(f"{epoch} , {data[1]}")
                previousvalue = value
            else:
                # malformed line
                pass
    return [f"{line}\n" for line in outlines]


def process_SMD_lines(lines, generator=True):
    def quote_bsys(items: List[str]) -> List[str]:
        """Ensure the data portion of BSYS lines is quoted if it contains commas"""
        if len(items) > 6 and items[2] == "BSYS":
            last_item = f'"{",".join(items[5:])}"'
            del items[5:]
            items.append(last_item)
        return items

    def toomany_rbr_items(items: List[str]):
        """Detect if the line is an RBRDT line and has too many items (corrupted line)"""
        return len(items) > 3 and items[3] == "RBRDT" and len(items) > 7

    def sortval(firstitem: str) -> float:
        """Callable to use for sorting lines by the numeric value of the first item"""
        try:
            return float(firstitem)
        except ValueError:
            return 0.0

    def process_line(line: str) -> Tuple[float, str]:
        """Process a line from an SMD file.

        Inputs
        ------
        line
            line from an SMD file, newline-terminated

        Returns
        -------
        tuple (sort_value, line)
            sort_value - line timestamp, for sorting. 0.0 if string line, -1 if invalid line
            line - processed line, newline terminated

        None is returned if the line is to be discarded
        """
        items = line.strip().split(",")
        # Discard extra headers lines lines with unknown time (0.0), lines with too few
        # items, and corrupt RBR lines
        if (
            not floatable(items[0])
            or len(items) < 5
            or float(items[0]) == 0.0
            or toomany_rbr_items(items)
        ):
            return (-1, "")
        items = quote_bsys(items)
        line = f'{",".join(items)}\n'
        return sortval(items[0]), line

    sorted_results = sorted((process_line(line) for line in lines), key=lambda r: r[0])
    sorted_lines = [result[1] for result in sorted_results if result[0] >= 0]
    return sorted_lines


def modeDetection(path: Path, filename: str):
    """Detect if we are in debug or in production mode. We do this based on the first few
    lines; in debug these will contain either FFT or SPEC, whereas in production only
    SPECA is encountered"""

    mode = "production"
    with open(path / filename) as infile:
        jline = 0
        for line in infile:
            if line[0:3] == "FFT" or line[0:5] == "SPEC,":
                mode = "debug"
                break
            jline = jline + 1

            if jline > 10:
                break
    return mode


def find_nth(haystack, needle, n):
    """Find nth and nth-1 occurences of a needle in the haystack"""
    start = haystack.find(needle)
    prev = start
    while start >= 0 and n > 1:
        prev = start
        start = haystack.find(needle, start + len(needle))
        n -= 1
    return (start, prev)


def log_corrupt_file(filepath):
    message = f"- ERROR:, file {filepath} is corrupt"
    log_errors(message)
    print(message)


First = True


def log_errors(error):
    global First
    if First:
        with open("error.txt", "w") as file:
            file.write(error + "\n")
        First = False
    else:
        with open("error.txt", "a") as file:
            file.write(error + "\n")
        First = False
