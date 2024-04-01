import gzip
import os.path

import numpy as np
import pandas as pd

from filenames import extensions, getFileNames
from versions import defaultVersion


def floatable(v):
    """Check if v can be converted to a float"""
    try:
        v = float(v)
    except ValueError:
        return False
    return True


def cat(
    path=None,
    outputFileName="displacement.CSV",
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
        path=path, suffix=Suffix, message="_" + Suffix, versionFileList=versionFileList
    )
    if len(fileNames) == 0:
        return False

    fname = os.path.splitext(outputFileName)[0]
    outputFileName = fname + "." + extensions(outputFileType)

    with Outfile(outputFileName, outputFileType) as outfile:
        for index, filename in enumerate(fileNames):
            if reportProgress:
                print(
                    "- "
                    + filename
                    + " (File {} out of {})".format(index + 1, len(fileNames))
                )

            if Suffix == "SPC":
                ip = 0
                prevLine = ""
                mode = modeDetection(path, filename)

                with open(os.path.join(path, filename)) as infile:
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
                        log_corrupt_file(os.path.join(path, filename))

            else:
                # Suffix is not 'SPC'
                fqfn = os.path.join(path, filename)
                with open(fqfn) as infile:
                    try:
                        lines = infile.readlines()
                    except:
                        log_corrupt_file(os.path.join(path, filename))
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
                        elif os.path.splitext(filename)[1].lower() == ".csv":
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
    def __init__(self, outputFileName, outputFileType):
        self.path = outputFileName
        self.gzip = outputFileType.lower() == "gz"

    def open(self):
        self.file = (
            gzip.open(self.path, "wb") if self.gzip else open(self.path, "w")
        )

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


def get_epoch_to_milis_relation(sst_file):
    # This function gets the relation between milis and epoch from the
    # FLT file. This assumes FLT exist, otherwise we get an error

    head, tail = os.path.split(sst_file)
    tail = tail.replace("SST", "FLT")

    flt_file = os.path.join(head, tail)
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


def process_sst_lines(lines, infile):
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
                outlines.append(str(epoch) + " , " + data[1])
                previousvalue = value
            else:
                # malformed line
                pass
    return [f"{line}\n" for line in outlines]


def process_SMD_lines(lines, generator=True):
    def iszero(v):
        return floatable(v) and float(v) == 0

    def lineitems(line):
        return line.split(",")

    def firstitem(line):
        return lineitems(line)[0]

    def toomany_items(line):
        items = lineitems(line)
        return len(items) > 3 and items[3] == "RBRDT" and len(items) > 7

    def sortval(line):
        try:
            return float(firstitem(line))
        except ValueError:
            return 0.0

    # Get header line, if any
    header = []
    if not floatable(firstitem(lines[0])):
        header = lines[0:1]
        lines = lines[1:]
    sorted_lines = sorted(
        [
            line
            for line in lines
            if not iszero(firstitem(line)) and not toomany_items(line)
        ],
        key=sortval,
    )
    return header + sorted_lines


def modeDetection(path, filename):
    """Detect if we are in debug or in production mode. We do this based on the first few
    lines; in debug these will contain either FFT or SPEC, whereas in production only
    SPECA is encountered"""

    mode = "production"
    with open(os.path.join(path, filename)) as infile:
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
