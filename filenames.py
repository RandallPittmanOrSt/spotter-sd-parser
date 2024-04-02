import re
from itertools import chain
from pathlib import Path
from typing import Optional, Union

PathLike = Union[Path, str]


def extensions(outputFileType):
    ext = {"csv": "csv", "matlab": "mat", "numpy": "npz", "pickle": "pickle", "gz": "gz"}
    if outputFileType.lower() in ext:
        return ext[outputFileType.lower()]
    else:
        raise Exception(
            "Unknown outputFileType; options are: numpy, matlab, pickle, csv"
        )


def getFileNames(path: Optional[PathLike], suffix, message, versionFileList=None):
    """This function returns all the filenames in a given *path* that conform to
    [D*]D_YYY.CSV where YYY is given by *suffix*."""

    path = Path(path) if path else Path()
    path = path.absolute()

    synonyms = [suffix]
    if suffix == "LOC":
        synonyms.append("GPS")
    # Get the file list from the directory, and select only those files that
    # match the Spotter output filename signature

    # initial file list is all filenames of the given suffix sorted by number
    exts = ["CSV", "csv", "log"]
    name_re = re.compile(r"\d+_\w{3}\.\w{3}")
    initial_fileNames = [
        p.name
        for p in sorted(
            chain.from_iterable(path.glob(f"*_{suffix}.{ext}") for ext in exts),
            key=lambda f: int(f.name.split("_")[0]),
        )
        if name_re.match(p.name)
    ]

    if versionFileList is not None:
        # Only add filenames that are in the present version file number list
        fileNames = [
            filename
            for filename in initial_fileNames
            if filename.split("_")[0].strip() in versionFileList
        ]
    else:
        fileNames = initial_fileNames
    # Are there valid Spotter files?
    if len(fileNames) < 1:
        # No files found; raise exception and exit
        print(f"  No {message} data files available.")
    return (path, fileNames)
