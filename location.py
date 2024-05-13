"""Parsing and preprocessing for *_LOC.CSV files"""
import io
import tempfile
from pathlib import Path
from typing import Union

import pandas as pd
from typing_extensions import TypeAlias

LOC_IN_HEADER_LINE = "GPS_Epoch_Time(s),lat(deg),lat(min*1e5),long(deg),long(min*1e5)\n"
"""Original header line of LOC files"""
LOC_OUT_HEADER_LINE = (
    "year,month,day,hour,min,sec,msec,"
    "latitude (decimal degrees),longitude (decimal degrees)"
)
"""header line for concatenated LOC file for np.savetxt"""
LOC_LINE_FORMAT = ",".join(["%d"]*7 + ["%13.8f"]*2)
"""field formats for np.savetxt"""

PathLike: TypeAlias = Union[str, Path]

def parse_location_file(loc_path: PathLike) -> pd.DataFrame:
    """Parse a *_LOC.csv spotter file and return a pandas dataframe with a datetime index"""
    with tempfile.TemporaryFile(mode="w+") as tmpfile:
        clean_loc_file(loc_path, tmpfile)
        tmpfile.seek(0)
        loc_df = (
            pd.read_csv(tmpfile, index_col=False)
            .dropna()
            .reset_index(drop=True)
            .astype(int)
            .sort_values("GPS_Epoch_Time(s)")
        )
    loc_df = loc_df.set_index(pd.to_datetime(loc_df["GPS_Epoch_Time(s)"], unit="s"))
    loc_df["latitude"] = loc_df["lat(deg)"] + loc_df["lat(min*1e5)"] / 6000000.0
    loc_df["longitude"] = loc_df["long(deg)"] + loc_df["long(min*1e5)"] / 6000000.0
    return loc_df.drop(columns=[
        "GPS_Epoch_Time(s)", "lat(deg)", "lat(min*1e5)", "long(deg)", "long(min*1e5)",
    ])


def merge_location_files(loc_dir: PathLike) -> pd.DataFrame:
    loc_dir_path = Path(loc_dir)
    loc_paths = [*loc_dir_path.glob("*_LOC.csv"), *loc_dir_path.glob("*_LOC.CSV")]
    loc_dfs = []
    for loc_path in loc_paths:
        loc_df = parse_location_file(loc_path)
        if len(loc_df) > 0:
            loc_dfs.append(loc_df)
    return pd.concat(loc_dfs, axis=0).sort_index()


def floatable(v):
    """Check if v can be converted to a float"""
    try:
        v = float(v)
    except ValueError:
        return False
    return True


def line_ok(line: str, nparts: int) -> bool:
    parts = line.split(",")
    return len(parts) == nparts and floatable(parts[0])


def clean_loc_file(loc_path: PathLike, tmpfile: io.TextIOWrapper) -> None:
    """Remove duplicate header lines, ensure five fields per line."""
    with open(loc_path) as fobj:
        lines = fobj.readlines()
    lines = [line for line in lines if line_ok(line, 5)]
    lines = [line.replace("\r\n", "\n").replace("\r", "\n") for line in lines]
    # Replace the header line that was removed (and make sure it's consistent)
    lines = [LOC_IN_HEADER_LINE, *lines]
    if lines[-1][-1] != "\n":
        lines[-1] = f"{lines[-1]}\n"
    tmpfile.writelines(lines)


# def split_df_tstamp(df: pd.DataFrame) -> pd.DataFrame:
#     """Change the datetime index into """
#     assert isinstance(df.index, pd.DatetimeIndex)
#     ncol_orig = len(df.columns)
#     for part in ["year", "month", "day", "hour", "minute", "second", "microsecond"]:
#         df[part] = getattr(df.index, part)
#     # convert microseconds to milliseconds
#     df["millisecond"] = (df["microsecond"]/1000).astype(int)
#     df = df.drop(columns=["microsecond"])
#     # move new columns to the start
#     cols = list(df.columns)
#     df = (
#         df[cols[ncol_orig:] + cols[:ncol_orig]]
#         .reset_index()
#         .drop(columns=["GPS_Epoch_Time(s)"])
#     )
#     return df


# def concat_loc(loc_dir: PathLike, loc_out_path: PathLike, overwrite_ok=False):
#     loc_dir = Path(loc_dir)
#     loc_out_path = Path(loc_out_path)
#     if not loc_dir.is_dir():
#         raise ValueError("loc_dir must be a path to a directory.")
#     if loc_out_path.exists() and not overwrite_ok:
#         raise ValueError(f"loc_out_path '{loc_out_path}' already exists.")
#     df = merge_location_files(loc_dir)
#     df = split_df_tstamp(df)
#     np.savetxt(loc_out_path, df.values, LOC_FORMATS, header=LOC_OUT_HEADER_LINE)

