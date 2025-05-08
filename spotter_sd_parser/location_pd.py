import logging
import sys
from itertools import repeat
from pathlib import Path
from typing import Annotated, NamedTuple

import pandas as pd
from cyclopts import Parameter
from cyclopts.types import Directory, ExistingDirectory

SCRIPTNAME = Path(__file__).name

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

LOC_FNAME = "location.csv"


class EpochRange(NamedTuple):
    min: float | None = None
    max: float | None = None


def _floatable(v):
    """Check if v can be converted to a float"""
    try:
        v = float(v)
    except ValueError:
        return False
    return True


def _int_able(v):
    """Check if v can be converted to an int"""
    try:
        v = int(v)
    except ValueError:
        return False
    return True


def _read_and_clean_loc_csv(
    loc_path: str | Path, epoch_range: EpochRange = EpochRange()
) -> pd.DataFrame:
    """Read a LOC CSV file into a Pandas DataFrame, including some basic cleaning.

    Parameters
    ----------
    loc_path
        Path to a LOC CSV file
    epoch_range
        Min and max values for the epoch_t (UNIX timestamp)

    Returns
    -------
    A Pandas dataframe with columns `epoch_t`, `lat_dd`, `lon_dd`
    """
    df = pd.read_csv(
        loc_path,
        names=["epoch_t", "lat_d", "lat_min_e5", "lon_d", "lon_min_e5"],
        dtype=str,  # fist assume is everything is a string; later convert to other types
        skiprows=1,  # don't use the included header
    )
    df = (
        # Keep rows where `epoch_t`` can be converted to `float`` and `link`` to `int`
        df[
            df["epoch_t"].map(_floatable)
            & df["lat_d"].map(_int_able)
            & df["lat_min_e5"].map(_int_able)
            & df["lon_d"].map(_int_able)
            & df["lon_min_e5"].map(_int_able)
        ]
        .astype(
            {
                "epoch_t": float,
                "lat_d": int,
                "lat_min_e5": int,
                "lon_d": int,
                "lon_min_e5": str,
            }
        )
        .dropna(axis=0, how="any")
    )
    df = df[df["epoch_t"] != 0]
    if epoch_range.min:
        df = df[df["epoch_t"] >= epoch_range.min]
    if epoch_range.max:
        df = df[df["epoch_t"] <= epoch_range.max]
    df["lat_dd"] = df["lat_d"] + df["lat_min_e5"].astype(float) / (60 * 1e5)
    df["lon_dd"] = df["lon_d"] + df["lon_min_e5"].astype(float) / (60 * 1e5)
    df = df.drop(["lat_d", "lat_min_e5", "lon_d", "lon_min_e5"], axis=1)
    return df


def parse_loc_csv(
    loc_path: str | Path, epoch_range: EpochRange = EpochRange()
) -> pd.DataFrame | None:
    loc_path = Path(loc_path)
    logger.info(
        "Parsing %s: %.2f MiB",
        loc_path.relative_to(loc_path.parent.parent),
        loc_path.stat().st_size / 1e6,
    )
    try:
        return _read_and_clean_loc_csv(loc_path, epoch_range)
    except Exception as exc:
        logger.error("ERROR with %s: %s", loc_path, exc)


def _merge_and_sort(loc_dfs: list[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(loc_dfs, ignore_index=True).sort_values("epoch_t", ignore_index=True)


def _epoch_t_to_split(df: pd.DataFrame) -> pd.DataFrame:
    timestamps = pd.to_datetime(df["epoch_t"], unit="s")
    del df["epoch_t"]
    for part in ["year", "month", "day", "hour", "minute", "second", "microsecond"]:
        df[part] = getattr(timestamps.dt, part)
    # convert microseconds to milliseconds
    df["millisecond"] = (df["microsecond"] / 1000).astype(int)
    df = df.drop(columns=["microsecond"])
    # move new columns to the start
    cols = list(df.columns)
    return df[cols[-7:] + cols[:-7]]


def write_merged_loc_data(out_dir: Path, df: pd.DataFrame):
    colnames = {
        "year": "# year",
        "minute": "min",
        "second": "sec",
        "millisecond": "msec",
        "lat_dd": "latitude (decimal degrees)",
        "lon_dd": "longitude (decimal degrees)",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / LOC_FNAME
    df.rename(columns=colnames).to_csv(csv_path, float_format="%.8f", index=False)


def main(
    raw_data_dir: ExistingDirectory,
    output_data_dir: Annotated[Directory, Parameter(name=["output-data-dir", "-d"])]
    | None = None,
    min_datetime: Annotated[str | None, Parameter(name=["min-datetime", "-n"])] = None,
    max_datetime: Annotated[str | None, Parameter(name=["max-datetime", "-x"])] = None,
) -> None:
    """Merge and reformat spotter location data from \\*_LOC.csv files into a single
    location.csv file.

    Parameters
    ----------
    raw_data_dir
        The existing directory where all the \\*_LOC.csv files are.
    output_data_dir
        The directory to which to save the merged location.csv file. If not
        provided, the file will be saved in a "parsed" subdirectory of `raw_data_dir`.
    min_datetime
        Minimum date/time to keep in output. May be provided as any string that can be
        interpreted by pandas.Timestamp(). Naive values are assumed to be UTC.
    max_datetime
        Maximum date/time to keep in output. May be provided as any string that can be
        interpreted by pandas.Timestamp(). Naive values are assumed to be UTC.
    """
    if output_data_dir is None:
        output_data_dir = raw_data_dir / "parsed"
    min_epoch_t = pd.Timestamp(min_datetime).timestamp() if min_datetime else None
    max_epoch_t = pd.Timestamp(max_datetime).timestamp() if max_datetime else None
    epoch_range = EpochRange(min_epoch_t, max_epoch_t)

    loc_paths = [*raw_data_dir.glob("*_LOC.csv"), *raw_data_dir.glob("*_LOC.CSV")]
    loc_dfs = [
        loc_df
        for loc_df in map(parse_loc_csv, loc_paths, repeat(epoch_range))
        if loc_df is not None
    ]
    logger.info("Merging and sorting data.")
    merged_df = _merge_and_sort(loc_dfs)
    merged_df = _epoch_t_to_split(merged_df)
    logger.info("Writing location data to %s", output_data_dir / LOC_FNAME)
    write_merged_loc_data(output_data_dir, merged_df)


if __name__ == "__main__":
    from cyclopts import App

    app = App(name="location_pd")
    app.default(main)
    app()
