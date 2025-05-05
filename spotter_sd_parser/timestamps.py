"""timestamps.py - Functions that add time-related columns to a dataframe with a
DatetimeIndex index."""

from typing import Literal

import pandas as pd


def add_unix_epoch_to_df(
    df: pd.DataFrame,
    colname="unix_epoch",
    precision: Literal["s", "ms", "us", "ns"] = "ms",
) -> tuple[pd.DataFrame, str]:
    """Use a DataFrame's DatetimeIndex to create a UNIX epoch column prepended to the
    DataFrame.

    Inputs
    ------
    df
        DataFrame with a DatetimeIndex
    colname
        Name for the new column with the epoch time
    precision
        Precision for the epoch time column: ["s", "ms", "us", "ns"]

    Returns
    -------
    df
        The updated Dataframe (the index is not modified)
    fmt_spec
        The format specifier appropriate for this precision
    """
    divisors = {"s": 1.0, "ms": 1e3, "us": 1e6, "ns": 1e9}
    fmt_specs = {"s": "%.f", "ms": "%.3f", "us": "%.6f", "ns": "%.9f"}
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(
            "The dataframe must have a DatetimeIndex index to use this function."
        )

    dt_type = f"datetime64[{precision}]"
    df[colname] = df.index.astype(dt_type).astype("int64") / divisors[precision]
    cols = list(df.columns)
    return df[cols[-1:] + cols[:-1]], fmt_specs[precision]


def add_timecols_to_df(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, tuple[str, ...], tuple[str, ...]]:
    """Use a DataFrame's DatetimeIndex to create integer columns
    "year", "month", "day", "hour", "minute", "second", "millisecond"

    Inputs
    ------
    df
        DataFrame with a DatetimeIndex

    Returns
    -------
    df
        The updated DataFrame with the time columns prepended (the index is not modified).
    colnames
        The names of the new columns.
    fmt_spec
        The format specifiers of the columns for use with np.savetxt.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(
            "The dataframe must have a DatetimeIndex index to use this function."
        )
    # Add time columns to the dataframe
    for part in ["year", "month", "day", "hour", "minute", "second", "microsecond"]:
        df[part] = getattr(df.index, part)
    # convert the microseconds column to a milliseconds column
    df["millisecond"] = (df["microsecond"] / 1000).astype(int)
    df = df.drop(columns=["microsecond"])
    # move new columns to the start
    cols = list(df.columns)
    df = df[cols[-7:] + cols[:-7]]
    return (
        df,
        ("year", "month", "day", "hourminute", "second", "millisecond"),
        ("%4d", "%2d", "%2d", "%2d", "%2d", "%2d", "%3d"),
    )
