import pandas as pd

from typing import Literal

def df_dtindex_to_unix_epoch(
    df: pd.DataFrame,
    colname="unix_epoch",
    precision: Literal["s", "ms", "us", "ns"] = "ms",
) -> tuple[pd.DataFrame, str]:
    """Use a DataFrame's DatetimeIndex to create a UNIX epoch column prepended to the DataFrame.

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
    assert isinstance(df.index, pd.DatetimeIndex)

    dt_type = f"datetime64[{precision}]"
    df[colname] = df.index.astype(dt_type).astype("int64") / divisors[precision]
    cols = list(df.columns)
    return df[cols[-1:] + cols[:-1]], fmt_specs[precision]


def df_dtindex_split(
        df: pd.DataFrame
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
        The updated DataFrame (the inde is not modified)
    fmt_spec
        A tuple of format specifiers for use with np.savetxt
    """
    assert isinstance(df.index, pd.DatetimeIndex)
    for part in ["year", "month", "day", "hour", "minute", "second", "microsecond"]:
        df[part] = getattr(df.index, part)
    # convert microseconds to milliseconds
    df["millisecond"] = (df["microsecond"]/1000).astype(int)
    df = df.drop(columns=["microsecond"])
    # move new columns to the start
    cols = list(df.columns)
    return (
        df[cols[-7:] + cols[:-7]],
        ("year", "month", "day", "hour" "minute", "second", "millisecond"),
        ("%4d", "%2d", "%2d", "%2d", "%2d", "%2d", "%3d")
    )
