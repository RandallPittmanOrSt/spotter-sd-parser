from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from spotter_sd_parser.location import (
    LOC_LINE_FORMAT,
    LOC_OUT_HEADER_LINE,
    merge_location_files,
)
from spotter_sd_parser.timestamps import add_timecols_to_df, add_unix_epoch_to_df


def concat_spotter_files(
    data_dir: Path,
    concat_outfile_path: Path,
    prefix_merger: Callable[[Path], pd.DataFrame],
    header: str,
    line_format: str,
    tstamp_type: Literal["UNIX epoch", "splitcols"] | None = None,
    overwrite_ok=False,
):
    """Concatenate a bunch of spotter data files into a single data file, with cleanup
    and sorting by date along the way

    Inputs
    ------
    data_dir
        Dir where all the CSV data files live
    concat_output_file
        Path to the CSV file to write
    prefix_merger
        A callable that takes data_dir and returns a pandas DataFrame with the merged data
        for the prefix (e.g. LOC)
    header
        Input for `header` kwarg of `np.savetext`. This will be different depending on
        the tstamp_type
    line_format
        string to use with `fmt` kwarg of `np.savetext` containing a %-format specifier
        for each field. This will be different depending on tstamp_type
    tstamp_type
        Optional "UNIX epoch" or "splitcols" to determine whether to return the intial timestamp
        column as seconds since midnight 1/1/1970 UTC or yyyy, mm, dd, etc. columns
    overwrite_ok
        If False (default) a prexisting `concat_output_file` results in an error.
    """
    data_dir = Path(data_dir)
    concat_outfile_path = Path(concat_outfile_path)
    if not data_dir.is_dir():
        raise ValueError("loc_dir must be a path to a directory.")
    if concat_outfile_path.exists() and not overwrite_ok:
        raise ValueError(f"loc_out_path '{concat_outfile_path}' already exists.")

    df = prefix_merger(data_dir)
    if tstamp_type == "splitcols":
        df = add_timecols_to_df(df)[0]
    elif tstamp_type == "UNIX epoch":
        df, _ = add_unix_epoch_to_df(df)

    concat_outfile_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(concat_outfile_path, df.values, fmt=line_format, header=header)


concat_fns = {
    "LOC": partial(
        concat_spotter_files,
        prefix_merger=merge_location_files,
        header=LOC_OUT_HEADER_LINE,
        line_format=LOC_LINE_FORMAT,
        tstamp_type="splitcols",
    )
}
