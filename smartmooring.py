#!/usr/bin/env python
"""Parsing and preprocessing for *_SMD.CSV files"""

import io
import logging
import os
import sys
import tempfile
import textwrap
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from typing_extensions import TypeAlias

from timestamps import df_dtindex_to_unix_epoch

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

PathLike: TypeAlias = Union[str, Path]

#: Original header line of SMD files
SMD_IN_HEADER_LINE = "epoch,link,log type,data\n"
#: Header line for concatenated SMD file for np.savetxt
SMD_OUT_HEADER_LINE = "epoch,link,log type,data1,data2,data3,data4"
#: Field formats for np.savetxt
SMD_LINE_FORMAT = "%s,%d,%s,%s,%s,%s,%s"

#: Types for the first three fields of all SMD rows
SMD_BASE_TYPES = {"epoch": float, "link": int, "log_type": str}
#: Types for BSYS rows
BSYS_TYPES = {"bridge_us": int, "log_level": str, "message": str}
#: Types for DATA rows, various module types
MODULES_TYPES = {
    "SOFT2": {"module_ms": int, "temp_cdegC": int},
    "RBRD": {"module_ms": int, "pressure_uBar": int},
    "RBRDT": {"module_ms": int, "pressure_uBar": int, "temp_udegC": int},
    "RBRU": {"module_ms": int},
}

#: Format specifiers for the first three fields of all SMD rows
SMD_BASE_FORMATS = {"epoch": "%.2f", "link": "%d", "log_type": "%s"}
#: Format specifiers for BSYS rows
BSYS_FORMATS = {"bridge_us": "%d", "log_level": "%s", "message": '"%s"'}
#: Format specifiers for DATA rows, various module types
MODULES_FORMATS = {
    "SOFT2": {"module_ms": "%d", "temp_cdegC": "%d"},
    "RBRD": {"module_ms": "%d", "pressure_uBar": "%d"},
    "RBRDT": {"module_ms": "%d", "pressure_uBar": "%d", "temp_udegC": "%d"},
    "RBRU": {"module_ms": "%d"},
}

#: Minimum number of fields in all SMD rows
MIN_SMD_NFIELDS = len(SMD_BASE_TYPES)
#: Number of fields in BSYS rows
BSYS_NFIELDS = MIN_SMD_NFIELDS + len(BSYS_TYPES)
#: Number of fields in DATA rows, by type
MODULE_NFIELDS = {
    # +1 is because field data1 is the module type
    name: MIN_SMD_NFIELDS + len(MODULES_TYPES[name]) + 1
    for name in MODULES_TYPES
}


@dataclass
class SMDData:
    """Dataframes for SmartMooring data returned from parse_SMD_file

    Attributes
    ----------
    bsys
        BSYS log message DataFrames
    modules
        dict of DataFrames from SmartMooring modules. Modules are only present in the dict
        if there is data for that module.
    other_sm
        DataFrame of any leftover SmartMooring data not in the first two entries
    """

    bsys: pd.DataFrame
    modules: dict[str, pd.DataFrame]
    other_sm: pd.DataFrame

    @classmethod
    def empty(cls):
        return cls(pd.DataFrame(), defaultdict(pd.DataFrame), pd.DataFrame())

    def __repr__(self):
        all_mod_txt = ""
        for mod_name in self.modules:
            all_mod_txt += (
                "\n"
                f"{mod_name}:\n"
                f"{textwrap.indent(f'{self.modules[mod_name]}', prefix='  ')}\n"
            )
        return (
            "bsys:\n"
            f"{textwrap.indent(f'{self.bsys}', prefix='  ')}\n"
            f"{all_mod_txt}\n"
        )


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


def _smd_line_ok(line: str) -> Optional[str]:
    parts = line.strip().split(",")
    if (
        not _floatable(parts[0])  # if first part is str, then its a header line
        or len(parts) < len(SMD_BASE_TYPES)  # no cut-off lines
        or not _int_able(parts[1])  # link field must be an integer
        or parts[2] not in ["BSYS", "DATA", "MSD", "BIN", "HB"]  # ensure valid log_type
        or (
            parts[2] == "BSYS" and len(parts) < BSYS_NFIELDS
        )  # last BSYS field may have a comma
        or (
            parts[2] == "DATA"
            and (parts[3] not in MODULES_TYPES or len(parts) != MODULE_NFIELDS[parts[3]])
        )
    ):
        # line is invalid
        return None
    if parts[2] == "BSYS" and len(parts) > BSYS_NFIELDS:
        # join all final parts of bsys line into one quoted message
        n_last_parts = (len(parts) - BSYS_NFIELDS) + 1
        last_parts = parts[-n_last_parts:]
        parts[-n_last_parts] = f'"{",".join(last_parts)}"'
        del parts[-(n_last_parts - 1) :]
    return f"{','.join(parts)}\n"


def _clean_smd_file(smd_path: PathLike, tmpfile: io.TextIOWrapper) -> None:
    """Remove duplicate header lines, ensure five fields per line."""
    with open(smd_path) as fobj:
        lines = fobj.readlines()
    tmpfile_lines = []
    for line in lines:
        if ok_line := _smd_line_ok(line):
            tmpfile_lines.append(ok_line.replace("\r\n", "\n").replace("\r", "\n"))
    if not len(tmpfile_lines):
        # don't write anything at all to tmpfile
        return
    tmpfile.write(SMD_IN_HEADER_LINE)
    tmpfile.writelines(tmpfile_lines)
    # ensure file ends in newline
    if tmpfile_lines[-1][-1] != "\n":
        tmpfile.write("\n")


def _isempty(fobj: io.TextIOWrapper):
    currpos = fobj.tell()
    end = fobj.seek(0, os.SEEK_END)
    fobj.seek(currpos)
    return end == 0


def _parse_SMD_file(smd_path: PathLike) -> SMDData:
    """Parse a *_SMD.csv spotter file and return pandas DataFrames with a datetime
    index"""
    with tempfile.TemporaryFile(mode="w+") as tmpfile:
        _clean_smd_file(smd_path, tmpfile)
        tmpfile.seek(0)
        if _isempty(tmpfile):
            return SMDData.empty()
        sm_df = (
            pd.read_csv(
                tmpfile,
                names=["epoch_t", "link", "log_type", "data1", "data2", "data3", "data4"],
                skiprows=1,
                memory_map=True,
            )
            .set_index("epoch_t")
            .convert_dtypes()
            .astype({"link": int})  # link is an int
            .drop(index=0, errors="ignore")  # drop rows with unknown time
        )
        sm_df.index = pd.to_datetime(sm_df.index, unit="s")
        sm_df = sm_df.sort_index()
    bsys_df = (
        sm_df[sm_df["log_type"] == "BSYS"]
        .drop(columns=["link", "log_type", "data4"])
        .rename(columns={"data1": "bridge_us", "data2": "log_level", "data3": "message"})
        .astype({name: type_ for name, type_ in BSYS_TYPES.items()})
    )
    modules_dfs: dict[str, pd.DataFrame] = {}
    for mod_name in MODULES_TYPES:
        mod_types = MODULES_TYPES[mod_name]
        colnames = {
            f"data{i+2}": fieldname for i, fieldname in enumerate(mod_types.keys())
        }
        dropped_colnames = [
            f"data{i+1}" for i in range(4) if f"data{i+1}" not in colnames
        ]
        mod_df = (
            sm_df[(sm_df["log_type"] == "DATA") & (sm_df["data1"] == mod_name)]
            .drop(columns=["log_type", *dropped_colnames])
            .rename(columns=colnames)
        )
        mod_df = mod_df.astype(
            {fieldname: type_ for fieldname, type_ in mod_types.items()}
        )
        if not mod_df.empty:
            modules_dfs[mod_name] = mod_df
    other_sm_df = sm_df[
        ~sm_df["log_type"].isin(["BSYS", "DATA"])
        | ((sm_df["log_type"] == "DATA") & ~sm_df["data1"].isin(MODULES_TYPES))
    ]
    return SMDData(bsys_df, modules_dfs, other_sm_df)


def _parse_SMD_files(smd_paths: list[Path]) -> list[SMDData]:
    all_results = []
    for path in smd_paths:
        logger.info("Parsing %s (%.1f MiB)", path, path.stat().st_size / 2**20)
        all_results.append(_parse_SMD_file(path))
    return all_results


def _merge_SMD_results(all_smd_data: list[SMDData]) -> SMDData:
    merged_smd_data = SMDData.empty()
    for smd_data in all_smd_data:
        # concatentate new data onto the existing dataframes
        if not smd_data.bsys.empty:
            merged_smd_data.bsys = pd.concat(
                [merged_smd_data.bsys, smd_data.bsys], axis=0
            )
        if not smd_data.other_sm.empty:
            merged_smd_data.other_sm = pd.concat(
                [merged_smd_data.other_sm, smd_data.other_sm], axis=0
            )
        for mod_name in smd_data.modules:
            merged_smd_data.modules[mod_name] = pd.concat(
                [merged_smd_data.modules[mod_name], smd_data.modules[mod_name]], axis=0
            )
    return merged_smd_data


def write_smd_results(out_dir: PathLike, smd_data: SMDData, outfile_prefix=""):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # BSYS
    if not smd_data.bsys.empty:
        bsys_df, _ = df_dtindex_to_unix_epoch(smd_data.bsys, "epoch")
        bsys_header = ",".join(["epoch", *BSYS_FORMATS.keys()])
        bsys_formatstr = ",".join([SMD_BASE_FORMATS["epoch"], *BSYS_FORMATS.values()])
        bsys_path = out_dir / f"{outfile_prefix}BSYS.csv"
        np.savetxt(bsys_path, bsys_df.values, fmt=bsys_formatstr, header=bsys_header)
    # modules
    for mod_name in smd_data.modules:
        mod_df, _ = df_dtindex_to_unix_epoch(smd_data.modules[mod_name], "epoch")
        mod_header = ",".join(["epoch", "link", *MODULES_FORMATS[mod_name].keys()])
        mod_formatstr = ",".join(
            [
                SMD_BASE_FORMATS["epoch"],
                SMD_BASE_FORMATS["link"],
                *MODULES_FORMATS[mod_name].values(),
            ]
        )
        mod_path = out_dir / f"{outfile_prefix}{mod_name.upper()}.csv"
        np.savetxt(mod_path, mod_df.values, fmt=mod_formatstr, header=mod_header)
    # other
    if not smd_data.other_sm.empty:
        other_df, _ = df_dtindex_to_unix_epoch(smd_data.other_sm, "epoch")
        other_header = ",".join(SMD_BASE_FORMATS.keys())
        other_formatstr = ",".join(SMD_BASE_FORMATS.values())
        other_path = out_dir / f"{outfile_prefix}other.csv"
        np.savetxt(other_path, other_df.values, fmt=other_formatstr, header=other_header)


def parse_and_merge_all_SMD_files(smd_dir: PathLike) -> SMDData:
    smd_dir_path = Path(smd_dir)
    smd_paths = [*smd_dir_path.glob("*_SMD.csv"), *smd_dir_path.glob("*_SMD.CSV")]
    all_smd_data = _parse_SMD_files(smd_paths)
    return _merge_SMD_results(all_smd_data)


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] in ["-h", "--help"]:
        logger.error(
            "Usage: %s sm_data_dir sm_out_dir [outfile_prefix]",
            Path(__file__).name
        )
        sys.exit(1)
    in_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    if len(sys.argv) > 3:
        sm_prefix = sys.argv[3]
    else:
        sm_prefix = ""

    merged_smd_data = parse_and_merge_all_SMD_files(in_dir)
    write_smd_results(out_dir, merged_smd_data, sm_prefix)
