import logging
import sys
import textwrap
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, NamedTuple

import pandas as pd

SCRIPTNAME = Path(__file__).name

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

PathLike = str | Path


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


def _read_and_clean_smd_csv(
    smd_path: PathLike, epoch_range: EpochRange = EpochRange()
) -> pd.DataFrame:
    """Read a Smart Mooring CSV file into a Pandas DataFrame, including some basic
    cleaning.

    Parameters
    ----------
    smd_path
        Path to a Smart Mooring CSV file
    epoch_range
        Min and max values for the epoch_t (UNIX timestamp)

    Returns
    -------
    A Pandas dataframe with columns `epoch_t`, `link`, `log_type`, and `data1..data5`
    """
    # data5 allows up to two commas within an unquoted BSYS message field
    data_names = [f"data{i+1}" for i in range(5)]
    df = pd.read_csv(
        smd_path,
        names=["epoch_t", "link", "log_type", *data_names],
        dtype=str,  # fist assume is everything is a string; later convert to other types
        skiprows=1,  # don't use the included header
    )
    df = (
        # Keep rows where `epoch_t`` can be converted to `float`` and `link`` to `int`
        df[df["epoch_t"].map(_floatable) & df["link"].map(_int_able)]
        .astype({"epoch_t": float, "link": int, "log_type": str})
        .dropna(axis=0, how="any", subset=("epoch_t", "link", "log_type"))
    )
    df = df[df["epoch_t"] != 0]
    if epoch_range.min:
        df = df[df["epoch_t"] >= epoch_range.min]
    if epoch_range.max:
        df = df[df["epoch_t"] <= epoch_range.max]
    return df


@dataclass
class SMDData:
    """Dataframes for SmartMooring data returned from parse_SMD_file.

    Attributes
    ----------
    bsys
        BSYS log message DataFrames
    modules
        dict of DataFrames from SmartMooring modules. Modules are only present in the dict
        if there is data for that module.
    other_sm
        DataFrame of any leftover SmartMooring data not in the first two entries
        (e.g. MSD, BIN, and HB messages)
    filename
        Optional filename if loaded from a single SMD csv file - just for logging purposes
    """

    bsys: pd.DataFrame
    modules: dict[str, pd.DataFrame]
    other_sm: pd.DataFrame
    filename: str | None = None

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
        other_repr = (
            (f"--other--\n" f"{textwrap.indent(f'{self.other_sm}', prefix='  ')}\n")
            if not self.other_sm.empty
            else ""
        )
        return (
            "BSYS:\n"
            f"{textwrap.indent(f'{self.bsys}', prefix='  ')}\n"
            f"{all_mod_txt}\n"
            f"{other_repr}"
        )


def _split_smd_csv(smd_df: pd.DataFrame) -> SMDData:
    """Split one SmartMooring dataframe into BSYS, DATA modules, and other dataframes."""
    smd_data = SMDData.empty()
    smd_data.bsys = smd_df[smd_df["log_type"] == "BSYS"]
    data_df = smd_df[smd_df["log_type"] == "DATA"].rename(columns={"data1": "mod_type"})
    for mod_type in data_df["mod_type"].unique():
        smd_data.modules[mod_type] = data_df[data_df["mod_type"] == mod_type]
    smd_data.other_sm = smd_df[~smd_df["log_type"].isin(["BSYS", "DATA"])]
    return smd_data


def _preprocess_bsys(bsys_df: pd.DataFrame) -> pd.DataFrame:
    bsys_colnames = {
        "data1": "bridge_us",
        "data2": "log_level",
        "data3": "message",
        "data4": "msg_extra1",
        "data5": "msg_extra2",
    }
    bsys_types = {
        "bridge_us": int,
        "log_level": str,
        "message": str,
    }
    bsys_df = (
        bsys_df.rename(columns=bsys_colnames)
        .dropna(axis=0, how="any", subset=("bridge_us", "log_level", "message"))
        .fillna({"msg_extra1": "", "msg_extra2": ""})
        .astype(bsys_types)
    )
    # Concat msg_extra1 & 2 to message, then drop them
    mask1 = ~(bsys_df["msg_extra1"] == "")  # non-empty msg_extra1 row mask
    mask2 = ~(bsys_df["msg_extra2"] == "")  # non-empty msg_extra2 row mask
    bsys_df.loc[mask1, "msg_extra1"] = "," + bsys_df.loc[mask1, "msg_extra1"]
    bsys_df.loc[mask2, "msg_extra2"] = "," + bsys_df.loc[mask2, "msg_extra2"]
    bsys_df["message"] += bsys_df["msg_extra1"] + bsys_df["msg_extra2"]
    del bsys_df["msg_extra1"]
    del bsys_df["msg_extra2"]
    return bsys_df


# column names and types for Smart Mooring module messages
mod_info = {
    "SOFT2": {
        "data2": ("module_ms", int),
        "data3": ("temp_cdegC", int),
    },
    "RBRD": {
        "data2": ("module_ms", int),
        "data3": ("pressure_uBar", int),
    },
    "RBRT": {
        "data2": ("module_ms", int),
        "data3": ("pressure_uBar", int),
    },
    "RBRDT": {
        "data2": ("module_ms", int),
        "data3": ("pressure_uBar", int),
        "data4": ("temp_udegC", int),
    },
    "RBRU": {
        "data2": ("module_ms", int),
    },
}


def _preprocess_data_df(data_df: pd.DataFrame, mod_type: str):
    """Convert a dataframe of DATA messages into a dataframe with just messages from a
    particular Smart Mooring module, with the proper fields."""
    colnames = {k: v[0] for k, v in mod_info[mod_type].items()}
    coltypes = {v[0]: v[1] for v in mod_info[mod_type].values()}
    dropped_columns = [
        colname
        for colname in (f"data{i + 1}" for i in range(5))
        if ((colname in data_df.columns) and (colname not in colnames))
    ]
    return (
        data_df.rename(columns=colnames)
        .drop(labels=dropped_columns, axis=1)
        .dropna(axis=0, how="any", subset=[*colnames.values()])
        .astype(coltypes)
    )


def _preprocess_smd_file(smd_path: PathLike, epoch_range: EpochRange) -> SMDData:
    """Convert a *SMD.csv file into a structure of Pandas dataframes

    Parameters
    ----------
    smd_path
        Path to a Smart Mooring CSV file
    epoch_range
        Min and max values for the epoch_t (UNIX timestamp)

    Returns
    -------
    SMDData
        A structure with the BSYS data as well as Smart Mooring individual module data
    """
    smd_df = _read_and_clean_smd_csv(smd_path, epoch_range=epoch_range)
    smd_data = _split_smd_csv(smd_df)
    smd_data.filename = Path(smd_path).name
    smd_data.bsys = _preprocess_bsys(smd_data.bsys)
    bad_mod_types = []
    for mod_type in smd_data.modules:
        if mod_type in mod_info:
            smd_data.modules[mod_type] = _preprocess_data_df(
                smd_data.modules[mod_type], mod_type
            )
        else:
            bad_mod_types.append(mod_type)
    for bmt in bad_mod_types:
        del smd_data.modules[bmt]
    return smd_data


class SMDMerger:
    """A class to wrap up the merging of a bunch of Smart Mooring data files"""

    def __init__(self, smd_paths: Iterable[Path], epoch_range: EpochRange) -> None:
        self._smd_paths = smd_paths
        self._merged_smd_data = SMDData.empty()
        self._epoch_range = epoch_range

    def _get_smd(self, smd_path: Path) -> SMDData | None:
        logger.info(
            "Parsing %s: %.2f MiB",
            smd_path.relative_to(smd_path.parent.parent),
            smd_path.stat().st_size / 1e6,
        )
        try:
            return _preprocess_smd_file(smd_path, epoch_range=self._epoch_range)
        except Exception as exc:
            logger.error("ERROR with %s: %s", smd_path, exc)

    def _merge_smd(self, smd_data: SMDData | None):
        if not smd_data:
            return
        logger.info(f"Merging {smd_data.filename}")
        merged = self._merged_smd_data
        merged.bsys = pd.concat([merged.bsys, smd_data.bsys], ignore_index=True)
        merged.other_sm = pd.concat(
            [merged.other_sm, smd_data.other_sm], ignore_index=True
        )
        for mod_type, mod_data in smd_data.modules.items():
            merged.modules[mod_type] = pd.concat(
                [merged.modules[mod_type], mod_data], ignore_index=True
            )

    def _sort_merged_data(self):
        logger.info("Sorting merged data.")
        merged = self._merged_smd_data
        if not merged.bsys.empty:
            merged.bsys = merged.bsys.sort_values("epoch_t", ignore_index=True)
        if not merged.other_sm.empty:
            merged.other_sm = merged.other_sm.sort_values("epoch_t", ignore_index=True)
        for mod_type, mod_data in merged.modules.items():
            merged.modules[mod_type] = mod_data.sort_values("epoch_t", ignore_index=True)

    def _merge_mp(self):
        with ProcessPoolExecutor() as pool:
            for smd_data in pool.map(self._get_smd, self._smd_paths):
                self._merge_smd(smd_data)

    def _merge_serial(self):
        for smd_path in self._smd_paths:
            smd_data = self._get_smd(smd_path)
            self._merge_smd(smd_data)

    def run(self, parallel=True) -> SMDData:
        if parallel:
            self._merge_mp()
        else:
            self._merge_serial()
        self._sort_merged_data()
        return self._merged_smd_data


def write_merged_smd_data(outdir: Path, merged_smd_data: SMDData):
    logger.info("Writing merged smartmooring CSVs to %s", outdir)
    if not outdir.is_dir():
        outdir.mkdir(parents=True, exist_ok=True)
    to_csv_kwargs = {"float_format": "%.2f", "index": False}
    if not merged_smd_data.bsys.empty:
        bsys_fname = outdir / "BSYS.csv"
        logger.debug("Writing %s", bsys_fname)
        merged_smd_data.bsys.to_csv(bsys_fname, **to_csv_kwargs)
    if not merged_smd_data.other_sm.empty:
        other_sm_fname = outdir / "other_sm.csv"
        logger.debug("Writing %s", other_sm_fname)
        merged_smd_data.other_sm.to_csv(other_sm_fname, **to_csv_kwargs)
    for mod_name, mod_data in merged_smd_data.modules.items():
        mod_fname = outdir / f"{mod_name.upper()}.csv"
        logger.debug("Writing %s", mod_fname)
        mod_data.to_csv(mod_fname, **to_csv_kwargs)


def _cli_err(logger_msg, *logger_args, code: int = 1):
    """Log an error message and exit."""
    logger.error(logger_msg, *logger_args)
    sys.exit(code)


def _usage_err():
    _cli_err(
        (
            "Usage:\n"
            " %s sm_data_dir [-o sm_out_dir] [-n min_datetime] [-x max_datetime]\n"
            "\n"
            "min_datetime and/or max_datetime can be any date/time string that can be\n"
            "interpreted by pandas.Timestamp, like 2024-04-01 or 2023-10-25T00:23:43Z\n"
            "Naive values are assumed to be UTC."
        ),
        SCRIPTNAME,
    )


def _cli_option(flag: str, argv: list[str]) -> str | None:
    if flag in argv:
        flag_idx = argv.index(flag)
        if flag_idx + 1 >= len(argv):
            _usage_err()
        return argv[flag_idx + 1]
    return None


def cli():
    if len(sys.argv) < 2 or sys.argv[1] in ["-h", "--help"]:
        _usage_err()
    spotter_dir = Path(sys.argv[1])
    out_dir = spotter_dir / "smartmooring"
    min_epoch_t = None
    max_epoch_t = None
    if opt := _cli_option("-o", sys.argv):
        out_dir = Path(opt)
    if opt := _cli_option("-n", sys.argv):
        min_epoch_t = pd.Timestamp(opt).timestamp()
    if opt := _cli_option("-x", sys.argv):
        max_epoch_t = pd.Timestamp(opt).timestamp()
    epoch_range = EpochRange(min_epoch_t, max_epoch_t)
    smd_paths = [*spotter_dir.glob("*_SMD.csv"), *spotter_dir.glob("*_SMD.CSV")]
    merged_smd_data = SMDMerger(smd_paths, epoch_range).run(parallel=True)
    write_merged_smd_data(out_dir, merged_smd_data)


if __name__ == "__main__":
    # basedir = Path(
    #     "/nfs/depot/cce_u1/haller/shared/FIELD_DATA/USACE/2023-2024/SD_card_data"
    # )
    # # smd_path = "/nfs/depot/cce_u1/haller/shared/FIELD_DATA/USACE/2023-2024/SD_card_data/S3_SPOT-30035R/0022_SMD.csv"
    # # smd_path = "/nfs/depot/cce_u1/haller/shared/FIELD_DATA/USACE/2023-2024/SD_card_data/S1_SPOT-1132/0319_SMD.CSV"
    # # smd_path = "/nfs/depot/cce_u1/haller/shared/FIELD_DATA/USACE/2023-2024/SD_card_data/S4_SPOT-30034R/1039_SMD.csv"  # 46M
    # # smd_path = "/nfs/depot/cce_u1/haller/shared/FIELD_DATA/USACE/2023-2024/SD_card_data/S4_SPOT-30034R/11184_SMD.csv"  # 103M
    # # smd_data = _preprocess_smd_file(smd_path)

    # # spotter_dir = basedir / "S1_SPOT-1132"
    # # spotter_dir = basedir / "S2_SPOT-1081"
    # # spotter_dir = basedir / "S3_SPOT-30035R"
    # date_range = ("2023-10-01T00:00:00Z", "2024-04-01T00:00:00Z")
    # epoch_range = (
    #     pd.Timestamp(date_range[0]).timestamp(),
    #     pd.Timestamp(date_range[1]).timestamp(),
    # )
    # spotter_dir = basedir / "S4_SPOT-30034R"
    # smd_paths = [*spotter_dir.glob("*_SMD.csv"), *spotter_dir.glob("*_SMD.CSV")]
    # merged_smd_data = SMDMerger(smd_paths, epoch_range=epoch_range).run(parallel=True)
    # write_merged_smd_data(Path("smartmooring"), merged_smd_data)
    try:
        cli()
    except Exception:
        logger.exception("There was an error running %s", SCRIPTNAME)
