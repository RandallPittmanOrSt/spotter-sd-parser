import logging
import textwrap
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Union

import pandas as pd
from typing_extensions import TypeAlias

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

PathLike = Union[Path, str]
EpochRange: TypeAlias = tuple[float, float]


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
    smd_path: PathLike, epoch_range: Optional[EpochRange]
) -> pd.DataFrame:
    """Read a Smart Mooring CSV file into a Pandas DataFrame, including some basic
    cleaning."""
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
    if epoch_range:
        df = df[(df["epoch_t"] >= epoch_range[0]) & (df["epoch_t"] <= epoch_range[1])]
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
    """

    bsys: pd.DataFrame
    modules: dict[str, pd.DataFrame]
    other_sm: pd.DataFrame
    filename: Optional[str] = None

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
    colnames = {k: v[0] for k, v in mod_info[mod_type].items()}
    coltypes = {v[0]: v[1] for v in mod_info[mod_type].values()}
    dropped_columns = [
        colname
        for colname in (f"data{i + 1}" for i in range(3, 5))
        if colname not in colnames
    ]

    return (
        data_df.rename(columns=colnames)
        .drop(labels=dropped_columns, axis=1)
        .dropna(axis=0, how="any", subset=[*colnames.values()])
        .astype(coltypes)
    )


def _preprocess_smd_file(
    smd_path: PathLike, epoch_range: Optional[EpochRange] = None
) -> SMDData:
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
    def __init__(
        self, smd_paths: Iterable[Path], epoch_range: Optional[EpochRange] = None
    ) -> None:
        self._smd_paths = smd_paths
        self._merged_smd_data = SMDData.empty()
        self._epoch_range = epoch_range

    def _get_smd(self, smd_path: Path) -> Optional[SMDData]:
        logger.info(
            "Parsing %s: %.2f MiB",
            smd_path.relative_to(smd_path.parent.parent),
            smd_path.stat().st_size / 1e6,
        )
        try:
            return _preprocess_smd_file(smd_path, epoch_range=self._epoch_range)
        except Exception as exc:
            logger.error("ERROR with %s: %s", smd_path, exc)

    def _merge_smd(self, smd_data: Optional[SMDData]):
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
        merged.bsys = merged.bsys.sort_values("epoch_t", ignore_index=True)
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


if __name__ == "__main__":
    basedir = Path(
        "/nfs/depot/cce_u1/haller/shared/FIELD_DATA/USACE/2023-2024/SD_card_data"
    )
    # smd_path = "/nfs/depot/cce_u1/haller/shared/FIELD_DATA/USACE/2023-2024/SD_card_data/S3_SPOT-30035R/0022_SMD.csv"
    # smd_path = "/nfs/depot/cce_u1/haller/shared/FIELD_DATA/USACE/2023-2024/SD_card_data/S1_SPOT-1132/0319_SMD.CSV"
    # smd_path = "/nfs/depot/cce_u1/haller/shared/FIELD_DATA/USACE/2023-2024/SD_card_data/S4_SPOT-30034R/1039_SMD.csv"  # 46M
    # smd_path = "/nfs/depot/cce_u1/haller/shared/FIELD_DATA/USACE/2023-2024/SD_card_data/S4_SPOT-30034R/11184_SMD.csv"  # 103M
    # smd_data = _preprocess_smd_file(smd_path)

    # spotter_dir = basedir / "S1_SPOT-1132"
    # spotter_dir = basedir / "S2_SPOT-1081"
    # spotter_dir = basedir / "S3_SPOT-30035R"
    date_range = ("2023-10-01T00:00:00Z", "2024-04-01T00:00:00Z")
    epoch_range = (
        pd.Timestamp(date_range[0]).timestamp(),
        pd.Timestamp(date_range[1]).timestamp(),
    )
    spotter_dir = basedir / "S4_SPOT-30034R"
    smd_paths = [*spotter_dir.glob("*_SMD.csv"), *spotter_dir.glob("*_SMD.CSV")]
    merged_smd_data = SMDMerger(smd_paths, epoch_range=epoch_range).run(parallel=True)
