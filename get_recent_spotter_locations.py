import time
from pathlib import Path
from textwrap import indent
from typing import Dict, NamedTuple

import pandas as pd

EAST_BOUND = -124.068905

data_basedir = Path(
    "/nfs/depot/cce_u1/haller/shared/FIELD_DATA/USACE/2023-2024/SD_card_data"
)

spotters = {
    "S1": "S1_SPOT-1132",
    "S2": "S2_SPOT-1081",
    "S3": "S3_SPOT-30035R",
    "S4": "S4_SPOT-30034R",
}


class SMData(NamedTuple):
    BSYS: pd.DataFrame
    SOFT2: pd.DataFrame
    RBRD: pd.DataFrame
    RBRT: pd.DataFrame
    RBRDT: pd.DataFrame
    RBRU: pd.DataFrame

    def __repr__(self):
        field_reprs = {
            fieldname: indent(repr(getattr(self, fieldname)), "\t")
            for fieldname in self._fields
        }
        return "\n".join(
            f"{fieldname}:\n{field_reprs[fieldname]}" for fieldname in self._fields
        )


def get_sm_df(sm_path: Path, epochs_to_datetimes=False):
    tstart = time.time()
    sm_df = pd.read_csv(
        sm_path,
        names=["epoch_t", "node", "log_type", "data1", "data2", "data3", "data4"],
        skiprows=1,
    ).set_index("epoch_t")
    if epochs_to_datetimes:
        sm_df.index = pd.to_datetime(sm_df.index, unit="s").rename("timestamp")
    print(f"--Reading took {time.time() - tstart:.1f} seconds")
    return sm_df


def get_rbr_data_from_sm_df(sm_df: pd.DataFrame) -> SMData:
    tstart = time.time()
    # BSYS messages
    bsys_df = (
        sm_df.loc[sm_df["log_type"] == "BSYS"]
        .drop(columns=["node", "log_type", "data4"])
        .rename(columns={"data1": "bridge_us", "data2": "log_level", "data3": "message"})
    )
    # DATA messages
    data_df = (
        sm_df.loc[sm_df["log_type"] == "DATA"]
        .drop(columns=["log_type"])
        .rename(columns={"data1": "sensor"})
    )
    # DATA, SOFT2 messages
    soft2_df = (
        data_df.loc[data_df["sensor"] == "SOFT2"]
        .drop(columns=["sensor", "data4"])
        .rename(columns={"data2": "soft_ms", "data3": "temp_cdegC"})
    )
    # DATA, RBR* messages
    rbr_df = data_df.loc[data_df["sensor"].str[:3] == "RBR"].rename(
        columns={"data2": "coda_ms"}
    )
    rbrd_df = (
        rbr_df.loc[rbr_df["sensor"] == "RBRD"]
        .drop(columns=["sensor", "data4"])
        .rename(columns={"data3": "pressure_uBar"})
    )
    rbrt_df = (
        rbr_df.loc[rbr_df["sensor"] == "RBRT"]
        .drop(columns=["sensor", "data4"])
        .rename(columns={"data3": "temp_udegC"})
    )
    rbrdt_df = (
        rbr_df.loc[rbr_df["sensor"] == "RBRDT"]
        .drop(columns=["sensor"])
        .rename(columns={"data3": "pressure_uBar", "data4": "temp_udegC"})
    )
    rbru_df = rbr_df.loc[rbr_df["sensor"] == "RBRU"].drop(
        columns=["sensor", "data3", "data4"]
    )
    print(f"--Splitting DFs took {time.time() - tstart:.1f} seconds")
    return SMData(bsys_df, soft2_df, rbrd_df, rbrt_df, rbrdt_df, rbru_df)


def get_loc_data(loc_csv_path: Path) -> pd.DataFrame:
    dt_columns = ["year", "month", "day", "hour", "minute", "second", "ms"]
    df = pd.read_csv(loc_csv_path).rename(
        columns={"# year": "year", "min": "minute", "sec": "second", "msec": "ms"}
    )
    df["timestamp"] = pd.to_datetime(pd.DataFrame(df, columns=dt_columns))
    df = df.set_index(["timestamp"]).drop(columns=dt_columns)
    return df


def get_all_sm_data() -> Dict[str, SMData]:
    tstart = time.time()
    sm_data: Dict[str, SMData] = {}
    for spotter_id in spotters:
        sm_path = data_basedir / spotters[spotter_id] / "parsed" / "smartmooring_data.csv"
        print(
            f"smartmooring_data.csv for {spotter_id} is"
            f" {sm_path.stat().st_size/2**20:.1f} MB"
        )
        sm_data[spotter_id] = get_rbr_data_from_sm_df(get_sm_df(sm_path, True))
    print(f"Overall time to get smartmooring data: {time.time() - tstart:.1f} seconds")
    return sm_data


def get_all_loc_data() -> Dict[str, pd.DataFrame]:
    tstart = time.time()
    loc_data: Dict[str, pd.DataFrame] = {}
    for spotter_id in spotters:
        loc_path = data_basedir / spotters[spotter_id] / "parsed" / "location.csv"
        print(
            f"location.csv for {spotter_id} is" f" {loc_path.stat().st_size/2**20:.1f} MB"
        )
        loc_data[spotter_id] = get_loc_data(loc_path)
    print(f"Overall time to get location data: {time.time() - tstart:.1f} seconds")
    return loc_data


if __name__ == "__main__":
    sm_data = get_all_sm_data()
    loc_data = get_all_loc_data()
