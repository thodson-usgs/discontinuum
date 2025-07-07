"""Helper functions for pulling USGS data."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd
import xarray as xr
from dataretrieval import nwis, wqp
from discontinuum.providers.base import MetaData
from loadest_gp.providers.usgs import get_metadata

if TYPE_CHECKING:
    # from pandas import DataFrame
    from typing import Dict, List, Optional, Union

    from xarray import Dataset

FT_TO_M = 0.3048
FT3_TO_M3 = 0.0283168

    # Quantitative values of "measured_rating_diff"
USGS_QUALITY_CODES = {
    'Excellent': '0.02',
    'Good': '0.05',
    'Fair': '0.08',
    'Poor': '0.12',
    'Unspecified': '0.12',
}


@dataclass
class NWISColumn:
    column_name: str
    standard_name: str
    long_name: [Optional[str]] = None
    units: [Optional[str]] = None
    conversion: float = 1.0

    @property
    def name(self):
        """
        Alias for standard_name.
        """
        return self.standard_name


@dataclass
class USGSParameter:
    pcode: str
    standard_name: str
    long_name: Optional[str] = ""
    units: Optional[str] = ""
    suffix: Optional[str] = ""
    conversion: Optional[float] = 1.0

    @property
    def ppcode(self):
        """
        Return the parameter code with a 'p' prefix, which is used by the QWData service.

        """
        return "p" + self.pcode

    @property
    def name(self):
        """
        Alias for standard_name.
        """
        return self.standard_name


USGSStage = USGSParameter(
    pcode="00065",
    standard_name="stage",
    long_name="Stream stage",
    units="meters",
    suffix="_Mean",
    conversion=FT_TO_M,
)

NWISStage = NWISColumn(
    column_name="gage_height_va",
    standard_name="stage",
    long_name="Stream stage",
    units="meters",
    conversion=FT_TO_M,
)

NWISDischarge = NWISColumn(
    column_name="discharge_va",
    standard_name="discharge",
    long_name="Stream discharge",
    units="cubic meters per second",
    conversion=FT3_TO_M3,
)

NWISDischargeUnc = NWISColumn(
    column_name="measured_rating_diff",
    standard_name="discharge_unc",
    long_name=("Stream discharge uncertainty estimated from qualitative "
               "measurement rating codes"),
    units="cubic meters per second",
    conversion=FT3_TO_M3,
)


def get_daily_stage(
    site: str,
    start_date: str,
    end_date: str,
) -> Dataset:
    """Get daily data from the USGS NWIS API.

    Parameters
    ----------
    site : str
        USGS site number.
    start_date : str
        Start date in the format 'yyyy-mm-dd'.
    end_date : str
        End date in the format 'yyyy-mm-dd'.
    params : List[USGSParameter], optional
        List of parameters to retrieve. The default is flow only `[USGSFlow]`.

    Returns
    -------
    Dataset
        Dataset with the requested data.
    """
    param = USGSStage
    df, _ = nwis.get_dv(
        sites=site,
        start=start_date,
        end=end_date,
        parameterCd=param.pcode,
        )

    if len(df) == 0:
        raise ValueError("No daily stage data is available for USGS site "
                         f"number: {site}")

    # rename columns
    df = df.rename(columns={param.pcode + param.suffix: param.name})
    # drop columns
    df = df[[param.name]]
    # remove timezone for xarray compatibility
    df.index = df.index.tz_localize(None)

    ds = xr.Dataset.from_dataframe(df)
    # rename "datetime" to "time", which is xarray convention
    ds = ds.rename({"datetime": "time"})

    # set metadata
    ds.attrs = get_metadata(site).__dict__
    # convert units
    ds[param.name] = ds[param.name] * param.conversion
    # xarray metadata assignment must come after all other operations
    ds[param.name].attrs = param.__dict__

    return ds


def get_measurements(
        site: str,
        start_date: str,
        end_date: str,
):
    """Get discharge measurements from the USGS NWIS API.

    Parameters
    ----------
    site : str
        Water Quality Portal site id; e.g., 'USGS-12345678'.
    start_date : str
        Start date in the format 'YYYY-MM-DD'.
    end_date : str
        End date in the format 'YYYY-MM-DD'.
    """
    # df, _ = nwis.get_discharge_measurements(
    #     sites=site,
    #     start=start_date,
    #     end=end_date,
    #     format="rdb_expanded",
    # )
    # Need this till get_discharge_measurements update is uploaded
    response = nwis.query_waterdata(
        'measurements', ssl_check=True, format="rdb_expanded",
        site_no=site, begin_date=start_date, end_date=end_date,
    )
    df = nwis._read_rdb(response.text)

    return read_measurements_df(df)    


def read_measurements_df(df: pd.DataFrame) -> xr.Dataset:
    """Read a DataFrame of USGS discharge measurements and convert to xarray Dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe from `dataretrieval.nwis.get_discharge_measurements()`

    Returns
    -------
    xr.Dataset
    
    Example
    -------
    >>> from dataretrieval import nwis
    >>> from rating_gp.providers.usgs import read_measurements_df
    >>> df, _ = nwis.get_discharge_measurements(
        sites='03339000',
        start='2020-01-01',
        end='2020-12-31',
        format='rdb_expanded',
        )
    >>> ds = read_measurements_df(df)
    """

    # assert the correct columns are present
    required_columns = [
        "measurement_dt",
        "gage_height_va",
        "discharge_va",
        "q_meas_used_fg",
        "control_type_cd",
        "measured_rating_diff",
        "streamflow_method",
        NWISStage.column_name,
        NWISDischarge.column_name,
    ]

    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns in the DataFrame: {missing_columns}"
        )

    # covert timezone to UTC? ignore for now
    df.index = pd.to_datetime(
        df["measurement_dt"],
        format="ISO8601",
    )
    df.index.name = "time"
    # df.index = df.index.tz_localize(None)
    df = df.rename(
        columns={
            NWISStage.column_name: NWISStage.standard_name,
            NWISDischarge.column_name: NWISDischarge.standard_name,
            }
        )
    
    # Process the control_type_cd column
    df["control_type_cd"] = (
        df["control_type_cd"]
        .fillna("Unspecified")
        .astype("category")
        )
    
    # Filter any measurements that are not used in the rating
    mask = df["q_meas_used_fg"].str.lower().isin(['yes', 'y'])
    df = df[mask]

    num_not_used = (~mask).sum()
    if num_not_used > 0:
        warnings.warn(
            f"{num_not_used} measurements were not used in the rating and "
            "will be dropped from the dataset.",
            UserWarning,
        )

    # Replace other values with 'Unspecified'
    df['measured_rating_diff'] = df['measured_rating_diff'].where(
        df['measured_rating_diff'].isin(USGS_QUALITY_CODES.keys()),
        'Unspecified'
    )

    df['discharge_unc_frac'] = (df['measured_rating_diff']
                                .replace(USGS_QUALITY_CODES)
                                .astype(float))

    # Set indirect measurements as 20% uncertain regardless of quality code
    df.loc[df['streamflow_method'] == 'QIDIR', 'discharge_unc_frac'] = 0.2

    # Convert fractional uncertainty to uncertainty assuming the uncertainty
    # fraction is a 2 sigma gse interval. (GSE = frac + 1)
    # (GSE -> exp(sigma_ln(Q)))
    df['discharge_unc'] = df['discharge_unc_frac'] / 2 + 1

    # drop data that is <= 0 as we need all positive data
    df = df[(df['stage'] > 0) & (df['discharge'] > 0)]

    ds = xr.Dataset.from_dataframe(
        df[[
            "stage",
            "discharge",
            "discharge_unc",
            "control_type_cd",
        ]]
        )

    for param in [NWISStage, NWISDischarge]:
        ds[param.name] = ds[param.name] * param.conversion
        ds[param.name].attrs = param.__dict__

    ds['discharge_unc'].attrs = NWISDischargeUnc.__dict__

    return ds
