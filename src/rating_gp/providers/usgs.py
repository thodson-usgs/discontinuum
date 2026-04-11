"""Helper functions for pulling USGS data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd
import xarray as xr
from dataretrieval import waterdata
from discontinuum.providers.base import MetaData, USGSParameter
from loadest_gp.providers.usgs import get_metadata

if TYPE_CHECKING:
    from typing import Optional

    from xarray import Dataset

FT_TO_M = 0.3048
FT3_TO_M3 = 0.0283168

USGS_QUALITY_CODES = {
    'Excellent': '0.02',
    'Good': '0.05',
    'Fair': '0.08',
    'Poor': '0.12',
    'Unspecified': '0.12',
}


@dataclass
class NWISColumn:
    """Column definition for NWIS field measurement data."""
    column_name: str
    standard_name: str
    long_name: Optional[str] = None
    units: Optional[str] = None
    conversion: float = 1.0

    @property
    def name(self):
        """Alias for standard_name."""
        return self.standard_name


USGSStage = USGSParameter(
    pcode="00065",
    standard_name="stage",
    long_name="Stream stage",
    units="meters",
    conversion=FT_TO_M,
)

NWISStage = NWISColumn(
    column_name="stage",
    standard_name="stage",
    long_name="Stream stage",
    units="meters",
    conversion=FT_TO_M,
)

NWISDischarge = NWISColumn(
    column_name="discharge",
    standard_name="discharge",
    long_name="Stream discharge",
    units="cubic meters per second",
    conversion=FT3_TO_M3,
)

NWISDischargeUnc = NWISColumn(
    column_name="measurement_rated",
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
    """Get daily data from the USGS Water Data API.

    Parameters
    ----------
    site : str
        USGS site number.
    start_date : str
        Start date in the format 'yyyy-mm-dd'.
    end_date : str
        End date in the format 'yyyy-mm-dd'.

    Returns
    -------
    Dataset
        Dataset with the requested data.
    """
    param = USGSStage

    if not site.startswith("USGS-"):
        monitoring_location_id = "USGS-" + site
    else:
        monitoring_location_id = site

    time_range = f"{start_date}/{end_date}"

    df, _ = waterdata.get_daily(
        monitoring_location_id=monitoring_location_id,
        parameter_code=param.pcode,
        time=time_range,
    )

    if len(df) == 0:
        raise ValueError("No daily stage data is available for USGS site "
                         f"number: {site}")

    # waterdata returns long format with 'time' and 'value' columns
    df = df[["time", "value"]].copy()
    df = df.rename(columns={"value": param.name})
    df = df.set_index("time")

    # remove timezone for xarray compatibility
    if hasattr(df.index, "tz") and df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    ds = xr.Dataset.from_dataframe(df)

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
    """Get discharge measurements from the USGS Water Data API.

    Parameters
    ----------
    site : str
        USGS site number; e.g., '03339000'.
    start_date : str
        Start date in the format 'YYYY-MM-DD'.
    end_date : str
        End date in the format 'YYYY-MM-DD'.
    """
    if not site.startswith("USGS-"):
        monitoring_location_id = "USGS-" + site
    else:
        monitoring_location_id = site

    time_range = f"{start_date}/{end_date}"

    df, _ = waterdata.get_field_measurements(
        monitoring_location_id=monitoring_location_id,
        parameter_code="00060,00065",
        time=time_range,
    )

    return read_measurements_df(df)


def read_measurements_df(df: pd.DataFrame) -> xr.Dataset:
    """Read a DataFrame of USGS discharge measurements and convert to xarray Dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe from `waterdata.get_field_measurements()`

    Returns
    -------
    xr.Dataset

    Example
    -------
    >>> from dataretrieval import waterdata
    >>> from rating_gp.providers.usgs import read_measurements_df
    >>> df, _ = waterdata.get_field_measurements(
        monitoring_location_id='USGS-03339000',
        parameter_code='00060,00065',
        time='2020-01-01/2020-12-31',
        )
    >>> ds = read_measurements_df(df)
    """
    # The waterdata API returns a long-format DataFrame with columns:
    # time, parameter_code, value, measurement_rated, control_condition, etc.

    # Pivot from long to wide format, aligning stage and discharge by time
    pivot_values = ["value", "measurement_rated"]
    if "control_condition" in df.columns:
        pivot_values.append("control_condition")

    pivot_df = df.pivot_table(
        index="time",
        columns="parameter_code",
        values=pivot_values,
        aggfunc="first",
    )

    # Flatten MultiIndex columns: ('value', '00060') -> 'value_00060'
    pivot_df.columns = [f"{c[0]}_{c[1]}" for c in pivot_df.columns]

    # Rename columns to standard names
    rename_map = {
        "value_00065": "stage",
        "value_00060": "discharge",
        "measurement_rated_00060": "measured_rating_diff",
        "control_condition_00060": "control_type_cd",
    }
    pivot_df = pivot_df.rename(columns=rename_map)

    # Timezone handling
    if hasattr(pivot_df.index, "tz") and pivot_df.index.tz is not None:
        pivot_df.index = pivot_df.index.tz_convert("UTC").tz_localize(None)
    pivot_df.index.name = "time"

    # Ensure required columns exist
    for col in ["stage", "discharge"]:
        if col not in pivot_df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Process the control_type_cd column
    if "control_type_cd" in pivot_df.columns:
        pivot_df["control_type_cd"] = (
            pivot_df["control_type_cd"]
            .fillna("Unspecified")
            .astype("category")
        )

    # Replace other values with 'Unspecified'
    if "measured_rating_diff" in pivot_df.columns:
        pivot_df['measured_rating_diff'] = pivot_df['measured_rating_diff'].where(
            pivot_df['measured_rating_diff'].isin(USGS_QUALITY_CODES.keys()),
            'Unspecified'
        )

        pivot_df['discharge_unc_frac'] = (pivot_df['measured_rating_diff']
                                          .replace(USGS_QUALITY_CODES)
                                          .astype(float))
    else:
        pivot_df['discharge_unc_frac'] = float(USGS_QUALITY_CODES['Unspecified'])

    # Convert fractional uncertainty to uncertainty assuming the uncertainty
    # fraction is a 2 sigma gse interval. (GSE = frac + 1)
    # (GSE -> exp(sigma_ln(Q)))
    pivot_df['discharge_unc'] = pivot_df['discharge_unc_frac'] / 2 + 1

    # drop data that is <= 0 as we need all positive data
    pivot_df = pivot_df[(pivot_df['stage'] > 0) & (pivot_df['discharge'] > 0)]

    data_cols = ["stage", "discharge", "discharge_unc"]
    if "control_type_cd" in pivot_df.columns:
        data_cols.append("control_type_cd")

    ds = xr.Dataset.from_dataframe(pivot_df[data_cols])

    for param in [NWISStage, NWISDischarge]:
        ds[param.name] = ds[param.name] * param.conversion
        ds[param.name].attrs = param.__dict__

    ds['discharge_unc'].attrs = NWISDischargeUnc.__dict__

    return ds
