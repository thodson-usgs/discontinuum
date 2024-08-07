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
    long_name: Optional[str] = None
    units: Optional[str] = None
    suffix: Optional[str] = None
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
    df, _ = nwis.get_discharge_measurements(
        sites=site,
        start=start_date,
        end=end_date,
    )
    # covert timezone to UTC? ignore for now
    df.index = pd.to_datetime(
        df["measurement_dt"],
        format="ISO8601",
    )
    df.index.name = "time"
    # df.index = df.index.tz_localize(None)
    # TODO set metadata and apply unit conversions
    # TODO OR use NWISColumn to rename and set metadata
    df = df.rename(
        columns={
            "gage_height_va": "stage",
            "discharge_va": "discharge",
            }
        )
    # parse uncertainty from measured "measured_rating_diff"
    ds = xr.Dataset.from_dataframe(df[["stage", "discharge"]])

    for param in [NWISStage, NWISDischarge]:
        ds[param.name] = ds[param.name] * param.conversion
        ds[param.name].attrs = param.__dict__

    return ds
