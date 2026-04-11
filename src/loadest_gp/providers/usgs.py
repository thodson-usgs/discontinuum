"""Helper functions for pulling USGS data."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd
import xarray as xr
from dataretrieval import waterdata
from discontinuum.providers.base import MetaData, USGSParameter

if TYPE_CHECKING:
    from typing import Dict, List, Optional, Union

    from xarray import Dataset
    from pandas import DataFrame

CFS_TO_M3 = 0.0283168


USGSFlow = USGSParameter(
    pcode="00060",
    standard_name="flow",
    long_name="Streamflow",
    units="cubic meters per second",
    conversion=0.0283168,
)


def get_parameters(
        pcodes: Dict[str, str],
        ) -> Union[USGSParameter, List[USGSParameter]]:
    """Get USGS parameters from a list of parameter codes.

    Parameters
    ----------
    pcodes : Dict[str, str]
        Mapping of standard names to USGS parameter codes.

    Returns
    -------
    List[USGSParameter]
        List of USGS parameters.
    """
    lookup_name = {value: key for key, value in pcodes.items()}

    params = []
    for pcode, name in lookup_name.items():
        params.append(
            USGSParameter(
                pcode=pcode,
                standard_name=name,
            )
        )

    if len(params) == 1:
        params = params[0]

    return params


def get_metadata(site: str) -> MetaData:
    """Get site metadata from the USGS Water Data API.

    Parameters
    ----------
    site : str
        USGS site number (e.g., '03339000' or 'USGS-03339000').

    Returns
    -------
    MetaData
        Dataclass with the site information.
    """
    if not site.startswith("USGS-"):
        site = "USGS-" + site

    df, _ = waterdata.get_monitoring_locations(
        monitoring_location_id=[site],
    )
    row = df.iloc[0]

    # geometry is [lon, lat] when geopandas is not installed
    geometry = row["geometry"]
    if isinstance(geometry, (list, tuple)):
        longitude, latitude = geometry[0], geometry[1]
    elif hasattr(geometry, "x") and hasattr(geometry, "y"):
        longitude, latitude = geometry.x, geometry.y
    else:
        longitude, latitude = None, None

    site_no = row.get("monitoring_location_number", site)

    return MetaData(
        id=site_no,
        name=row["monitoring_location_name"],
        latitude=latitude,
        longitude=longitude,
    )


# TODO pass a dict not a list of params
def get_daily(
    site: str,
    start_date: str,
    end_date: str,
    params: Union[List[USGSParameter], USGSParameter] = USGSFlow,
    **kwargs,
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
    params : List[USGSParameter], optional
        List of parameters to retrieve. The default is flow only `[USGSFlow]`.
    kwargs : Dict
        Additional keyword arguments to pass to the API.

    Returns
    -------
    Dataset
        Dataset with the requested data.
    """
    if not isinstance(params, list):
        params = [params]

    pcodes = [param.pcode for param in params]

    if not site.startswith("USGS-"):
        monitoring_location_id = "USGS-" + site
    else:
        monitoring_location_id = site

    time_range = f"{start_date}/{end_date}"

    df, _ = waterdata.get_daily(
        monitoring_location_id=monitoring_location_id,
        parameter_code=pcodes,
        time=time_range,
        **kwargs,
    )

    return format_daily(df, site, params)


def format_daily(
        df: DataFrame,
        site_id: Optional[str] = None,
        params: Union[List[USGSParameter], USGSParameter] = None,
) -> Dataset:
    """
    Format results of waterdata.get_daily.

    Parameters
    ----------
    df : DataFrame
        Dataframe returned from waterdata.get_daily.
    site_id : str, optional
        USGS site number for populating metadata. The default is None.
    params : List[USGSParameter], optional
        List of parameters to retrieve. The default is flow only `[USGSFlow]`.
    """
    if params is None:
        params = [USGSFlow]
    elif not isinstance(params, list):
        params = [params]

    # The waterdata API returns a long-format DataFrame with columns:
    # time, value, parameter_code, etc.
    # Pivot to wide format with one column per parameter
    pcode_to_param = {param.pcode: param for param in params}

    # Filter to requested parameter codes
    df = df[df["parameter_code"].isin(pcode_to_param.keys())].copy()

    # Pivot from long to wide format
    pivot_df = df.pivot_table(
        index="time",
        columns="parameter_code",
        values="value",
        aggfunc="first",
    )

    # Rename columns from pcode to standard name
    pivot_df = pivot_df.rename(
        columns={pcode: pcode_to_param[pcode].name for pcode in pivot_df.columns}
    )

    # Ensure time index is timezone-naive for xarray compatibility
    if hasattr(pivot_df.index, "tz") and pivot_df.index.tz is not None:
        pivot_df.index = pivot_df.index.tz_localize(None)

    ds = xr.Dataset.from_dataframe(pivot_df)

    # set metadata
    if site_id:
        ds.attrs = get_metadata(site_id).__dict__

    for param in params:
        if param.name in ds:
            ds[param.name] = ds[param.name] * param.conversion
            # xarray metadata assignment must come after all other operations
            ds[param.name].attrs = param.__dict__

    return ds


def format_wqp_samples(
        df: DataFrame,
        name: str = "concentration",
        pcode: Optional[str] = None,
) -> Dataset:
    """Format results of waterdata.get_samples.

    Parameters
    ----------
    df : DataFrame
        Dataframe returned from waterdata.get_samples.
    name : str
        Short name for the parameter. Default is 'concentration'.
    pcode : str
        USGS parameter code for populating metadata.

    Returns
    -------
    Dataset
    """
    # create datetime index
    df.index = pd.to_datetime(
        df["Activity_StartDate"] + " " + df["Activity_StartTime"]
    )

    df[name] = df["Result_Measure"].astype(float)
    df.index.name = "time"
    df.index = df.index.tz_localize(None)

    # create xarray dataset and remove unnecessary columns
    # TODO include censoring flag
    ds = xr.Dataset.from_dataframe(df[[name]])

    # drop censored values and warn user
    if any(ds[name].isnull()):
        ds = ds.dropna(dim="time")
        warnings.warn(
            "Censored values have been removed from the dataset.",
            stacklevel=1,
            )

    if pcode:
        attrs = get_parameters({name: pcode})
        ds[name].attrs = attrs.__dict__

    return ds


def get_samples(
        site: str,
        start_date: str,
        end_date: str,
        characteristic: str,
        fraction: str,
        provider: str = "NWIS",
        name: str = "concentration",
        filter_pcodes: Optional[List[str]] = None,
        **kwargs,
) -> Dataset:
    """Get sample data from the Water Quality Portal API.

    Parameters
    ----------
    site : str
        Water Quality Portal site id; e.g., 'USGS-12345678'.
    start_date : str
        Start date in the format 'YYYY-MM-DD'.
    end_date : str
        End date in the format 'YYYY-MM-DD'.
    characteristic : str
        The name of the parameter to retrieve.
    fraction : str
        The fraction of the parameter to retrieve. Options are
        'Filtered field and/or lab', 'Unfiltered', etc.
    provider : str
        The data provider. Options are 'NWIS' or 'STORET'.
    name : str
        Short name for the parameter. Default is 'concentration'.
    filter_pcodes : List[str]
        List of parameter codes to filter the dataset.
    kwargs : Dict
        Additional keyword arguments to pass to the WQP API.
    """
    if provider not in ["NWIS", "STORET"]:
        raise ValueError("Provider must be 'NWIS' or 'STORET'")

    if provider == "NWIS" and not site.startswith("USGS-"):
        site = "USGS-" + site

    df, _ = waterdata.get_samples(
        monitoringLocationIdentifier=site,
        characteristic=characteristic,
        activityStartDateLower=start_date,
        activityStartDateUpper=end_date,
        **kwargs,
    )

    # filter by fraction
    if fraction:
        df = df[df["Result_SampleFraction"] == fraction]

    if provider == "NWIS":
        if filter_pcodes:
            # filter df by list of pcodes
            filter_pcodes_int = [int(p) for p in filter_pcodes]
            df = df[df["USGSpcode"].isin(filter_pcodes_int)]
            pcode = filter_pcodes[0]

        elif len(set(df["USGSpcode"])) != 1:
            raise ValueError("Multiple parameters returned from NWIS.")

        else:
            # only one pcode returned, so take the first entry
            pcode = df["USGSpcode"].iloc[0]

        pcode = str(int(pcode)).zfill(5)

    else:
        pcode = None

    ds = format_wqp_samples(df, name, pcode)

    if provider == "NWIS":
        # strip the "USGS-" prefix from the site number
        site_id = site[5:] if site.startswith("USGS-") else site
        ds.attrs = get_metadata(site_id).__dict__

    if provider == "STORET":
        # TODO check for "USGS-" prefix, then get metadata
        pass

    return ds
