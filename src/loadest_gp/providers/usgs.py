"""Helper functions for pulling USGS data."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd
import xarray as xr
from dataretrieval import nwis, wqp
from discontinuum.providers.base import MetaData

if TYPE_CHECKING:
    # from pandas import DataFrame
    from typing import Dict, List, Optional, Union

    from xarray import Dataset

CFS_TO_M3 = 0.0283168


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


USGSFlow = USGSParameter(
    pcode="00060",
    standard_name="flow",
    long_name="Streamflow",
    units="cubic meters per second",
    suffix="_Mean",
    conversion=0.0283168,
)


# better to use a dictionary here where the key becomes the name of the parameter
def get_parameters(pcodes: Dict[str, str]) -> Union[USGSParameter, List[USGSParameter]]:
    """Get USGS parameters from a list of parameter codes.

    Parameters
    ----------
    pcodes : List[str]
        List of USGS parameter codes.

    Returns
    -------
    List[USGSParameter]
        List of USGS parameters.
    """
    lookup_name = {value: key for key, value in pcodes.items()}
    pcode_list = [v for _, v in pcodes.items()]

    df, _ = nwis.get_pmcodes(pcode_list)

    # convert to USGSParameter
    params = []
    for _, row in df.iterrows():
        params.append(
            USGSParameter(
                pcode=row["parameter_cd"],
                standard_name=lookup_name[row["parameter_cd"]],
                long_name=row["SRSName"],
                units=row["parm_unit"],
            )
        )

    if len(params) == 1:
        params = params[0]

    return params


def get_metadata(site: str) -> MetaData:
    """Get site metadata from the USGS NWIS API.

    Parameters
    ----------
    site : str
        USGS site number.

    Returns
    -------
    pandas.DataFrame
        Dataframe with the site information.
    """
    df, _ = nwis.get_info(sites=site)
    row = df.iloc[0]

    return MetaData(
        id=row["site_no"],
        name=row["station_nm"],
        latitude=row["dec_lat_va"],
        longitude=row["dec_long_va"],
    )


# TODO pass a dict not a list of params
def get_daily(
    site: str,
    start_date: str,
    end_date: str,
    params: Union[List[USGSParameter], USGSParameter] = USGSFlow,
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
    if not isinstance(params, list):
        params = [params]

    pcodes = [param.pcode for param in params]

    df, _ = nwis.get_dv(sites=site, start=start_date, end=end_date, parameterCd=pcodes)

    # rename columns
    df = df.rename(columns={param.pcode + param.suffix: param.name for param in params})
    # drop columns
    df = df[[param.name for param in params]]
    # remove timezone for xarray compatibility
    df.index = df.index.tz_localize(None)

    ds = xr.Dataset.from_dataframe(df)
    # rename "datetime" to "time", which is xarray convention
    ds = ds.rename({"datetime": "time"})
    # ds["date"] = ds["date"].dt.date

    # set metadata
    ds.attrs = get_metadata(site).__dict__

    for param in params:
        ds[param.name] = ds[param.name] * param.conversion
        # xarray metadata assignment must come after all other operations
        ds[param.name].attrs = param.__dict__

    return ds


def format_wqp_date(date: str) -> str:
    """Reformat date from 'YYYY-MM-DD' to 'MM-DD-YYYY'."""
    return "-".join(date.split("-")[1:] + [date.split("-")[0]])


def get_samples(
        site: str,
        start_date: str,
        end_date: str,
        characteristic: str,
        fraction: str,
        provider: str = "NWIS",
        name: str = "concentration",
        filter_pcodes: Optional[List[str]] = None,
):
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
        The fraction of the parameter to retrieve. Options are 'Total',
        'Dissolved', 'Suspended'.
    provider : str
        The data provider. Options are 'NWIS' or 'STORET'.
    name : str
        Short name for the parameter. Default is 'concentration'.
    """
    if fraction and fraction not in ["Total", "Dissolved", "Suspended"]:
        raise ValueError("Fraction must be 'Total', 'Dissolved', 'Suspended'")

    if provider not in ["NWIS", "STORET"]:
        raise ValueError("Provider must be 'NWIS' or 'STORET'")

    if provider == "NWIS" and not site.startswith("USGS-"):
        site = "USGS-" + site

    # reformat dates from 'YYYY-MM-DD' to 'MM-DD-YYYY'
    start_date = format_wqp_date(start_date)
    end_date = format_wqp_date(end_date)

    df, _ = wqp.get_results(
        siteid=site,
        startDateLo=start_date,
        startDateHi=end_date,
        characteristicName=characteristic,
        provider=provider,
    )

    # filter by fraction
    if fraction:
        df = df[df["ResultSampleFractionText"] == fraction]

    # create datetime index
    df.index = pd.to_datetime(
        df["ActivityStartDate"] + " " + df["ActivityStartTime/Time"]
    )

    df[name] = df["ResultMeasureValue"].astype(float)
    df.index.name = "time"
    df.index = df.index.tz_localize(None)

    if provider == "NWIS":
        # add parameter metadata
        if not filter_pcodes and len(set(df["USGSPCode"])) != 1:
            # TODO print the pcodes
            raise ValueError("Multiple parameters returned from NWIS.")

        elif filter_pcodes:
            # filter df by list of pcodes
            df = df[df["USGSPCode"].isin(filter_pcodes)]

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

    if provider == "NWIS":
        # strip the "USGS-" prefix from the site number
        site_id = site[5:]
        ds.attrs = get_metadata(site_id).__dict__

        if filter_pcodes:
            pcode = filter_pcodes[0]

        else:
            pcode = df["USGSPCode"].iloc[0]

        pcode = str(pcode).zfill(5)

        attrs = get_parameters({name: pcode})
        ds[name].attrs = attrs.__dict__

    if provider == "STORET":
        pass

    return ds


def get_qwdata_samples(
    site: str, start_date: str, end_date: str, pcode: str, name: str = "concentration"
) -> Dataset:
    """Get sample data from the USGS NWIS API.

    Warning: The QWData service is deprecated and no longer receives data.

    Parameters
    ----------
    site : str
        USGS site number.
    start_date : str
        Start date in the format 'yyyy-mm-dd'.
    end_date : str
        End date in the format 'yyyy-mm-dd'.
    pcode : str
        USGS parameter to retrieve.
    name : str
        Short name for the parameter.

    Returns
    -------
    pandas.DataFrame
        Dataframe with the requested sample data.
    """
    df, _ = nwis.get_qwdata(
        sites=site, start=start_date, end=end_date, parameterCd=pcode
    )
    attrs = get_parameters({name: pcode})
    ppcode = "p" + pcode

    # check if data are strings
    if df[ppcode].dtype == "O":
        # remove "<" and ">" from values and convert to float
        # TODO handle censoring
        df[ppcode] = df[ppcode].str.extract("(\d+.\d+)", expand=False).astype(float)

    df = df.rename(columns={ppcode: name})

    df = df[[name]]
    # remove timezone for xarray compatibility
    df.index = df.index.tz_localize(None)

    ds = xr.Dataset.from_dataframe(df)
    # rename "datetime" to "time", which is xarray convention
    ds = ds.rename({"datetime": "time"})

    ds.attrs = get_metadata(site).__dict__
    ds[name].attrs = attrs.__dict__

    return ds
