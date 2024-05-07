"""Helper functions for pulling USGS data."""

from __future__ import annotations
from typing import TYPE_CHECKING

import xarray as xr

from dataclasses import dataclass
from dataretrieval import nwis

from discontinuum.providers.base import Location

if TYPE_CHECKING:
    # from pandas import DataFrame
    from xarray import DataArray, Dataset
    from typing import Optional, List, Union, Dict

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
    units="m^3/s",
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


def get_location(site: str) -> Location:
    """Get site information from the USGS NWIS API.

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

    return Location(
        id=row["site_no"],
        name=row["station_nm"],
        latitude=row["dec_lat_va"],
        longitude=row["dec_long_va"],
    )


# TODO pass a dict not a list of params
def get_daily(
    site: str, start_date: str, end_date: str, params: List[USGSParameter] = [USGSFlow]
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
    ds.attrs = get_location(site).__dict__

    for param in params:
        ds[param.name] = ds[param.name] * param.conversion
        # xarray metadata assignment must come after all other operations
        ds[param.name].attrs = param.__dict__

    return ds


def get_samples(
    site: str, start_date: str, end_date: str, pcode: str, name: str = "concentration"
) -> Dataset:
    """Get sample data from the USGS NWIS API.

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
    if df[ppcode].dtype == 'O':
        # remove "<" and ">" from values and convert to float
        # TODO handle censoring
        df[ppcode] = df[ppcode].str.extract('(\d+.\d+)', expand=False).astype(float)

    df = df.rename(columns={ppcode: name})

    df = df[[name]]
    # remove timezone for xarray compatibility
    df.index = df.index.tz_localize(None)

    ds = xr.Dataset.from_dataframe(df)
    # rename "datetime" to "time", which is xarray convention
    ds = ds.rename({"datetime": "time"})

    ds.attrs = get_location(site).__dict__
    ds[name].attrs = attrs.__dict__

    return ds
