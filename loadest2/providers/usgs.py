"""Helper functions for pulling USGS data."""

from __future__ import annotations
from typing import TYPE_CHECKING

import pandas as pd
from dataclasses import dataclass
from dataretrieval import nwis


if TYPE_CHECKING:
    from pandas import DataFrame
    from typing import Optional, List

CFS_TO_M3 = 0.0283168


@dataclass
class NWISParameter:
    pcode: str
    name: str
    unit: str
    suffix: Optional[str] = None
    conversion: Optional[float] = 1.0

    @property
    def ppcode(self):
        """
        Return the parameter code with a 'p' prefix, which is used by the QWData service.

        """
        return "p" + self.pcode


NWISFlow = NWISParameter("00060", "Streamflow, mean daily", "m3/s" "_Mean", 0.0283168)


def get_nwis_parameters(pcodes: List[str]) -> List[NWISParameter]:
    """Get NWIS parameters from a list of parameter codes.

    Parameters
    ----------
    pcodes : List[str]
        List of NWIS parameter codes.

    Returns
    -------
    List[NWISParameter]
        List of NWIS parameters.
    """
    df, _ = nwis.get_pmcoodes(pcodes)
    # convert to NWISParameter
    params = []
    for _, row in df.iterrows():
        params.append(
            NWISParameter(
                pcode=row["parameter_cd"],
                name=row["SRSName"],
                unit=row["unit"],
            )
        )

    return params


def get_nwis_daily_data(
    site: str, start_date: str, end_date: str, params: List[NWISParameter] = [NWISFlow]
) -> DataFrame:
    """Get daily data from the USGS NWIS API.

    Parameters
    ----------
    site : str
        USGS site number.
    start_date : str
        Start date in the format 'yyyy-mm-dd'.
    end_date : str
        End date in the format 'yyyy-mm-dd'.
    params : List[NWISParameter], optional
        List of parameters to retrieve. The default is flow only [NWISFlow].

    Returns
    -------
    pandas.DataFrame
        Dataframe with the requested data.
    """
    pcodes = [param.pcode for param in params]

    data, _ = nwis.get_dv(
        sites=site, start=start_date, end=end_date, parameterCd=pcodes
    )

    # rename columns
    data = data.rename(
        columns={param.pcode + param.suffix: param.name for param in params}
    )
    # convert units
    for param in params:
        data[param.name] = data[param.name] * param.conversion

    # drop columns
    data = data[[param.name for param in params]]

    return data


def get_nwis_sample_data(
    site: str, start_date: str, end_date: str, param: NWISParameter
) -> DataFrame:
    """Get sample data from the USGS NWIS API.

    Parameters
    ----------
    site : str
        USGS site number.
    start_date : str
        Start date in the format 'yyyy-mm-dd'.
    end_date : str
        End date in the format 'yyyy-mm-dd'.
    param : NWISParameter
        NWIS parameter to retrieve.

    Returns
    -------
    pandas.DataFrame
        Dataframe with the requested sample data.
    """
    pcode = param.pcode
    data, _ = nwis.get_qwdata(
        site=site, start=start_date, end=end_date, parameterCd=pcode
    )
    # check if data are strings
    if data[param.ppcode].dtype == 'O':
        # remove "<" and ">" from values and convert to float
        # TODO handle censoring
        data[param.ppcode] = (
            data[param.ppcode].str.extract('(\d+.\d+)', expand=False).astype(float)
        )

    data = data.rename(columns={param.ppcode: param.name})

    data = data[[param.name]]

    return data
