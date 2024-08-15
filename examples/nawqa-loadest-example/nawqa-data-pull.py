# Retrieve data from the National Water Quality Assessment Program (NAWQA)

import lithops
import os
import pandas as pd
import math

from dataclasses import dataclass
from dataretrieval import nwis, wqp, nldi

from loadest_gp.providers.usgs import format_wqp_date

# TODO move this information to a configuration file
PROJECT = "National Water Quality Assessment Program (NAWQA)"
START_DATE = "1991-01-01"  # verify start date
END_DATE = "2023-12-31"  # verify end date
CHARACTERISTIC = "Phosphorus"
FRACTION = "Total"
DESTINATION_BUCKET = os.environ.get('DESTINATION_BUCKET')


@dataclass
class SiteRecord:
    site_id: str
    start_date: str
    end_date: str
    characteristic: str
    fraction: str
    project: str


def map_retrieval(record: SiteRecord):
    """Map function to pull data from NWIS and WQP"""
    # search for neighboring sites
    site_list = find_neighboring_sites(record.site_id)
    wqp_site_list = [f"USGS-{site}" for site in site_list]

    # download covariates (daily streamflow)
    feature_df = nwis.get_dv(
        sites=record.site_id,
        start=record.start_date,
        end=record.end_date,
        parameterCd="00060",
        )

    target_df = wqp.get_results(
        siteid=wqp_site_list,
        startDateLo=format_wqp_date(record.start_date),
        startDateHi=format_wqp_date(record.end_date),
        characteristicName=record.characteristic,
        project=record.project,
        )

    # merge results from all sites
    target_df['MonitoringLocationIdentifier'] = f"USGS-{record.site_id}"

    # drop other fractions
    target_df = target_df[target_df["ResultSampleFractionText"] == record.fraction]

    # print the number of samples and features at the site
    print(
        f"Site {record.site_id} has {len(target_df)} samples and",
        f"{len(feature_df)} daily flow values meeting the search criteria.",
    )

    # save the results to parquet on S3
    if len(target_df) != 0 and len(feature_df) != 0:
        feature_df.astype(str).to_parquet(
            f's3://{DESTINATION_BUCKET}/nawqa-samples.parquet',
            engine='pyarrow',
            partition_cols=['MonitoringLocationIdentifier'],
            compression='zstd',
        )

        target_df.astype(str).to_parquet(
            f's3://{DESTINATION_BUCKET}/nawqa-flow.parquet',
            engine='pyarrow',
            partition_cols=['site_no'],
            compression='zstd',
        )


def find_neighboring_sites(site, search_factor=0.05):
    """Find sites upstream and downstream of the given site within a certain
    distance.

    Parameters
    ----------
    site : str
        8-digit site number.
    search_factor : float, optional
    """
    site_df, _ = nwis.get_info(sites=site)
    drain_area_sq_mi = site_df["drain_area_va"].values[0]
    length = _estimate_watershed_length_km(drain_area_sq_mi)
    search_distance = length * search_factor
    # clip between 1 and 9999km
    search_distance = max(1.0, min(9999.0, search_distance))

    # get upstream and downstream sites
    gdfs = [
        nldi.get_features(
            feature_source="WQP",
            feature_id=f"USGS-{site}",
            navigation_mode=mode,
            distance=search_distance,
            data_source="nwissite",
            )
        for mode in ["UM", "DM"]  # upstream and downstream
    ]

    features = pd.concat([gdfs], ignore_index=True)

    df, _ = nwis.get_info(sites=list(features.identifier.str.strip('USGS-')))
    # drop sites with disimilar different drainage areas
    df = df.where(
        (df["drain_area_va"] / drain_area_sq_mi) > search_factor,
        ).dropna(how="all")

    return df["site_no"].to_list()


def _estimate_watershed_length_km(drain_area_sq_mi):
    """Estimate the diameter assuming a circular watershed.

    Parameters
    ----------
    drain_area_sq_mi : float
        The drainage area in square miles.

    Returns
    -------
    float
        The diameter of the watershed in kilometers.
    """
    # assume a circular watershed
    length_miles = 2 * (drain_area_sq_mi / math.pi) ** 0.5
    # convert to km
    return length_miles * 1.60934


if __name__ == "__main__":
    project = "National Water Quality Assessment Program (NAWQA)"

    site_df = pd.read_csv(
        'NWQN_sites.csv',
        comment='#',
        dtype={'SITE_QW_ID': str, 'SITE_FLOW_ID': str},
        )

    site_list = site_df['SITE_QW_ID'].to_list()

    sites = [
        SiteRecord(
            site_id=site,
            start_date=START_DATE,
            end_date=END_DATE,
            characteristic=CHARACTERISTIC,
            fraction=FRACTION,
            project=PROJECT,
            )
        for site in site_list
    ]

    sites = sites[:4]  # prune for testing

    fexec = lithops.FunctionExecutor(config_file="lithops.yaml")
    futures = fexec.map(map_retrieval, sites)

    futures.get_result()
