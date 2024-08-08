# Retrieve data from the National Water Quality Assessment Program (NAWQA)

import lithops
import math
import os
import pandas as pd

from dataclasses import dataclass
from loadest_gp.providers import get_samples, get_daily_flow

PROJECT = "National Water Quality Assessment Program (NAWQA)"
START_DATE = "1991-01-01"  # verify start date
END_DATE = "2024-12-31"  # verify end date
PARAMETER_CD = "00665"
DESTINATION_BUCKET = os.environ.get('DESTINATION_BUCKET')

@dataclass
class SiteRecord:
    site_id: str
    start_date: str
    end_date: str
    characteristic: str

def map_retrieval(site):
    """Map function to pull data from NWIS and WQP"""
    # download covariates (daily streamflow)
    daily = usgs.get_daily(site=site, start_date=start_date, end_date=end_date)

    # download target (concentration)
    samples = usgs.get_samples(site=record.site_id, 
                               start_date=record.start_date
                               end_date=END_DATE, 
                               characteristic=characteristic,
                               fraction=fraction)




    flow_ds = get_daily_flow(site, start_date=START_DATE, end_date=END_DATE)

    df, _ = wqp.get_results(siteid=site_list,
                            project=PROJECT,
                            )

    # merge sites
    df['MonitoringLocationIdentifier'] = f"USGS-{site}"

    if len(df) != 0:
        df.astype(str).to_parquet(f's3://{DESTINATION_BUCKET}/nwqn-samples.parquet',
                                  engine='pyarrow',
                                  partition_cols=['MonitoringLocationIdentifier'],
                                  compression='zstd')
        # optionally, `return df` for further processing


if __name__ == "__main__":
    project = "National Water Quality Assessment Program (NAWQA)"

    site_df = pd.read_csv(
        'NWQN_sites.csv',
        comment='#',
        dtype={'SITE_QW_ID': str, 'SITE_FLOW_ID': str},
        )

    site_list = site_df['SITE_QW_ID'].to_list()

    sites = [
        SiteRecord(site_id=site, start_date=START_DATE, end_date=END_DATE, parameter_cd=PARAMETER_CD )
        for site in site_list
    ]

    sites = sites[:4]  # prune for testing

    fexec = lithops.FunctionExecutor(config_file="lithops.yaml")
    futures = fexec.map(map_retrieval, sites)

    futures.get_result()
