# Retrieve data from the National Water Quality Assessment Program (NAWQA)

import boto3
import io
import lithops
import os
import pandas as pd
import xarray as xr

from dataclasses import dataclass
from matplotlib import pyplot as plt

from discontinuum.utils import aggregate_to_daily
from loadest_gp.providers import usgs
from loadest_gp import LoadestGPMarginalGPyTorch as LoadestGP

PROJECT = "National Water Quality Assessment Program (NAWQA)"
START_DATE = "1991-01-01"  # verify start date
END_DATE = "2024-12-31"  # verify end date
PCODE = "631"  # Nitrate plus nitrite
CHARACTERISTIC = "Inorganic nitrogen (nitrate and nitrite)"
FRACTION = "Dissolved"
DESTINATION_BUCKET = "s3://wma-uncertainty/nwqn-loadest-example/loadest-gp-output/"
SAMPLES_BUCKET = "s3://wma-uncertainty/nwqn-loadest-example/nwqn-samples.parquet/"
DAILY_BUCKET = "s3://wma-uncertainty/nwqn-loadest-example/nwqn-daily.parquet/"


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
    # download covariates (daily streamflow)
    #daily = usgs.get_daily(
    #    site=record.site_id,
    #    start_date=record.start_date,
    #    end_date=record.end_date
    #    )

    # download target (concentration)
    #samples = usgs.get_samples(
    #    site=record.site_id,
    #    start_date=record.start_date,
    #    end_date=record.end_date,
    #    characteristic=record.characteristic,
    #    fraction=record.fraction,
    #    project=record.project
    #    )

    # read data from s3
    site = record.site_id

    daily = pd.read_parquet(f"{DAILY_BUCKET}/MonitoringLocationIdentifier=USGS-{site}")

    samples = pd.read_parquet(f"{SAMPLES_BUCKET}/site_no={site}")
    samples = samples[samples["USGSPCode"] == PCODE]

    if samples.empty or daily.empty:
        print(f"Site {site} has no data.")
        return

    samples = aggregate_to_daily(samples)

    training_data = xr.merge([samples, daily], join="inner")

    model = LoadestGP()

    fig, ax = plt.subplots()
    model.fit(
        target=training_data["concentration"],
        covariates=training_data[["time", "flow"]]
        )

    # plot the result
    model.plot(daily[["time", "flow"]], ax=ax)

    # save the figure to S3
    img_data = io.BytesIO()
    fig.savefig(img_data, format='png')
    img_data.seek(0)

    s3 = boto3.resource('s3')
    bucket = s3.Bucket(DESTINATION_BUCKET)
    bucket.put_object(
        Body=img_data,
        ContentType='image/png',
        Key="nwqn-{site_id}-{characteristic}-{fraction}.png".format(
            **record.__dict__
            )
    )


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
