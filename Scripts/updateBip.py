# %% Imports

import os
import datetime

import requests
from pyspark import SparkContext

from statcast.bip import Bip


# %% Create Spark Context

sc = SparkContext(appName="updateBip")

# %% Load data, plot histograms of statcast data

years = (2015, 2016)

for year in years:
    bip = Bip(years=(year,), n_jobs=sc)

# %% Transfer results to S3

instanceID = requests. \
        get('http://169.254.169.254/latest/meta-data/instance-id').text
dtStr = datetime.datetime.utcnow().strftime('%Y-%m-%d--%H-%M-%S')
os.system('aws s3 sync . s3://mf-first-bucket/output/{}/{}'.
          format(instanceID, dtStr))

# %% Stop Spark Context

sc.stop()
