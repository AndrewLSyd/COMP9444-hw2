"""
Usage: python gsheets.py [fw]

Add this account to gsheets sharing permissions: argo-service-account@gcp-wow-rwds-ai-dschapter-dev.iam.gserviceaccount.com
Authenticate with GCP service account first:
    gcloud auth activate-service-account argo-service-account@gcp-wow-rwds-ai-dschapter-dev.iam.gserviceaccount.com --key-file=argo-dschapter-dev-c9b15ee5d63f.json
    export GOOGLE_APPLICATION_CREDENTIALS=argo-dschapter-dev-c9b15ee5d63f.json
References: https://developers.google.com/sheets/api/quickstart/python
"""
from __future__ import print_function
import os.path
import os
import sys

from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

import google.auth
from google.cloud import bigquery
from google.cloud import bigquery_storage
from google.cloud.exceptions import NotFound

import pandas as pd

# set credentials to GCP service account to access gsheets
os.environ['GOOGLE_APPLICATION_CREDENTIALS']="argo-dschapter-dev-c9b15ee5d63f.json"

# grab creentials from default login, use gcloud auth login
credentials, your_project_id = google.auth.default(
    scopes=["https://www.googleapis.com/auth/cloud-platform"]
)

SAMPLE_SPREADSHEET_ID = '1x-BDCMig4xmOxJfoSZ0TUHQOJNpTFV1_YbyNiKxooWo'  # grab from URL
SAMPLE_RANGE_NAME = 'Sheet1!B1:H1000'

def write_to_gsheets(row_to_write):
    """
    
    row_to_write: a list of items to append row-wise at the end e.g. [1,2,3,4,5,6]
    
    """
    # build interface with gsheets
    service = build('sheets', 'v4')

    # The ID of the spreadsheet to update.
    spreadsheet_id = SAMPLE_SPREADSHEET_ID  # TODO: Update placeholder value.

    # The A1 notation of a range to search for a logical table of data.
    # Values will be appended after the last row of the table.
    range_ = SAMPLE_RANGE_NAME  # TODO: Update placeholder value.

    # How the input data should be interpreted.
    value_input_option = 'USER_ENTERED'  # TODO: Update placeholder value.

    # How the input data should be inserted.
    insert_data_option = 'OVERWRITE'  # TODO: Update placeholder value.

    value_range_body = {
      "range": SAMPLE_RANGE_NAME,
      "majorDimension": "ROWS",
      "values": [row_to_write]
    }

    request = service.spreadsheets().values().append(spreadsheetId=spreadsheet_id, range=range_, valueInputOption=value_input_option, insertDataOption=insert_data_option, body=value_range_body)
    response = request.execute()

    print(response)
    
if __name__ == '__main__':
    write_to_gsheets([1,  # training start time
                      2,  # network structure
                      3,  # optimiser
                      4,  # loss
                      5,  # tra
                      6])
    

