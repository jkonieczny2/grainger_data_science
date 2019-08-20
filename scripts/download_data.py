#!/usr/bin/python

import requests
import shutil


url = 'https://s3-us-west-2.amazonaws.com/pcadsassessment/parking_citations.corrupted.csv'
filename = '../data/parking_citations.corrupted.csv'

def download_file(url, filename):
    with requests.get(url, stream=True) as resp:
        with open(filename, 'wb') as f:
            shutil.copyfileobj(resp.raw, f)
    return filename

download_file(url, filename)


