#!/usr/bin/bash

# Create necessary directories
mkdir -p data
mkdir -p app/models

# Download CSV file from S3
cd scripts
URL="https://s3-us-west-2.amazonaws.com/pcadsassessment/parking_citations.corrupted.csv"
echo "Downloading data from ${URL}"
python download_data.py
echo "Finished downloading data from ${URL}"

# Split file into corrupted and uncorrupted
echo "Parsing CSV into corrupted and uncorrupted data."
python parse_citations_file.py
echo "Finished parsing CSV"


# Train model
echo "Running Model Training Script"
python SKLearnRandomForest-ForScript.py
echo "Finished training model"

# Start Server
cd ../app/
echo "Starting Flask server"
python app.py
echo "Started Flask server"
