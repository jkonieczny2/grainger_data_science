#!/usr/bin/env python
# coding: utf-8
from __future__ import division

import pandas as pd
import numpy as np
import csv
from pprint import pprint
import random
from sklearn import preprocessing
import pickle
from math import sqrt 
from datetime import datetime

# Load data from "uncorrupted" part of dataset
filename = '../data/parking_citations_uncorrupted.csv'
random.seed(42)

# Get headers and data types
columns = []
with open(filename) as f:
    reader = csv.reader(f)
    columns = reader.next()
    
names = [col.replace(" ","_") for col in columns]

dtypes = {
    'Ticket_number': 'unicode' ,
    'Issue_Date': 'unicode' ,
    'Issue_Time':'unicode'  ,
    'Meter_Id': 'unicode',
    'Marked_Time': 'unicode' ,
    'RP_State_Plate': 'unicode',
    'Plate_Expiry_Date': 'unicode' ,
    'VIN': 'unicode' ,
    'Make': 'unicode' ,
    'Body Style': 'unicode' ,
    'Color': 'unicode' ,
    'Location': 'unicode' ,
    'Route': 'unicode' ,
    'Agency': 'unicode' ,
    'Violation_Code': 'unicode' ,
    'Violation_Description': 'unicode' ,
    'Fine_amount': np.float64 ,
    'Latitude': np.float64 ,
    'Longitude': np.float64 ,
}


# Load data from file
print("Reading data from file " + filename)

citations = pd.read_csv(
    filename ,
    header = 0 ,
    names = names ,
    dtype = dtypes ,
)


# Label dataset by top 25 makes
top_25_makes = citations.groupby(['Make']).size().sort_values(ascending=False)
make_names = set(top_25_makes.index[:25])

citations['top_25_makes'] = citations['Make'].apply(lambda x: (x in make_names)*1)
citations['top_25_makes'] = citations['top_25_makes'].astype('category')

# Replace null Latitude & Longitude with 99999.0
citations['Latitude'] = citations['Latitude'].fillna(99999.0)
citations['Longitude'] = citations['Longitude'].fillna(99999.0)

print("Formatting and transforming data")
# Date formatting and coversion to days since epoch
citations['Issue_Date'] = pd.to_datetime(citations['Issue_Date'])

citations['Plate_Expiry_Date'] = citations['Plate_Expiry_Date'].fillna('')
citations['Plate_Expiry_Date'] = pd.to_datetime(citations['Plate_Expiry_Date'].str[:-2], format='%Y%m', errors='coerce')

citations['Issue_Date'] = (citations['Issue_Date'] - datetime(2000,01,01)).dt.days
citations['Plate_Expiry_Date'] = (citations['Plate_Expiry_Date'].fillna(datetime(2000,01,01)) - datetime(2000,01,01)).dt.days

# Drop vars with too many NA's
drop_vars = [
    'VIN' ,
    'Marked_Time' ,
    'Meter_Id'
]
citations.drop(drop_vars , axis=1, inplace=True)

# Fill in nulls for continuous variables
citations['Issue_time'] = citations['Issue_time'].fillna(citations['Issue_time'].median())
citations['Fine_amount'] = citations['Fine_amount'].fillna(citations['Fine_amount'].median())

# Fill in nulls for categorical variables
categorical_vars = [
    'RP_State_Plate' ,
    'Body_Style' ,
    'Color' ,
    'Route' ,
    'Agency' ,
    'Violation_code' ,
    'Violation_Description' ,
    'Location'
]

for v in categorical_vars:
    citations[v] = citations[v].astype('category')
    citations[v] = citations[v].fillna(citations[v].mode().values[0])


# Convert latitude/longitude into distance from center of LA
# Using US Feet Projection, should be Cartesian formula
# http://www.earthpoint.us/StatePlane.aspx, (34.0522,-118.2437)
la_lat = 6487847
la_lon = 1841468

citations['distance_from_la'] = ((citations['Latitude'] - la_lat)**2 + (citations['Longitude'] - la_lon)**2).apply(lambda x: sqrt(x))/5280


# Label-encode all high cardinality categoricals - worked better than one-hot
categorical_vars = [
    'Agency' , 
    'Color' ,
    'Route' ,
    'Violation_code' ,
    'Violation_Description' ,
    'Location' ,
    'Body_Style' ,
    'RP_State_Plate'
]

for v in categorical_vars:
    le = preprocessing.LabelEncoder()
    le.fit(citations[v])

    # Write Label Encoder to app directory
    filename = '../app/models/%s_label_encoder.pkl'%v
    print("Writing Label Encoder for %s"%v)
    with open(filename, 'wb') as f:
        pickle.dump(le, f)

    citations[v] = le.transform(citations[v])


feature_cols = [
    'Issue_Date' ,
    'Issue_time' ,
    'Plate_Expiry_Date' ,
    'Color' ,
    'Location' ,
    'Route' ,
    'Agency' ,
    'Violation_code' ,
    'Violation_Description' ,
    'Fine_amount' ,
    'Body_Style' ,
    'RP_State_Plate' ,
    'distance_from_la'
] 

target_cols = [
    'top_25_makes' ,
]


labels = np.array(citations[target_cols])
features = np.array(citations[feature_cols])


print("Training Model")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Train-test split
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)

# Instantiate model with 100 decision trees
rf = RandomForestClassifier(
    n_estimators = 100, 
    random_state = 42,
    max_depth = 20 ,
)
# Train the model on training data
rf.fit(train_features, train_labels)

print("Saving Model")

with open('../app/models/random_forest_classifier_final.pkl' ,'wb') as f:
    pickle.dump(rf, f)

