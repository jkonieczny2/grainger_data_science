from flask import Flask, request, json
import pickle
import numpy as np
from flask import jsonify
from math import sqrt
import pandas as pd
from datetime import datetime
#from sklearn.externals import joblib
app = Flask(__name__)


with open('models/random_forest_classifier_final.pkl') as f:
    model = pickle.load(f)

categorical_variables = [
    'Agency' ,
    'Color' ,
    'Route' ,
    'Violation_code' ,
    'Violation_Description' ,
    'Location' ,
    'Body_Style' ,
    'RP_State_Plate' ,
]

encoders = {}

for v in categorical_variables:
    print('loading encoder for %s'%v)
    with open('models/%s_label_encoder.pkl'%v, 'rb') as f:
        encoders[v] = pickle.load(f)

variables = [
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
    #'distance_from_la'
] 


@app.route('/', methods=['POST'])
def index():    
    # Parse request body for model input 
    data = request.get_json()
    print(data)    

    # Check to make sure all fields present; otherwise raise HTTP error
    
    # Convert date string to pd date
    start_epoch = datetime(2000,01,01)

    issue_dt = data['Issue_Date']
    issue_dt = datetime.strptime(issue_dt, "%Y-%m-%dT%M:%H:%S")
    issue_dt = (issue_dt - start_epoch).days
    data['Issue_Date'] = issue_dt

    exp_dt = data['Plate_Expiry_Date']
    exp_dt = datetime.strptime(exp_dt , "%Y%m")
    exp_dt = (exp_dt - start_epoch).days
    data['Plate_Expiry_Date'] = exp_dt

    # Append un_transformed variables
    prediction_data = []
    for v in variables:
        d = data[v]

        if v in encoders:
            encoder = encoders[v]
            d = encoder.transform([d])[0]

        prediction_data.append(d)

    # Calculate distance_from_la and add to data
    la_lat = 6487847
    la_lon = 1841468

    d = sqrt(((data['Latitude'] - la_lat)**2 + (data['Longitude'] - la_lon)**2))/5280
    prediction_data.append(d)

    # Transform into numpy array for prediction
    prediction_data = np.array([prediction_data])

    # Make prediction 
    prediction = model.predict_proba(prediction_data)

    pred_class = np.argmax(prediction)
    pred_prb = np.amax(prediction)

    # Respond with prediction result
    result = {
        'prediction': str(pred_class) ,
        'probability': str(pred_prb) ,
    }    
    
    return jsonify(result)

if __name__ == '__main__':    
    # model = load_model(MODEL_FILE_NAME)
    # listen on all IPs 
    app.run(host='0.0.0.0')
