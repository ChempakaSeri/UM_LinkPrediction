from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
import sklearn
import xgboost as xgb

import pickle
import pandas as pd



header = ['MM','MC','WN','WMN(GB)','WCN','DS(GB)','AC','MC(%)']
retName = ['Probability_1', 'Probability_2', 'Probability_3', 'Probability_6','Probability_7']

app = Flask (__name__)
CORS(app)

@app.route('/api/link', methods=['GET'])
def api_sentiment():
    global error
    error = False

    if 'data' in request.args:
        data = str(request.args['data'])
        if data is None:
            return "Error: No data provided"
        result = predict(data)
        return(result)

def predict(data):
    return_dict = {}
    
    
    model = xgb.Booster({'nthread':4})

    model.load_model('C:/Users/Ameer/Desktop/cuya/model_v3.pkl')

    if data is not None:
        feature = [float(i) for i in data.split(',')]

        values = pd.DataFrame(  [feature],
                                columns=header,
                                dtype=float,
                                index=['input']
                                )

        input_variable = xgb.DMatrix(values)

        prob_1 = model.predict(input_variable)[0][0]
        prob_2 = model.predict(input_variable)[0][1]
        prob_3 = model.predict(input_variable)[0][2]
        prob_6 = model.predict(input_variable)[0][3]
        prob_7 = model.predict(input_variable)[0][4]

        return_dict.update({
            retName[0]:str(prob_1),
            retName[1]:str(prob_2),
            retName[2]:str(prob_3),
            retName[3]:str(prob_6),
            retName[4]:str(prob_7)
        })

        return(jsonify(return_dict))

   

def main():
    #   Load model into a dictionary
    app.run(debug=True, host='localhost', port=5000)

if __name__ == "__main__":
    main()
