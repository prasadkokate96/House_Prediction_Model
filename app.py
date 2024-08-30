import pickle
import json

from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model and scaler
regmodel = pickle.load(open('regmodel.pkl', 'rb'))
scalar = pickle.load(open('scaling.pkl', 'rb'))

@app.route('/')  # Route for the home page
def home():
    return render_template('Home.html')

@app.route('/predict_api', methods=['POST'])  # API route for prediction
def predict_api():
    data = request.json['data']  # Get data from the JSON request
    print(data)
    print(np.array(list(data.values())).reshape(1, -1))
    new_data = scalar.transform(np.array(list(data.values())).reshape(1, -1))
    output = regmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])

@app.route('/predict', methods=['POST'])  # Route for the form submission
def predict():
    data = [float(x) for x in request.form.values()]  # Get form data
    final_input = scalar.transform(np.array(data).reshape(1, -1))
    print(final_input)
    output = regmodel.predict(final_input)[0]  # Make prediction
    return render_template("Home.html", prediction_text="The house price prediction is {}".format(output))

if __name__ == "__main__":
    app.run(debug=True)
