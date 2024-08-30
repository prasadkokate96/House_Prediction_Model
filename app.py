import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np 
import pandas as pd 

app=Flask(__name__)

# Below we load the model 
regmodel=pickle.load(open('regmodel.pkl','rb'))
scalar=pickle.load(open('scaling.pkl','rb'))
@app.route('/') # this is first route used to go on home page
def home():
    return render_template('Home.html')


@app.route('/predict_api',methods=['post']) #used to predict but before we get some inputs from user to predict 
def predict_api():
    data=request.json['data'] #whenever user hit to predict api and try to give data it will save in json file
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data=scalar.transform(np.array(list(data.values())).reshape(1,-1))
    output=regmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])

if __name__=="__main__":
    app.run(debug=True)
