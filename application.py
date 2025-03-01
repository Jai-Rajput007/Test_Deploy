from flask import Flask, request, jsonify, render_template, redirect, url_for
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

# Load models with error handling
try:
    ridge_model = pickle.load(open('Models/ridge.pkl', 'rb'))
    scaler_model = pickle.load(open('Models/scaler.pkl', 'rb'))
except FileNotFoundError:
    print("Error: Model files not found!")
    ridge_model = None
    scaler_model = None
except Exception as e:
    print(f"Error loading models: {str(e)}")
    ridge_model = None
    scaler_model = None

if ridge_model is None or scaler_model is None:
    raise Exception("Failed to load models. Check file paths and contents.")

@app.route("/")
def index():
    return redirect(url_for('predict_datapoint'))

@app.route('/predict_datapoint', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        try:
            Temperature = float(request.form.get('Temperature', 0))
            RH = float(request.form.get('RH', 0))
            Ws = float(request.form.get('Ws', 0))
            Rain = float(request.form.get('Rain', 0))
            FFMC = float(request.form.get('FFMC', 0))
            DMC = float(request.form.get('DMC', 0))
            ISI = float(request.form.get('ISI', 0))
            Classes = float(request.form.get('Classes', 0))
            Region = float(request.form.get('Region', 0))
            
            new_data = scaler_model.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
            result = ridge_model.predict(new_data)
            return render_template('home.html', results=result[0])
        except ValueError as e:
            return render_template('home.html', error=f"Invalid input: {str(e)}")
        except Exception as e:
            return render_template('home.html', error=f"An error occurred: {str(e)}")
    else:
        return render_template('home.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)