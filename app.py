from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd  # Added pandas

app = Flask(__name__)

# Load models
rf_model = joblib.load("xgb_classifier.joblib")  # Model 1
impulse_model = joblib.load("impulse_buying_model.joblib")  # Model 2

# Load transformers
rf_transformer = joblib.load(r'C:\Users\thenn\Desktop\MLProject\MLproject\ct_transformer')  # For Model 1
impulse_transformer = joblib.load("transformer.joblib")  # For Model 2

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/model1', methods=['GET', 'POST'])
def model1():
    prediction = None
    if request.method == 'POST':
        form_values = request.form.to_dict()  # Convert to dictionary

        # Convert form values to DataFrame
        input_df = pd.DataFrame([form_values])
  

        # Apply transformation
        transformed_features = rf_transformer.transform(input_df)

        # Make prediction
        prediction = rf_model.predict(transformed_features)[0]

        return render_template('result1.html', prediction=prediction)

    return render_template('model1.html')

@app.route('/model2', methods=['GET', 'POST'])
def model2():
    prediction = None
    if request.method == 'POST':
        form_values = request.form.to_dict()  # Convert to dictionary

        # Convert form values to DataFrame
        input_df = pd.DataFrame([form_values])

        # Apply transformation
        transformed_features = impulse_transformer.transform(input_df)
       

        # Make prediction
        prediction = impulse_model.predict(transformed_features)[0]

        return render_template('result2.html', prediction=prediction)

    return render_template('model2.html')

if __name__ == '__main__':
    app.run(debug=True)
