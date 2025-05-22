import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load model and preprocessor
model = pickle.load(open('dtr.pkl', 'rb'))
preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input values
        input_data = pd.DataFrame([[
            int(request.form['Year']),
            float(request.form['average_rain_fall_mm_per_year']),
            float(request.form['pesticides_tonnes']),
            float(request.form['avg_temp']),
            request.form['Area'],
            request.form['Item']
        ]], columns=['Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp', 'Area', 'Item'])

        # Transform input
        transformed_input = preprocessor.transform(input_data)

        # Make prediction
        prediction = model.predict(transformed_input)

        return render_template('index.html', prediction=f"{prediction[0]:.2f}")

    except Exception as e:
        return render_template('index.html', error=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
