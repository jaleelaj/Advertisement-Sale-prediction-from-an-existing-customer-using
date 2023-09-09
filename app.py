from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

# Estimated scaling parameters from your sample data
estimated_age_mean = 35.0  # Mean of Age in your sample data
estimated_age_std = 8.4    # Standard Deviation of Age in your sample data
estimated_salary_mean = 52390.5  # Mean of Salary in your sample data
estimated_salary_std = 36793.9   # Standard Deviation of Salary in your sample data

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the input data from the form
        age = float(request.form['age'])
        salary = float(request.form['salary'])

        # Preprocess the input data using the estimated scaling parameters
        age_scaled = (age - estimated_age_mean) / estimated_age_std
        salary_scaled = (salary - estimated_salary_mean) / estimated_salary_std

        print("Input Age Scaled:", age_scaled)
        print("Input Salary Scaled:", salary_scaled)

        # Make a prediction
        input_data = np.array([[age_scaled, salary_scaled]])
        prediction = model.predict(input_data)

        print("Prediction:", prediction)

        # Return the prediction as JSON
        return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
