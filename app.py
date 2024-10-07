from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load('iq_model.pkl')

# Home page route
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle form submission and make predictions
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get form data
            age = int(request.form.get('age', 0))  # Default to 0 if missing
            gender = request.form.get('gender', 'Male')  # Default to 'Male'
            ethnicity = request.form.get('ethnicity', 'Caucasian')  # Default value
            socioeconomic_status = request.form.get('socioeconomic_status', 'Middle')
            location = request.form.get('location', 'Urban')
            years_of_education = int(request.form.get('years_of_education', 12))
            highest_education_level = request.form.get('highest_education_level', 'Bachelor')
            standardized_test_score = int(request.form.get('standardized_test_score', 0))
            memory_test_score = int(request.form.get('memory_test_score', 0))
            problem_solving_score = int(request.form.get('problem_solving_score', 0))
            verbal_reasoning_score = int(request.form.get('verbal_reasoning_score', 0))
            math_reasoning_score = int(request.form.get('math_reasoning_score', 0))
            processing_speed_score = int(request.form.get('processing_speed_score', 0))
            spatial_reasoning_score = int(request.form.get('spatial_reasoning_score', 0))

            # Prepare input data in the correct format
            input_data = pd.DataFrame({
                'Age': [age],
                'Gender': [gender],
                'Ethnicity': [ethnicity],
                'Socioeconomic Status': [socioeconomic_status],
                'Location': [location],
                'Years of Education': [years_of_education],
                'Highest Education Level': [highest_education_level],
                'Standardized Test Score': [standardized_test_score],
                'Memory Test Score': [memory_test_score],
                'Problem Solving Score': [problem_solving_score],
                'Verbal Reasoning Score': [verbal_reasoning_score],
                'Math Reasoning Score': [math_reasoning_score],
                'Processing Speed Score': [processing_speed_score],
                'Spatial Reasoning Score': [spatial_reasoning_score]
            })

            # Make prediction
            prediction = model.predict(input_data)

            # Return the result to the user
            return render_template('index.html', predicted_iq=prediction[0])

        except Exception as e:
            return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
