from flask import Flask, render_template, request
import pickle
import pandas as pd

# Load the model
with open('GB_boosted_clf.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input features from the form
    strdist = float(request.form['strdist'])
    basarea = float(request.form['basarea'])
    fos = float(request.form['fos'])
    cohesion = float(request.form['cohesion'])
    scarpdist = float(request.form['scarpdist'])
    scarps = float(request.form['scarps'])
    frictang = float(request.form['frictang'])
    rainfall = float(request.form['rainfall'])
    
    # Create a DataFrame for prediction
    input_features = pd.DataFrame([[strdist, basarea, fos, cohesion, scarpdist, scarps, frictang,rainfall]],
                                   columns=['strdist', 'basarea', 'fos', 'cohesion', 'scarpdist', 'scarps', 'frictang','rainfall'])
    
    # Make prediction (assuming binary classification 0 = low risk, 1 = high risk)
    prediction_prob = model.predict_proba(input_features)[0]
    
    # Determine the result based on the probability threshold
    if prediction_prob[1] >= 0.5:
        prediction = "Chances are high for a landslide"
    else:
        prediction = "Chances are low for a landslide"
    
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
