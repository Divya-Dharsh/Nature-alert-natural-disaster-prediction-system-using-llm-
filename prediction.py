import numpy as np
import pickle
from flask import Flask, request, render_template
from dotenv import load_dotenv
from gpt4all import GPT4All
import os
import joblib

# Load environment variables
load_dotenv()
app = Flask(__name__)

# Load GPT4All model
try:
    gpt4all_model = GPT4All("mistral-7b-instruct-v0.1.Q4_0.gguf")
except Exception as e:
    print("Error loading GPT4All model:", e)
    gpt4all_model = None

# Correct load_model function
def load_model(file_path):
    try:
        model = joblib.load(file_path)
        print(f"Loaded model from {file_path} -> type: {type(model)}")  # Debug
        return model
    except Exception as e:
        print(f"Error loading model from {file_path}: {e}")
        return None

# Load ML models
earthquake_model = load_model('models/models/earthquake_model.pkl')
flood_model = load_model('models/models/flood_model.pkl')
cyclone_model = load_model('models/models/cyclone_model.pkl')

# Routes
@app.route('/')
def home():
    return render_template('homepage.html')

@app.route('/ai_query')
def ai_query():
    return render_template('ai_query.html')

@app.route('/query')
def query():
    return render_template('query.html')

@app.route('/earth')
def earth():
    return render_template('earthquake.html')

@app.route('/cyclone')
def cyclone():
    return render_template('cyclone.html')

@app.route('/flood')
def flood():
    return render_template('flood.html')

@app.route('/contact')
def contact():
    return render_template('contactus.html')

@app.route('/esm')
def esm():
    return render_template('esm.html')

@app.route('/fs')
def fs():
    return render_template('Safety.html')

@app.route('/cs')
def cs():
    return render_template('CS.html')

@app.route('/tech')
def tech():
    return render_template('tech.html')

# Prediction Routes
@app.route('/pre', methods=['POST'])
def pre():
    if earthquake_model is None:
        return render_template('earthquake.html', prediction_text="Model not available.")
    try:
        # Convert input values to float
        val = [float(x) for x in request.form.values()]
        # Predict category directly ('Low', 'Medium', 'High')
        prediction = earthquake_model.predict([val])
        pred = prediction[0]  # No need to convert to float, it's already a string

        # Message based on predicted category
        if pred == 'Low':
            msg = "LOW chances of earthquake"
        elif pred == 'Medium':
            msg = "MODERATE chances of earthquake"
        else:
            msg = "HIGH chances of earthquake"

        return render_template('earthquake.html', prediction_text=f"Predicted Risk Level: {pred}. {msg}")
    except Exception as e:
        return render_template('earthquake.html', prediction_text=f"Error: {e}")

@app.route('/flo', methods=['POST'])
def flo():
    if flood_model is None:
        return render_template('flood.html', prediction_text="Model not available.")
    try:
        val = [float(x) for x in request.form.values()]
        prediction = flood_model.predict([val])[0]

        if prediction == 1:
            msg = "⚠️ HIGH chances of flood occurrence!"
        else:
            msg = "✅ No flood risk detected."

        return render_template('flood.html', prediction_text=msg)
    except Exception as e:
        return render_template('flood.html', prediction_text=f"Error: {e}")

@app.route('/cyl', methods=['POST'])
def cyl():
    if cyclone_model is None:
        return render_template('cyclone.html', prediction_text="Model not available.")
    try:
        lat = float(request.form.get('latitude'))
        lon = float(request.form.get('longitude'))
        wind = float(request.form.get('wind'))
        pressure = float(request.form.get('pressure'))

        val = [[lat, lon, wind, pressure]]
        prediction = cyclone_model.predict(val)
        pred = int(prediction[0])

        risk_map = {
            0: "LOW chances of hurricane occurrence",
            1: "MODERATE hurricane, possible tropical depression",
            2: "HIGH chances of a strong hurricane, be alert!"
        }

        return render_template('cyclone.html', prediction_text=risk_map.get(pred, "Unknown risk level"))
    except Exception as e:
        return render_template('cyclone.html', prediction_text=f"Error: {e}")

# AI Query Route
@app.route('/ask', methods=['POST'])
def ask():
    user_question = request.form.get("question")

    if not user_question:
        return render_template('query.html', user_question=user_question, answer="Please enter a question.")

    if not gpt4all_model:
        return render_template('query.html', user_question=user_question, answer="AI model not available.")

    try:
        prompt = f"You are an assistant that gives helpful and safety-related answers about natural disasters.\nUser: {user_question}\nAssistant:"
        with gpt4all_model.chat_session():
            response = gpt4all_model.generate(prompt, temp=0.7)
        return render_template('query.html', user_question=user_question, answer=response.strip())
    except Exception as e:
        return render_template('query.html', user_question=user_question, answer=f"Error fetching response: {str(e)}")

# Start Flask App
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
