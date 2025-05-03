# Nature-alert: natural-disaster-prediction-system-with-llm

A **Flask-based AI web application** that predicts the risk levels of **earthquakes, floods, and cyclones** using machine learning models. It also includes an AI chatbot powered by a **local GPT4All LLM (Mistral-7B-Instruct)** to provide safety-related answers to disaster-related queries — all offline and secure.

---

## 🚀 Features

- 🌍 Earthquake risk level prediction (`Low`, `Medium`, `High`)
- 🌊 Flood occurrence detection (Yes/No)
- 🌀 Cyclone severity classification (0: Low, 1: Medium, 2: High)
- 🤖 AI chatbot using GPT4All (Mistral 7B) for disaster safety Q&A
- 🔐 Runs locally with no cloud dependencies
- 🧠 ML models loaded from `.pkl` files using `joblib`

---

## 🛠️ Tech Stack

- Python 3.x
- Flask (Web framework)
- GPT4All (Mistral 7B Instruct model)
- Joblib, NumPy, Python-dotenv
- HTML (Jinja2 templates)
- Local machine learning models (`.pkl`)

---

## 📁 Folder Structure

disaster-prediction-ai/

├── app.py                           # Main Flask application

├── requirements.txt                # Required Python packages

├── .env.example                    # Sample environment variables (safe for GitHub)

├── README.md                       # Project documentation

├── Dockerfile                      # (Optional) For containerizing the app

├── .dockerignore                   # Ignore unnecessary files in Docker builds


├── models/                         # Machine Learning models and training code

│   ├── cyclone_model.pkl           # Cyclone prediction model

│   ├── earthquake_model.pkl        # Earthquake prediction model

│   ├── flood_model.pkl             # Flood prediction model

│   └── train_models/               # Training code for models

│       ├── train_cyclone_model.py  # Cyclone model training script

│       ├── train_earthquake_model.py # Earthquake model training script

│       └── train_flood_model.py    # Flood model training script

├── static/                         # Static files (CSS, images, etc.)

│   └── css/                        # CSS files for styling

│       ├── cyclone.css             # Styling for cyclone page

│       ├── earthquake.css          # Styling for earthquake page

│       ├── flood.css               # Styling for flood page

│       ├── homepage.css            # Homepage styling

│       ├── query.css               # Query page styling

│       ├── safety.css              # Safety information page styling

│       └── style.css               # General styling

├── templates/                      # HTML templates for the app

│   ├── ai_query.html               # Template for AI query page

│   ├── cyclone.html                # Template for cyclone prediction page

│   ├── earthquake.html             # Template for earthquake prediction page

│   ├── flood.html                  # Template for flood prediction page

│   ├── homepage.html               # Homepage template

│   ├── query.html                  # Template for query page

│   └── safety.html                 # Template for safety information page


-----

# 🚀 **Setup & Installation**
Follow the steps below to set up and run the project locally on your machine:

# 1️⃣ **Clone the Repository**

git clone https://github.com/Divya-Dharsh/Nature-alert-natural-disaster-prediction-system-using-llm-.git

cd disaster-prediction-ai

# 2️⃣ **Set Up Virtual Environment**

# For Windows
python -m venv .venv

.venv\Scripts\activate

# For macOS/Linux
python3 -m venv .venv

source .venv/bin/activate

# 3️⃣ **Install Dependencies**
pip install -r requirements.txt

# 4️⃣ **Download GPT4All Model**
You need the mistral-7b-instruct-v0.1.Q4_0.gguf model file.

Download it from GPT4All Downloads

Place it in the root of your project (same folder as app.py)

⚠️ Note: GPT4All model can be large (~4-8GB). Make sure your machine has sufficient memory.

# 5️⃣ **Configure Environment Variables**
Create a .env file in the root directory (you can copy from .env.example):

**.env**

FLASK_ENV=development

SECRET_KEY=your-secret-key

# 6️⃣ **Run the App**

python app.py

Visit http://localhost:5000 in your browser to access the app.

------
so that's it , feel happy to share !!!
