import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import os

# Load and clean data
df = pd.read_csv("../atlantic.csv")

# Drop non-numeric features
df = df.drop(columns=['ID', 'Name', 'Date', 'Time', 'Event', 'Status'])

# Clean and convert coordinates
df['Latitude'] = df['Latitude'].str.replace('[^0-9.]', '', regex=True).astype(float)
df['Longitude'] = df['Longitude'].str.replace('[^0-9.-]', '', regex=True).astype(float)

# Binning Maximum Wind into Risk Levels
def wind_to_risk(wind):
    if wind < 64:
        return 0  # Low
    elif wind < 96:
        return 1  # Medium
    else:
        return 2  # High

df['RiskLevel'] = df['Maximum Wind'].apply(wind_to_risk)

# Features and target
X = df[['Latitude', 'Longitude', 'Maximum Wind', 'Minimum Pressure']]
y = df['RiskLevel']

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/cyclone_model.pkl")
print("Model saved to models/cyclone_model.pkl")
