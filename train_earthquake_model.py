import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import os

# Load and clean
df = pd.read_csv("../earthquake (1).csv")
df = df.dropna()
df.columns = df.columns.str.strip()

# Convert 'Magnitude' column from "6 MW" ➝ 6.0 (float)
df['Magnitude'] = df['Magnitude'].astype(str).str.replace('MW', '', regex=False).astype(float)

# Map Magnitude ➝ Risk Category
def label_risk(mag):
    if mag <= 4.0:
        return 'Low'
    elif mag <= 6.0:
        return 'Medium'
    else:
        return 'High'

df['Risk'] = df['Magnitude'].apply(label_risk)

# Rename 'Depth' ➝ 'Gap'
df.rename(columns={'Depth': 'Gap'}, inplace=True)

# Features and labels
X = df[['Latitude', 'Longitude', 'Gap']]
y = df['Risk']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train classifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluation (optional)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/earthquake_model.pkl")
print("saved! to models/earthquake_model.pkl ")
