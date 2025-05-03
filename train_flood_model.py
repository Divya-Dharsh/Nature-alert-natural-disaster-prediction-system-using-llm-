import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Load the dataset
df = pd.read_csv("../FloodPrediction.csv")
df.columns = df.columns.str.strip()  # Clean column names

# Drop rows where 'Flood?' is empty
df = df.dropna(subset=['Flood?'])

# Select features and target
X = df[['Max_Temp', 'ALT', 'Rainfall']]
y = df['Flood?'].astype(int)  # Convert target to integer

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train classification model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save the model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/flood_model.pkl")
print("Saved to models/flood_model.pkl")
