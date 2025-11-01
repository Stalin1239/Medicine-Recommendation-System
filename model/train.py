import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Load data
df = pd.read_csv('data/symptoms_medicines.csv')
X = df[['fever', 'cough', 'headache', 'fatigue']]
y = df['medicine']

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model
os.makedirs('model', exist_ok=True)
joblib.dump(model, 'model/model.pkl')
print("✅ Model trained!")
