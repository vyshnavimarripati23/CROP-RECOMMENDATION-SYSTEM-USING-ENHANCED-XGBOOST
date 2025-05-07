import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("Crop_recommendation.csv")

# Check column names (ensure correct spelling)
print("Column Names:", df.columns)

# Features and target
X = df.drop(columns=['label'])  # Ensure correct column name
y = df['label']

# Apply MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)


# Train model
model = RandomForestClassifier()
model.fit(X_scaled, y)

# Save the trained model
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Save the MinMaxScaler
with open('minmaxscaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

print("✅ Model and MinMaxScaler Saved Successfully!")
