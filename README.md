# WEARABLE-COACH-FOR-OPTIMAL-EXERCISE-PERFORMANCE
Spinal cord injuries (SCI) often lead to profound loss of mobility and sensation, underscoring  the critical need for effective rehabilitation.
##ML MODELS
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
data = pd.read_csv('/content/pad_dataset1.csv')
X = data.drop(columns=['output','time index'])
y = data['output']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = XGBClassifier()
model.fit(X_train, y_train)
y_proba = model.predict_proba(X_test)
y_pred_max_proba = y_proba.max(axis=1)
y_pred_classes = y_proba.argmax(axis=1)
for i, (pred_class, max_prob) in enumerate(zip(y_pred_classes, y_pred_max_proba)):
 print(f"Instance {i + 1}: Predicted class is {pred_class} with the probability {max_prob * 
100:.2f}%")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))
new_sample = [[20.61, -25.3, -0.06, -23.93, 40.54, -48.07]]
new_df = pd.DataFrame(new_sample, columns=X.columns)
new_proba = model.predict_proba(new_df)
new_pred_max_proba = new_proba.max(axis=1)
new_pred_classes = new_proba.argmax(axis=1)
for i, (pred_class, max_prob) in enumerate(zip(new_pred_classes, new_pred_max_proba)):
 print(f"New instance {i + 1}: Predicted class is {pred_class} with the probability {max_prob 
* 100:.2f}%")
import pickle
# Assuming 'model' is your trained XGBoost model
with open('spi_shirt.pkl', 'wb') as f:
 pickle.dump(model, f)
with open('spi_shirt.pkl', 'rb') as f:
 loaded_model = pickle.load(f)
import numpy as np
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import requests
import pickle
# Load your machine learning model using pickle
with open('/content/spi_shirt.pkl', 'rb') as f:
 model = pickle.load(f)
# Define the scope and credentials to access Google Sheets and Google Drive
scope = ['https://www.googleapis.com/auth/spreadsheets',
 'https://www.googleapis.com/auth/drive']
credentials = ServiceAccountCredentials.from_json_keyfile_name('/content/credentials.json', 
scope)
client = gspread.authorize(credentials)
sheet = client.open("sshirt")
worksheet = sheet.get_worksheet(0)
try:
 sensor_data = worksheet.get_all_values()
except requests.exceptions.ReadTimeout as e:
 print("Timeout while reading data from Google Sheet:", e)
 sensor_data = []
processed_data = []
# Iterate through each row in sensor_data
for row in sensor_data:
 # Check if all elements in the row are empty strings
 if all(cell == '' for cell in row):
 continue # Skip empty rows
 # Convert each element in the row to float
 processed_row = []
 for cell in row:
 processed_row.append(float(cell))
 processed_data.append(processed_row)
if processed_data:
 y_proba = model.predict_proba(np.array(processed_data))
 y_pred_max_proba = y_proba.max(axis=1)
 y_pred_classes = y_proba.argmax(axis=1)
 for i, (pred_class, max_prob) in enumerate(zip(y_pred_classes, y_pred_max_proba)):
 print(f"Instance {i + 1}: Predicted class is {pred_class} with the probability {max_prob * 
100:.2f}%")
 y_pred = model.predict(np.array(processed_data))
else:
 print("No valid sensor data found in the Google Sheet.")
