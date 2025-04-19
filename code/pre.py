import pandas as pd
import numpy as np
import re
import tkinter as tk
from tkinter import ttk
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from joblib import dump, load

# Load Datasets
team_perf_df = pd.read_csv("team_performance_dataset_2008to2024.csv")
players_info_df = pd.read_csv("Players_Info_2024.csv")
ipl_teams_df = pd.read_csv("ipl_teams_2024_info.csv")

# Data Cleaning
def clean_salary(salary):
    if isinstance(salary, str):
        salary = salary.lower().replace("us$", "").replace("\xa0", "").strip()
        salary = re.sub(r"[^\d\.a-z ]", "", salary)
        match = re.search(r"[\d\.]+", salary)
        if match:
            num = float(match.group())
            if "crore" in salary:
                return num * 10_000_000
            elif "lakh" in salary:
                return num * 100_000
            else:
                return num
    return 0

players_info_df["Player Salary"] = players_info_df["Player Salary"].apply(clean_salary)
scaler = MinMaxScaler()
players_info_df["Player Salary"] = scaler.fit_transform(players_info_df[["Player Salary"]])

# Encoding Team Names
encoder = LabelEncoder()
team_perf_df["Match_Winner"] = encoder.fit_transform(team_perf_df["Match_Winner"])

# Feature Selection
X = team_perf_df.drop(columns=["Match_Winner"])
y = team_perf_df["Match_Winner"]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Random Forest Model
rf = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=2, random_state=42)
rf.fit(X_train, y_train)
dump(rf, "rf_model.joblib")

# XGBoost Model
xgb_model = xgb.XGBClassifier(objective="multi:softmax", num_class=len(set(y)), eval_metric="mlogloss")
xgb_model.fit(X_train, y_train)
dump(xgb_model, "xgb_model.joblib")

# LSTM Model
X_train_lstm = np.expand_dims(X_train, axis=-1)
X_test_lstm = np.expand_dims(X_test, axis=-1)
lstm_model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(50),
    Dense(len(set(y)), activation="softmax")
])
lstm_model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
lstm_model.fit(X_train_lstm, y_train, epochs=10, batch_size=16)
lstm_model.save("lstm_model.h5")

# GUI Application
def predict_score():
    team = team_var.get()
    model_choice = model_var.get()
    input_features = np.array([[float(entry.get()) for entry in feature_entries]])
    
    if model_choice == "Random Forest":
        model = load("rf_model.joblib")
        prediction = model.predict(input_features)[0]
    elif model_choice == "XGBoost":
        model = load("xgb_model.joblib")
        prediction = model.predict(input_features)[0]
    elif model_choice == "LSTM":
        model = load_model("lstm_model.h5")
        input_features_expanded = np.expand_dims(input_features, axis=-1)
        prediction = np.argmax(model.predict(input_features_expanded))
    
    result_label.config(text=f"Predicted Score: {prediction}")

root = tk.Tk()
root.title("IPL Score Prediction")
root.geometry("400x400")

tk.Label(root, text="Select Team:").pack()
team_var = tk.StringVar()
team_dropdown = ttk.Combobox(root, textvariable=team_var, values=ipl_teams_df["Team"].tolist())
team_dropdown.pack()

tk.Label(root, text="Enter Features:").pack()
feature_entries = []
for col in X.columns:
    tk.Label(root, text=col).pack()
    entry = tk.Entry(root)
    entry.pack()
    feature_entries.append(entry)

tk.Label(root, text="Select Model:").pack()
model_var = tk.StringVar()
model_dropdown = ttk.Combobox(root, textvariable=model_var, values=["Random Forest", "XGBoost", "LSTM"])
model_dropdown.pack()

tk.Button(root, text="Predict", command=predict_score).pack()
result_label = tk.Label(root, text="")
result_label.pack()

root.mainloop()
