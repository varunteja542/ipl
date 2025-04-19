import os
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import joblib

# ======= Load Model =======
try:
    model = tf.keras.models.load_model("ipllstm.keras")
    print("Model loaded successfully!")
except Exception as e:
    messagebox.showerror("Error", f"Failed to load model: {e}")
    exit()

# ======= Load Data & Preprocess =======
df = pd.read_csv("ipl_data.csv")
df = df.drop(columns=["mid", "date", "batsman", "bowler"], axis=1, errors="ignore")

# Convert overs to int
df["overs"] = pd.to_numeric(df["overs"], errors="coerce").astype(int)

# One-hot encode categorical features
df = pd.get_dummies(df, columns=["bat_team", "bowl_team", "venue"], drop_first=True)

# Drop NaN values
df = df.apply(pd.to_numeric, errors='coerce').dropna()

# Extract unique team names
teams = sorted(set(df.filter(like="bat_team_").columns.str.replace("bat_team_", "")))

# Ensure required columns exist
expected_columns = ['overs', 'runs', 'wickets', 'runs_last5', 'wickets_last5'] + list(df.columns[5:])
df = df.reindex(columns=expected_columns, fill_value=0)

# ======= Load or Train Scalers =======
try:
    scaler_x = joblib.load("scaler_x.pkl")
    scaler_y = joblib.load("scaler_y.pkl")
except FileNotFoundError:
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_x.fit_transform(df.drop(columns=["total"]))
    y_scaled = scaler_y.fit_transform(df[["total"]])
    joblib.dump(scaler_x, "scaler_x.pkl")
    joblib.dump(scaler_y, "scaler_y.pkl")

# ======= Prediction Function =======
def predict_score():
    try:
        # Get user input values
        overs = float(overs_entry.get())
        runs = float(runs_entry.get())
        wickets = float(wickets_entry.get())
        runs_last5 = float(runs_last5_entry.get())
        wickets_last5 = float(wickets_last5_entry.get())
        bat_team = bat_team_var.get()
        bowl_team = bowl_team_var.get()

        # Prepare input
        input_data = [overs, runs, wickets, runs_last5, wickets_last5]

        # Add one-hot encoding for selected teams
        team_features = [f"bat_team_{bat_team}", f"bowl_team_{bowl_team}"]
        for col in df.columns[5:]:
            input_data.append(1 if col in team_features else 0)

        # Scale input and reshape for LSTM
        input_scaled = scaler_x.transform([input_data])
        input_seq = input_scaled.reshape(1, input_scaled.shape[1], 1)

        # Make prediction
        pred_scaled = model.predict(input_seq)
        pred_actual = scaler_y.inverse_transform(pred_scaled)[0][0]

        # Show result
        messagebox.showinfo("Prediction", f"🏏 Predicted Total Score: {pred_actual:.2f}")
    except Exception as e:
        messagebox.showerror("Error", str(e))

# ======= GUI Setup =======
root = tk.Tk()
root.title("IPL Score Predictor")
root.geometry("400x450")

# Input Fields
tk.Label(root, text="Overs").pack()
overs_entry = tk.Entry(root)
overs_entry.pack()

tk.Label(root, text="Current Runs").pack()
runs_entry = tk.Entry(root)
runs_entry.pack()

tk.Label(root, text="Wickets").pack()
wickets_entry = tk.Entry(root)
wickets_entry.pack()

tk.Label(root, text="Runs in Last 5 Overs").pack()
runs_last5_entry = tk.Entry(root)
runs_last5_entry.pack()

tk.Label(root, text="Wickets in Last 5 Overs").pack()
wickets_last5_entry = tk.Entry(root)
wickets_last5_entry.pack()

# Batting Team Dropdown
tk.Label(root, text="Batting Team").pack()
bat_team_var = tk.StringVar()
bat_team_dropdown = ttk.Combobox(root, textvariable=bat_team_var, values=teams, state="readonly")
bat_team_dropdown.pack()

# Bowling Team Dropdown
tk.Label(root, text="Bowling Team").pack()
bowl_team_var = tk.StringVar()
bowl_team_dropdown = ttk.Combobox(root, textvariable=bowl_team_var, values=teams, state="readonly")
bowl_team_dropdown.pack()

# Predict Button
predict_btn = tk.Button(root, text="Predict Score", command=predict_score)
predict_btn.pack(pady=10)

root.mainloop()
