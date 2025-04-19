import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

# === Load trained model and preprocessing tools ===
model = tf.keras.models.load_model("ipl_lstm_best_model.h5", compile=False)
#model = tf.keras.models.load_model("ipl_lstm_model_two.h5", custom_objects={})
model.compile(optimizer="adam", loss="mse")

preprocessor = joblib.load("preprocessor.save")
scaler_x = joblib.load("scaler_x.save")
scaler_y = joblib.load("scaler_y.save")
le_batsman = joblib.load("le_batsman.save")
le_bowler = joblib.load("le_bowler.save")

df = pd.read_csv("ipl_dataset_2.csv")

venues = sorted(df["venue"].unique().tolist())
bat_teams = sorted(df["bat_team"].unique().tolist())
bowl_teams = sorted(df["bowl_team"].unique().tolist())
batsmen = sorted(df["batsman"].unique().tolist())
bowlers = sorted(df["bowler"].unique().tolist())
# === GUI Window ===
root = tk.Tk()
root.title("IPL Score Predictor")
root.geometry("400x600")

def safe_encode(label, encoder):
    if label in encoder.classes_:
        return encoder.transform([label])[0]
    else:
        return np.nanmean(encoder.transform(encoder.classes_))

def predict():
    try:
        actual_overs = int(over_var.get()) + (int(ball_var.get()) / 6)
        input_dict = {
            "venue": [venue_var.get()],
            "bat_team": [bat_team_var.get()],
            "bowl_team": [bowl_team_var.get()],
            "batsman": [safe_encode(batsman_var.get(), le_batsman)],
            "bowler": [safe_encode(bowler_var.get(), le_bowler)],
            "overs": [actual_overs],
            "runs": [int(runs_var.get())],
            "wickets": [int(wickets_var.get())],
            "runs_last_5": [int(runs_last5_var.get())],
            "wickets_last_5": [int(wickets_last5_var.get())],
            "runrate": [float(runrate_var.get())]
        }
        
        input_df = pd.DataFrame(input_dict)
        X_encoded = preprocessor.transform(input_df).toarray()
        X_scaled = scaler_x.transform(X_encoded)
        SEQ_LEN = 10
        X_seq = np.repeat(X_scaled, SEQ_LEN, axis=0).reshape(1, SEQ_LEN, -1).astype(np.float32)
        
        y_pred = model.predict(X_seq)
        y_pred_actual = scaler_y.inverse_transform(y_pred)
        
        current_score = int(runs_var.get())
        if y_pred_actual[0][0] < current_score:
            y_pred_actual[0][0] = current_score + np.random.uniform(5, 15)
        
        messagebox.showinfo("Prediction", f"🏏 Predicted Total Score: {y_pred_actual[0][0]:.2f}")
    except Exception as e:
        messagebox.showerror("Error", f"Something went wrong: {e}")

# === GUI Components ===
venue_var = ttk.Combobox(root, values=venues, state="readonly")
venue_var.set("Select Venue")
venue_var.pack(pady=5)

bat_team_var = ttk.Combobox(root, values=bat_teams, state="readonly")
bat_team_var.set("Select Batting Team")
bat_team_var.pack(pady=5)

bowl_team_var = ttk.Combobox(root, values=bowl_teams, state="readonly")
bowl_team_var.set("Select Bowling Team")
bowl_team_var.pack(pady=5)

batsman_var = ttk.Combobox(root, values=batsmen, state="readonly")
batsman_var.set("Select Batsman")
batsman_var.pack(pady=5)

bowler_var = ttk.Combobox(root, values=bowlers, state="readonly")
bowler_var.set("Select Bowler")
bowler_var.pack(pady=5)

over_var = tk.Entry(root)
over_var.insert(0, "0")
over_var.pack(pady=5)

tk.Label(root, text="Over").pack()
ball_var = tk.Entry(root)
ball_var.insert(0, "0")
ball_var.pack(pady=5)

tk.Label(root, text="Ball").pack()
runs_var = tk.Entry(root)
runs_var.insert(0, "0")
runs_var.pack(pady=5)

tk.Label(root, text="Runs").pack()
wickets_var = tk.Entry(root)
wickets_var.insert(0, "0")
wickets_var.pack(pady=5)

tk.Label(root, text="Wickets").pack()
runs_last5_var = tk.Entry(root)
runs_last5_var.insert(0, "0")
runs_last5_var.pack(pady=5)

tk.Label(root, text="Runs Last 5 Overs").pack()
wickets_last5_var = tk.Entry(root)
wickets_last5_var.insert(0, "0")
wickets_last5_var.pack(pady=5)

tk.Label(root, text="Wickets Last 5 Overs").pack()
runrate_var = tk.Entry(root)
runrate_var.insert(0, "0")
runrate_var.pack(pady=5)

tk.Label(root, text="Runrate").pack()

predict_btn = tk.Button(root, text="Predict Total Score 🚩", command=predict, bg="green", fg="white")
predict_btn.pack(pady=10)

root.mainloop()