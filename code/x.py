import numpy as np
import tkinter as tk
from tkinter import ttk
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import pickle

# Load the trained model
model = load_model("ipllstm.h5")

# Load pre-trained scalers (Ensure you have these saved from training)
with open("scaler_x.pkl", "rb") as f:
    scaler_x = pickle.load(f)
with open("scaler_y.pkl", "rb") as f:
    scaler_y = pickle.load(f)

def predict_score():
    try:
        # Get user inputs
        bat_team = team_var.get()
        bowl_team = team_var2.get()
        venue = venue_var.get()
        overs = float(overs_entry.get())
        runs = int(runs_entry.get())
        wickets = int(wickets_entry.get())
        runs_last5 = int(runs_last5_entry.get())
        wickets_last5 = int(wickets_last5_entry.get())

        # One-hot encoding for teams and venue
        bat_team_encoded = [1 if team == bat_team else 0 for team in team_options]
        bowl_team_encoded = [1 if team == bowl_team else 0 for team in team_options]
        venue_encoded = [1 if v == venue else 0 for v in venue_options]

        # Combine features into an input array
        input_data = [overs, runs, wickets, runs_last5, wickets_last5] + bat_team_encoded + bowl_team_encoded + venue_encoded
        input_data = np.array(input_data).reshape(1, -1)
        
        # Normalize input
        input_data = scaler_x.transform(input_data)
        
        # Reshape for LSTM (Assuming timesteps=1)
        input_data = np.reshape(input_data, (1, 1, input_data.shape[1]))  
        
        # Make prediction
        predicted_score = model.predict(input_data)
        predicted_score = scaler_y.inverse_transform(predicted_score.reshape(-1, 1))[0][0]
        
        # Display result
        result_label.config(text=f"Predicted Score: {predicted_score:.2f}")
    except Exception as e:
        result_label.config(text=f"Error: {str(e)}")

# GUI Setup
root = tk.Tk()
root.title("IPL Score Predictor")
root.geometry("500x400")

team_options = ["MI", "CSK", "RCB", "KKR", "SRH", "DC", "KXIP", "RR", "GT", "LSG"]
venue_options = ["Wankhede", "Chinnaswamy", "Eden Gardens", "Feroz Shah Kotla", "M. A. Chidambaram"]

tk.Label(root, text="Batting Team").pack()
team_var = ttk.Combobox(root, values=team_options)
team_var.pack()

tk.Label(root, text="Bowling Team").pack()
team_var2 = ttk.Combobox(root, values=team_options)
team_var2.pack()

tk.Label(root, text="Venue").pack()
venue_var = ttk.Combobox(root, values=venue_options)
venue_var.pack()

tk.Label(root, text="Overs").pack()
overs_entry = tk.Entry(root)
overs_entry.pack()

tk.Label(root, text="Runs").pack()
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

tk.Button(root, text="Predict", command=predict_score).pack()

result_label = tk.Label(root, text="", font=("Arial", 14))
result_label.pack()

root.mainloop()
