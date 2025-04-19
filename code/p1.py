import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import ttk, messagebox
import ttkbootstrap as tb  # Modern UI theme
from ttkbootstrap.constants import *
from PIL import Image, ImageTk  # Image handling
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load dataset
df = pd.read_csv("ipl_data.csv")  # Ensure this file exists

# Load trained model
model = load_model("ip.h5")

# Create and fit encoders
venue_encoder = LabelEncoder()
batting_team_encoder = LabelEncoder()
bowling_team_encoder = LabelEncoder()
striker_encoder = LabelEncoder()
bowler_encoder = LabelEncoder()

df['venue'] = venue_encoder.fit_transform(df['venue'])
df['bat_team'] = batting_team_encoder.fit_transform(df['bat_team'])
df['bowl_team'] = bowling_team_encoder.fit_transform(df['bowl_team'])
df['batsman'] = striker_encoder.fit_transform(df['batsman'])
df['bowler'] = bowler_encoder.fit_transform(df['bowler'])

# Fit the scaler
scaler = StandardScaler()
scaler.fit(df[['venue', 'bat_team', 'bowl_team', 'batsman', 'bowler']])

# Create GUI window
root = tb.Window(themename="superhero")
root.title("IPL Score Predictor")
root.geometry("600x600")
root.resizable(True, True)

# Background Image with Transparency
bg_image = Image.open("ipl.jpeg").convert("RGBA")
bg_image = bg_image.resize((1600, 1800), Image.Resampling.LANCZOS)

data = np.array(bg_image)
data[..., 3] = 150  # Set transparency level
bg_image = Image.fromarray(data)
bg_photo = ImageTk.PhotoImage(bg_image)

canvas = tk.Canvas(root, width=600, height=600)
canvas.pack(fill="both", expand=True)
canvas.create_image(0, 0, image=bg_photo, anchor="nw")

# UI Frame (to keep content above background)
frame = tb.Frame(root, padding=10, bootstyle=LIGHT)
frame.place(relx=0.5, rely=0.5, anchor="center")

tb.Label(frame, text="IPL Score Predictor", font=("Arial", 18, "bold"), bootstyle=PRIMARY).pack(pady=10)

# Dropdown function
def create_dropdown(label, values):
    tb.Label(frame, text=label, font=("Arial", 12)).pack(pady=5)
    var = tk.StringVar()
    formatted_values = [v.strip() for v in values]  # Ensure full names display properly
    dropdown = tb.Combobox(frame, textvariable=var, values=formatted_values, state="readonly", width=40, bootstyle=SUCCESS)
    dropdown.pack(pady=5)
    dropdown.current(0)
    return var, dropdown

venue_var, venue_dropdown = create_dropdown("Select Venue:", venue_encoder.classes_)
batting_team_var, batting_team_dropdown = create_dropdown("Select Batting Team:", batting_team_encoder.classes_)
bowling_team_var, bowling_team_dropdown = create_dropdown("Select Bowling Team:", bowling_team_encoder.classes_)
striker_var, striker_dropdown = create_dropdown("Select Striker:", striker_encoder.classes_)
bowler_var, bowler_dropdown = create_dropdown("Select Bowler:", bowler_encoder.classes_)

# Prediction Label
prediction_label = tb.Label(frame, text="", font=("Arial", 16, "bold"), bootstyle=DANGER)
prediction_label.pack(pady=10)

# Prediction Function
def predict_score():
    try:
        venue_value = venue_encoder.transform([venue_var.get()])[0]
        batting_team_value = batting_team_encoder.transform([batting_team_var.get()])[0]
        bowling_team_value = bowling_team_encoder.transform([bowling_team_var.get()])[0]
        striker_value = striker_encoder.transform([striker_var.get()])[0]
        bowler_value = bowler_encoder.transform([bowler_var.get()])[0]

        input_array = np.array([[venue_value, batting_team_value, bowling_team_value, striker_value, bowler_value]])
        input_scaled = scaler.transform(input_array)
        predicted_score = model.predict(input_scaled)
        predicted_score = int(predicted_score[0, 0])

        prediction_label.config(text=f"Predicted Score: {predicted_score}", bootstyle=SUCCESS)
    except Exception as e:
        messagebox.showerror("Error", "Please select valid inputs.")

# Predict Button with Hover Effect
def on_enter(e):
    predict_button.config(bootstyle=INFO)

def on_leave(e):
    predict_button.config(bootstyle=PRIMARY)

predict_button = tb.Button(frame, text="Predict Score", bootstyle=PRIMARY, command=predict_score)
predict_button.pack(pady=20)

predict_button.bind("<Enter>", on_enter)
predict_button.bind("<Leave>", on_leave)

# Run the GUI
root.mainloop()
