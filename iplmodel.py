import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import keras_tuner as kt
import matplotlib.pyplot as plt

# === LOAD DATA ===
df = pd.read_csv("ipl_dataset_2.csv")

# Drop unnecessary columns
drop_cols = ["mid", "date", "delivery_type"]
df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

# Fix datatypes
df["overs"] = pd.to_numeric(df["overs"], errors="coerce").fillna(0).astype(float)

# Handle Outliers
num_cols = ["runs", "wickets", "runs_last_5", "wickets_last_5", "runrate"]
for col in num_cols:
    df[col] = np.clip(df[col], df[col].quantile(0.01), df[col].quantile(0.99))

# === LABEL ENCODING ===
le_batsman = LabelEncoder()
le_bowler = LabelEncoder()
df['batsman'] = le_batsman.fit_transform(df['batsman'])
df['bowler'] = le_bowler.fit_transform(df['bowler'])

joblib.dump(le_batsman, "le_batsman.save")
joblib.dump(le_bowler, "le_bowler.save")

# === ONE HOT ENCODER ===
categorical_cols = ["bat_team", "bowl_team", "venue"]
preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)],
    remainder='passthrough'
)

# Prepare X and y
X_raw = df.drop(columns=["total"], errors='ignore')
y_raw = df.get("total", pd.Series()).values.reshape(-1, 1)

# Encode X
X_encoded = preprocessor.fit_transform(X_raw)
joblib.dump(preprocessor, "preprocessor.save")

# === SCALING ===
scaler_x = StandardScaler(with_mean=False)
scaler_y = StandardScaler()
X_scaled = scaler_x.fit_transform(X_encoded).toarray()
y_scaled = scaler_y.fit_transform(y_raw)

joblib.dump(scaler_x, "scaler_x.save")
joblib.dump(scaler_y, "scaler_y.save")

# === CREATE SEQUENCES ===
SEQ_LEN = 10
X_seq, y_seq = [], []

for i in range(SEQ_LEN, X_scaled.shape[0]):
    X_seq.append(X_scaled[i-SEQ_LEN:i])
    y_seq.append(y_scaled[i])

X_seq, y_seq = np.array(X_seq), np.array(y_seq)

# === TRAIN TEST SPLIT ===
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

# === HYPERPARAMETER TUNING FUNCTION ===
def build_model(hp):
    model = Sequential([
        LSTM(hp.Int('lstm1_units', min_value=64, max_value=256, step=64), 
             activation="tanh", return_sequences=True, input_shape=(SEQ_LEN, X_seq.shape[2])),
        Dropout(hp.Float('dropout1', 0.1, 0.3, step=0.05)),
        LSTM(hp.Int('lstm2_units', min_value=32, max_value=128, step=32), activation="tanh"),
        Dropout(hp.Float('dropout2', 0.1, 0.3, step=0.05)),
        Dense(hp.Int('dense_units', min_value=16, max_value=64, step=16), activation="relu"),
        Dense(1, activation="linear")  # Output layer for regression
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=hp.Choice('learning_rate', [0.001, 0.0005, 0.0001])
        ),
        loss="mse"
    )
    
    return model

# === RUN HYPERPARAMETER SEARCH ===
tuner = kt.RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=4,  # Number of hyperparameter combinations to try
    executions_per_trial=1,
    directory='hyperparam_tuning',
    project_name='ipl_lstm_tuning'
)

# Perform hyperparameter search
tuner.search(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=32)

# Get the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# === BUILD AND TRAIN BEST MODEL ===
best_model = tuner.hypermodel.build(best_hps)

history = best_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32)

# === SAVE THE BEST MODEL ===
best_model.save("ipl_lstm_best_model.h5")

# === EVALUATION ===
y_pred = best_model.predict(X_test)
y_pred_actual = scaler_y.inverse_transform(y_pred)
y_test_actual = scaler_y.inverse_transform(y_test)

rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
mae = mean_absolute_error(y_test_actual, y_pred_actual)
r2 = r2_score(y_test_actual, y_pred_actual)
mape = np.mean(np.abs((y_test_actual - y_pred_actual) / y_test_actual)) * 100
accuracy = 100 - mape

print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R² Score: {r2:.2f}")
print(f"Accuracy (100 - MAPE): {accuracy:.2f}%")

# === PLOT TRAINING LOSS ===
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training & Validation Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)
plt.show()

# === PLOT ACTUAL vs PREDICTED ===
plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_test_actual)), y_test_actual, label='Actual', marker='o', alpha=0.7)
plt.scatter(range(len(y_pred_actual)), y_pred_actual, label='Predicted', marker='x', alpha=0.7)
plt.title("Actual vs Predicted Total Score")
plt.xlabel("Sample Index")
plt.ylabel("Total Score")
plt.legend()
plt.grid(True)
plt.show()
