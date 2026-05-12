import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load everything ONCE
scaler = joblib.load("scaler.pkl")
xgb_model = joblib.load("xgb_model.pkl")
kmeans = joblib.load("kmeans.pkl")
cluster_density = joblib.load("cluster_density.pkl")
autoencoder = load_model("autoencoder.h5", compile=False)

# Simple memory for delta_time (not perfect, but works)
last_time = None

def feature_engineering(transaction):
    global last_time

    x = np.array(transaction).reshape(1, -1)

    time = x[0][0]
    amount = x[0][1]

    # ---- delta_time ----
    if last_time is None:
        delta_time = 0
    else:
        delta_time = time - last_time

    last_time = time

    # ---- global_velocity (simplified) ----
    global_velocity = 1

    # ---- hour ----
    hour = (time // 3600) % 24

    # ---- amount_zscore (simplified placeholder) ----
    amount_zscore = 0

    # Append new features (IMPORTANT: same order as training)
    x = np.append(x, [delta_time, global_velocity, hour, amount_zscore])
    return x.reshape(1, -1)


def predict_transaction(transaction):
    # Step 1: feature engineering
    x = feature_engineering(transaction)

    # Step 2: scaling
    x_scaled = scaler.transform(x)

    # Step 3: autoencoder error
    recon = autoencoder.predict(x_scaled)
    error = np.mean(np.square(x_scaled - recon))

    # Step 4: clustering
    cluster_id = kmeans.predict(x_scaled)[0]
    density = cluster_density.get(cluster_id, 0)
    distance = np.min(kmeans.transform(x_scaled))

    # Step 5: final features
    x_final = np.hstack((x_scaled, [[error, density, distance]]))

    # Step 6: prediction
    pred = xgb_model.predict(x_final)[0]

    return {
        "fraud": int(pred),
        "status": "BLOCKED" if pred == 1 else "ALLOWED"
    }