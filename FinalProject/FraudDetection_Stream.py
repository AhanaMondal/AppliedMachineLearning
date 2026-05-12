import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

import xgboost as xgb

# =========================
# 1. Load Dataset
# =========================
df = pd.read_csv(r"C:\Users\Ahana\Downloads\creditcard_fraud_detection.csv")
df = df.sort_values(by="Time").reset_index(drop=True)

# =========================
# 2. Feature Engineering (Velocity)
# =========================

# Delta Time
df["delta_time"] = df["Time"].diff().fillna(0)

# Global Velocity (transactions in last 30 seconds)
window = 30
df["global_velocity"] = df["Time"].rolling(window=window).count().fillna(0)

# Hour of day
df["hour"] = (df["Time"] // 3600) % 24

# Amount Rarity (z-score per hour)
df["amount_mean_hour"] = df.groupby("hour")["Amount"].transform("mean")
df["amount_std_hour"] = df.groupby("hour")["Amount"].transform("std").replace(0, 1)

df["amount_zscore"] = (df["Amount"] - df["amount_mean_hour"]) / df["amount_std_hour"]

# Drop helper cols
df.drop(["amount_mean_hour", "amount_std_hour"], axis=1, inplace=True)

# =========================
# 3. Train / Stream Split
# =========================
split_idx = int(len(df) * 0.8)

train_df = df.iloc[:split_idx]
stream_df = df.iloc[split_idx:]

X_train = train_df.drop("Class", axis=1)
y_train = train_df["Class"]

X_stream = stream_df.drop("Class", axis=1)
y_stream = stream_df["Class"]

# =========================
# 4. Scaling
# =========================
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_stream_scaled = scaler.transform(X_stream)

# =========================
# 5. Clustering (on training only)
# =========================
kmeans = KMeans(n_clusters=10, random_state=42)
clusters = kmeans.fit_predict(X_train_scaled)

train_df["cluster"] = clusters

# Fraud density per cluster
cluster_fraud_density = train_df.groupby("cluster")["Class"].mean().to_dict()

# =========================
# 6. Autoencoder (normal only)
# =========================
X_train_normal = X_train_scaled[y_train == 0]

input_dim = X_train_normal.shape[1]

input_layer = Input(shape=(input_dim,))
encoded = Dense(16, activation="relu")(input_layer)
encoded = Dense(8, activation="relu")(encoded)

decoded = Dense(16, activation="relu")(encoded)
decoded = Dense(input_dim, activation="sigmoid")(decoded)

autoencoder = Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss="mse")

autoencoder.fit(
    X_train_normal,
    X_train_normal,
    epochs=10,
    batch_size=256,
    shuffle=True,
    verbose=1
)

# =========================
# 7. Reconstruction Error (train)
# =========================
reconstructions = autoencoder.predict(X_train_scaled)
train_errors = np.mean(np.square(X_train_scaled - reconstructions), axis=1)

# =========================
# 8. Cluster Features (train)
# =========================
cluster_ids = kmeans.predict(X_train_scaled)

cluster_density_feature = np.array([
    cluster_fraud_density.get(cid, 0) for cid in cluster_ids
])

cluster_distance = np.min(
    kmeans.transform(X_train_scaled), axis=1
)

# =========================
# 9. Final Training Features
# =========================
X_train_final = np.hstack((
    X_train_scaled,
    train_errors.reshape(-1, 1),
    cluster_density_feature.reshape(-1, 1),
    cluster_distance.reshape(-1, 1)
))

# =========================
# 10. Train XGBoost
# =========================
scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)

xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    use_label_encoder=False
)

xgb_model.fit(X_train_final, y_train)

# =========================
# 11. STREAMING SIMULATION
# =========================
y_preds = []
fraud_detected = 0
total_fraud = 0

for i in range(len(X_stream_scaled)):
    x = X_stream_scaled[i].reshape(1, -1)

    # Autoencoder error
    recon = autoencoder.predict(x, verbose=0)
    error = np.mean(np.square(x - recon))

    # Cluster features
    cluster_id = kmeans.predict(x)[0]
    density = cluster_fraud_density.get(cluster_id, 0)
    distance = np.min(kmeans.transform(x))

    # Final feature vector
    x_final = np.hstack((x, [[error, density, distance]]))

    # Predict
    pred = xgb_model.predict(x_final)[0]
    y_preds.append(pred)

    actual = y_stream.iloc[i]

    if actual == 1:
        total_fraud += 1

    if pred == 1:
        fraud_detected += 1
        print(f"[BLOCKED] Transaction {i} flagged as FRAUD")
    else:
        print(f"[ALLOWED] Transaction {i} is normal")

# =========================
# 12. Evaluation
# =========================
print("\n=== Streaming Performance ===")
print(classification_report(y_stream, y_preds))
print(f"Fraud detected: {fraud_detected}/{total_fraud}")

# =========================
# 13. Save Models
# =========================
import joblib

# Save scaler
joblib.dump(scaler, "scaler.pkl")

# Save XGBoost model
joblib.dump(xgb_model, "xgb_model.pkl")

# Save Autoencoder
autoencoder.save("autoencoder.h5")

# Save KMeans
joblib.dump(kmeans, "kmeans.pkl")

# Save cluster fraud density
joblib.dump(cluster_fraud_density, "cluster_density.pkl")

print("Models saved successfully.")