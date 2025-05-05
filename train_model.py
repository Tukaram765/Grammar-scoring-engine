import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import joblib

# --- Load Data ---
train_transcripts = pd.read_csv("train_transcripts.csv")
test_transcripts = pd.read_csv("test_transcripts.csv")
train_audio = pd.read_csv("train_audio_features.csv")
test_audio = pd.read_csv("test_audio_features.csv")
labels_df = pd.read_csv("dataset/train.csv").rename(columns={"label": "grammar"})

# Merge audio + transcript features
train_df = pd.merge(train_transcripts, train_audio, on="filename")
test_df = pd.merge(test_transcripts, test_audio, on="filename")

# Merge grammar labels
train_df = pd.merge(train_df, labels_df[["filename", "grammar"]], on="filename")

# --- Preprocessing ---
X = train_df.drop(columns=["filename", "transcript", "grammar"])
y = train_df["grammar"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# --- Model Training ---
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# --- Evaluation ---
y_pred = model.predict(X_val_scaled)
pearson_corr, _ = pearsonr(y_val, y_pred)
rmse = mean_squared_error(y_val, y_pred) ** 0.5

print(f"✅ Pearson Correlation: {pearson_corr:.4f}")
print(f"✅ RMSE: {rmse:.4f}")

# --- Inference on Test Set ---
X_test = test_df.drop(columns=["filename", "transcript"])
X_test_scaled = scaler.transform(X_test)
test_predictions = model.predict(X_test_scaled)

# --- Save Submission ---
submission_df = pd.DataFrame({
    "filename": test_df["filename"],
    "label": test_predictions.round(1)
})
submission_df.to_csv("submission.csv", index=False)
print("✅ submission.csv saved.")

# --- Save Model and Scaler (Optional but useful) ---
joblib.dump(model, "grammar_model.pkl")
joblib.dump(scaler, "scaler.pkl")
