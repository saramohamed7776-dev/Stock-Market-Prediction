"""
Stock Market Prediction using LSTM
===================================
Author: Sarsora | Uneeq Interns - AI/ML Internship Task
Stock: Apple Inc. (AAPL) - You can change this to any stock ticker!

This project:
- Downloads real stock data using yfinance
- Prepares data with MinMaxScaler
- Builds an LSTM model with TensorFlow/Keras
- Trains on 80% data, tests on 20%
- Predicts and visualizes results
- Saves chart as output image
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving files
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# STEP 1: CONFIGURATION — Change stock here!
# ─────────────────────────────────────────────
STOCK_TICKER = "AAPL"       # Try: GOOGL, TSLA, MSFT, AMZN
START_DATE   = "2018-01-01"
END_DATE     = "2024-01-01"
SEQUENCE_LEN = 60           # Days of history the model looks back at
EPOCHS       = 25
BATCH_SIZE   = 32

print(f"\n{'='*55}")
print(f"  📈 Stock Market Prediction — {STOCK_TICKER}")
print(f"{'='*55}\n")

# ─────────────────────────────────────────────
# STEP 2: DOWNLOAD DATA
# ─────────────────────────────────────────────
print(f"[1/6] Downloading {STOCK_TICKER} stock data ({START_DATE} → {END_DATE})...")
df = yf.download(STOCK_TICKER, start=START_DATE, end=END_DATE, progress=False)

if df.empty:
    raise ValueError(f"No data found for ticker '{STOCK_TICKER}'. Check the ticker symbol.")

# Flatten MultiIndex columns if present
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

print(f"    ✅ Downloaded {len(df)} trading days of data")
print(f"    📅 Date range: {df.index[0].date()} → {df.index[-1].date()}")
print(f"    💰 Price range: ${df['Close'].min():.2f} — ${df['Close'].max():.2f}\n")

# Use Closing Price
close_prices = df[['Close']].values

# ─────────────────────────────────────────────
# STEP 3: PREPROCESS — Scale & Create Sequences
# ─────────────────────────────────────────────
print("[2/6] Preprocessing data...")

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices)

# Create sequences: X = past 60 days, y = next day's price
X, y = [], []
for i in range(SEQUENCE_LEN, len(scaled_data)):
    X.append(scaled_data[i - SEQUENCE_LEN:i, 0])
    y.append(scaled_data[i, 0])

X, y = np.array(X), np.array(y)

# Split: 80% train, 20% test
split = int(len(X) * 0.80)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Reshape for LSTM: (samples, timesteps, features)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test  = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

print(f"    ✅ Training samples:   {len(X_train)}")
print(f"    ✅ Testing  samples:   {len(X_test)}")
print(f"    ✅ Sequence length:    {SEQUENCE_LEN} days\n")

# ─────────────────────────────────────────────
# STEP 4: BUILD LSTM MODEL
# ─────────────────────────────────────────────
print("[3/6] Building LSTM model...")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

model = Sequential([
    LSTM(units=64, return_sequences=True, input_shape=(SEQUENCE_LEN, 1)),
    Dropout(0.2),
    LSTM(units=64, return_sequences=False),
    Dropout(0.2),
    Dense(units=32, activation='relu'),
    Dense(units=1)  # Predict next closing price
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()
print()

# ─────────────────────────────────────────────
# STEP 5: TRAIN MODEL
# ─────────────────────────────────────────────
print(f"[4/6] Training model ({EPOCHS} epochs)...")

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)
print(f"\n    ✅ Training complete! Ran {len(history.history['loss'])} epochs\n")

# ─────────────────────────────────────────────
# STEP 6: PREDICT & EVALUATE
# ─────────────────────────────────────────────
print("[5/6] Making predictions and evaluating...")

y_pred_scaled = model.predict(X_test, verbose=0)

# Inverse transform to get real dollar values
y_pred = scaler.inverse_transform(y_pred_scaled)
y_true = scaler.inverse_transform(y_test.reshape(-1, 1))

# Metrics
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae  = mean_absolute_error(y_true, y_pred)
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

print(f"\n    📊 Model Performance:")
print(f"    {'─'*35}")
print(f"    RMSE  (Root Mean Sq Error): ${rmse:.2f}")
print(f"    MAE   (Mean Abs Error):     ${mae:.2f}")
print(f"    MAPE  (Mean Abs % Error):   {mape:.2f}%\n")

# ─────────────────────────────────────────────
# STEP 7: VISUALIZE — Full Dashboard Chart
# ─────────────────────────────────────────────
print("[6/6] Creating visualization...")

# Get test dates for x-axis
test_start_idx = split + SEQUENCE_LEN
test_dates = df.index[test_start_idx: test_start_idx + len(y_true)]

fig, axes = plt.subplots(3, 1, figsize=(14, 14))
fig.patch.set_facecolor('#0D0D0D')
plt.suptitle(f'{STOCK_TICKER} Stock Price Prediction — LSTM Model',
             fontsize=16, color='#FFD700', fontweight='bold', y=0.98)

# ── Plot 1: Full History + Predicted Zone ──
ax1 = axes[0]
ax1.set_facecolor('#111111')
ax1.plot(df.index, df['Close'], color='#4A90D9', linewidth=1.2, label='Full Historical Close')
ax1.set_title('Full Historical Stock Price', color='white', fontsize=12, pad=10)
ax1.set_ylabel('Price (USD)', color='#AAAAAA')
ax1.tick_params(colors='#AAAAAA')
ax1.spines[:].set_color('#333333')
ax1.legend(facecolor='#1A1A1A', labelcolor='white')
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# ── Plot 2: Actual vs Predicted (Test Period) ──
ax2 = axes[1]
ax2.set_facecolor('#111111')
ax2.plot(test_dates, y_true, color='#00E5FF', linewidth=1.5, label='Actual Price')
ax2.plot(test_dates, y_pred, color='#FFD700', linewidth=1.5, linestyle='--', label='Predicted Price')
ax2.fill_between(test_dates, y_true.flatten(), y_pred.flatten(),
                 alpha=0.15, color='#FF6B35')
ax2.set_title('Actual vs Predicted — Test Period (20%)', color='white', fontsize=12, pad=10)
ax2.set_ylabel('Price (USD)', color='#AAAAAA')
ax2.tick_params(colors='#AAAAAA')
ax2.spines[:].set_color('#333333')
ax2.legend(facecolor='#1A1A1A', labelcolor='white')
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

# ── Plot 3: Training Loss ──
ax3 = axes[2]
ax3.set_facecolor('#111111')
ax3.plot(history.history['loss'],     color='#FF6B6B', linewidth=2, label='Training Loss')
ax3.plot(history.history['val_loss'], color='#4ECDC4', linewidth=2, label='Validation Loss')
ax3.set_title('Model Training Loss', color='white', fontsize=12, pad=10)
ax3.set_xlabel('Epoch', color='#AAAAAA')
ax3.set_ylabel('MSE Loss', color='#AAAAAA')
ax3.tick_params(colors='#AAAAAA')
ax3.spines[:].set_color('#333333')
ax3.legend(facecolor='#1A1A1A', labelcolor='white')

# Metrics annotation box
metrics_text = f"RMSE: ${rmse:.2f}  |  MAE: ${mae:.2f}  |  MAPE: {mape:.2f}%"
fig.text(0.5, 0.01, metrics_text, ha='center', color='#FFD700',
         fontsize=11, style='italic',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#1A1A1A', edgecolor='#FFD700'))

plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.savefig('/mnt/user-data/outputs/stock_prediction_output.png',
            dpi=150, bbox_inches='tight', facecolor='#0D0D0D')
plt.close()

print("    ✅ Chart saved → stock_prediction_output.png\n")

print(f"{'='*55}")
print("  🎉 ALL DONE! Stock Market Prediction Complete.")
print(f"{'='*55}")
print(f"\n  Stock:    {STOCK_TICKER}")
print(f"  RMSE:     ${rmse:.2f}")
print(f"  MAE:      ${mae:.2f}")
print(f"  MAPE:     {mape:.2f}%")
print(f"\n  Output:   stock_prediction_output.png")
print(f"  Model:    LSTM (2-layer) with Dropout")
print(f"{'='*55}\n")
