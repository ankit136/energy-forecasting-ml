# ==========================================
# AI-Powered Energy Consumption Forecasting
# ==========================================

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------------
# 1. Data Generation
# -----------------------------
def generate_data():
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=180*24, freq='H')

    df = pd.DataFrame({'datetime': dates})

    df['hour'] = df['datetime'].dt.hour
    df['day'] = df['datetime'].dt.day
    df['month'] = df['datetime'].dt.month
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    df['temperature'] = 25 + 10*np.sin(2*np.pi*df['hour']/24) + np.random.normal(0,2,len(df))
    df['humidity'] = 60 + 20*np.cos(2*np.pi*df['hour']/24) + np.random.normal(0,5,len(df))

    df['energy'] = (
        100
        + 20*np.sin(2*np.pi*(df['hour']-8)/24)
        - 10*df['is_weekend']
        + 2.5*np.abs(df['temperature'] - 22)
        + np.random.normal(0,5,len(df))
    )

    return df

# -----------------------------
# 2. Preprocessing
# -----------------------------
def preprocess(df):
    df['lag_1'] = df['energy'].shift(1)
    df['rolling_mean_3'] = df['energy'].shift(1).rolling(3).mean()
    df.dropna(inplace=True)
    return df

# -----------------------------
# 3. Train Model
# -----------------------------
def train_model(df):
    features = [
        'hour','day','month','day_of_week','is_weekend',
        'temperature','humidity','lag_1','rolling_mean_3'
    ]

    X = df[features]
    y = df['energy']

    split = int(len(df)*0.8)

    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("RMSE:", round(rmse,2))
    print("R2:", round(r2,4))

    return y_test, y_pred

# -----------------------------
# 4. Save Results
# -----------------------------
def save_results(y_test, y_pred):
    df_out = pd.DataFrame({
        'Actual': y_test.values,
        'Predicted': y_pred
    })

    df_out.to_csv("outputs/predictions.csv", index=False)
    print("Saved predictions.csv")

# -----------------------------
# 5. Main
# -----------------------------
def main():
    df = generate_data()
    df = preprocess(df)
    y_test, y_pred = train_model(df)
    save_results(y_test, y_pred)

if __name__ == "__main__":
    main()
