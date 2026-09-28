import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error
import joblib

model = load_model("autoEncoder.h5", compile=False)
scaler = joblib.load("scaler.save")

df = pd.read_csv("synhetic_series.csv")
values = df["value"].values.reshape(-1, 1)

values_scaled = scaler.transform(values)

def create_sliding_windows(data, window_size):
    x = []

    for i in range(len(data) - window_size + 1):
        x.append(data[i:i + window_size])

    return np.array(x)

window_size = 10

x_all = create_sliding_windows(
    values_scaled,
    window_size=window_size
)

x_all = x_all.reshape(
    (x_all.shape[0], x_all.shape[1], 1)
)

x_prediction = model.predict(x_all)

mse_list = np.mean(
    np.square(x_all - x_prediction),
    axis=(1, 2)
)

threshold = np.percentile(mse_list, 95)

print(f"Eşik değeri belirleme: {threshold}")

anomalies = mse_list > threshold

df_result = df.iloc[window_size - 1:].copy()

df_result["reconstruction_error"] = mse_list
df_result["anomaly"] = anomalies

df_result.to_csv(
    "anomaly_results.csv",
    index=False
)

plt.figure()
plt.plot(df_result.index, df_result["value"], label="Veri")

# Anomalileri kırmızı noktalar ile işaretle
plt.scatter(df_result[df_result["anomaly"]].index,
            df_result[df_result["anomaly"]]["value"],
            color="red", label="Anomali", s=20)

plt.title("Zaman serisi ve tespit edilen anomaliler")
plt.xlabel("zaman")
plt.ylabel("değer")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()