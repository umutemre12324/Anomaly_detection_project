import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

t=np.arange(0,1000)

data=np.sin(0.02*t)+0.1+np.random.normal(size=len(t))

plt.figure()
plt.plot(t,data)
plt.title("Synthetic Time Series Data")
plt.xlabel("Time")
plt.ylabel("Value")
plt.show()

anomaly_indices = np.random.randint(0, len(t), size=20)

data[anomaly_indices] += np.random.normal(10, 2, size=20)

plt.figure()
plt.plot(t, data)
plt.scatter(t[anomaly_indices], data[anomaly_indices], color='red', label='Anomalies')
plt.title("Synthetic Time Series Data with Anomalies")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.show()

df=pd.DataFrame(
    {
        "timeStamp": pd.date_range(start="2023-01-01", periods=len(t), freq="h"),
        "value": data
    }
)

print(df.head())

df.to_csv("synhetic_series.csv",index=False)
