import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler

window_size = 10

df = pd.read_csv("synhetic_series.csv")
values = df["value"].values.reshape(-1, 1)

scaler = MinMaxScaler()
values_scaled = scaler.fit_transform(values)

joblib.dump(scaler, "scaler.save")


def create_sliding_windows(data, window_size):
    x = []

    for i in range(len(data) - window_size + 1):
        x.append(data[i:i + window_size])

    return np.array(x)


x = create_sliding_windows(
    data=values_scaled,
    window_size=window_size
)

x = x.reshape(x.shape[0], x.shape[1], 1)

np.save("x_train.npy", x)