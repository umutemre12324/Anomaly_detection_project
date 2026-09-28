import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.callbacks import EarlyStopping

x_train = np.load("x_train.npy")

timeSteps = x_train.shape[1]
input_dim = x_train.shape[2]

inputs = Input(shape=(timeSteps, input_dim))

encoded = LSTM(
    64,
    activation="relu",
    return_sequences=False
)(inputs)

latent = RepeatVector(timeSteps)(encoded)

decoded = LSTM(
    64,
    activation="relu",
    return_sequences=True
)(latent)

output = TimeDistributed(Dense(1))(decoded)

autoEncoder = Model(inputs, output)

autoEncoder.compile(
    optimizer="adam",
    loss="mse"
)

earlyStop = EarlyStopping(
    monitor="loss",
    patience=5,
    restore_best_weights=True
)

history = autoEncoder.fit(
    x_train,
    x_train,
    epochs=50,
    batch_size=32,
    callbacks=[earlyStop]
)

plt.figure()

plt.plot(
    history.history["loss"],
    label="Eğitim Kaybı"
)

plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Eğitim Süreci")
plt.legend()
plt.grid(True)
plt.show()

autoEncoder.save("autoEncoder.h5")