import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


# -------------------------------
# BUILD MODEL
# -------------------------------
def build_lstm(input_shape):
    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=input_shape))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(5))  # predicting 5 features

    model.compile(optimizer='adam', loss='mse')
    return model


# -------------------------------
# PREPARE INPUT SEQUENCE
# -------------------------------
def prepare_sequence(user_input):
    # Repeat input to simulate time steps
    seq = np.array([user_input, user_input, user_input])
    return seq.reshape((1, 3, len(user_input)))


# -------------------------------
# PREDICT FUTURE VALUES
# -------------------------------
def predict_future(model, user_input):
    seq = prepare_sequence(user_input)
    prediction = model.predict(seq, verbose=0)
    return prediction[0]