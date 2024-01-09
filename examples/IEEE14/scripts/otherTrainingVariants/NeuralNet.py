#https://github.com/onnx/tensorflow-onnx/blob/main/tutorials/keras-resnet50.ipynb
import pandas as pd
import time
import matplotlib.pyplot as plt
from random import randrange
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import tf2onnx

from utils import IEEE14_data_preparation, plot_against_reference

# read input/output data and reference solution
ref_sol = pd.read_csv('/home/fbrandt3/arbeit/NonLinearSystemNeuralNetworkFMU.jl/examples/IEEE14/scripts/otherTrainingVariants/IEEE_14_Buses_res.csv')
num_inputs = 16
num_outputs = 110

X_train, y_train, X_test, y_test, x_scaler, y_scaler, inp_cols, out_cols = IEEE14_data_preparation('/home/fbrandt3/arbeit/NonLinearSystemNeuralNetworkFMU.jl/examples/IEEE14/scripts/otherTrainingVariants/eq_1403.csv')


n_inputs = X_train.shape[1]
n_outputs = y_train.shape[1]

# hyperparams (should be tuned)
n_hidden_layers = 4
n_hidden_units = 64
num_epochs = 100
start_schedule = num_epochs//2
batch_size = 32
initial_learning_rate = 0.001


hidden_units = [n_hidden_units]*n_hidden_layers
n_hidden_layers = len(hidden_units)
fnn_model = keras.Sequential()

fnn_model.add(keras.Input(shape=(n_inputs,)))

for i in range(n_hidden_layers):
  fnn_model.add(layers.Dense(hidden_units[i]))
  fnn_model.add(layers.PReLU())

fnn_model.add(layers.Dense(n_outputs))

fnn_model.compile(loss='mean_squared_error',
                optimizer=tf.keras.optimizers.Adam(initial_learning_rate))

def scheduler(epoch, lr):
  if epoch < start_schedule:
    return lr
  else:
    return lr * tf.math.exp(-0.01)
lrs_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

t0 = time.time()
history = fnn_model.fit(
    X_train,
    y_train,
    batch_size=batch_size,
    validation_split=0.05,
    callbacks=[lrs_callback],
    verbose=1, epochs=num_epochs)
t1 = time.time()

print(f"Total fitting time: {t1-t0} s")
print(f"MSE on Test Set after fitting: {fnn_model.evaluate(X_test, y_test)}")

spec = (tf.TensorSpec((None, num_inputs), tf.float32, name="input"),)
output_path = fnn_model.name + ".onnx"

model_proto, _ = tf2onnx.convert.from_keras(fnn_model, input_signature=spec, opset=13, output_path=output_path)


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(('train_loss', 'val_loss'))
plt.title("MSE for clustered dataset")
plt.xlabel("Number of Epochs")
plt.ylabel("MSE")
plt.show()

plot_against_reference(fnn_model, ref_sol, inp_cols, out_cols, randrange(50), x_scaler, y_scaler)
  
  
#TODO:
# showcase other approaches from Drive
# train IEEE14 with resiudal approach
# maybe Reservoir Computing