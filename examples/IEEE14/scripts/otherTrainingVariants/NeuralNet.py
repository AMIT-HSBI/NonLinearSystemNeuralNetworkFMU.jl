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
ref_sol = pd.read_csv(r'examples\IEEE14\scripts\otherTrainingVariants\IEEE_14_Buses_res.csv')
num_inputs = 16
num_outputs = 110

X_train, y_train, X_test, y_test, x_scaler, y_scaler, inp_cols, out_cols = IEEE14_data_preparation(r'examples\IEEE14\scripts\otherTrainingVariants\eq_1403.csv')


# input and first dense layer
n_inputs = X_train.shape[1]
n_outputs = y_train.shape[1]
n_hidden_units = 64

t0 = time.time()
inputs = keras.Input(shape=(n_inputs,))
x = layers.Dense(n_hidden_units)(inputs)
layer_1_out = keras.layers.PReLU()(x)

# number of residual blocks
n_blocks = 4
last_layer = layer_1_out
for i in range(n_blocks):
  x = layers.Dense(n_hidden_units)(last_layer)
  x = keras.layers.PReLU()(x)
  next_layer = layers.add([x, last_layer])

  last_layer = next_layer

outputs = layers.Dense(n_outputs)(last_layer)

residual_model = keras.Model(inputs, outputs)


residual_model.compile(loss='mean_squared_error',
                optimizer=tf.keras.optimizers.Adam(0.001))


# initial learning rate: 0.01
num_epochs = 1000
start_schedule = num_epochs//2
batch_size = 32
def scheduler(epoch, lr):
  if epoch < start_schedule:
    return lr
  else:
    return lr * tf.math.exp(-0.01)
lrs_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

history = residual_model.fit(
    X_train,
    y_train,
    batch_size=batch_size,
    validation_split=0.05,
    callbacks=[lrs_callback],
    verbose=1, epochs=num_epochs)
t1 = time.time()

print(f"Total fitting time: {t1-t0}")
print(f"MSE on Test Set after fitting: {residual_model.evaluate(X_test, y_test)}")

spec = (tf.TensorSpec((None, num_inputs), tf.float32, name="input"),)
output_path = residual_model.name + ".onnx"

model_proto, _ = tf2onnx.convert.from_keras(residual_model, input_signature=spec, opset=13, output_path=output_path)


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(('train_loss', 'val_loss'))
plt.title("MSE for clustered dataset")
plt.xlabel("Number of Epochs")
plt.ylabel("MSE")
plt.show()


plot_against_reference(residual_model, ref_sol, inp_cols, out_cols, randrange(50), x_scaler, y_scaler)

#TODO:
# what is with the NN Architecture, try different options