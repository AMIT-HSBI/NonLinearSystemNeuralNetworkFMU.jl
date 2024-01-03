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


n_inputs = X_train.shape[1]
n_outputs = y_train.shape[1]

# hyperparams (should be tuned)
n_hidden_layers = 4
n_hidden_units = 64
num_epochs = 1000
start_schedule = num_epochs//2
batch_size = 32
initial_learning_rate = 0.001

t0 = time.time()
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

history = fnn_model.fit(
    X_train,
    y_train,
    batch_size=batch_size,
    validation_split=0.05,
    callbacks=[lrs_callback],
    verbose=1, epochs=num_epochs)
t1 = time.time()

print(f"Total fitting time: {t1-t0}")
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
# what is with the NN Architecture, try different options
# employ Bayesian Optimization using Ax


# now hyperparam opt
import keras_tuner

def build_fnn_model(hp):
  model = keras.Sequential()
  model.add(keras.Input(shape=(n_inputs,)))
  
  
  num_hidden_units = hp.Int("n_units",min_value=16,max_value=128,step=16)
  num_hidden_layers = hp.Int("n_layers",min_value=1,max_value=10)
  
  use_PReLU = hp.Boolean("use_PReLU")
  if not use_PReLU:
    act_func = hp.Choice("act_func", ['relu','tanh','selu','elu','leaky_relu','gelu','mish'])
  
  for i in range(num_hidden_layers):
    if use_PReLU:
      model.add(layers.Dense(units=num_hidden_units))
      model.add(layers.PReLU())
    else:
      model.add(layers.Dense(units=num_hidden_units, activation=act_func))
  model.add(layers.Dense(n_outputs))
  
  learning_rate = hp.Float("learning_rate",min_value=0.001,max_value=10,step=10,sampling="log")
  # add optimizer choice later
  # optimizer_str = hp.Choice("optimizer",['Adam','AdamW','Adadelta','Adagrad','Adamax'])
  # optimizer = getattr(tf.keras.optimizers, optimizer_str)
  # model.compile(loss='mean_squared_error',optimizer=optimizer(learning_rate))
  model.compile(loss='mean_squared_error',optimizer=tf.keras.optimizers.Adam(learning_rate))
  
  return model

tuner = keras_tuner.BayesianOptimization(
    build_fnn_model,
    objective='val_loss',
    max_trials=10)


# maybe use validation_split instead of validation_data
tuner.search(X_train, y_train, epochs=15, validation_data=(X_test, y_test))
best_model = tuner.get_best_models()[0]
  