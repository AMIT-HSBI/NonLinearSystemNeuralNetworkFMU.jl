# now hyperparam opt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner

from utils import IEEE14_data_preparation

ref_sol = pd.read_csv('/home/fbrandt3/arbeit/NonLinearSystemNeuralNetworkFMU.jl/examples/IEEE14/scripts/otherTrainingVariants/IEEE_14_Buses_res.csv')
num_inputs = 16
num_outputs = 110

X_train, y_train, X_test, y_test, x_scaler, y_scaler, inp_cols, out_cols = IEEE14_data_preparation('/home/fbrandt3/arbeit/NonLinearSystemNeuralNetworkFMU.jl/examples/IEEE14/scripts/otherTrainingVariants/eq_1403.csv')


n_inputs = X_train.shape[1]
n_outputs = y_train.shape[1]

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
  
  learning_rate = hp.Float("learning_rate",min_value=0.000001,max_value=0.001,step=10,sampling="log")
  # add optimizer choice later
  # optimizer_str = hp.Choice("optimizer",['Adam','AdamW','Adadelta','Adagrad','Adamax'])
  # optimizer = getattr(tf.keras.optimizers, optimizer_str)
  # model.compile(loss='mean_squared_error',optimizer=optimizer(learning_rate))
  model.compile(loss='mean_squared_error',optimizer=tf.keras.optimizers.Adam(learning_rate))
  
  return model

tuner = keras_tuner.BayesianOptimization(
    build_fnn_model,
    objective='val_loss',
    max_trials=50)


# maybe use validation_split instead of validation_data
tuner.search(X_train, y_train, epochs=30, validation_data=(X_test, y_test))
best_model = tuner.get_best_models()[0]

print(best_model)