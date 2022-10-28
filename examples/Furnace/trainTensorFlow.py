import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import onnxruntime
import tf2onnx

print("TensorFlow version:", tf.__version__)

def getNInput(eq):
  if eq == "2134":
    return 14
  elif eq == "1933":
    return 20
  else:
    raise(IndexError('Not allowed eq value ' + eq))

for eq in ["2134", "1933"]:
  # Load data from CSV
  eq_df = pd.read_csv("Furnace/data/eq_"+eq+".csv")
  eq_n_inputs = getNInput(eq)
  eq_n_outputs = len(eq_df.columns)-eq_n_inputs

  eq_input_names = eq_df.columns[:eq_n_inputs]
  eq_output_names = eq_df.columns[eq_n_inputs:]

  eq_train_inputs = np.array(eq_df[eq_input_names])
  eq_train_outputs = np.array(eq_df[eq_output_names])

  # Set up model
  model = tf.keras.models.Sequential([
    layers.Dense(eq_n_inputs, activation='sigmoid'),
    layers.Dense(eq_n_inputs*10, activation='tanh'),
    layers.Dense(eq_n_outputs)
  ])
  model.compile(loss = tf.keras.losses.MeanSquaredError(),
                optimizer = tf.keras.optimizers.Adam())

  # Train model
  model.fit(eq_train_inputs, eq_train_outputs, epochs=10)

  # Export to ONNX
  spec = (tf.TensorSpec((None, eq_n_inputs,), tf.float32, name="input"),)
  output_path = os.path.join("Furnace", "onnx", "eq_"+eq+".onnx")
  model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=12, output_path=output_path)
