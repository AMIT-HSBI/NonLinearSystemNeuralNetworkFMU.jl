#nohup time python3 train.py &

import os
import sys

import pandas as pd
import numpy as  np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from clusterData import cluster

def train(eqName, workdir, nInputs, nOutputs, csvFile, batchSize=32, nEpochs=1000, nLayers=10, nHiddenUnits=64):
  """
  Train surrogate of equation with TensorFlow.

  Parameters
  ----------
  eqName: str
    Index of equation
  """

  os.makedirs(workdir, exist_ok=True)
  os.chdir(workdir)

  print("eqName:" + eqName)
  print("workdir:" + workdir)
  print("nInputs:" + str(nInputs))
  print("nOutputs:" + str(nOutputs))
  print("csvFile:" + csvFile)

  tf_model_name = "tf_"+eqName
  nInputsReal = nInputs

  # Read train/test data
  df = pd.read_csv(csvFile)

  # cluster output data
  inp = df.iloc[:, :nInputs].to_numpy()
  out = df.iloc[:, nInputsReal:nInputsReal+nOutputs].to_numpy()
  label_indices, _ = cluster(out)

  # select cluster
  cluster_ind = 0
  inp = inp[label_indices[cluster_ind]]
  out = out[label_indices[cluster_ind]]

  x_train, x_test, y_train, y_test = train_test_split(inp, out, test_size=0.1, shuffle=True)

  # Scale inputs and outputs between (0,1)
  x_scaler = MinMaxScaler(feature_range=(0,1))
  y_scaler = MinMaxScaler(feature_range=(0,1))

  # fit transform on train data
  x_train = x_scaler.fit_transform(x_train)
  y_train = y_scaler.fit_transform(y_train)

  # transform on train data
  x_test = x_scaler.fit_transform(x_test)
  y_test = y_scaler.fit_transform(y_test)


  train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
  train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batchSize)

  test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
  test_dataset = test_dataset.shuffle(buffer_size=1024).batch(batchSize)

  # Setup model
  #normalize = layers.Normalization()
  #normalize.adapt(x_train)

  # create model
  hiddenLayers = [nHiddenUnits] * nLayers

  model = keras.Sequential()

  model.add(keras.Input(shape=(nInputs,)))
  #model.add(normalize) #?
  for i in range(nLayers):
    model.add(layers.Dense(hiddenLayers[i]))
    model.add(layers.PReLU())
  model.add(layers.Dense(nOutputs))

  model.compile(loss = tf.keras.losses.MeanSquaredError(),
                optimizer = tf.keras.optimizers.Adam(0.001))
  
  def scheduler(epoch, lr):
    if epoch < nEpochs//2:
        return lr
    else:
        return lr * tf.math.exp(-0.01)

  # Train model
  callbacks = [
    keras.callbacks.ModelCheckpoint(
      # Path where to save the model
      # The two parameters below mean that we will overwrite
      # the current checkpoint if and only if
      # the `val_loss` score has improved.
      # The saved model name will include the current epoch.
      filepath=os.path.join(workdir, tf_model_name+"_{epoch}"),
      save_best_only=True,  # Only save a model if `val_loss` has improved.
      monitor="val_loss",
      verbose=1,
    ),
    keras.callbacks.TensorBoard(
      log_dir=os.path.join(workdir, "logs"),
      histogram_freq=0,  # How often to log histogram visualizations
      embeddings_freq=0,  # How often to log embedding visualizations
      update_freq="epoch",
    ),  # How often to write logs (default: once per epoch)
    keras.callbacks.LearningRateScheduler(scheduler)
  ]

  history = model.fit(
    train_dataset,
    epochs=nEpochs,
    callbacks=callbacks,
    validation_data=test_dataset
    )

  model.save(os.path.join(workdir, eqName+"_final"))

def test_result(trainedModelPath):
  """
  Compute model(x)-y of trained model.
  """
  model = keras.models.load_model(trainedModelPath)

  # Evaluate the model on the test data using `evaluate`
  print("Evaluate on test data")
  results = model.evaluate(x_test, y_test, batch_size=128)
  print("test loss, test acc:", results)


if __name__ == "__main__":
  if len(sys.argv) == 6:
    eqName   = sys.argv[1]
    workdir  = sys.argv[2]
    nInputs  = int(sys.argv[3])
    nOutputs = int(sys.argv[4])
    csvFile  = sys.argv[5]
    train(eqName, workdir, nInputs, nOutputs, csvFile)
