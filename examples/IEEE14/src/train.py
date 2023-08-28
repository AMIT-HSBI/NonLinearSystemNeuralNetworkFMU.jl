#nohup time python3 train.py &

import os
import sys

import pandas as pd
import numpy as  np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def scale(x, min, max, a, b):
  """
  Scale x to [a, b]
  """
  return ((x - min)/(max - min)) * (b-a) + a


def train(eqName, workdir, nInputs, nOutputs, csvFile):
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

  nSamples = len(df)
  nTrain = round(0.9*nSamples)
  nTest = nSamples - nTrain

  x_train = np.array(df.iloc[:nTrain, :nInputs]).astype("float32")
  y_train = np.array(df.iloc[:nTrain, nInputsReal:nInputsReal+nOutputs]).astype("float32")
  assert len(x_train[1,:]) == nInputs
  assert len(y_train[1,:]) == nOutputs

  x_test = np.array(df.iloc[nTrain:, :nInputs]).astype("float32")
  y_test = np.array(df.iloc[nTrain:, nInputsReal:nInputsReal+nOutputs]).astype("float32")
  assert len(x_test[1,:]) == nInputs
  assert len(y_train[1,:]) == nOutputs

  # Normalize data
  #for i in range(nOutputs):
  #  min_train = min(y_train[:,i])
  #  max_train = max(y_train[:,i])
  #  y_train[:,i] = scale(y_train[:,i], min_train, max_train, -0.9, 0.9)
  #  y_test[:,i] = scale(y_test[:,i], min_train, max_train, -0.9, 0.9)

  train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
  train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)

  test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
  test_dataset = test_dataset.shuffle(buffer_size=1024).batch(64)

  # Setup model
  normalize = layers.Normalization()
  normalize.adapt(x_train)

  model = tf.keras.Sequential([
    normalize,
    layers.Dropout(0.02, input_shape=(nInputs,)),
    layers.Dense(nInputs),
    layers.Dense(100, activation="tanh"),
    layers.Dense(nOutputs*10, activation="sigmoid"),
    layers.Dense(nOutputs*4,   activation="tanh"),
    layers.Dense(nOutputs)
  ])

  def residual_loss(y_true, y_pred):
    return tf.norm()


  model.compile(loss = tf.keras.losses.MeanSquaredError(),
                optimizer = tf.keras.optimizers.Adam())

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
    )  # How often to write logs (default: once per epoch)
  ]

  history = model.fit(
    train_dataset,
    epochs=1000,
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

  x = x_test[1,:]
  y = y_test[1,:]
  model(x) - y

if __name__ == "__main__":
  if len(sys.argv) == 6:
    eqName   = sys.argv[1]
    workdir  = sys.argv[2]
    nInputs  = int(sys.argv[3])
    nOutputs = int(sys.argv[4])
    csvFile  = sys.argv[5]
    train(eqName, workdir, nInputs, nOutputs, csvFile)
