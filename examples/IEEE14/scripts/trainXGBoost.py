import os
import sys

import pandas as pd
import numpy as np
import onnxruntime as rt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from skl2onnx import to_onnx, update_registered_converter
from skl2onnx.common.shape_calculator import calculate_linear_regressor_output_shapes
from skl2onnx.helpers.onnx_helper import save_onnx_model, load_onnx_model
from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost


# inp_1403 = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/csv_files/inp_1403')
# out_1403 = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/csv_files/out_1403')



def train_xgb(eqName, workdir, nInputs, nOutputs, csvFile, test_size=0.1):
  """
  Train surrogate of equation with XGBoost.

  Parameters
  ----------
  eqName: str
    Index of equation
  workdir str
    Directory where to save model
  nInputs int
    Number of inputs/features
  nOutputs int
    Number of outputs
  csvFile str
    path to train/test data

  Returns
  ---------
  trained xgb model
  """


  os.makedirs(workdir, exist_ok=True)
  os.chdir(workdir)

  print("eqName: " + eqName)
  print("workdir: " + workdir)
  print("nInputs: " + str(nInputs))
  print("nOutputs: " + str(nOutputs))
  print("csvFile: " + csvFile)

  nInputsReal = nInputs

  # Read train/test data
  df = pd.read_csv(csvFile)

  nSamples = len(df)
  nTrain = round((1-test_size)*nSamples)
  nTest = nSamples - nTrain

  x_train = np.array(df.iloc[:nTrain, :nInputs]).astype("float32")
  y_train = np.array(df.iloc[:nTrain, nInputsReal:nInputsReal+nOutputs]).astype("float32")
  assert len(x_train[1,:]) == nInputs
  assert len(y_train[1,:]) == nOutputs

  x_test = np.array(df.iloc[nTrain:, :nInputs]).astype("float32")
  y_test = np.array(df.iloc[nTrain:, nInputsReal:nInputsReal+nOutputs]).astype("float32")
  assert len(x_test[1,:]) == nInputs
  assert len(y_train[1,:]) == nOutputs

  # normalize inputs to 0 mean and 1 variance
  # fit xgboost regressor with default hyperparams
  pipe = Pipeline([("scaler", StandardScaler()), ("xgb", XGBRegressor())])
  pipe.fit(x_train, y_train)

  return pipe, x_train, y_train, x_test, y_test


def test_xgb(pipe, x_test, y_test, metric=mean_absolute_error):
  """
  Test trained xgb model on test data

  Parameters
  ----------
  pipe object
    trained xgb_model with scaler
  x_test, y_test nparray
    test data
  metric function
    evalutation metric for test performance 
    default: MAE

  Returns
  -------
  test score evaluated by metric
  """
  return metric(y_test, pipe.predict(x_test))


def save_xgb_onnx(pipe, x_train, save_path):
  """
  Save trained xgb model into onnx format

  Parameters
  ----------
  pipe object
    pipeline to be saved (xgboost + scaler)
  x_train nparray
    training input
  save_path str
    path to saved model
  
  Returns
  -------
  ...
  """

  # Register the converter for XGBRegressor
  # defined in onnxmltools
  update_registered_converter(
    XGBRegressor,
    "XGBoostXGBRegressor",
    calculate_linear_regressor_output_shapes,
    convert_xgboost,
    )
  # convert to onnx object
  onx = to_onnx(pipe, x_train.astype(np.float32), target_opset={"": 12, "ai.onnx.ml": 2})
  # save onnx model at save_path
  save_onnx_model(onx, save_path)
  print(f'model saved at: {save_path}')

def load_xgb_onnx(save_path):
  # load saved model
  return load_onnx_model(save_path)
  


if __name__ == "__main__":
  # if len(sys.argv) == 6:
  #   eqName   = sys.argv[1]
  #   workdir  = sys.argv[2]
  #   nInputs  = int(sys.argv[3])
  #   nOutputs = int(sys.argv[4])
  #   csvFile  = sys.argv[5]

  eqName = "1304"
  workdir = '/home/fbrandt3/arbeit/NonLinearSystemNeuralNetworkFMU.jl/examples/IEEE14/scripts'
  nInputs = 16
  nOutputs = 110
  csvFile = '/home/fbrandt3/arbeit/NonLinearSystemNeuralNetworkFMU.jl/examples/IEEE14/data/sims/IEEE_14_Buses_1000/data/eq_1403.csv'

  trained_pipe, x_train, y_train, x_test, y_test = train_xgb(eqName, workdir, nInputs, nOutputs, csvFile)

  print(f'test result: {test_xgb(trained_pipe, x_test, y_test, mean_absolute_error)}')

  # saves trained pipeline in workdir with name "xgb_pipe"
  save_xgb_onnx(trained_pipe, x_train, "xgb_pipe.onnx")

  loaded_model = load_xgb_onnx("xgb_pipe.onnx")

  # prediction using the saved model
  sess = rt.InferenceSession(loaded_model.SerializeToString(), providers=["CPUExecutionProvider"])
  pred_onx = sess.run(None, {"X": x_test[:5].astype(np.float32)})
  print("predict", pred_onx[0].ravel())










