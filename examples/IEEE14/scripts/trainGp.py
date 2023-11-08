# train GP and export to ONNX
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

from clusterData import cluster

import onnx
import onnxruntime as rt
import skl2onnx
from skl2onnx.common.data_types import FloatTensorType, DoubleTensorType
from skl2onnx import convert_sklearn
from skl2onnx.helpers.onnx_helper import save_onnx_model, load_onnx_model

# read data
eg_1403_data = pd.read_csv('/home/fbrandt3/arbeit/NonLinearSystemNeuralNetworkFMU.jl/examples/IEEE14/scripts/eq_1403_10000.csv')
inp = eg_1403_data.iloc[:, :16]
out = eg_1403_data.iloc[:, 16:-1]

# convert input and output data to numpy arrays
inp_np = inp.to_numpy()
out_np = out.to_numpy()

print(inp_np.shape)
print(out_np.shape)


# cluster output data using DBSCAN
label_indices, _ = cluster(out_np)


# transform input and output data using min-max scaling between (0,1)
x_scaler_1403 = MinMaxScaler(feature_range=(0,1))
y_scaler_1403 = MinMaxScaler(feature_range=(0,1))

# cluster_ind = 0 for best results
cluster_ind = 0

# only use data from one cluster for training
X_1403 = inp_np[label_indices[cluster_ind]]
y_1403 = out_np[label_indices[cluster_ind]]

X_train_1403, X_test_1403, y_train_1403, y_test_1403 = train_test_split(X_1403, y_1403, test_size=0.1, shuffle=True)

X_train_1403_scaled = X_train_1403
X_test_1403_scaled = X_test_1403
X_train_1403_scaled = x_scaler_1403.fit_transform(X_train_1403)
X_test_1403_scaled = x_scaler_1403.transform(X_test_1403)

y_train_1403_scaled = y_scaler_1403.fit_transform(y_train_1403)
y_test_1403_scaled = y_scaler_1403.transform(y_test_1403)

print(X_train_1403_scaled.shape)
print(y_train_1403_scaled.shape)


# train GP
noise_variance = 1.0
lengthscale = 1.0
kernel = noise_variance * RBF(lengthscale)

gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=0)
gpr.fit(X_train_1403_scaled[:100], y_train_1403_scaled[:100,1])

print('mae on test data', mean_absolute_error(y_test_1403_scaled[:,1], gpr.predict(X_test_1403_scaled)))
print('mse on test', mean_squared_error(y_test_1403_scaled[:,1], gpr.predict(X_test_1403_scaled)))
print('r2 score on test', gpr.score(X_test_1403_scaled, y_test_1403_scaled[:,1]))


def predict_with_onnxruntime(onx, X):
    sess = rt.InferenceSession(onx.SerializeToString(), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    res = sess.run(None, {input_name: X.astype(np.float64)})
    return res[0]


gpr_mixin = skl2onnx.wrap_as_onnx_mixin(gpr, target_opset=12)

onx = gpr_mixin.to_onnx(X_train_1403_scaled.astype(np.float64))
print(predict_with_onnxruntime(onx, X_test_1403_scaled) - gpr.predict(X_test_1403_scaled))

# # save trained gpr as onnx model using Double instead of float
# initial_type = [("X_1403", DoubleTensorType([None, None]))] # may be wrong with X_1403
# onx64 = convert_sklearn(gpr, initial_types=initial_type, target_opset=12)
# save_onnx_model(onx64, save_path)