import pandas as pd
import numpy as np
import time
from random import randrange
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA

from utils import plot_against_reference, cluster

from sklearn.multioutput import MultiOutputRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import DoubleTensorType



# read input/output data and reference solution
ref_sol = pd.read_csv(r'examples\IEEE14\scripts\otherTrainingVariants\IEEE_14_Buses_res.csv')
# ninputs = 16, noutputs = 110
data = pd.read_csv(r'examples\IEEE14\scripts\otherTrainingVariants\eq_1403.csv')
inp = data.iloc[:, :16]
out = data.iloc[:, 16:-1]

inp_columns = inp.columns
out_columns = out.columns

num_inputs = 16
num_outputs = 110

# convert input and output data to numpy arrays
inp_np = inp.to_numpy()
out_np = out.to_numpy()

# standard scaling the data before clustering is important
out_np_standard_scaler = StandardScaler()
transformed_out_np = out_np_standard_scaler.fit_transform(out_np)
# eps = 10. is a pretty good choice
label_indices, cluster_labels = cluster(transformed_out_np, min_samples=5, eps=5.)
  
# transform input and output data using min-max scaling between (0,1)
x_scaler = MinMaxScaler(feature_range=(0,1))
y_scaler = MinMaxScaler(feature_range=(0,1))

# cluster_ind = 0 for best results
cluster_ind = 0

# only use data from one cluster for training
X_1403 = inp_np[label_indices[cluster_ind]]
y_1403 = out_np[label_indices[cluster_ind]]

X_train_1403, X_test_1403, y_train_1403, y_test_1403 = train_test_split(X_1403, y_1403, test_size=0.1, shuffle=True)

X_train_1403_scaled = x_scaler.fit_transform(X_train_1403)
X_test_1403_scaled = x_scaler.transform(X_test_1403)

y_train_1403_scaled = y_scaler.fit_transform(y_train_1403)
y_test_1403_scaled = y_scaler.transform(y_test_1403)


pca = PCA(n_components=2)
transformed = pca.fit_transform(transformed_out_np)

# plot pca'd data in class label colors
if -1 in label_indices:
  for i in range(-1, len(label_indices)-1):
    plt.scatter(transformed[label_indices[i]][:,0], transformed[label_indices[i]][:,1], label=f'Class {i}')
else:
  for i in range(len(label_indices)):
    plt.scatter(transformed[label_indices[i]][:,0], transformed[label_indices[i]][:,1], label=f'Class {i}')

plt.legend()
plt.show()


noise_variance = 0.9
lengthscale = 1.0

t0 = time.time()
kernel = noise_variance * RBF(lengthscale)
#gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=0)
gpr = MultiOutputRegressor(GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=0))
gpr.fit(X_train_1403_scaled, y_train_1403_scaled)
t1 = time.time()  

print(f"Total fitting time: {t1-t0}")
print(f"MSE on Test Set after fitting: {mean_squared_error(y_test_1403_scaled, gpr.predict(X_test_1403_scaled))}")


model_onnx = convert_sklearn(
    gpr,
    initial_types=[("X_train_1403_scaled", DoubleTensorType([None, num_inputs]))],
    target_opset={"": 12, "ai.onnx.ml": 2},
)

with open("mp_gp.onnx", "wb") as f:
    f.write(model_onnx.SerializeToString())
    

plot_against_reference(gpr, ref_sol, inp_columns, out_columns, randrange(50), x_scaler, y_scaler)




#TODO:
# comment the code
# change clustering to KMeans
    
