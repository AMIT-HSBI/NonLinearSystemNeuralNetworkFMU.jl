import pandas as pd
import time
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from random import randrange
from sklearn.metrics import mean_squared_error

from utils import IEEE14_data_preparation, plot_against_reference
from trainXGBoost import save_xgb_onnx, load_xgb_onnx

# read input/output data and reference solution
ref_sol = pd.read_csv(r'examples\IEEE14\scripts\otherTrainingVariants\IEEE_14_Buses_res.csv')
# ninputs = 16, noutputs = 110

X_train, y_train, X_test, y_test, x_scaler, y_scaler, inp_cols, out_cols = IEEE14_data_preparation(r'examples\IEEE14\scripts\otherTrainingVariants\eq_1403.csv')

t0 = time.time()
xgb = XGBRegressor()
xgb.fit(X_train, y_train)
t1 = time.time()

print(f"Total fitting time: {t1-t0}")
print(f"MSE on Test Set after fitting: {mean_squared_error(y_test, xgb.predict(X_test))}")

save_xgb_onnx(xgb, X_train, "xgb.onnx")

plot_against_reference(xgb, ref_sol, inp_cols, out_cols, randrange(50), x_scaler, y_scaler)

#TODO:
# for all methods:
    # data preparation (same for all), could maybe be improved
    # fitting/training of the model with explanation of method
    # testing on a test set and report of results
    # save to onnx
    # method specific plotting of loss, refsol etc.
# test results after training