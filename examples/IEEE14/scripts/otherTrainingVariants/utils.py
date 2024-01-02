import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import mean_absolute_error, mean_squared_error

def plot_against_reference(model, ref_sol, inp_vars, out_vars, ind, x_scaler, y_scaler, predict_function=None, feature_function=None):
  # test against reference solution
  if predict_function is not None:
    ref_sol_pred = predict_function(x_scaler.transform(ref_sol[inp_vars]), model)
  if feature_function is not None:
    ref_sol_pred = model.predict(x_scaler.transform(feature_function(ref_sol[inp_vars].to_numpy())))
  else:
    ref_sol_pred = model.predict(x_scaler.transform(ref_sol[inp_vars]))
  ref_sol_pred_unscaled = y_scaler.inverse_transform(ref_sol_pred)

  # indices 94-99 dont exist in the reference solution (['lPQ9.a', 'lPQ5.a', 'lPQ12.a', 'lPQ3.a', 'lPQ2.a', 'lPQ4.a'])
  out_ref_sol = ref_sol[out_vars[ind]]
  ref_sol_pred_unscaled = ref_sol_pred_unscaled[:,ind]
  
  print('mae', mean_absolute_error(out_ref_sol, ref_sol_pred_unscaled))
  print('mse', mean_squared_error(out_ref_sol, ref_sol_pred_unscaled))
  
  figure, axis = plt.subplots(2, 2)
  axis[0,0] = plt.plot(ref_sol_pred_unscaled, label="ref_solution")
  axis[0,0] = plt.plot(out_ref_sol, label="prediction")
  
  axis[1,0] = plt.plot(ref_sol_pred_unscaled, label="ref_solution")
  axis[1,0] = plt.set_title("ref_solution")
  
  axis[0,1] = plt.plot(out_ref_sol, label="prediction")
  axis[0,1] = plt.set_title("prediction")

  plt.legend()
  plt.show()
  
# clusters data and returns a dictionary with the array indices for all members of each cluster
# label_indices[0] = [0,1,2,3]
# label_indices[1] = [4,7,12]
def cluster(data, min_samples, eps):
  # cluster data using the DBSCAN algorithm
  clustering = DBSCAN(min_samples=min_samples, eps=eps).fit(data)
  labels = clustering.labels_

  # Sample list of cluster labels
  cluster_labels = labels

  # Create a dictionary to store the indices for each label
  label_indices = {}

  # Iterate through the list and populate the dictionary
  for index, label in enumerate(cluster_labels):
      if label not in label_indices:
          label_indices[label] = []
      label_indices[label].append(index)

  # Now you can access the indices for each label
  print("Number of clusters:", len(label_indices))
  return label_indices, cluster_labels