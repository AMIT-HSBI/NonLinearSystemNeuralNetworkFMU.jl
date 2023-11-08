import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# clusters data and returns a dictionary with the array indices for all members of each cluster
# label_indices[0] = [0,1,2,3]
# label_indices[1] = [4,7,12]

# data is output data!!!
def cluster(data, min_samples=5, eps=5.):
  data_scaler = StandardScaler()

  # cluster data using the DBSCAN algorithm
  clustering = DBSCAN(min_samples=min_samples, eps=eps).fit(data_scaler.fit_transform(data))
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


def plot_clusters(data, label_indices):
  pca = PCA(n_components=2)
  data = pca.fit_transform(data)
  # plot pca'd data in class label colors
  if -1 in label_indices:
    for i in range(-1, len(label_indices)-1):
      plt.scatter(data[label_indices[i]][:,0], data[label_indices[i]][:,1], label=f'Class {i}')
  else:
    for i in range(len(label_indices)):
      plt.scatter(data[label_indices[i]][:,0], data[label_indices[i]][:,1], label=f'Class {i}')

  plt.legend()
  plt.show()


def extract_cluster(data, label_indices, cluster_index):
   return data[label_indices[cluster_index]]

