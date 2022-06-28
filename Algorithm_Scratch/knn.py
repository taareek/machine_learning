import numpy as np
from sklearn import datasets 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

cmap= ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# getting iris daataset
iris = datasets.load_iris()

# getting features and labels
X, y = iris.data, iris.target

# Total no. of samples
print("Total no. of samples contains in iris dataset: ", len(X))

# splitting dataset into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=1)

# Total no. of train and test samples
# print("Total train samples: ", len(X_train))
# print("Total no. of test samples: ", len(X_test))

# # getting shape of training samples and training tables 
# print("Shape of training samples: ", X_train.shape)
# print("Shape of training labels: ", y_train.shape)

# # getting shape of test samples and training tables 
# print("Shape of test samples: ", X_test.shape)
# print("Shape of test labels: ", y_test.shape)

# # getting shape of a single training sample and training tables 
# print("Shape of a single training sample: ", X_train[0].shape)
# print("Shape of a single training labels: ", y_train[0].shape)

# # print a label
# print("A sample from training set: \n", X_train[0])
# print("A label from training set: \n", y_train[0])

# # scatter plot whole datasets
# plt.figure()
# plt.scatter(X[:, 0], X[:, 1], c= y, cmap= cmap, edgecolors='k', s= 20)
# plt.show()

# ofu = [1, 1, 1, 2, 2, 2, 3, 4, 5, 3, 4, 6, 7]
# from collections import Counter
# most_common = Counter(ofu).most_common(1)
# print(most_common)
# print(most_common[0][0])

# imporitng knn class
from k_nearest_neighbor import KNN

clf = KNN(k=9)

#fit the train data into our classifier
clf.fit(X_train, y_train)

#predict 
predictions = clf.predict(X_test)

# getting accuracy
acc = np.sum(predictions == y_test) / len(y_test)

print("Accuracy: ", round(acc, 2)*100, "%")