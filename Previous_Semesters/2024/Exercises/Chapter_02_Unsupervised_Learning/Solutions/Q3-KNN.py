import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut

X = np.array([
    [1, 5], [2, 6], [3, 7], [4, 8], [5, 9], [7, 2], [8, 3],
    [5, 1], [6, 2], [7, 3], [8, 4], [9, 5], [2, 7], [3, 8]
])

y = np.array([
    0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1, 1
])

best_k = None
best_accuracy = 0

for k in range(1, len(X)):
    loo = LeaveOneOut()
    accuracies = []

    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        accuracy = knn.score(X_test, y_test)
        accuracies.append(accuracy)
    
    mean_accuracy = np.mean(accuracies)
    print(f"k = {k}, error = {1 - mean_accuracy}")
    
    if mean_accuracy > best_accuracy:
        best_accuracy = mean_accuracy
        best_k = k

error = 1 - best_accuracy

print(f"Best value for k: {best_k}")
print(f"Error of the best k value: {error}")
