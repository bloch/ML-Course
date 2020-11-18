import numpy as np
from matplotlib import pyplot as plt
from collections import Counter
from scipy.spatial import distance
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784')
data = mnist['data']
labels = mnist['target']

idx = np.random.RandomState(0).choice(70000, 11000)
train = data[idx[:10000], :].astype(int)
train_labels = labels[idx[:10000]]
test = data[idx[10000:], :].astype(int)
test_labels = labels[idx[10000:]]

# Pre-processing: Calculating distances between all test images and all train images. Done for efficiency reasons.
dists = distance.cdist(test, train)
# test_list variable is for inner use, for time-complexity issues.
test_list = list()
for i in range(0, len(test)):
    test_list.append(list(test[i]))



# parameters: train_imgs - train images, train_labels - train labels, image_query - query image, k - integer parameter
def kNN(train_imgs, train_labels, image_query, k):
    i = test_list.index(list(image_query))                                       # finds index of current image query
    label_counter = Counter()                                                    # useful data structure to count labels
    dists_i = [(dists[i][j], train_labels[j]) for j in range(len(train_imgs))]   # distances of query img from data imgs
    dists_i.sort(key=lambda tup: tup[0])                                         # sorting the distances
    for j in range(k):                                                   # choosing k nearest(first k in sorted list)
        label_counter[int(dists_i[j][1])] += 1                           # counting labels of k nearest neighbors
    return (label_counter.most_common(1))[0][0]                          # returns most common label among k nearest


# Section b
print("Section (b) runs: ")
n = 1000
k = 10
correct = 0
train_b = train[:n]
train_labels_b = train_labels[:n]
for i in range(0, n):
    prediction = kNN(train_b, train_labels_b, test[i], k)
    if prediction == int(test_labels[i]):
        correct += 1

print("\tCorrect =", correct, "/", n, "=", correct / n)
print("\tPercentage is", 100 * correct / n, "%.")
print("Section (b) done.")

# Section c
print("Section (c) runs: ")
n = 1000
train_c = train[:n]
train_labels_c = train_labels[:n]
x_labels = [i for i in range(1, 101)]               # x_labels[1,2,..., 100] to be used as k
y_labels = []
for k in x_labels:
    predictions = np.zeros(len(test))
    for i in range(len(test)):
        predictions[i] = kNN(train_c, train_labels_c, test[i], k)
    correct = np.sum(predictions == np.array(test_labels[:n], dtype=int))
    y_labels.append(correct / n)

plt.plot(x_labels, y_labels)
plt.xlabel("k")
plt.ylabel("accuracy")
plt.show()

maximumC = max(y_labels)
index = y_labels.index(maximumC) + 1       # + 1 as i'th index represtents k=i+1
print("The best k is", index, "with a value of", maximumC)
print("Section (c) done.")

# Section d
print("Section (d) runs: ")
k = 1
x_labels = [n for n in range(100, 5001, 100)]   # x_labels=[100,..., 5000] to be used as first n elems in train images
y_labels = []
for n in x_labels:
    train_d = train[:n]
    train_labels_d = train_labels[:n]
    predictions = np.zeros(len(test))
    for i in range(len(test)):
        predictions[i] = kNN(train_d, train_labels_d, test[i], k)
    correct = np.sum(predictions == np.array(test_labels, dtype=int))
    y_labels.append(correct / 1000)


plt.plot(x_labels, y_labels)
plt.xlabel("n")
plt.ylabel("accuracy")
plt.show()
print("Section d done.")
