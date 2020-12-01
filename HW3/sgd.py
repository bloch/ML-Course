#################################
# Your name: Nathan Bloch
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib

import numpy as np
import numpy.random
from sklearn.datasets import fetch_openml
import sklearn.preprocessing
from matplotlib import pyplot as plt

"""
Assignment 3 question 2 skeleton.

Please use the provided function signature for the SGD implementation.
Feel free to add functions and other code, and submit this file with the name sgd.py
"""

def helper_hinge():
	mnist = fetch_openml('mnist_784')
	data = mnist['data']
	labels = mnist['target']

	neg, pos = "0", "8"
	train_idx = numpy.random.RandomState(0).permutation(np.where((labels[:60000] == neg) | (labels[:60000] == pos))[0])
	test_idx = numpy.random.RandomState(0).permutation(np.where((labels[60000:] == neg) | (labels[60000:] == pos))[0])

	train_data_unscaled = data[train_idx[:6000], :].astype(float)
	train_labels = (labels[train_idx[:6000]] == pos)*2-1

	validation_data_unscaled = data[train_idx[6000:], :].astype(float)
	validation_labels = (labels[train_idx[6000:]] == pos)*2-1

	test_data_unscaled = data[60000+test_idx, :].astype(float)
	test_labels = (labels[60000+test_idx] == pos)*2-1

	# Preprocessing
	train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
	validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
	test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)
	return train_data, train_labels, validation_data, validation_labels, test_data, test_labels

def helper_ce():
	mnist = fetch_openml('mnist_784')
	data = mnist['data']
	labels = mnist['target']
	
	train_idx = numpy.random.RandomState(0).permutation(np.where((labels[:8000] != 'a'))[0])
	test_idx = numpy.random.RandomState(0).permutation(np.where((labels[8000:10000] != 'a'))[0])

	train_data_unscaled = data[train_idx[:6000], :].astype(float)
	train_labels = labels[train_idx[:6000]]

	validation_data_unscaled = data[train_idx[6000:8000], :].astype(float)
	validation_labels = labels[train_idx[6000:8000]]

	test_data_unscaled = data[8000+test_idx, :].astype(float)
	test_labels = labels[8000+test_idx]

	# Preprocessing
	train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
	validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
	test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)
	return train_data, train_labels, validation_data, validation_labels, test_data, test_labels

def SGD_hinge(data, labels, C, eta_0, T):
	"""
	Implements Hinge loss using SGD.
	"""
	# TODO: Implement me
	w = np.zeros(784, dtype=np.float64)
	for t in range(1, T+1):
		i = np.random.randint(0, data.shape[0], 1)
		eta_t = eta_0 / t
		if(labels[i] * np.dot(data[i], w) < 1):
			w = (1 - eta_t)*w + eta_t*C*labels[i]*np.squeeze(data[i])
		else:
			w = (1 - eta_t)*w
		
	return w


def SGD_ce(data, labels, eta_0, T):
	"""
	Implements multi-class cross entropy loss using SGD.
	"""
	# TODO: Implement me
	w = np.zeros((10, 784), dtype=np.float64)

	for t in range(1, T+1):
		i = np.random.randint(0, data.shape[0], 1)
		gradient = ce_gradient(w, data[i], labels[i])
		w = w - eta_0 * gradient
	return w

#################################

# Place for additional code

################################### Question 1 ##########################################################

def set_accuracy(data, labels, w):
	accuracy = 0
	for i in range(data.shape[0]):
		if(np.dot(data[i], w) >= 0):
			prediction = 1
		else:
			prediction = -1

		if(prediction == labels[i]):
			accuracy = accuracy + 1

	return accuracy / data.shape[0]


def Q1a():
	train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper_hinge()
	x_labels = [pow(10, i) for i in range(-5, 6)]
	y_labels = list()

	for eta_0 in x_labels:
		validiation_result = 0
		for i in range(10):
			w = SGD_hinge(train_data, train_labels, 1, eta_0, 1000)
			validiation_result += set_accuracy(validation_data, validation_labels, w)
		y_labels.append(validiation_result / 10)

	plt.plot(x_labels, y_labels)
	plt.xscale('log')
	plt.xlabel('eta_0 value')
	plt.ylabel('validation set accuracy')
	plt.xticks(x_labels)
	plt.xlim(x_labels[0], x_labels[-1])
	# plt.show()

# Q1a()

# best eta_0 was pow(10, 0) = 1.
def Q1b():
	train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper_hinge()
	x_labels = [pow(10, i) for i in range(-7, 8)]
	y_labels = list()

	for C in x_labels:
		validiation_result = 0
		for i in range(10):
			w = SGD_hinge(train_data, train_labels, C, 1, 1000)
			validiation_result += set_accuracy(validation_data, validation_labels, w)
		y_labels.append(validiation_result / 10)

	plt.plot(x_labels, y_labels)
	plt.xscale('log')
	plt.xlabel('C value')
	plt.ylabel('validation set accuracy')
	plt.xticks(x_labels)
	plt.xlim(x_labels[0], x_labels[-1])
	# plt.show()

# Q1b()

# best C was pow(10, -4) = 10^-4.

def Q1c():
	train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper_hinge()
	C = pow(10, -4)
	eta_0 = 1
	T = 20000
	w = SGD_hinge(train_data, train_labels, C, eta_0, T)
	plt.imshow(np.reshape(w, (28,28)), interpolation='nearest')
	# plt.show()

# Q1c()

def Q1d():
	train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper_hinge()
	C = pow(10, -4)
	eta_0 = 1
	T = 20000
	w = SGD_hinge(train_data, train_labels, C, eta_0, T)
	test_set_accuracy = set_accuracy(test_data, test_labels, w)
	print("The accuracy of the best classifier on the test set is", test_set_accuracy)

# Q1d()

################################### Question 2 ##########################################################


def ce_accuracy(data, labels, w):
	accuracy = 0
	results = np.dot(data, np.transpose(w))
	for i in range(data.shape[0]):
		prediction = np.argmax(results[i])
		if(prediction == int(labels[i])):
			accuracy += 1
	return accuracy / data.shape[0]

def ce_gradient(w, x, y):
	gradient = np.zeros((10, 784), dtype=np.float64)
	y = int(y)
	dot_products = np.dot(w, np.squeeze(x))
	max_dot = np.max(dot_products)
	dot_products = dot_products - max_dot
	dot_products = np.exp(dot_products)
	for i in range(10):
		if(i == y):
			gradient[i] = ((dot_products[y]/np.sum(dot_products))-1)*x
		else:
			gradient[i] = (dot_products[i]/np.sum(dot_products))*x
	return gradient


def Q2a():
	train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper_ce()
	x_labels = [pow(10, i) for i in range(-10,11)]
	y_labels = list()

	for eta_0 in x_labels:
		validiation_result = 0
		for i in range(10):
			w = SGD_ce(train_data, train_labels, eta_0, 1000)
			validiation_result += ce_accuracy(validation_data, validation_labels, w)
		y_labels.append(validiation_result / 10)

	plt.plot(x_labels, y_labels)
	plt.xscale('log')
	plt.xlabel('eta0 Value')
	plt.ylabel('Validation Set Accuracy')
	plt.xticks(x_labels, fontsize=6)
	plt.xlim(x_labels[0], x_labels[-1])
	# plt.show()

# Q2a()

def Q2b():
	train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper_ce()
	w = SGD_ce(train_data, train_labels, pow(10, -6), 20000)
	fig, ax = plt.subplots(2,5)
	for i, axi in enumerate(ax.flat):
		axi.imshow(np.reshape(w[i], (28, 28)), interpolation='nearest')
		axi.axis('off')
		axi.set_title(i)
	# plt.show()

# Q2b()

def Q2c():
	train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper_ce()
	w = SGD_ce(train_data, train_labels, pow(10, -6), 20000)
	test_set_accuracy = ce_accuracy(test_data, test_labels, w)
	print("The accuracy of the best classifier on the test set is", test_set_accuracy)

# Q2c()
###############################################