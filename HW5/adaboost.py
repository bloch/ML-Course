#################################
# Your name: Nathan Bloch
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib.
from matplotlib import pyplot as plt
import numpy as np
from process_data import parse_data

np.random.seed(7)

def run_adaboost(X_train, y_train, T):
    """ Returns: 
        hypotheses : 
            A list of T tuples describing the hypotheses chosen by the algorithm. 
            Each tuple has 3 elements (h_pred, h_index, h_theta), where h_pred is 
            the returned value (+1 or -1) if the count at index h_index is <= h_theta.
        alpha_vals : 
            A list of T float values, which are the alpha values obtained in every 
            iteration of the algorithm.
    """
    D = [(1/len(X_train)) for _ in X_train]
    classifier = []
    alpha_vals = []
    for i in range(1, T+1):
    	epsilon_t, h = findBestWL(X_train, y_train, D)
    	classifier.append(h)
    	omega_t = (1/2)*(np.log((1-epsilon_t)/epsilon_t))
    	alpha_vals.append(omega_t)
    	updateDistribution(X_train, y_train, D, omega_t, h[2], h[1], h[0])

    return (classifier, alpha_vals)

##############################################
# You can add more methods here, if needed.

def predict(x, theta, typ):
	if typ == 1:
		if(x <= theta):
			return 1
		else:
			return -1
	else:
		if(x >= theta):
			return 1
		else:
			return -1

def updateDistribution(X_train, y_train, D, omega_t, theta, j, typ):
	for i in range(len(X_train)):
		prediction = predict(X_train[i][j],theta, typ)
		D[i] *= np.exp(-1*omega_t*y_train[i]*prediction)
	sumD = sum(D)
	for i in range(len(X_train)):
		D[i] /= sumD
	
def findBestWL(X_train, Y_train, D):
    wlp = WL1(X_train, Y_train, D)
    wlm = WL2(X_train, Y_train, D)
    if wlp[0] <= wlm[0]:
        return wlp
    else:
        return wlm

def WL1(X_train, Y_train, D):
    S = [[X_train[i], Y_train[i], D[i]] for i in range(len(X_train))]
    d = len(X_train)
    m = len(S)
    F = np.inf
    J = 0
    O = 0
    for j in range(d):
        sortedS = sorted(S, key=lambda x: x[0][j])
        lastx = sortedS[m - 1][0][j] + 1
        tmp = [sortedS[i][2] for i in range(m) if sortedS[i][1] == 1]
        f = sum(tmp)
        if f < F:
            F = f
            O = sortedS[0][0][j] - 1
            J = j
        for i in range(m):
            f = f - sortedS[i][1] * sortedS[i][2]
            if i != m - 1:
                if (f < F) and sortedS[i][0][j] != sortedS[i + 1][0][j]:
                    F = f
                    O = 0.5 * (sortedS[i][0][j] + sortedS[i + 1][0][j])
                    J = j
            if i == m - 1:
                if (f < F) and sortedS[i][0][j] != lastx:
                    F = f
                    O = 0.5 * (sortedS[i][0][j] + lastx)
                    J = j
    h = (1, J, O)
    return F, h

def WL2(X_train, Y_train, D):  # S = {(xi,yi)}_i=1...n is th data set and D is the distribution
    S = [[X_train[i], Y_train[i], D[i]] for i in range(len(X_train))]
    d = len(X_train)
    m = len(S)
    F = np.inf
    J = 0
    O = 0
    for j in range(d):
        sortedS = sorted(S, key=lambda x: x[0][j])
        lastx = sortedS[m - 1][0][j] + 1
        tmp = [sortedS[i][2] for i in range(m) if sortedS[i][1] == -1]
        f = sum(tmp)
        if f < F:
            F = f
            O = sortedS[0][0][j] - 1
            J = j
        for i in range(m):
            f = f + sortedS[i][1] * sortedS[i][2]
            if i != m - 1:
                if (f < F) and sortedS[i][0][j] != sortedS[i + 1][0][j]:
                    F = f
                    O = 0.5 * (sortedS[i][0][j] + sortedS[i + 1][0][j])
                    J = j
            if i == m - 1:
                if (f < F) and sortedS[i][0][j] != lastx:
                    F = f
                    O = 0.5 * (sortedS[i][0][j] + lastx)
                    J = j
    h = (-1, J, O)
    return F, h

def ErrorCalc(X, Y, classifier, alpha_vals):
	T = [i for i in range(1, len(classifier)+1)]
	accs = []
	for i in range(len(classifier)):
		accuracy = 0
		for j in range(len(X)):
			sum_pred = 0
			for k in range(i+1):
				prediction = predict(X[j][classifier[k][1]], classifier[k][2], classifier[k][0])
				sum_pred += alpha_vals[k] * prediction

			if sum_pred >= 0:
				prediction = 1
			else:
				prediction = -1

			if(prediction == int(Y[j])):
				accuracy += 1

		accs.append(1 - (accuracy/len(X)))

	return accs

def A(classifier, alpha_vals, X_train, y_train, X_test, y_test):
    T = range(1, len(classifier)+1)
    testErrors = ErrorCalc(X_test, y_test, classifier, alpha_vals)
    trainErrors = ErrorCalc(X_train, y_train, classifier, alpha_vals)
    test_line, = plt.plot(T, testErrors)
    test_line.set_label("Test Data")
    train_line, = plt.plot(T, trainErrors)
    train_line.set_label("Train Data")
    plt.legend()
    #plt.show()

def B(classifier, alpha_vals, X_train, y_train, X_test, y_test, vocab):
	counter = 1
	for h in classifier:
		#print("WL -", counter ,"is the hypothesis", h, "and the word is", vocab[h[1]])
		counter += 1


def C(classifier, alpha_vals, X_train, y_train, X_test, y_test):
	loss_test = []
	loss_train = []
	Times = 80

	for T in range(1, Times+1):
		res_train = 0
		for i in range(len(X_train)):
			tmp = 0
			for j in range(0, T):
				prediction = predict(X_train[i][classifier[j][1]], classifier[j][2], classifier[j][0])
				tmp += prediction*alpha_vals[j]

			tmp *= (-1)*y_train[i]
			res_train += np.exp(tmp)

		res_train /= len(X_train)
		loss_train.append(res_train)

		res_test = 0
		for i in range(len(X_test)):
			tmp = 0
			for j in range(0, T):
				prediction = predict(X_test[i][classifier[j][1]], classifier[j][2], classifier[j][0])
				tmp += prediction*alpha_vals[j]

			tmp *= (-1)*y_test[i]
			res_test += np.exp(tmp)

		res_test /= len(X_test)
		loss_test.append(res_test)

	T = range(1, Times+1)

	train_data, = plt.plot(T, loss_train)
	train_data.set_label("Train Loss")

	test_data, = plt.plot(T, loss_test)
	test_data.set_label("Test Loss")

	plt.legend()
	#plt.show()

##############################################

def main():
    data = parse_data()
    if not data:
        return
    (X_train, y_train, X_test, y_test, vocab) = data
    
    #Section (a)
    # classifier, alpha_vals = run_adaboost(X_train, y_train, 80)
    # A(classifier, alpha_vals, X_train, y_train, X_test, y_test)
    
    #Section (b)
    # classifier, alpha_vals = run_adaboost(X_train, y_train, 10)
    # B(classifier, alpha_vals, X_train, y_train, X_test, y_test, vocab)

    #Section (c)
    # classifier, alpha_vals = run_adaboost(X_train, y_train, 80)
    # C(classifier, alpha_vals, X_train, y_train, X_test, y_test)

if __name__ == '__main__':
    main()
