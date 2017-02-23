# Iris Flower Data Set via
# http://archive.ics.uci.edu/ml/datasets/Iris

import numpy as np
import pso

data = []
labels = {
	"Iris-setosa": 1.0,
	"Iris-versicolor": 2.0,
	"Iris-virginica": 3.0,
}
with open("iris.txt") as file:
	for l in file.readlines():
		l = l.rstrip().split(",")
		for i in range(4):
			l[i] = float(l[i])
		l[4] = labels[l[4]]
		data.append(np.array(l))

training = []
testing = []
for i in range(3):
	for j in range(50):
		idx = j + 50*i
		if np.random.random() < 0.50:
			training.append(data[idx])
		else:
			testing.append(data[idx])

def plot(d):
	import matplotlib.pyplot as plt
	c = ["red", "green", "blue"]
	for di in d:
		plt.scatter(di[0], di[2], color=c[int(di[4]-1)], marker="o")
		plt.scatter(di[1], di[3], color=c[int(di[4]-1)], marker="x")
		plt.scatter(di[0], di[3], color=c[int(di[4]-1)], marker="^")
		plt.scatter(di[1], di[2], color=c[int(di[4]-1)], marker="v")
	plt.show()

def softmax(x):
	e = np.exp(x)
	s = e / sum(e)
	return s

def nn(w, x):
	# 4 inputs + bias; 4 nodes + bias; 3 outputs
	# 5*4 + 5*3 = 35 weights
	x = np.append(x, 1)
	h0 = np.tanh(np.dot(w[0:5], x))
	h1 = np.tanh(np.dot(w[5:10], x))
	h2 = np.tanh(np.dot(w[10:15], x))
	h3 = np.tanh(np.dot(w[15:20], x))
	h = np.array([h0, h1, h2, h3, 1.0])
	o0 = np.tanh(np.dot(w[20:25], h))
	o1 = np.tanh(np.dot(w[25:30], h))
	o2 = np.tanh(np.dot(w[30:35], h))
	return softmax(np.array([o0, o1, o2]))

def train(w):
	score = 0
	for i in range(len(training)):
		correct = int(training[i][4] - 1)
		predicted = nn(w, training[i][:4])
		score -= predicted[correct]
	return score

def test(w):
	score = 0
	for i in range(len(testing)):
		correct = int(testing[i][4] - 1)
		predicted = nn(w, testing[i][:4])
		predictedIndex = np.argmax(predicted)
		print predicted, predictedIndex, correct
		if predictedIndex == correct:
			score += 1
	print "%d / %d"%(score, len(testing))

bounds = [[-1.0, 1.0]]*35
optimizer = pso.Optimizer(train, bounds)
print optimizer.populationSize

optimizer.stepUntil(1e-6, logging=True, waitIterations=20)

print optimizer.globalBestFitness
print optimizer.globalBestValue
print len(optimizer.globalBestHistory)

test(optimizer.globalBestValue)