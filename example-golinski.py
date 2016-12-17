import pso
import numpy as np

def golinski(x):
	return 0.7854*x[0]*x[1]**2*(3.3333*x[2]**2 + 14.9334*x[2] - 43.0934) - 1.508*x[0]*(x[5]**2 + x[6]**2) + 7.4777*(x[5]**3 + x[6]**3) + 0.7854*(x[3]*x[5]**2 + x[4]*x[6]**2)

bounds = [[2.6, 3.6], [0.7, 0.8], [17, 28], [7.3, 8.3], [7.3, 8.3], [2.9, 3.9], [5, 5.5]]

optimizer = pso.Optimizer(golinski, bounds)

optimizer.addBounds(bounds)
optimizer.addConstraint(lambda x: 27.0/(x[0] * x[1]**2 * x[2]) <= 1)
optimizer.addConstraint(lambda x: 397.5/(x[0] * x[1]**2 * x[2]**2) <= 1)
optimizer.addConstraint(lambda x: 1.93*x[3]**3/(x[1] * x[2] * x[5]**4) <= 1)
optimizer.addConstraint(lambda x: 1.93*x[4]**3/(x[1] * x[2] * x[6]**4) <= 1)
optimizer.addConstraint(lambda x: np.sqrt((745*x[3]/x[1]/x[2])**2 + 16.9*1e6)/(110*x[5]**3) <= 1)
optimizer.addConstraint(lambda x: np.sqrt((745*x[4]/x[1]/x[2])**2 + 157.5*1e6)/(85*x[6]**3) <= 1)
optimizer.addConstraint(lambda x: x[1]*x[2]/40 <= 1)
optimizer.addConstraint(lambda x: 5*x[1]/x[0] <= 1)
optimizer.addConstraint(lambda x: x[0]/12/x[1] <= 1)
optimizer.addConstraint(lambda x: (1.5*x[5] + 1.9)/x[3] <= 1)
optimizer.addConstraint(lambda x: (1.1*x[6] + 1.9)/x[4] <= 1)

optimizer.stepUntil(1e-6, logging=True)

print optimizer.populationSize
print optimizer.globalBestFitness
print optimizer.globalBestValue
print len(optimizer.globalBestHistory)