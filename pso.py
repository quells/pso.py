import sys
import numpy as np

class Optimizer(object):
	def __init__(self, fitnessFunction, parameterShape, populationSize=None, localSize=25, inertia=0.95, particleStep=0.75, localStep=0.5, globalStep=0.1):
		self.fitnessFunction = fitnessFunction
		self.parameterShape = parameterShape
		if not populationSize:
			self.populationSize = int(4*localSize*len(parameterShape))
		else:
			self.populationSize = populationSize
		self.localSize = localSize
		
		self.constraints = []
		self.bounds = []
		
		self.inertia = inertia
		self.particleStep = particleStep
		self.localStep = localStep
		self.globalStep = globalStep
		
		self.reset()
	
	def reset(self):
		maxFloat = sys.float_info.max
		
		self.averageFitness = maxFloat
		self.averageHistory = []
		
		values = []
		velocities = []
		for ps in self.parameterShape:
			values.append(np.random.uniform(ps[0], ps[1], self.populationSize))
			delta = (ps[1] - ps[0])
			velocities.append(np.random.uniform(-delta, delta, self.populationSize))
		
		self.values = np.array(values).transpose()
		self.velocities = np.array(velocities).transpose()
		
		self.particleBestValue = np.array(values).transpose()
		self.particleBestFitness = [maxFloat] * self.populationSize
		
		localGroupingCount = int(np.ceil(float(self.populationSize)/self.localSize))
		self.localBestValue = np.array([self.values[0]] * localGroupingCount)
		self.localBestFitness = [maxFloat] * localGroupingCount
		
		self.globalBestValue = self.values[0]
		self.globalBestFitness = maxFloat
		self.globalBestHistory = []
		self.globalBestValueHistory = []
	
	def addConstraint(self, c):
		self.constraints.append(c)
	
	def addBounds(self, b):
		self.bounds += b
	
	def updateFitness(self):
		avg = 0
		
		for i in range(len(self.values)):
			cFlag = False
			for c in self.constraints:
				if not c(self.values[i]):
					cFlag = True
					break
			if cFlag:
				continue
			
			j = i % len(self.localBestValue)
			f = self.fitnessFunction(self.values[i])
			avg += f
			
			if f < self.particleBestFitness[i]:
				self.particleBestFitness[i] = f
				self.particleBestValue[i] = self.values[i]
			
			if f < self.localBestFitness[j]:
				self.localBestFitness[j] = f
				self.localBestValue[j] = self.values[i]
			
			if f < self.globalBestFitness:
				self.globalBestFitness = f
				self.globalBestValue = self.values[i]
				
		self.averageFitness = avg / float(len(self.values))
	
	def step(self):
		self.updateFitness()
		
		for i in range(len(self.values)):
			j = i % len(self.localBestValue)
			
			ri = self.values[i]
			rp = (self.particleBestValue[i] - ri) * np.random.uniform()
			rl = (self.localBestValue[j] - ri) * np.random.uniform()
			rg = (self.globalBestValue - ri) * np.random.uniform()
			vi = self.velocities[i]
			vi = self.inertia*vi + self.particleStep*rp + self.localStep*rl + self.globalStep*rg
			self.velocities[i] = vi
			self.values[i] += vi
		
		if len(self.bounds) > 0:
			clip = self.values.transpose()
			for i in range(len(clip)):
				clip[i] = np.clip(clip[i], self.bounds[i][0], self.bounds[i][1])
			self.values = clip.transpose()
	
	def stepUntil(self, epsilon, waitIterations=200, maxIterations=3000, logging=False):
		self.step()
		
		last = self.globalBestFitness
		sinceLastImprovement = 0
		
		self.averageHistory.append(self.averageFitness)
		self.globalBestHistory.append(self.globalBestFitness)
		self.globalBestValueHistory.append(self.globalBestValue)
		
		i = 0
		while True:
			self.step()
			
			self.averageHistory.append(self.averageFitness)
			self.globalBestHistory.append(self.globalBestFitness)
			self.globalBestValueHistory.append(self.globalBestValue)
			
			if logging:
				print self.globalBestFitness, self.globalBestValue
			
			if abs(last - self.globalBestFitness) < epsilon or i > maxIterations:
				sinceLastImprovement += 1
				if sinceLastImprovement > waitIterations:
					break
			else:
				sinceLastImprovement = 0
			
			last = self.globalBestFitness
			i += 1
		
		self.globalBestValueHistory = np.array(self.globalBestValueHistory)