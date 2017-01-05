# pso.py

A generic particle swarm optimization (PSO) Python class.

**Requires NumPy.**

PSO is a minimizing algorithm, so ensure that your fitness function is written accordingly. A common technique to maximize a function with PSO is to return a negative value.

Hard parameter bounds and constraints are implemented.

### Example Walkthrough

This example finds a solution to the Golinski Speed Reducer problem. It illustrates the use of bounds and constraints on parameters in the problem.

The known solution has a fitness of about 2987.41 with parameters [3.50, 0.7, 17, 7.3, 7.30, 3.35, 5.29]. The PSO with these settings typically achieves a fitness of around 2994.4.

`def golinski(x):`

The fitness function definition. Must take an array (native list or NumPy 1D array) and return a float.

`bounds = [...]`

The range of values that guesses should be pulled from for each parameter. Used just for initialization; this does not impose hard bounds on the parameters.

`optimizer = pso.Optimizer(golinski, bounds)`

Initializes the optimizer with some default (overridable) hyperparameters. These include:

**populationSize**: The number of particles to include in the system. Scales with the number of parameters unless specified. Choosing this number to balance parameter-space-exploration against execution time is itself a difficult problem.

**localSize**: The size of groups of particles that can "see" one another. Defaults to 25.

**inertia, particleStep, localStep, globalStep**: Hyperparameters that affect how quickly the particles converge on the best solution. Choosing these values is itself a difficult problem; quick convergence is generally at odds with sufficient exploration of the parameter space.

`optimizer.addBounds(bounds)`

Sets hard limits on the range of values each parameter can take.

`optimizer.addConstraint(...)`

Takes a function (or nameless lambda) that sets hard limits on the range of values each parameter can take relative to other parameters. The function should take an array (native list or NumPy 1D array) and return a boolean.

`optimizer.stepUntil(1e-6, logging=True)`

Step until the change in the best fitness so far has been less than this value for some period of time. Overridable constants:

**waitIterations**: Even if the fitness doesn't change, keep stepping at least this many iterations. The optimizer will often plateau for long periods of time while the parameter space is searched. Defaults to 200.

**maxIterations**: Finish stepping after this many iterations even if the system is improving. Mostly useful when logging is turned off. Defaults to 3000.

**logging**: Print the best fitness and the best particle values found so far after each step. Defaults to false.
