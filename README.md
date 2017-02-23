# pso.py

A generic particle swarm optimization (PSO) Python class.

**Requires NumPy.**

PSO is a minimizing algorithm, so ensure that your fitness function is written accordingly. A common technique to maximize a function with PSO is to return a negative value.

Hard parameter bounds and constraints are implemented.

## Examples

For a comprehensive walkthrough of how to use the `Optimizer` class, see [the Golinski example](https://github.com/quells/pso.py/blob/master/example-golinski.md).

For an example of using the `Optimizer` class to tune a simple [feedforward neural network](https://en.wikipedia.org/wiki/Feedforward_neural_network), see [the Iris example](https://github.com/quells/pso.py/blob/master/example-iris.py).

## Discussion

Note that other optimization techniques such as gradient descent may be more suitable for training large networks; the iris example is more of a proof-of-concept. On the other hand, PSO is useful for problems that are not differentiable.