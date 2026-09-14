## Basic Concepts

- **Discrete variables** have separate, countable values, e.g., integers (2 follows 1).
- **Continuous variables** are real numbers. There is no clear "next" value, since there is always another number in between (1.1, 1.01, 1.001, ... all lie between 1 and 1.2).
- **Partial functions** are not defined for all inputs (not every input yields a valid result).
- **Derivative**: the rate of change of a function.

---

## Optimization Problem

- **Function**: describes what to optimize.
- **Input parameters**: the factors that influence the function.
- **Goal**: find the input parameters that maximize/minimize the function (we will assume maximization).

### State Space

The search space of an optimization problem is the space of its input parameters.

It is usually not possible to test all combinations of variables, so we need to estimate which ones are the best to test.

---

## How to Find a Solution

1. Mathematics
2. Local Search Algorithms
3. Global Search Algorithms

---

## Algorithms

### Hill Climbing

Starting from an initial point, it looks at its neighbors, picks the best one, and makes it the new current state. It repeats this until no better neighbor is found. It is only guaranteed to find a local maximum (not necessarily the global one).

### Gradient Ascent

Based on the shape of the function at our current position, we compute the rate of change (the derivative) and move uphill, in the direction of the slope, with a step size proportional to the derivative.

### Grid Search

![[3d-grid-dearch.png]]

Evaluates all the points of a grid (e.g., in a 3D space) and chooses the best result. Optionally, we can then apply a local search algorithm (e.g., Hill Climbing) starting from that point.

### Simulated Annealing (Research Algorithm)

A probabilistic optimization algorithm that tries to find the global optimum by exploring the search space and occasionally accepting a worse solution.

It usually runs until it reaches a state that is good enough for the application, or until the computational budget is exhausted.

---

## Population-Based Algorithms

### Iterative Grid

Repeatedly updates values and adjusts points across a structured grid.

We start with a grid search, find the quadrant with the best solutions, zoom into it, subdivide it, and continue.

![[iterative-grid-alg.png]]

### Particle Swarm

The idea is that, on the way to a good solution, we might come across an even better one.

### Evolutionary Algorithms

Algorithms that solve problems by mimicking biological evolution and natural selection.
