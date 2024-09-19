# discontinuum

`discontinuum` is a middleware for developing Gaussian process (GP) models.
Why might we want a middleware? 
GP's are a flexible and elegant approach to modeling dynamical systems
for which we have sparse and uncertain observations.
In this arena, simple GP models, specified in several lines of math,
can often achieve state-of-the-art predictive performance.
However, fitting GP's is numerically intense, $\mathcal{O}(n^3)$ complexity.
They have several optimizations that take advantage of simplifying assumptions,
different algorithms, or GPUs, but each has tradeoffs.
Ideally, we could quickly write mathematical models, then run them on whichever
"engine" is best suited for a particular problem.

Furthermore, most models include a lot of relatively standard utility functions
for plotting, managing metadata, data pre-processing, and other "boiler plate."
`discontinum` packages engines and utilities within a single ecosystem,
such that creating a new model is just a matter of writing a little math without 
too much boilerplate.

# Site Contents

```{toctree}
:maxdepth: 2

self
getting_started
notebooks/loadest-gp-demo
notebooks/rating-gp-demo
api_reference
```