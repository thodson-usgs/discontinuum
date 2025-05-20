# discontinuum

GP's are a flexible approach to machine learning,
which are naturally suited for applications with sparse and noisy data
or for uncertainty analysis.
However, fitting GP's is numerically expensive,
which has led to a range of optimizations with different tradeoffs.
Ideally, we could quickly write mathematical models, then run them on whichever
"engine" is best suited for a particular problem.

Most models applications also include a fair amount of "boiler plate"
in the form of utility functions for plotting, managing metadata, data pre-processing, etc.
`discontinum` packages several engines and helper utilities into a single ecosystem
to simplify the process of prototyping GP models.

# Site Contents

```{toctree}
:maxdepth: 2

self
getting_started
notebooks/loadest-gp-demo
notebooks/rating-gp-demo
api_reference
```
