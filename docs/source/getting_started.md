# Getting Started

![PyPI - Version](https://img.shields.io/pypi/v/discontinuum)

```{warning}
Experimental.
```

## Installation
Install using pip:
```{code-block} bash
pip install discontinuum
```

## Models

### loadset-gp
`loadest-gp` is Gaussian-process model for estimating river constituent time series,
which borrows its namesake from the venerable LOAD ESTimator (LOADEST) software program.
However, LOADEST has several serious limitations
---it's essentially a linear regression---and it has been all but replaced by
the more flexible Weighted Regression on Time Discharge and Season (WRTDS),
which allows the relation between target and covariate to vary through time.
`loadest-gp` takes the WRTDS idea and reimplements it as a GP.
Try it out in the [loadest-gp demo](notebooks/loadest-gp-demo.ipynb).

### rating-gp
`rating-gp` is a Gaussian-process model for estimating river flow from stage time series.
Try it out in the [rating-gp demo](notebooks/rating-gp-demo.ipynb).


## Engines
Currently, the only supported engines are the marginal likelihood implementation in `pymc` and `gpytorch`.
Latent GP implementations could be added in the future.
In general, the `gpytorch` implementation is faster and provides a lot of powerful features,
like GPU support, whereas `pymc` is a more complete probabilistic-programming framework,
which can be "friendlier" for certain use cases.


## Roadmap
```{mermaid}
mindmap
  root((discontinuum))
    Data Providers
      USGS
      etc
    Engines
      PyMC
      PyTorch
    Utilities
      [Pre-processing]
      [Post-processing]
      [Plotting]
    Models
      [loadest-gp]
      [rating-gp]
```
