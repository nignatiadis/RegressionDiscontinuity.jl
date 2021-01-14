# RegressionDiscontinuity

[![Build Status](https://github.com/nignatiadis/RegressionDiscontinuity.jl/workflows/CI/badge.svg)](https://github.com/nignatiadis/RegressionDiscontinuity.jl/actions)
[![Coverage](https://codecov.io/gh/nignatiadis/RegressionDiscontinuity.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/nignatiadis/RegressionDiscontinuity.jl)

A Julia package for Regression Discontinuity analyses.

## Examples

### Naive Local Linear Regression
The following estimates the sharp RDD estimate using local linear regression
without any kind of bias correction. It uses a rectangular kernel and the
Imbens-Kalyanaraman bandwidth.
```
using RegressionDiscontinuity
data = load_rdd_data(:lee08)

result = fit(NaiveLocalLinearRD(kernel = Rectangular(), bandwidth = ImbensKalyanaraman()), data)
```
### Min-Max Optimal Estimator

The following estimates the sharp RDD estimate for the min-max optimal
estimator of [Imbens and Wager (2019)](https://arxiv.org/abs/1705.01677).

The estimate assumes a bound of 14.28 on the second derivative of the conditional
mean functions for the outcome in the Lee data. The optimization uses a user specified solver. The fastest option is [Mosek](https://docs.mosek.com/9.2/install/installation.html), which is free for academics. An open source alternative is [Hypatia.jl](https://github.com/chriscoey/Hypatia.jl), but it is currently slower for this problem.  

```
using RegressionDiscontinuity, MosekTools
data = load_rdd_data(:lee08)

result = fit(ImbensWagerOptRD(B=14.28, solver=Mosek.Optimizer), data)
```

### Density Test.

The following estimates a test of manipulation of the running variable based on [McCrary (2008)](https://www.sciencedirect.com/science/article/abs/pii/S0304407607001133). 

```
using RegressionDiscontinuity
data = load_rdd_data(:lee08)

lee08_mccrary = fit(McCraryTest(), data.ZsR)
```

