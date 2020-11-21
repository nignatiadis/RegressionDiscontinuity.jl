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

result = NaiveLocalLinearRD(kernel = Rectangular(), bandwidth = ImbensKalyanaraman())
```
### Min-Max Optimal Estimator

The following estimates the sharp RDD estimate for the min-max optimal
estimator of [Imbens and Wager (2019)](https://arxiv.org/abs/1705.01677).

The estimate assumes a bound of 14.28 on the second derivative of the conditional
mean functions for the outcome in the Lee data. The optimization uses the
solver from [Mosek](https://docs.mosek.com/9.2/install/installation.html).

Mosek is free for academics. An open source solver option include [Hypatia.jl](https://github.com/chriscoey/Hypatia.jl), but it is currently slower for this problem.  

```
using RegressionDiscontinuity, MosekTools
data = load_rdd_data(:lee08)

result = fit(MinMaxOptRD(B=14.28, solver=Mosek.Optimizer), data)
```
