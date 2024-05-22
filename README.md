<div align="center">
    <img src="docs/src/assets/logo.svg" alt="RegressionDiscontinuity.jl" width="220">
</div>

<h2 align="center">RegressionDiscontinuity.jl</h2>
<p align="center"> A Julia package for Regression Discontinuity analyses.</p>
<p align="center">
  <a href="https://github.com/nignatiadis/RegressionDiscontinuity.jl/workflows/CI/badge.svg">
    <img src="https://github.com/nignatiadis/RegressionDiscontinuity.jl/workflows/CI/badge.svg"
         alt="Build Status">
  </a>
  <a href="https://codecov.io/gh/nignatiadis/RegressionDiscontinuity.jl/">
    <img src="https://codecov.io/gh/nignatiadis/RegressionDiscontinuity.jl/branch/master/graph/badge.svg"
         alt="Coverage">
  </a>
</p>


## **Warning**

This package is still experimental (and results should be double-checked for any empirical work). Please report any issues you encounter.


## Examples

We demonstrate the basic functionality of the package using the U.S. House Elections dataset of [Lee (2008)](https://www.sciencedirect.com/science/article/abs/pii/S0304407607001121).

```julia
julia> using RegressionDiscontinuity
julia> data = RDData(RegressionDiscontinuity.Lee08())
```

### Naive Local Linear Regression
The following estimates the sharp RDD estimate using local linear regression
without any kind of bias correction. It uses a rectangular kernel and the
Imbens-Kalyanaraman bandwidth.

```julia
julia> fit(NaiveLocalLinearRD(kernel = Rectangular(), bandwidth = ImbensKalyanaraman()), data.ZsR, data.Ys)
```

```
Local linear regression for regression discontinuity design
       ⋅⋅⋅⋅ Naive inference (not accounting for bias)
       ⋅⋅⋅⋅ Rectangular kernel (U[-0.5,0.5])
       ⋅⋅⋅⋅ Imbens Kalyanaraman bandwidth
       ⋅⋅⋅⋅ Eicker White Huber variance
────────────────────────────────────────────────────────────────────────────────────────────────
                          h        τ̂         se         bias     z   p-val  Lower 95%  Upper 95%
────────────────────────────────────────────────────────────────────────────────────────────────
Sharp RD estimand  0.462024  0.08077  0.0087317  unaccounted  9.25  <1e-99  0.0636562  0.0978838
────────────────────────────────────────────────────────────────────────────────────────────────
```

### Min-Max Optimal Estimator

The following estimates the sharp RDD estimate for the min-max optimal
estimator of [Imbens and Wager (2019)](https://arxiv.org/abs/1705.01677).

The estimate assumes a bound of 14.28 on the second derivative of the conditional
mean functions for the outcome in the Lee data. The optimization uses a user specified solver. The fastest option is [Mosek](https://docs.mosek.com/9.2/install/installation.html), which is free for academics. An open source alternative is [Hypatia.jl](https://github.com/chriscoey/Hypatia.jl), but it is currently slower for this problem.  

```julia
julia> using RegressionDiscontinuity
julia> using MosekTools
julia> fit(ImbensWagerOptRD(B=14.28, solver=Mosek.Optimizer), data.ZsR, data.Ys)
```

```
Imbens-Wager (2019) optimized regression discontinuity design
Max Second Derivative Bound: 14.28
────────────────────────────────────────────────────────────────────────
                          τ̂         se   max bias  Lower 95%  Upper 95%
────────────────────────────────────────────────────────────────────────
Sharp RD estimand  0.0592235  0.0197343  0.0101986  0.0159069    0.10254
────────────────────────────────────────────────────────────────────────
```

### Noise-Induced Randomization

We conduct inference using Noise-Induced Randomization (NIR), as described in [EIWW](https://arxiv.org/abs/2004.09458). We apply NIR, assuming the sensitivity model 𝒯₀ on a regression discontinuity design artificially constructed from test scores in early childhood (data from the Early Childhood Longitudinal Study).

```julia
julia> using RegressionDiscontinuity
julia> using Empirikos
julia> using MosekTools
julia> ecls_tbl = RegressionDiscontinuity.raw_table(RegressionDiscontinuity.ECLS_EIWW())
julia> Zs = NormalSample.(ecls_tbl.Z, minimum(ecls_tbl.SE))
julia> ZsR = RunningVariable(Zs, -0.2, :≥)
julia> Ys = ecls_tbl.Y
julia> nir = NoiseInducedRandomization(; solver=Mosek.Optimizer)
julia> nir_fit = fit(nir, ZsR, Ys)
```

```
RD analysis with Noise Induced Randomization (NIR)
────────────────────────────────────────────────────────────────────────────────────────
                            τ̂         se   max bias  Lower 95%  Upper 95%  CI halfwidth
────────────────────────────────────────────────────────────────────────────────────────
Weighted RD estimand  0.355071  0.0232147  0.0109893   0.304914   0.405229     0.0501575
────────────────────────────────────────────────────────────────────────────────────────
```

### McCrary Density Test

The following estimates a test of manipulation of the running variable based on [McCrary (2008)](https://www.sciencedirect.com/science/article/abs/pii/S0304407607001133). 

```julia
julia> using RegressionDiscontinuity
julia> fit(McCraryTest(), data.ZsR)
```

```
The McCrary (2008) test for manipulation in the
running variable for RDD.
          ⋅⋅⋅⋅ Bin size: 0.0112
          ⋅⋅⋅⋅ Bandwidth size: 0.2426
───────────────────────────────────────────────
                    θ̂         σ̂     z   p-val
───────────────────────────────────────────────
McCrary Test  0.102688  0.0798507  1.29  0.1984
───────────────────────────────────────────────
```