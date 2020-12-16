# RegressionDiscontinuity

[![Build Status](https://github.com/nignatiadis/RegressionDiscontinuity.jl/workflows/CI/badge.svg)](https://github.com/nignatiadis/RegressionDiscontinuity.jl/actions)
[![Coverage](https://codecov.io/gh/nignatiadis/RegressionDiscontinuity.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/nignatiadis/RegressionDiscontinuity.jl)

A Julia package for Regression Discontinuity analyses.

## Examples

### Density Test.
The following estimates a test of manipulation of the running variable based on [McCrary (2008)](https://www.sciencedirect.com/science/article/abs/pii/S0304407607001133). 

```
using RegressionDiscontinuity, DataFrames
df = load_rdd_data(:lee08) |> DataFrame

(pval, plot,  θhat, σθ,  b, h, df) = density_test(df.Zs)
```

### Visual comparison:

I benchmarked my density test with R and Stata. For a 100_000 draws from a random normal the results are:
* Julia

![image](/figures/data1_julia.png)

* Stata

![image](/figures/data1_stata.png)

* R

![image](/figures/data1R.png)

Allowing for some manipulation in the running variable:
* Julia

![image](/figures/data3_julia.png)

* Stata

![image](/figures/data3_stata.png)

* R

![image](/figures/data3R.png)

Finally, looking at [Lee (2008)](https://www.sciencedirect.com/science/article/pii/S0304407607001121) data:
* Julia

![image](/figures/lee08_julia.png)

* Stata

![image](/figures/lee08_stata.png)

* R

![image](/figures/lee08R.png)

 Overall, they look practically the same. The `R` figures looks a little bit different because the package selects the limits of y-axis differently. However, it is just a zoomed-in version of the others.

----
### References
Lee, D. S. (2008). Randomized experiments from non-random selection in US House elections. Journal of Econometrics, 142(2), 675-697.

McCrary, J. (2008). Manipulation of the running variable in the regression discontinuity design: A density test. Journal of econometrics, 142(2), 698-714.