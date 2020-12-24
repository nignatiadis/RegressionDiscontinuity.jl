using RegressionDiscontinuity
using Hypatia
using Roots, Distributions
using GLM, DataFrames
using Test

data = load_rdd_data(:lee08)
result = fit(
    ImbensWagerOptRD(
        B = 14.28,
        solver = Hypatia.Optimizer,
        num_buckets = 2000,
        variance = Homoskedastic(),
    ),
    data,
)

#test that the result is close to the R package result with the same settings
@test result.tau_est ≈ 0.05936146 atol = 0.001
@test result.se_est ≈ 0.01975009 atol = 0.0001
@test result.maxbias ≈ 0.01019907 atol = 0.0001

level = 0.95
rel_bias = result.maxbias / result.se_est
zz = fzero(
    z -> cdf(Normal(), rel_bias - z) + cdf(Normal(), -rel_bias - z) + level - 1,
    0,
    rel_bias - quantile(Normal(), (1 - level) / 2.1),
)

result_plusminus = result.tau_est - result.ci[1]
est_plusminus = zz * result.se_est

@test result_plusminus ≈ est_plusminus atol = 0.00001
