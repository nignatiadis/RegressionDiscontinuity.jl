# Testing the function
using Test
using Feather

using RegressionDiscontinuity
using Plots
using Distributions
using GLM
using DataFrames


# Testing that the package doesn't detect false positives:
R1 = rand(Normal(), 100_000)
R2 = rand(Uniform(-1, 1), 100_000)

@test density_test(R1; verbose=false, plot=false, generate=false)[1] >= 0.05
@test density_test(R2; verbose=false, plot=false, generate=false)[1] >= 0.05

# Testing that the package detect true positives
R3 = rand(Uniform(-1, 1), 100_000)
R3 = R3 .+ 2 * (1.0 * (rand(Uniform(-1, 1), 100_000) .> 0) .* (R3 .< 0))
@test density_test(R3; verbose=false, plot=false, generate=false)[1] < 0.05




# Testing the package with real data.
lee08_path = joinpath(dirname(@__FILE__), "..", "data", "lee08.feather")
lee08 = Feather.read(lee08_path)
lee08_rdd = load_rdd_data(:lee08)
Z = lee08.margin ./ 100
pval, plot,  θhat, σθ,  b, h, df = density_test(Z)

# I will compare the results with the rdd package.
pvalR = 0.1982771
binR = 0.011
bwR = 0.242
θhatR = 0.103

@test pval ≈ pvalR atol = 0.001
@test b ≈ binR atol = 0.001
@test h ≈ bwR atol = 0.001
@test θhat ≈ θhatR atol = 0.001