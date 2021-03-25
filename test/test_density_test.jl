# Testing the function
using Test

using RegressionDiscontinuity
using Distributions
using GLM
using DataFrames


# Testing that the package doesn't detect false positives:
R1 = rand(Normal(), 100_000)

rv1 = RunningVariable(R1)
test1 = fit(McCraryTest(), rv1)

R2 = rand(Uniform(-1, 1), 100_000)
rv2 = RunningVariable(R2)
test2 = fit(McCraryTest(), rv2)



#@test test1.pval >= 0.05
#@test test2.pval >= 0.05

# Testing that the package detect true positives
R3 = rand(Uniform(-1, 1), 100_000)
R3 = R3 .+ 2 * (1.0 * (rand(Uniform(-1, 1), 100_000) .> 0) .* (R3 .< 0))
rv3 = RunningVariable(R3)
test3 = fit(McCraryTest(), rv3)

#@test test3.pval < 0.05




# Testing the package with real data.
lee08 = RDData(RegressionDiscontinuity.Lee08())

lee08.ZsR
lee08_mccrary = fit(McCraryTest(), lee08.ZsR)


# I will compare the results with the rdd package.
pvalR = 0.1982771
binR = 0.011
bwR = 0.242
θhatR = 0.103

@test lee08_mccrary.pval ≈ pvalR atol = 0.001
@test lee08_mccrary.b ≈ binR atol = 0.001
@test lee08_mccrary.h ≈ bwR atol = 0.001
@test lee08_mccrary.θhat ≈ θhatR atol = 0.001
