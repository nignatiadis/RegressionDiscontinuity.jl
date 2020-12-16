# Testing the function
using Test
using RegressionDiscontinuity
using Plots
using Distributions
using GLM
using DataFrames


# Testing that the package doesn't detect false positives:
R = rand(Normal(), 100_000)


