module RegressionDiscontinuity

using Reexport

import Base: size, getindex, getproperty, propertynames, show

using DataFrames
@reexport using Distributions
using FastGaussQuadrature
using Feather
using Intervals
using JuMP
using LinearAlgebra
using GLM
using OffsetArrays
using Plots 
using QuadGK
using RecipesBase
using Roots
using Setfield
using Statistics
import Statistics: var
@reexport using StatsBase
import StatsBase: fit, weights, nobs
using StatsModels
using Tables

using UnPack





include("running_variable.jl")
include("load_example_data.jl")
include("kernels.jl")
include("imbens_kalyanaraman.jl")
include("local_linear.jl")
include("minmax_optimal.jl")
include("density_test.jl")

export RunningVariable,
    RDData,
    Treated,
    Untreated,
    load_rdd_data,
    Rectangular,
    bandwidth,
    ImbensKalyanaraman,
    linearweights,
    EickerHuberWhite,
    Homoskedastic,
    NaiveLocalLinearRD,
    ImbensWagerOptRD,
    estimate_second_deriv_bound,
    density_test
end
