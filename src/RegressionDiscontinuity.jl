module RegressionDiscontinuity

using Reexport

import Base: size, getindex, getproperty, propertynames, show

using DataFrames
@reexport using Distributions
using FastGaussQuadrature
using Feather
using IntervalSets
using JuMP
using LaTeXStrings
using LinearAlgebra
using GLM
using OffsetArrays
using QuadGK
using RecipesBase
using Requires
using Roots
using Setfield
using Statistics
import Statistics: var
@reexport using StatsBase
import StatsBase: fit, weights, nobs
using StatsDiscretizations
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

function __init__()
    @require Empirikos="cab608d6-c565-4ea1-96d6-ce5441ba21b0" include("nir.jl")
end


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
    McCraryTest
end
