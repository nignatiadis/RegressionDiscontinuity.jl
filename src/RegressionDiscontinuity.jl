module RegressionDiscontinuity



# Julia Standard Library packages and imports
import Base: size, getindex, getproperty, propertynames, show
using DelimitedFiles
using LinearAlgebra
using Statistics; import Statistics: var
using DataFrames 
# Julia Registry packages
using Distributions
using IntervalSets
using JuMP
using LaTeXStrings
using GLM
using QuadGK
using RangeHelpers
using RecipesBase
using Requires
using Roots
using Setfield
using StatsBase; import StatsBase: fit, weights, nobs
using StatsDiscretizations
using Tables


include("types.jl")
include("running_variable.jl")
include("discretization.jl")
include("load_example_data.jl")
include("kernels.jl")
include("imbens_kalyanaraman.jl")
include("local_linear.jl")
include("minmax_optimal.jl")
include("density_test.jl")
include("nir.jl")


export fit

export RunningVariable,
    RDData,
    Treated,
    Untreated,
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
