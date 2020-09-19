module RegressionDiscontinuity

import Base:size,getindex
import CodecBzip2

using DataFrames
using Distributions

using GLM

using RData
using RecipesBase

using Statistics
using StatsBase
import StatsBase:fit
using StatsModels

using Tables

using UnPack


include("running_variable.jl")
include("load_example_data.jl")
include("imbens_kalyanaraman.jl")
include("kernels.jl")

export RunningVariable,
	   load_rdd_data
       ik_bandwidth,
	   Rectangular

end
