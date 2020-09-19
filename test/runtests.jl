using RegressionDiscontinuity
using Test

@testset "RegressionDiscontinuity.jl" begin
    include("test_runningvariable.jl")
    include("test_bandwidth.jl")
end

