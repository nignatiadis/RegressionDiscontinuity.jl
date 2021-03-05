using RegressionDiscontinuity
using Test

@testset "RegressionDiscontinuity.jl" begin
    include("test_runningvariable.jl")
    include("test_bandwidth.jl")
    include("test_density_test.jl")
    include("test_minmax.jl")
end
