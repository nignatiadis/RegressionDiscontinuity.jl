Base.@kwdef struct MinMaxOptRD <: SharpRD
    B::Float64
	solver
	num_buckets = 2000
end

Base.@kwdef struct FittedMinMaxOptRD
	tau_est::AbstractFloat
    se_est::AbstractFloat
    weights::Array{Float64}
end

#make subtype?
struct GridRunningVariable
	ZsR::Centered
	edges::Array{Float64, 1}
	values::Array{Float64, 1}
	weights::Array{Int, 1}
	widths::Array{Float64, 1}
	binmap::Array{Int, 1}
	ix_cutoff::Int
end

function GridRunningVariable(ZsR::Centered, num_buckets)
	xmax = maximum(ZsR)
 	xmin = minimum(ZsR)
	h = (xmax - xmin)/num_buckets
	grid = (xmin-h/2):h:(xmax+h)
	grid = vcat(grid, repeat([0], 2 - sum(grid.==0)))
	sort!(grid)
	grid_values = midpoints(grid)
	grid_width  = grid_values[2:length(grid_values)] .- grid_values[1:(length(grid_values)-1)]
	ix_cutoff = argmin(abs.(grid_values))
	binned = fit(Histogram, ZsR, grid)
	grid_weights = binned.weights
	binmap = StatsBase.binindex.(Ref(binned), ZsR)
	return GridRunningVariable(ZsR, grid, grid_values,  grid_weights,
							   grid_width, binmap, ix_cutoff)
end

#double check other options for this
function estimate_var(data::RDData)
	fitted_lm = fit(LinearModel, @formula(Ys ~ Ws * Zs), data)
    yhat = predict(fitted_lm)
    σ² = mean((data.Ys - yhat).^2) * length(data.Zs) / (length(data.Zs) - 4)
	return σ²
end

function fit(method::MinMaxOptRD, data::RDData; level=0.95)
	B = method.B
    xDiscr = GridRunningVariable(center(data.ZsR), method.num_buckets)
    X = xDiscr.values; h = xDiscr.widths; d = length(X); n= xDiscr.weights; ixc = xDiscr.ix_cutoff
	#to do check the treatment thing
    c = 0.0; W = X .> c
    σ² = zeros(d) .+ estimate_var(data)
    model =  Model(method.solver)

    @variable(model, G[1:d])
    @variable(model, f[1:d])
    @variable(model, l1 >=0)
    @variable(model, l2)
    @variable(model, l3)
    @variable(model, l4)
    @variable(model, l5)

    for i in 1:d
        @constraint(model, G[i] == 2*B*f[i] + l2*(1-W[i]) + l3*W[i] + l4*(X[i] - c) + l5*(W[i] - 0.5)*(X[i]-c))
    end

    @constraint(model, f[ixc]==0)
    @constraint(model, (f[ixc+1] - f[ixc])/h[ixc]== 0)

    for i in 3:d
        @constraint(model, (f[i] - 2*f[i-1] + f[i-2])/(h[i-1]*h[i-2]) - l1 <=0)
        @constraint(model, (f[i] - 2*f[i-1] + f[i-2])/(h[i-1]*h[i-2]) + l1 >=0)
    end

    @objective(model, Min, 1/4*(sum(n[i]*G[i]^2/σ²[i] for i in 1:d)) + l1^2 - l2 + l3)

    @suppress_out begin
        optimize!(model)
    end

    γ_xx = -value.(G)./(2 .*σ²)
    γ = γ_xx[xDiscr.binmap]
    #w = x.>0
    #γ[w.==0] = -γ[w.==0] ./ sum(γ[w.==0])
    #γ[w.==1] = γ[w.==1] ./ sum(γ[w.==1])

    FittedMinMaxOptRD(
        tau_est = sum(γ.*data.Ys),
        se_est = 0.0,
        weights = γ
    )
end


function Base.show(io::IO, rdd_fit::RegressionDiscontinuity.FittedMinMaxOptRD)
    println(io, "Local linear regression for regression discontinuity design")
    println(io, "       ⋅⋅⋅⋅ " * "Naive inference (not accounting for bias)")
    println(io, "       ⋅⋅⋅⋅ " * _string(rdd_fit.rdd_setting.kernel))
    println(io, "       ⋅⋅⋅⋅ " * _string(rdd_fit.rdd_setting.bandwidth))
    println(io, "       ⋅⋅⋅⋅ " * _string(rdd_fit.rdd_setting.variance))
    Base.show(io, rdd_fit.coeftable)
end
