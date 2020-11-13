

Base.@kwdef struct MinMaxOptRD{B,V<:VarianceEstimator} <: SharpRD
    max_second_deriv::B
    variance::V = EickerHuberWhite()
	num_buckets = 2000
end

Base.@kwdef struct FittedMinMaxOptRD{W,T}
    weights::W
    tau_est::T
    se_est::T
end

struct DiscretizedRDData
	y::Array{Float64, 1}
	x::Array{Float64, 1}
	xx::Array{Float64, 1}
	weights::Array{Int, 1}
	ixc::Int
	h::Array{Float64, 1}
	xmap::Array{Int, 1}
    grid::Array{Float64, 1}
end

function DiscretizedRDData(y, x, num_buckets)
	xmax = maximum(x)
 	xmin = minimum(x)
	h = (xmax - xmin)/num_buckets
	grid = (xmin-h/2):h:(xmax+h)
	grid = vcat(grid, repeat([0], 2 - sum(grid.==0)))
	sort!(grid)
	xx = midpoints(grid)
	h  = xx[2:length(xx)] .- xx[1:(length(xx)-1)]
	ixc = argmin(abs.(xx))
	binned = fit(Histogram, x, grid)
	weights = binned.weights
	xmap = StatsBase.binindex.(Ref(binned), x)
	return DiscretizedRDData(y, x, midpoints(grid), weights, ixc, h, xmap, grid)
end

function fit(method::MinMaxOptRD, rddata::RDData; level=0.95)

	B = estimate_B2(x, y)
    #B = 14.28
    data = DiscretizedRDData(y, x, num_buckets)
    X = data.xx; h = data.h; d = length(X); n= data.weights; ixc = data.ixc
    c = 0.0; W = X .> c
    σ2 = zeros(d) .+ estimate_σ2(x ,y)
    model =  Model(Mosek.Optimizer)

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

    @objective(model, Min, 1/4*(sum(n[i]*G[i]^2/σ2[i] for i in 1:d)) + l1^2 - l2 + l3)

    @suppress_out begin
        optimize!(model)
    end

    γ_xx = -value.(G)./(2 .*σ2)
    γ = γ_xx[data.xmap]
    w = x.>0
    γ[w.==0] = -γ[w.==0] ./ sum(γ[w.==0])
    γ[w.==1] = γ[w.==1] ./ sum(γ[w.==1])

    FittedMinMaxOptRD(
        tau_est = τ,
        se_est = 0.0,
        weights = γ
    )
end
