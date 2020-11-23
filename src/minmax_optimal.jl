Base.@kwdef struct MinMaxOptRD <: SharpRD
    B::Float64
	solver
	num_buckets = 2000
end

Base.@kwdef struct FittedMinMaxOptRD
	tau_est::AbstractFloat
    se_est::AbstractFloat
	ci::Array{Float64}
    weights::Array{Float64}
	B::Float64
	coeftable
end


#TODO: allow user to specify?
function estimate_var(data::RDData)
	fitted_lm = fit(LinearModel, @formula(Ys ~ Ws * ZsC), data)
    yhat = predict(fitted_lm)
    σ² = mean((data.Ys - yhat).^2) * length(data.Zs) / (length(data.Zs) - 4)
	return σ²
end


function bias_adjusted_gaussian_ci(se; maxbias=0.0, level=0.95)
    rel_bias = maxbias/se
    zz = fzero( z-> cdf(Normal(), rel_bias-z) + cdf(Normal(), -rel_bias-z) +  level -1,
        0, rel_bias - quantile(Normal(),(1- level)/2.1))
    zz*se
end

function fit(method::MinMaxOptRD, data::RDData; level=0.95)
	B = method.B
    ZsD = DiscretizedRunningVariable(data.ZsR, method.num_buckets)
    X = ZsD.Zs; h = ZsD.h; d = length(X); n= ZsD.weights
    ixc = argmin(abs.(ZsD.Zs .- ZsD.cutoff)); c = ZsD.cutoff; W = ZsD.Ws
	sig2 = estimate_var(data)
    σ² = zeros(d) .+ sig2
    model =  Model(method.solver)

    @variable(model, f[1:d])
    @variable(model, l1 >=0)
    @variable(model, l2)
    @variable(model, l3)
    @variable(model, l4)
    @variable(model, l5)
	@variable(model, s)

	G = @expression(model, 2*B.*f .+ l2.*(1 .-W) + l3.*W +l4.*(X .- c) + l5.*(W .- 0.5).*(X .-c))

    @constraint(model, f[ixc]==0)
    @constraint(model, (f[ixc+1] - f[ixc])/h[ixc]== 0)

    for i in 3:d
        @constraint(model, (f[i] - 2*f[i-1] + f[i-2])/(h[i-1]*h[i-2]) - l1 <=0)
        @constraint(model, (f[i] - 2*f[i-1] + f[i-2])/(h[i-1]*h[i-2]) + l1 >=0)
    end

	qobj = @expression(model, [1/2 .*sqrt.(n).*G ./sqrt.(σ²); l1])
	@constraint(model, [s; qobj] in SecondOrderCone())

    @objective(model, Min, s^2 - l2 + l3)
	optimize!(model)
    γ_xx = -value.(G)./(2 .*σ²)
    γ = γ_xx[ZsD.binmap]

	τ = sum(γ.*data.Ys)
	maxbias = value(l1)
	se = sqrt(sum(γ.^2)*sig2)
	ci = bias_adjusted_gaussian_ci(se; maxbias=maxbias, level=level)
	ci = [τ - ci, τ + ci]
	# TODO: add p-values and z-values?
	levstr = isinteger(level*100) ? string(Integer(level*100)) : string(level*100)
	res = [τ se maxbias first(ci) last(ci)]
    colnms = ["τ̂"; "se"; "max bias"; "Lower $levstr%"; "Upper $levstr%"]
    rownms = ["Sharp RD estimand"]
    coeftbl = CoefTable(res, colnms, rownms, 0, 0)

    FittedMinMaxOptRD(
        tau_est = sum(γ.*data.Ys),
        se_est = se,
		ci = ci,
        weights = γ,
		B = B,
		coeftable = coeftbl
    )
end


function Base.show(io::IO, rdd_fit::RegressionDiscontinuity.FittedMinMaxOptRD)
    println(io, "Imbens-Wager (2019) optimized regression discontinuity design")
    println(io, "Max Second Derivative Bound: ", string(rdd_fit.B))
    Base.show(io, rdd_fit.coeftable)
end
