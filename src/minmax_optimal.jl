"""
    ImbensWagerOptRD(B::Float64, solver, num_buckets=2000, variance=Homoskedastic()) <: SharpRD

A `SharpRD` estimator.

`B` is a bound on the second derivative of the mean of
the outcome variable, conditional on the running variable above and below the cutoff.

The `solver` object is a `MathOptInterface` compatible optimizer such as `Mosek.Optimizer` that will be
used to solve the numerical optimization problem described in Imbens & Wager (2019).

The number of bins for the discretization of the running variable that is required to solve
the numerical optimization problem is `num_buckets`.
"""
Base.@kwdef struct ImbensWagerOptRD <: SharpRD
    B::Float64
	solver
	num_buckets = 2000
	variance = Homoskedastic()
end

Base.@kwdef struct FittedOptRD
	tau_est::AbstractFloat
    se_est::AbstractFloat
	maxbias::AbstractFloat
	ci::Array{Float64}
    weights::Array{Float64}
	B::Float64
	coeftable
end
"""
    Homoskedastic()

Variance estimator for the RD estimator that assumes homoskedasticity.
"""
struct Homoskedastic <: VarianceEstimator end

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

function var(::Homoskedastic, γ::Vector, sig2::Float64)
	sqrt(sum(γ.^2)*sig2)
end

"""
    fit(method::ImbensWagerOptRD, data::RDData; level=0.95)

Find the min-max optimal RD estimator over a convex class of conditional mean
functions with a bounded second derivative, based on the numerical optimization
procedure described in Imbens & Wager (2019).

Returns a FittedOptRD object with fields `tau_est`, `se_est`, `maxbias`, `ci`,
`weights` the RD estimator weights for each observation, `B` the bound on
the second derivative of the conditional mean functions used in estimation, and
`coeftable` providing an object of type CoefTable for the estimator.
"""
function fit(method::ImbensWagerOptRD, data::RDData; level=0.95)
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
	se = var(method.variance, γ, sig2)
	ci = bias_adjusted_gaussian_ci(se; maxbias=maxbias, level=level)
	ci = [τ - ci, τ + ci]
	levstr = isinteger(level*100) ? string(Integer(level*100)) : string(level*100)
	res = [τ se maxbias first(ci) last(ci)]
    colnms = ["τ̂"; "se"; "max bias"; "Lower $levstr%"; "Upper $levstr%"]
    rownms = ["Sharp RD estimand"]
    coeftbl = CoefTable(res, colnms, rownms, 0, 0)

    FittedOptRD(
        tau_est = sum(γ.*data.Ys),
        se_est = se,
		maxbias = maxbias,
		ci = ci,
        weights = γ,
		B = B,
		coeftable = coeftbl
    )
end


function Base.show(io::IO, rdd_fit::RegressionDiscontinuity.FittedOptRD)
    println(io, "Imbens-Wager (2019) optimized regression discontinuity design")
    println(io, "Max Second Derivative Bound: ", string(rdd_fit.B))
    Base.show(io, rdd_fit.coeftable)
end
