using Hypatia
using RegressionDiscontinuity
using Empirikos
using JuMP 
using Ipopt
using StatsBase
using LinearAlgebra
using Test
using IntervalSets
ecls_tbl = RegressionDiscontinuity.raw_table(RegressionDiscontinuity.ECLS_EIWW())
Zs = NormalSample.(ecls_tbl.Z, minimum(ecls_tbl.SE))
ZsR = RunningVariable(Zs, -0.2, :≥)

Ys = ecls_tbl.Y
nir = NoiseInducedRandomization(; solver=Hypatia.Optimizer)
nir_fit = fit(nir, ZsR, Ys)


γs = nir_fit.γs
nir = nir_fit.nir


m = Model(Ipopt.Optimizer)

M = nir.response_upper_bound - nir.response_lower_bound
convexclass = nir.convexclass
us = Empirikos.support(convexclass)

hplus  = γs.h₊.(us)
hminus = γs.h₋.(us)

π = Empirikos.prior_variable!(m, convexclass)
_nparams = Empirikos.nparams(convexclass)

@variable(m, α[1:_nparams] >= 0)

dkw_band = StatsBase.fit(nir.flocalization, Zs)
Empirikos.flocalization_constraint!(m, dkw_band, π)


@constraint(m, α .<= M .* π.finite_param)
@variable(m, hplus_expectation)
@variable(m, hminus_expectation)
@variable(m, hplus_α_expectation)
@variable(m, hminus_α_expectation)

@constraint(m, hplus_expectation == dot(π.finite_param, hplus))
@constraint(m, hminus_expectation == dot(π.finite_param, hminus))

@constraint(m, hplus_α_expectation == dot(α, hplus))
@constraint(m, hminus_α_expectation == dot(α, hminus))

@NLobjective(m, Max, hplus_α_expectation/(hplus_expectation + 1e-3) - hminus_α_expectation/(hminus_expectation + 1e-3))


JuMP.optimize!(m)

@test nir_fit.confint.maxbias ≈ JuMP.objective_value(m) rtol = 0.05


# # check with some offset


# ν = minimum(ecls_tbl.SE)


# sharp_target = RegressionDiscontinuity.SharpRegressionDiscontinuityTarget(;
#     cutoff = NormalSample(-0.25, ν),
# )

# sharp_target = RegressionDiscontinuity.SharpRegressionDiscontinuityTarget(;
#     cutoff = NormalSample(IntervalSets.Interval(-0.26, -0.2), ν),
# )

# nir_sharp_target = NoiseInducedRandomization(;
#     solver = Mosek.Optimizer,
#     target = sharp_target,
#     τ_range = 1.0,
# )

# nir_sharp_fit = fit(nir_sharp_target, ZsR, Ys)

# @test nir_sharp_fit.confint.maxbias/nir_fit.confint.maxbias > 1


# γs = nir_sharp_fit.γs
# nir = nir_sharp_fit.nir


# nir_refit = RegressionDiscontinuity.nir_sensitivity(nir_sharp_fit, ZsR, Ys, 0.6)

# @test nir_refit.confint.maxbias < nir_sharp_fit.confint.maxbias

# γs = nir_refit.γs
# nir = nir_refit.nir


# m = Model(Ipopt.Optimizer)

# M = nir.response_upper_bound - nir.response_lower_bound
# τ_range = nir.τ_range
# convexclass = nir.convexclass
# us = Empirikos.support(convexclass)

# hplus  = γs.h₊.(us)
# hminus = γs.h₋.(us)
# ws = nir.target.(us)

# π = Empirikos.prior_variable!(m, convexclass)
# _nparams = Empirikos.nparams(convexclass)

# @variable(m, α[1:_nparams] >= 0)
# @variable(m, dΤ[1:_nparams] >= 0)

# dkw_band = StatsBase.fit(nir.flocalization, Zs)
# Empirikos.flocalization_constraint!(m, dkw_band, π)


# @constraint(m, α .<= M .* π.finite_param)
# @constraint(m, dΤ .<= τ_range .* π.finite_param)


# @variable(m, hplus_expectation)
# @variable(m, hminus_expectation)
# @variable(m, w_expectation)

# @variable(m, hplus_α_expectation)
# @variable(m, hminus_α_expectation)

# @variable(m, hplus_dΤ_expectation)
# @variable(m, w_dΤ_expectation)

# @constraint(m, hplus_expectation == dot(π.finite_param, hplus))
# @constraint(m, hminus_expectation == dot(π.finite_param, hminus))

# @constraint(m, w_expectation == dot(π.finite_param, ws))

# @constraint(m, hplus_α_expectation == dot(α, hplus))
# @constraint(m, hminus_α_expectation == dot(α, hminus))


# @constraint(m, w_dΤ_expectation == dot(dΤ, ws))
# @constraint(m, hplus_dΤ_expectation == dot(dΤ, hplus))


# @NLobjective(m, Max, hplus_α_expectation/(hplus_expectation + 1e-3) - hminus_α_expectation/(hminus_expectation + 1e-3)  + hplus_dΤ_expectation/(hplus_expectation + 1e-3) - w_dΤ_expectation/(w_expectation + 1e-3))



# JuMP.optimize!(m)


# JuMP.objective_value(m)
# nir_sharp_fit.confint.maxbias

# nir_refit.confint.maxbias
# @test nir_refit.confint.maxbias ≈ JuMP.objective_value(m) rtol = 0.01

# @test nir_sharp_fit.confint.maxbias ≈ JuMP.objective_value(m) rtol = 0.01
