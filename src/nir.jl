using .Empirikos
using ForwardDiff
using LinearFractional
using StatsFuns

abstract type AbstractRegressionDiscontinuityTarget end
abstract type TargetedRegressionDiscontinuityTarget <: AbstractRegressionDiscontinuityTarget end

struct ConstantTarget <: AbstractRegressionDiscontinuityTarget end

Base.@kwdef struct SharpRegressionDiscontinuityTarget{T,S} <: TargetedRegressionDiscontinuityTarget
    cutoff::T
    denom_est::S = 1.0
end

function (sharp_target::SharpRegressionDiscontinuityTarget)(u)
    likelihood( sharp_target.cutoff, u) / sharp_target.denom_est
end

function update_target(sharp_target::SharpRegressionDiscontinuityTarget, prior)
    denom_est = pdf(prior, sharp_target.cutoff)
    @set sharp_target.denom_est = denom_est
end


function Base.getproperty(obj::RunningVariable{<:EBayesSample}, sym::Symbol)
    if sym === :ZsC
        return response.(obj.Zs) .- obj.cutoff
    else
        return getfield(obj, sym)
    end
end


function StatsDiscretizations.unwrap(Z::EBayesSample)
    response(Z)
end

function StatsDiscretizations.wrap(Z::EBayesSample, val)
    @set Z.Z = val
end

export NoiseInducedRandomization

struct NoiseInducedRandomization{RDT, T,D,F,G,P,V,S} <: RegressionDiscontinuity.SharpRD
    response_lower_bound::T
    response_upper_bound::T
    solver
    convexclass::G
    plugin_G::P
    flocalization::F
    target::RDT
    discretizer::D
    bias_opt_multiplier::S
    σ_squared::V
    τ_range::S
    skip_bias::Bool
    maxbias_opt_bisection::Int
    α::S
end

function NoiseInducedRandomization(;
    response_lower_bound = 0.0,
    response_upper_bound = 1.0,
    convexclass = :default,
    solver,
    plugin_G = :default,
    flocalization = :default,
    target = ConstantTarget(),
    discretizer = :default,
    bias_opt_multiplier = 1.0,
    σ_squared = abs2(response_upper_bound - response_lower_bound),
    τ_range = 2*(response_upper_bound - response_lower_bound),
    skip_bias=false,
    maxbias_opt_bisection = 50,
    α = 0.05
    )
    NoiseInducedRandomization(
        response_lower_bound,
        response_upper_bound,
        solver,
        convexclass,
        plugin_G,
        flocalization,
        target,
        discretizer,
        bias_opt_multiplier,
        σ_squared,
        τ_range,
        skip_bias,
        maxbias_opt_bisection,
        α
    )
end

Base.@kwdef struct NoiseInducedRandomizationWeights{EB, Γ₊, Γ₋, H₊, H₋, M, T}
    γ₊::Γ₊
    γ₋::Γ₋
    h₊::H₊
    h₋::H₋
    eb_sample::EB
    model::M
    marginal_probs_treated::T
    marginal_probs_untreated::T
end


function nir_default_convexclass(Zs::AbstractVector{<:BinomialSample})
    DiscretePriorClass(range(1e-4, stop=1-1e-4, length=400))
end


function nir_default_discretizer(ZsR::RunningVariable{<:BinomialSample})
    cutoff = Int(ZsR.cutoff)
    Zs = ZsR.Zs
    skedasticity(Zs) === Empirikos.Homoskedastic() || throw("Only homoskedastic case supported.")
    K = ntrials(Zs[1])
    # TODO: add cases dpeending on >= who is treated
    ℓ = 0 #max(0, ceil(Int, cutoff - window_size))
    u = K #min(K, floor(Int, cutoff + window_size))
    #TODO: add checks
    discr_all = FiniteSupportDiscretizer(0:1:K)
    if ZsR.treated === :≥
        discr_untreated = FiniteSupportDiscretizer(ℓ:1:(cutoff-1))
        discr_treated = FiniteSupportDiscretizer(cutoff:1:u)
    elseif ZsR.treated === :>
        discr_untreated = FiniteSupportDiscretizer(ℓ:1:cutoff)
        discr_treated = FiniteSupportDiscretizer((cutoff+1):1:u)
    elseif ZsR.treated === :≤
        discr_treated = FiniteSupportDiscretizer(ℓ:1:cutoff)
        discr_untreated = FiniteSupportDiscretizer((cutoff+1):1:u)
    elseif ZsR.treated === :<
        discr_treated = FiniteSupportDiscretizer(ℓ:1:(cutoff-1))
        discr_untreated = FiniteSupportDiscretizer(cutoff:1:u)
    else
        throw(ArgumentError("RunningVariable:treated can only be one of :>, :<, :≥, :≤"))
    end

    (untreated = discr_untreated,
     treated = discr_treated,
     all = discr_all)
end

function nir_default_convexclass(Zs::AbstractVector{<:Empirikos.AbstractNormalSample})
    z_min, z_max = extrema(response.(Zs))
    DiscretePriorClass(range(z_min, stop=z_max, length=400))
end

function nir_default_discretizer(ZsR::RunningVariable{<:Empirikos.AbstractNormalSample})
    cutoff = ZsR.cutoff
    Zs = ZsR.Zs
    skedasticity(Zs) === Empirikos.Homoskedastic() || throw("Only homoskedastic case supported.")
    ν = Empirikos.std(Zs[1])
    window_size = 15*ν
    # TODO: add cases dpeending on >= who is treated
    ℓ = cutoff-window_size
    u = cutoff+window_size

    _range_left = range(ℓ; stop=cutoff, length=1000)
    _range_right = range(cutoff; stop=u, length=1000)
    _range_all = [_range_left[1:(end-1)]; cutoff; _range_right[2:end]]

    if ZsR.treated === :≥
        discr_untreated = BoundedIntervalDiscretizer{:closed,:open}(_range_left)
        discr_treated = BoundedIntervalDiscretizer{:closed,:open}(_range_right)
        discr_all = RealLineDiscretizer{:closed,:open}(_range_all)
    elseif ZsR.treated === :>
        discr_untreated = BoundedIntervalDiscretizer{:open,:closed}(_range_left)
        discr_treated = BoundedIntervalDiscretizer{:open,:closed}(_range_right)
        discr_all = RealLineDiscretizer{:open,:closed}(_range_all)
    elseif ZsR.treated === :≤
        discr_treated = BoundedIntervalDiscretizer{:open,:closed}(_range_left)
        discr_untreated = BoundedIntervalDiscretizer{:open,:closed}(_range_right)
        discr_all = RealLineDiscretizer{:open,:closed}(_range_all)
    elseif ZsR.treated === :<
        discr_treated = BoundedIntervalDiscretizer{:closed,:open}(_range_left)
        discr_untreated = BoundedIntervalDiscretizer{:closed,:open}(_range_right)
        discr_all = RealLineDiscretizer{:closed,:open}(_range_all)
    else
        throw(ArgumentError("RunningVariable:treated can only be one of :>, :<, :≥, :≤"))
    end

    (untreated = discr_untreated,
     treated = discr_treated,
     all = discr_all)
end


function initialize(nir::NoiseInducedRandomization, ZsR, Ys)
    Zs = ZsR.Zs
    cutoff = ZsR.cutoff

    n = nobs(Zs)
    if nir.flocalization === :default
        α_min = min(n^(-1/4),nir.α)
        nir = @set nir.flocalization = DvoretzkyKieferWolfowitz(α_min)
    end

    if nir.convexclass === :default
        #TODO: customize by sample type
        nir = @set nir.convexclass = nir_default_convexclass(Zs)
    end

    if nir.plugin_G === :default
        nir = @set nir.plugin_G = NPMLE(; solver = nir.solver, convexclass = nir.convexclass)
    end

    if nir.discretizer === :default
        # check for Homoskedastic
        nir = @set nir.discretizer = nir_default_discretizer(ZsR)
    end

    if nir.σ_squared === :lm
        lm_fit = fit(LinearModel, @formula(Ys~response(Zs)*Ws), RDData(Ys, ZsR))
        σ_squared = mean(abs2, residuals(lm_fit))
        nir = @set nir.σ_squared = σ_squared
    end
    nir
end

function nir_h(γ, Zs_levels, disc = Zs_levels)
    function h(u)
        dot(likelihood.(Zs_levels, u), γ.(disc))
    end
end


function add_bias_constraint!(model, nir::NoiseInducedRandomization{<:ConstantTarget}, h₊, h₋, t)
    M = nir.bias_opt_multiplier * (nir.response_upper_bound - nir.response_lower_bound) / 2
    for u in Distributions.support(nir.convexclass)
        @constraint(model,  M*(h₊(u) - h₋(u)) <=  t)
        @constraint(model,  M*(h₊(u) - h₋(u)) >= -t)
    end
end

function add_bias_constraint!(model, nir::NoiseInducedRandomization{<:TargetedRegressionDiscontinuityTarget}, h₊, h₋, t)
    target = nir.target
    M = nir.bias_opt_multiplier * (nir.response_upper_bound - nir.response_lower_bound) / 2
    τ_range = nir.τ_range/2

    @variable(model, t1)
    @variable(model, t2)

    for u in Distributions.support(nir.convexclass)
        @constraint(model,  M*(h₊(u) - h₋(u)) <=  t1)
        @constraint(model,  M*(h₊(u) - h₋(u)) >= -t1)
        @constraint(model,  τ_range*(h₊(u) - target(u)) <=  t2)
        @constraint(model,  τ_range*(h₊(u) - target(u)) >= -t2)
    end
    @constraint(model, t1 + t2 <= t)
end

function nir_weights_quadprog(nir::NoiseInducedRandomization, Zs)
    model = Model(nir.solver)
    σ_squared = nir.σ_squared
    n = nobs(Zs)
    eb_sample = skedasticity(Zs) === Empirikos.Homoskedastic() ? Zs[1] : throw("Only implemented for homoskedastic samples.")

    γ₊ = StatsDiscretizations.add_discretized_function!(model, nir.discretizer.treated)
    γ₋ = StatsDiscretizations.add_discretized_function!(model, nir.discretizer.untreated)

    Zs_levels_treated = Empirikos.set_response.(eb_sample, nir.discretizer.treated)
    Zs_levels_untreated = Empirikos.set_response.(eb_sample, nir.discretizer.untreated)

    h₊ = nir_h(γ₊, Zs_levels_treated, nir.discretizer.treated)
    h₋ = nir_h(γ₋, Zs_levels_untreated, nir.discretizer.untreated)

    @variable(model, t)

    add_bias_constraint!(model, nir, h₊, h₋, t)

    marginal_probs_treated = pdf.(nir.plugin_G, Zs_levels_treated)
    marginal_probs_untreated = pdf.(nir.plugin_G, Zs_levels_untreated)

    #return (marginal_probs_treated,marginal_probs_untreated, nir.plugin_G, Zs_levels_treated, Zs_levels_untreated)
    @constraint(model, dot(marginal_probs_treated, γ₊.(Zs_levels_treated)) ==  1)
    @constraint(model, dot(marginal_probs_untreated, γ₋.(Zs_levels_untreated)) ==  1)

    @variable(model, s)
    @constraint(model, [s;
                        t;
                        sqrt(σ_squared/n) .* γ₊.(nir.discretizer.treated) .* sqrt.(marginal_probs_treated);
                        sqrt(σ_squared/n) .* γ₋.(nir.discretizer.untreated) .* sqrt.(marginal_probs_untreated)
                    ]  ∈ SecondOrderCone())

    @objective(model, Min, s)
    optimize!(model)

    γ₊_fun = JuMP.value(γ₊, z->Empirikos.set_response(eb_sample, z))
    γ₊_fun = @set γ₊_fun.discretizer = nir.discretizer.all
    γ₋_fun = JuMP.value(γ₋, z->Empirikos.set_response(eb_sample, z))
    γ₋_fun = @set γ₋_fun.discretizer = nir.discretizer.all

    #return γ₊_fun, Zs_levels_treated
    h₊ = nir_h(γ₊_fun, Zs_levels_treated)
    h₋ = nir_h(γ₋_fun, Zs_levels_untreated)

    NoiseInducedRandomizationWeights(;γ₊=γ₊_fun, γ₋=γ₋_fun,
        h₊ = h₊, h₋ = h₋,
        model=model,
        eb_sample = eb_sample,
        marginal_probs_treated=marginal_probs_treated,
        marginal_probs_untreated=marginal_probs_untreated
    )
end



function nir_maxbias(nir::NoiseInducedRandomization{<:ConstantTarget}, γs, Zs)
    M = nir.response_upper_bound - nir.response_lower_bound

    bias_model = LinearFractionalModel(nir.solver)
    convexclass = nir.convexclass
    us = support(convexclass)

    hplus  = γs.h₊.(us)
    hminus = γs.h₋.(us)

    π = Empirikos.prior_variable!(bias_model, convexclass)
    _nparams = Empirikos.nparams(convexclass)

    @variable(bias_model, α[1:_nparams] >= 0)

    dkw_band = StatsBase.fit(nir.flocalization, Zs)
    Empirikos.flocalization_constraint!(bias_model, dkw_band, π)


    @constraint(bias_model.model, dot(π.finite_param, hplus) == 1)

    @objective(bias_model.model, JuMP.MOI.MAX_SENSE, dot(π.finite_param, hminus))
    optimize!(bias_model)

    _ξ_max = JuMP.objective_value(bias_model)

    @objective(bias_model.model, JuMP.MOI.MIN_SENSE, dot(π.finite_param, hminus))
    optimize!(bias_model)
    _ξ_min = JuMP.objective_value(bias_model)

    @constraint(bias_model.model, α .<= M.*π.finite_param)

    ξ = _ξ_min
    ξs = range(_ξ_min; stop=_ξ_max, length=nir.maxbias_opt_bisection)
    max_vals = zeros(length(ξs))
    #min_vals = zeros(length(ξs))

    @constraint(bias_model.model, parametric_constraint, dot(π.finite_param, hminus) == _ξ_min)

    for (i, ξ) in enumerate(ξs)
        JuMP.set_normalized_rhs(parametric_constraint, ξ)
        @objective(bias_model.model, JuMP.MOI.MAX_SENSE, dot(α, hplus) - dot(α, hminus)/ξ)
        optimize!(bias_model)
        max_vals[i] = JuMP.objective_value(bias_model)

        #@objective(bias_model.model, JuMP.MOI.MIN_SENSE, dot(α, hplus) - dot(α, hminus)/ξ)
        #optimize!(bias_model)
        #min_vals[i] = JuMP.objective_value(bias_model)
    end

    ξ = ξs[argmax(max_vals)]
    JuMP.set_normalized_rhs(parametric_constraint, ξ)
    @objective(bias_model.model, JuMP.MOI.MAX_SENSE, dot(α, hplus) - dot(α, hminus)/ξ)
    optimize!(bias_model)
    maxbias = JuMP.objective_value(bias_model)

    (maxbias=maxbias, bias_model=bias_model, ξ=ξ, ξ_min = _ξ_min, ξ_max = _ξ_max,
     ξs = ξs, grid_vals = max_vals)
end


function nir_maxbias(nir::NoiseInducedRandomization{<:TargetedRegressionDiscontinuityTarget}, γs, Zs)

    M = nir.response_upper_bound - nir.response_lower_bound
    τ_range = nir.τ_range

    bias_model = LinearFractionalModel(nir.solver)
    convexclass = nir.convexclass
    us = support(convexclass)

    hplus  = γs.h₊.(us)
    hminus = γs.h₋.(us)
    ws = nir.target.(us)


    π = Empirikos.prior_variable!(bias_model, convexclass)
    _nparams = Empirikos.nparams(convexclass)

    @variable(bias_model, α[1:_nparams] >= 0)
    @variable(bias_model, dΤ[1:_nparams] >= 0)

    dkw_band = StatsBase.fit(nir.flocalization, Zs)
    Empirikos.flocalization_constraint!(bias_model, dkw_band, π)


    @constraint(bias_model.model, dot(π.finite_param, hplus) == 1)

    @objective(bias_model.model, JuMP.MOI.MAX_SENSE, dot(π.finite_param, hminus))
    optimize!(bias_model)

    _ξ_max = JuMP.objective_value(bias_model)

    @objective(bias_model.model, JuMP.MOI.MIN_SENSE, dot(π.finite_param, hminus))
    optimize!(bias_model)
    _ξ_min = JuMP.objective_value(bias_model)

    @constraint(bias_model.model, α .<= M.*π.finite_param)
    @constraint(bias_model.model, dΤ .<= τ_range .* π.finite_param)

    ξ = _ξ_min
    ξs = range(_ξ_min; stop=_ξ_max, length=nir.maxbias_opt_bisection)

    @constraint(bias_model.model, parametric_constraint, dot(π.finite_param, hminus) == ξ)



    max_ws = zeros(length(ξs))
    min_ws = zeros(length(ξs))

    for (i, ξ) in enumerate(ξs)
        JuMP.set_normalized_rhs(parametric_constraint, ξ)
        @objective(bias_model.model, JuMP.MOI.MAX_SENSE, dot(π.finite_param, ws))
        optimize!(bias_model)
        max_ws[i] = JuMP.objective_value(bias_model)

        @objective(bias_model.model, JuMP.MOI.MIN_SENSE, dot(π.finite_param, ws))
        optimize!(bias_model)
        min_ws[i] = JuMP.objective_value(bias_model)
    end

    @constraint(bias_model.model, weight_constraint, dot(π.finite_param, ws) ==  max_ws[1])

    argmax_ws = zeros(length(ξs))
    maxvals_ws = zeros(length(ξs))

    #TODOs: also minimize.
    #TODOs: last resolve.
    for (i, ξ) in enumerate(ξs[2:(end-1)])
        JuMP.set_normalized_rhs(parametric_constraint, ξ)
        _w_list = range(min_ws[i]; stop=max(min_ws[i], max_ws[i]), step=0.02)
        tmp_vals = zeros(length(_w_list))
        for (j,w) in enumerate(_w_list)
            JuMP.set_normalized_rhs(weight_constraint, w)
            @objective(
                bias_model.model,
                JuMP.MOI.MAX_SENSE,
                dot(α, hplus) - dot(α, hminus)/ξ - dot(dΤ, hplus) + dot(dΤ, ws) / w
            )
            optimize!(bias_model)
            tmp_vals[j] = JuMP.objective_value(bias_model)
        end
        argmax_w_idx = argmax(tmp_vals)
        argmax_ws[i] = _w_list[argmax_w_idx]
        maxvals_ws[i] = tmp_vals[argmax_w_idx]
    end

    maxbias = maximum(maxvals_ws)

    (maxbias=maxbias, bias_model=bias_model, ξ=ξ, ξ_min = _ξ_min, ξ_max = _ξ_max,
     ξs = ξs, max_ws=max_ws, min_ws=min_ws, argmax_ws=argmax_ws, grid_vals = maxvals_ws)
end



Base.@kwdef struct FittedNoiseInducedRandomization
    coeftable
    τ̂
    se
    maxbias
    γs
    maxbias_fit
    nir
end

function Base.show(io::IO, nir::FittedNoiseInducedRandomization)
    Base.println(io::IO, "RD analysis with Noise Induced Randomization (NIR)")
    Base.show(io, nir.coeftable)
end

function StatsBase.fit(nir::NoiseInducedRandomization, ZsR::RunningVariable, Ys)
    nir = initialize(nir, ZsR, Ys)
    convexclass = nir.convexclass
    Zs = ZsR.Zs
    nir = @set nir.plugin_G = StatsBase.fit(nir.plugin_G, summarize(nir.discretizer.all.(Zs)))

    if !isa(nir.target, ConstantTarget)
        nir = @set nir.target = update_target(nir.target, nir.plugin_G)
    end


    γs = nir_weights_quadprog(nir, Zs)

    #return γs
    γ₊_denom = sum( γs.γ₊.(Zs,0))
    γ₋_denom = sum( γs.γ₋.(Zs,0))

    μ̂γ₊ = dot( γs.γ₊.(Zs,0), Ys) / γ₊_denom
    μ̂γ₋ = dot( γs.γ₋.(Zs,0), Ys) / γ₋_denom
    τ̂ = μ̂γ₊ - μ̂γ₋


    σ̂_squared =   dot( γs.γ₊.(Zs,0).^2, (Ys .- μ̂γ₊).^2)/γ₊_denom^2 +
                  dot( γs.γ₋.(Zs,0).^2, (Ys .- μ̂γ₋).^2)/γ₋_denom^2

    se = sqrt(σ̂_squared)

    if !nir.skip_bias
        maxbias_fit =  nir_maxbias(nir, γs, Zs)
        maxbias = maxbias_fit.maxbias
    else
        maxbias_fit = nothing
        maxbias = 0.0
    end

    level = 1-nir.α
    ci_halfwidth = bias_adjusted_gaussian_ci(se; maxbias = maxbias, level = level)
    ci = [τ̂ - ci_halfwidth, τ̂ + ci_halfwidth]
    levstr = isinteger(level * 100) ? string(Integer(level * 100)) : string(level * 100)
    res = [τ̂ se maxbias first(ci) last(ci) ci_halfwidth]
    colnms = ["τ̂"; "se"; "max bias"; "Lower $levstr%"; "Upper $levstr%"; "CI halfwidth"]
    rownms = ["Weighted RD estimand"]
    coeftbl = CoefTable(res, colnms, rownms, 0, 0)


    FittedNoiseInducedRandomization(coeftable = coeftbl, τ̂ = τ̂,
        γs = γs, se = se, maxbias = maxbias, maxbias_fit = maxbias_fit,
        nir = nir)
end


@recipe function f(rdd_weights::RegressionDiscontinuity.NoiseInducedRandomizationWeights{<:BinomialSample})
    γ₋ = rdd_weights.γ₋
    γ₊ = rdd_weights.γ₊
    ys = [.- collect(γ₋.dictionary); collect(γ₊.dictionary)]
    xs = response.([collect(keys(γ₋.dictionary)); collect(keys(γ₊.dictionary))])

    @series begin
        seriestype := :scatter
        seriescolor := :lightblue
        markerstrokewidth := 0
        markersize :=5
        label := ""
        xs, ys
    end

    label --> nothing
    seriestype --> :sticks
    linecolor --> :lightblue
    linewidth --> 2
    yguide --> L"\gamma(z)"
    xguide --> L"z"
    xs, ys
end



### Curvature Bound ###



Base.@kwdef struct NoiseInducedRandomizationCurvature{EB, F, G<:DiscretePriorClass, T, S, TT} <: WorstCaseCurvatureSelector
    response_lower_bound::T = 0.0
    response_upper_bound::T = 1.0
    density_lower_bound::S
    ebayes_sample::EB
    flocalization::F = nothing
    solver
    convexclass::G =  DiscretePriorClass(-2.5:0.01:2.5)
    n_grid_f_prime::Int = 200
    width_grid_f_double_prime::TT = 0.2
end


Base.@kwdef struct FittedNoiseInducedRandomizationCurvature{S,T, C<:NoiseInducedRandomizationCurvature}
    max_curvature::S
    us::T
    πs::T
    α0s::T
    f
    f′
    f′′
    nir_curvature::C
end

function curvature(nir_curvature::FittedNoiseInducedRandomizationCurvature)
    nir_curvature.max_curvature
end

function curvature(nir_curvature::NoiseInducedRandomizationCurvature)
    curvature(fit( nir_curvature ))
end

function fit(nir_curvature::NoiseInducedRandomizationCurvature)
    convexclass = nir_curvature.convexclass
    us = support(convexclass)
    _nparams = Empirikos.nparams(convexclass)
    M = nir_curvature.response_upper_bound - nir_curvature.response_lower_bound
    Z = nir_curvature.ebayes_sample
    model = Model(nir_curvature.solver)

    @variable(model, π[1:_nparams] >= 0)
    @variable(model, t >= 0)
    @constraint(model, sum(π) == t)
    @variable(model, H[1:_nparams] >= 0)
    @constraint(model, H .<=  M.* π)

    if isa(Z, BinomialSample)
        @show "binom"
        n_trials = ntrials(Z)
        function quasibinom_pdf(n, p, k)
            m = clamp(k, 0, n)
            val = betalogpdf(m + 1, n - m + 1, p) - log(n + 1)
            exp(val)
        end
        f_vec(z) = quasibinom_pdf.(n_trials, us, z)
        tmp_cutoff = response(Z)
        h = 1e-5
        f_vec_0 = f_vec(tmp_cutoff)
        f_vec_prime_0 = (f_vec(tmp_cutoff + h ) .- f_vec(tmp_cutoff - h ))./(2h)
        f_vec_double_prime_0 = (f_vec(tmp_cutoff + h ) .- 2f_vec_0 .+ f_vec(tmp_cutoff - h ))/abs2(h)
    else
        f_vec(z) = likelihood.(Empirikos.set_response(nir_curvature.ebayes_sample, z), us)
        f_vec_prime(z) = ForwardDiff.derivative(f_vec, z)
        f_vec_0 = f_vec(0.0)
        f_vec_prime_0 = f_vec_prime(0.0)
        f_vec_double_prime_0 = ForwardDiff.derivative(f_vec_prime, 0.0)
    end


    f = @expression(model, dot(π, f_vec_0))
    f′ = @expression(model, dot(π, f_vec_prime_0))
    f′′ = @expression(model, dot(π, f_vec_double_prime_0))

    α = @expression(model, dot(H, f_vec_0))
    α′ = @expression(model, dot(H, f_vec_prime_0))
    α′′ = @expression(model, dot(H, f_vec_double_prime_0))


    f_min = nir_curvature.density_lower_bound
    @constraint(model, f >= f_min*t)
    @constraint(model, f == 1)

    @objective(model, JuMP.MOI.MAX_SENSE,  f′)
    optimize!(model)
    _ξ_max = JuMP.value(f′)
    ξs = range(0.0; stop = _ξ_max, length = nir_curvature.n_grid_f_prime)

    @constraint(model, parametric_constraint, f′ == ξs[1])


    # range for f' and f''
    max_ws = zeros(length(ξs))
    min_ws = zeros(length(ξs))


    for (i, ξ) in enumerate(ξs)
        JuMP.set_normalized_rhs(parametric_constraint, ξ)
        @objective(model, JuMP.MOI.MAX_SENSE, f′′)
        optimize!(model)
        max_ws[i] = JuMP.value(f′′)

        @objective(model, JuMP.MOI.MIN_SENSE, f′′)
        optimize!(model)
        min_ws[i] = JuMP.value(f′′)
    end


    @constraint(model, weight_constraint, f′′ ==  max_ws[1])

    argmax_ws = zeros(length(ξs))
    maxvals_ws = zeros(length(ξs))

    for (i, ξ) in enumerate(ξs[2:(end-1)])
        JuMP.set_normalized_rhs(parametric_constraint, ξ)
        _w_list = range(min_ws[i]; stop=max(min_ws[i], max_ws[i]),
                     step=nir_curvature.width_grid_f_double_prime)
        tmp_vals = zeros(length(_w_list))
        for (j,w) in enumerate(_w_list)
            JuMP.set_normalized_rhs(weight_constraint, w)
            @objective(
                model,
                JuMP.MOI.MAX_SENSE,
                α′′ - 2*α′*ξ - α*w + 2α*ξ^2
            )
            optimize!(model)
            tmp_vals[j] = JuMP.objective_value(model)
        end
        argmax_w_idx = argmax(tmp_vals)
        argmax_ws[i] = _w_list[argmax_w_idx]
        maxvals_ws[i] = tmp_vals[argmax_w_idx]
    end

    max_curvature = maximum(maxvals_ws)
    _idx_ξ_worst_case = argmax(maxvals_ws)
    _ξ_worst_case= ξs[_idx_ξ_worst_case]
    w_worst_case = argmax_ws[_idx_ξ_worst_case]
    JuMP.set_normalized_rhs(parametric_constraint, _ξ_worst_case)
    JuMP.set_normalized_rhs(weight_constraint, w_worst_case)
    @objective(
        model,
        JuMP.MOI.MAX_SENSE,
        α′′ - 2*α′*_ξ_worst_case - α*w_worst_case + 2α*_ξ_worst_case^2
    )
    optimize!(model)

    π_worst = JuMP.value.(π)
    H_worst = JuMP.value.(H)

    πs = π_worst ./ sum(π_worst)
    α0s = ifelse.( iszero.(π_worst), zero(eltype(π_worst)) , H_worst ./ π_worst)


    myf(z) = dot(H_worst, f_vec(z) )/dot(π_worst, f_vec(z) )
    myf′(z) = ForwardDiff.derivative(myf, z)
    myf′′(z) = ForwardDiff.derivative(myf′, z)

    FittedNoiseInducedRandomizationCurvature(;
        max_curvature = f_vec, #myf′′(0),
        us=collect(us),
        πs=πs,
        α0s=α0s,
        f=myf,
        f′=myf′,
        f′′=myf′′,
        nir_curvature = nir_curvature
    )
end
