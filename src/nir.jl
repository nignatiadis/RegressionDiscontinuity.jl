using .Empirikos
using LinearFractional

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

struct NoiseInducedRandomization{T,D,F,G,P,V,S} <: RegressionDiscontinuity.SharpRD
    response_lower_bound::T
    response_upper_bound::T
    solver
    convexclass::G
    plugin_G::P
    flocalization::F
    discretizer::D
    bias_opt_multiplier::S
    σ_squared::V
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
    discretizer = :default,
    bias_opt_multiplier = 1.0,
    σ_squared = :default,
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
        discretizer,
        bias_opt_multiplier,
        σ_squared,
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
    window_size = 3*0.5*sqrt(K)
    # TODO: add cases dpeending on >= who is treated
    ℓ = max(0, ceil(Int, cutoff - window_size))
    u = min(K, floor(Int, cutoff + window_size))
    #TODO: add checks
    discr_all = FiniteSupportDiscretizer(0:1:K)
    discr_untreated = FiniteSupportDiscretizer(ℓ:1:(cutoff-1))
    discr_treated = FiniteSupportDiscretizer(cutoff:1:u)
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
    window_size = 3*ν
    # TODO: add cases dpeending on >= who is treated
    ℓ = cutoff-window_size
    u = cutoff+window_size
    #TODO: add checks
    discr_treated = BoundedIntervalDiscretizer{:closed,:open}(range(ℓ; stop=cutoff, length=200))
    discr_untreated = BoundedIntervalDiscretizer{:closed,:open}(range(cutoff; stop=u, length=200))
    discr_all = RealLineDiscretizer{:closed,:open}([discr_treated.grid[1:(end-1)]; discr_untreated.grid])
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

    if nir.σ_squared === :default
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


function nir_weights_quadprog(nir, Zs)
    model = Model(nir.solver)
    M = nir.bias_opt_multiplier * (nir.response_upper_bound - nir.response_lower_bound) / 2
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

    for u in Distributions.support(nir.convexclass)
        @constraint(model,  h₊(u) - h₋(u) <=  t)
        @constraint(model,  h₊(u) - h₋(u) >= -t)
    end

    marginal_probs_treated = pdf.(nir.plugin_G, Zs_levels_treated)
    marginal_probs_untreated = pdf.(nir.plugin_G, Zs_levels_untreated)

    #return (marginal_probs_treated,marginal_probs_untreated, nir.plugin_G, Zs_levels_treated, Zs_levels_untreated)
    @constraint(model, dot(marginal_probs_treated, γ₊.(Zs_levels_treated)) ==  1)
    @constraint(model, dot(marginal_probs_untreated, γ₋.(Zs_levels_untreated)) ==  1)

    @variable(model, s)
    @constraint(model, [s;
                        M*t;
                        sqrt(σ_squared/n) .* γ₊.(nir.discretizer.treated) .* sqrt.(marginal_probs_treated);
                        sqrt(σ_squared/n) .* γ₋.(nir.discretizer.untreated) .* sqrt.(marginal_probs_untreated)
                    ]  ∈ SecondOrderCone())

    @objective(model, Min, s)
    optimize!(model)

    γ₊_fun = JuMP.value(γ₊, z->Empirikos.set_response(eb_sample, z))
    γ₊_fun = @set γ₊_fun.discretizer = nir.discretizer.all
    γ₋_fun = JuMP.value(γ₋, z->Empirikos.set_response(eb_sample, z))
    γ₋_fun = @set γ₋_fun.discretizer = nir.discretizer.all

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


function StatsBase.fit(nir::NoiseInducedRandomization, ZsR::RunningVariable, Ys)
    nir = initialize(nir, ZsR, Ys)
    @unpack convexclass = nir
    Zs = ZsR.Zs
    nir = @set nir.plugin_G = StatsBase.fit(nir.plugin_G, summarize(nir.discretizer.all.(Zs)))

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

    maxbias_fit =  nir_maxbias(nir, γs, ZsR.Zs)
    maxbias = maxbias_fit.maxbias

    level = 1-nir.α
    ci_halfwidth = bias_adjusted_gaussian_ci(se; maxbias = maxbias, level = level)
    ci = [τ̂ - ci_halfwidth, τ̂ + ci_halfwidth]
    levstr = isinteger(level * 100) ? string(Integer(level * 100)) : string(level * 100)
    res = [τ̂ se maxbias first(ci) last(ci) ci_halfwidth]
    colnms = ["τ̂"; "se"; "max bias"; "Lower $levstr%"; "Upper $levstr%"; "CI halfwidth"]
    rownms = ["Weighted RD estimand"]
    coeftbl = CoefTable(res, colnms, rownms, 0, 0)


    (coeftable=coeftbl, τ̂=τ̂, γs=γs, se=se, maxbias=maxbias, maxbias_fit = maxbias_fit)
end

function nir_maxbias(nir, γs, Zs)
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

@recipe function f(rdd_weights::RegressionDiscontinuity.NoiseInducedRandomizationWeights{<:BinomialSample})
    γ₋ = rdd_weights.γ₋
    γ₊ = rdd_weights.γ₊
    ys = [.- collect(γ₋.dictionary); collect(γ₊.dictionary)]
    xs = [γ₋.discretizer; γ₊.discretizer]

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
