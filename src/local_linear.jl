abstract type SharpRD end

abstract type VarianceEstimator end

struct EickerHuberWhite <: VarianceEstimator end
_string(::EickerHuberWhite) = "Eicker White Huber variance"

Base.@kwdef struct NaiveLocalLinearRD{K,B,V<:VarianceEstimator} <: SharpRD
    kernel::K
    bandwidth::B
    variance::V = EickerHuberWhite()
end

Base.@kwdef struct FittedLocalLinearRD{R,F,K,B,S,T,C}
    rdd_setting::R
    fitted_lm::F
    fitted_kernel::K
    fitted_bandwidth::B
    data_subset::S
    tau_est::T
    se_est::T
    coeftable::C
end

linearweights(fitted::FittedLocalLinearRD) = linearweights(fitted.fitted_lm)

function linearweights(fitted_lm::RegressionModel; idx = 2)
    wts = fitted_lm.model.rr.wts
    γs = (fitted_lm.model.pp.chol\fitted_lm.model.pp.X'*Diagonal(wts))[idx, :]
    γs
end

function var(::EickerHuberWhite, fitted_lm::RegressionModel)
    γs = linearweights(fitted_lm)
    dot(abs2.(γs), abs2.(residuals(fitted_lm)))
end


fit(method::SharpRD, ZsR::RunningVariable, Y) = fit(method, RDData(Y, ZsR))

function fit(method::NaiveLocalLinearRD, rddata::RDData; level=0.95)
    c = rddata.cutoff
    @unpack kernel, variance = method
    h = bandwidth(method.bandwidth, kernel, rddata)
    fitted_kernel = setbandwidth(kernel, h)

    @unpack lb, ub = support(fitted_kernel)
    new_support = Interval(lb + c, ub + c)

    rddata_filt = rddata[new_support]
    wts = weights(fitted_kernel, rddata_filt.ZsR)

    fitted_lm = fit(LinearModel, @formula(Ys ~ Ws * ZsC), rddata_filt, wts = wts)

    tau_est = coef(fitted_lm)[2]
    se_est = sqrt(var(variance, fitted_lm))
    γs = linearweights(fitted_lm)

    z = tau_est/se_est
    pval = 1-cdf(Normal(), abs(z))
    z_quantile = quantile(Normal(), 1-level/2)
    ci = Interval(tau_est - se_est*z_quantile, tau_est + se_est*z_quantile)
    # as in GLM.jl
    levstr = isinteger(level*100) ? string(Integer(level*100)) : string(level*100)

    res = [h tau_est se_est "unaccounted" z pval leftendpoint(ci) rightendpoint(ci)]
    colnms = ["h"; "τ̂"; "se"; "bias"; "z"; "p-val"; "Lower $levstr%"; "Upper $levstr%"]
    rownms = ["Sharp RD estimand"]
    coeftbl = CoefTable(res, colnms, rownms, 6, 5)

    FittedLocalLinearRD(
        rdd_setting = method,
        fitted_lm = fitted_lm,
        fitted_kernel = fitted_kernel,
        fitted_bandwidth = h,
        data_subset = rddata_filt,
        tau_est = tau_est,
        se_est = se_est,
        coeftable = coeftbl,
    )
end





function Base.show(io::IO, rdd_fit::RegressionDiscontinuity.FittedLocalLinearRD)
    println(io, "Local linear regression for regression discontinuity design")
    println(io, "       ⋅⋅⋅⋅ " * "Naive inference (not accounting for bias)")
    println(io, "       ⋅⋅⋅⋅ " * _string(rdd_fit.rdd_setting.kernel))
    println(io, "       ⋅⋅⋅⋅ " * _string(rdd_fit.rdd_setting.bandwidth))
    println(io, "       ⋅⋅⋅⋅ " * _string(rdd_fit.rdd_setting.variance))
    Base.show(io, rdd_fit.coeftable)
end

@recipe function f(rdd_fit::FittedLocalLinearRD; show_local_support = false)
    sorted_subset = rdd_fit.data_subset |> DataFrame |> df -> sort(df, [:Zs])
    sorted_preds = predict(rdd_fit.fitted_lm, sorted_subset)
    if show_local_support
        @unpack lb, ub = support(rdd_fit.fitted_kernel)
        @series begin
            label := nothing
            seriestype := :vline
            linecolor := :lightgrey
            inestyle := :dot
            linewidth := 2
            [lb; ub]
        end
    end
    seriestype = :path
    linecolor --> :lightblue
    linewidth --> 2
    label --> "Local linear fit"
    sorted_subset.Zs, sorted_preds
end
