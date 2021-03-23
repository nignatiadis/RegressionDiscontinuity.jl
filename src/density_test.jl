# TODO MOVE to StatsDiscretizations

function _midpoints(discr::BoundedIntervalDiscretizer)
    StatsBase.midpoints(discr.grid)
end

function Base.count(discr::BoundedIntervalDiscretizer, zs::AbstractArray)
    cnts = zeros(Int, length(discr))
    for z in zs
        idx = findfirst(discr, z)
        cnts[idx] += 1
    end
    cnts
end



struct McCraryFanGijbels <: BandwidthSelector end

"""
    McCraryTest(bin::Union{Real,Nothing} = nothing, bw::Union{Real,Nothing} = nothing)

The McCrary (2008) test for manipulation in the running variable for a sharp
regression discontinuity estimator.

The `binwidth` object is the width of each bin. The default is 2*sd(runvar)*length(runvar)^(-0.5), following McCrary 2008.

The `bandwidth` object is the bandwidth. The default uses the calculation from McCrary (2008), pg. 705.

## Reference
McCrary, J. (2008).
Manipulation of the running variable in the regression discontinuity design: A density test.
Journal of econometrics, 142(2), 698-714.
"""
Base.@kwdef struct McCraryTest{K,H,W}
    binwidth::W = McCraryBinwidth()
    bandwidth::H = McCraryFanGijbels()
    kernel::K = SymTriangularDist()
end


# TODO: General kernels
# 3.20 and 3.22 of Fan and Gijbels.
#bandwidth(h::Number, kernel, rddata)
function _fan_gijbels_kernel_constant(kernel::SymTriangularDist)
    3.348
end




function bandwidth(
    method::McCraryTest{K, McCraryFanGijbels},
    left_tbl,
    right_tbl
    ) where {K}

    κ = _fan_gijbels_kernel_constant(method.kernel)
    c = left_tbl.cutoff

    #left part
    left_ols = lm(@formula(Ys ~ Zs + Zs^2 + Zs^3 + Zs^4 ), left_tbl)
    left_coef = coef(left_ols)
    left_mse =  deviance(left_ols) / dof_residual(left_ols)

    lf = 2 * left_coef[3] .+ 6 * left_coef[4] * left_tbl.Zs + 12 * left_coef[5] * left_tbl.Zs.^2
    lh = κ * (left_mse * (c - left_tbl.Zs[1]) / sum(abs2, lf))^(1/5)

    # Right part
    right_ols = lm(@formula(Ys ~ Zs + Zs^2 + Zs^3 + Zs^4 ), right_tbl)
    right_coef = coef(right_ols)
    right_mse =  deviance(right_ols) / dof_residual(right_ols)

    rf = 2 * right_coef[3] .+ 6 * right_coef[4] * right_tbl.Zs + 12 * right_coef[5] * right_tbl.Zs.^2
    rh = κ * (right_mse * (right_tbl.Zs[end] - c) / sum(abs2, rf))^(1/5)


    @show length(left_tbl.Zs), length(right_tbl.Zs), left_tbl.Zs[1], right_tbl.Zs[end]
    # Taking the average
    h = (rh + lh)/2
end

struct McCraryBinwidth end

function binwidth(b::Number, args...)
    b
end

function binwidth(::McCraryBinwidth, Zs::RunningVariable)
    2 * std(Zs) / sqrt(length(Zs))
end


function _mccrary_discretizer(b, Zs::RunningVariable)
    b = binwidth(b, Zs)
    rmin, rmax = extrema(Zs)
    cutoff = Zs.cutoff

    leftend = floor(rmin/b - eps(rmin/b))*b
    leftgrid_length =  round(Int,(cutoff-leftend)/b) + 1

    rightend = (1+ceil(rmax/b + eps(rmax/b)))*b
    rightgrid_length =  round(Int,(rightend-cutoff)/b)

    if Zs.treated ∈ (:≥, :<)
        L, R = :open, :closed
    elseif Zs.treated ∈ (:≤, :>)
        L, R = :closed, :open
    end

    discr_left =  BoundedIntervalDiscretizer{L,R}(
        range(leftend; stop=cutoff, length=leftgrid_length)
    )
    discr_right = BoundedIntervalDiscretizer{L,R}(
        range(cutoff; stop=rightend, length=rightgrid_length + 1)
    )

    if Zs.treated ∈ (:≥, :>)
        discr = (untreated = discr_left, treated = discr_right)
    elseif Zs.treated ∈ (:<, :≤)
        discr = (untreated = discr_rightt, treated = discr_left)
    end
    discr
end



Base.@kwdef struct FittedMcCraryTest{T,V,C,M}
    b::T
    h::T
    θhat::T
    σθ::T
    pval::T
    fitted_res::V
    coeftable::C
    method::M
end


"""
    fit(method::McCraryTest, data::RunningVariable)

Test for manipulation in the running variable following McCrary (2008).

# Arguments

- `data::RunningVariable`: The running variable.

# Returns
Returns  a FittedMcCraryTest object with fields `b` and `h` which are the bin size
and the bandwidth used, respectively. Also, it returns `θhat`, `σθ`, `pval`, `Js`,
and `fitted_res` which are the estimated difference in the intercepts,
the standard error of the estimate, the pval of the test H0: θhat = 0,
the number of bins used, and the fitted values for the support of the running variable
(as a DataFrame). Finally, `coeftable` provides an object of type CoefTable for the estimator.
"""
function fit(method::McCraryTest, Zs::RunningVariable)

    cutoff = Zs.cutoff
    treated_symbol = Zs.treated

    n = length(Zs)

    kernel(t) = pdf(method.kernel, t)

    b = binwidth(method.binwidth, Zs)
    discr = _mccrary_discretizer(b, Zs)


    Zs_untreated = Zs[Untreated()]
    Zs_treated = Zs[Treated()]

    wts_untreated = count(discr.untreated, Zs_untreated)
    wts_treated = count(discr.treated, Zs_treated)

    untreated_tbl = RDData(
        wts_untreated ./ (n*b),
        RunningVariable(_midpoints(discr.untreated), cutoff, treated_symbol)
    )

    treated_tbl = RDData(
        wts_treated ./ (n*b),
        RunningVariable(_midpoints(discr.treated), cutoff, treated_symbol)
    )

    if Zs.treated ∈ (:≥, :>)
        h = bandwidth(method, untreated_tbl, treated_tbl)
    elseif Zs.treated ∈ (:<, :≤)
        h = bandwidth(method, treated_tbl, untreated_tbl)
    end


    kernel_wts_untreated = kernel.( untreated_tbl.ZsC ./ h)
    kernel_wts_treated = kernel.( treated_tbl.ZsC ./ h)

    fitted_lm_untreated = fit(LinearModel, @formula(Ys ~ ZsC), untreated_tbl, wts = kernel_wts_untreated)
    fitted_lm_treated = fit(LinearModel, @formula(Ys ~ ZsC), treated_tbl, wts = kernel_wts_treated)

    untreated_f_hat = coef(fitted_lm_untreated)[1]
    treated_f_hat = coef(fitted_lm_treated)[1]

    # The estimates:
    θhat = log(treated_f_hat) - log(untreated_f_hat)
    σθ = sqrt((1 / (n * h)) * (24 / 5) * ( (1 / untreated_f_hat) + (1 / treated_f_hat)))
    z = θhat / σθ
    pval = 2(1 -  cdf(Normal(), abs(z)))

    # Creating the table
    res = [θhat σθ z pval]
    colnms = ["θ̂"; "σ̂"; "z"; "p-val"]
    rownms = ["McCrary Test"]
    coeftbl = CoefTable(res, colnms, rownms, 4, 3)


FittedMcCraryTest(
        b=b,
        h=h,
        θhat=θhat,
        σθ=σθ,
        pval=pval,
        fitted_res=nothing,
        coeftable=coeftbl,
        method=method
    )

end

function Base.show(io::IO, rdd_fit::FittedMcCraryTest)
    println(io, "The McCrary (2008) test for manipulation in the")
    println(io, "running variable for RDD.")
    println(io, "          ⋅⋅⋅⋅ " * "Bin size: ", string(round(rdd_fit.b, digits=4)))
    println(io, "          ⋅⋅⋅⋅ " * "Bandwidth size: ", string(round(rdd_fit.h, digits=4)))
    Base.show(io, rdd_fit.coeftable)
end
