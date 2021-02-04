# Density test for manipulation in the running variable. 
# This work is based on McCrary, J. (2008). Manipulation of the running variable in the regression discontinuity design: A density test. Journal of econometrics, 142(2), 698-714.

"""
    McCraryTest(bin::Union{Real,Nothing} = nothing, bw::Union{Real,Nothing} = nothing) 

A McCrary (2008) test for manipulation in the running variable for `SharpRD` estimator.

The `bin` object is the width of each bin. The default is 2*sd(runvar)*length(runvar)^(-0.5), following McCrary 2008.

The `bw` object is the bandwidth. The default uses the calculation from McCrary (2008), pg. 705.
"""
Base.@kwdef struct McCraryTest{K}
    bin::Union{Real,Nothing} = nothing 
    bw::Union{Real,Nothing} = nothing 
    kernel::K = SymTriangularDist()
end

Base.@kwdef struct FittedMcCraryTest{T,M,V,C}
    b::T
    h::T
    θhat::T
    σθ::T
    pval::T
    Js::M
    fitted_res::V
    coeftable::C
end


# 1. Descriptive Statistics

function descriptive_stats(RunVar::RunningVariable)
    n = Base.length(RunVar)
    sd = StatsBase.std(RunVar)
    rmax = Base.maximum(RunVar)
    rmin = Base.minimum(RunVar)
    return n, sd, rmax, rmin
end

# 2. Binsize (uses descriptive_stats)
# function Binsize(n, sd, bin::Union{Real,Nothing}=nothing)
function bin_width(n, sd)
    b = 2 * sd * n^(-.5)
end

function bin_width(Zs::RunningVariable)
    n, sd, rmax, rmin = descriptive_stats(Zs)
    return b = 2 * sd * (n)^(-.5)
end


# 3. Number of bins
function bin_number(c, b, rmin, rmax)
    Jl = convert(Int64, round(floor((c - rmin) / b) + 1))
    J = convert(Int64, floor((rmax - rmin) / b) + 2)
    Jr = J - Jl
    return J, Jl, Jr
end

function bin_number(Zs::RunningVariable)
    n, sd, rmax, rmin = descriptive_stats(Zs)
    b = bin_width(Zs)
    c = Zs.cutoff
    Jl = convert(Int64, round(floor((c - rmin) / b) + 1))
    J = convert(Int64, floor((rmax - rmin) / b) + 2)
    Jr = J - Jl
    return J, Jl, Jr
end

function bin_number(method::McCraryTest, Zs::RunningVariable)
    n, sd, rmax, rmin = descriptive_stats(Zs)

    if method.bin === nothing
        b = bin_width(Zs) 
    else
        b = method.bin
    end
    c = Zs.cutoff
    Jl = convert(Int64, round(floor((c - rmin) / b) + 1))
    J = convert(Int64, floor((rmax - rmin) / b) + 2)
    Jr = J - Jl
    return J, Jl, Jr
end

# 4. Kernel Constant for McCrary
### According to footnote 18 of McCrary (2008), this constant cannot be computed by the standard formula in the book (Eq 4.3) because is a boundary case. What I understand is that the constant is calculted from 3.20 and 3.22 of Fan and Gijbels. 

function kernel_constant(method::McCraryTest)
    # Here I need to see how to calculate this. For now its output is just for the Triangular Kernel 
    # kernel = EquivalentKernel(method.kernel)

    κ = 3.348
end


# 5. Bandwidth, needs (4), (3), (2), (1)
# function bandwidth(mccrary::McCrary, kernel::SymTriangularDist, X, Y, Jl, c)

function bandwidth(method::McCraryTest, Zs::RunningVariable)
        # I will proceed to estimate the optimal bandwidth using McCrary (2008) optimal method.
    κ = kernel_constant(method)
    c = Zs.cutoff

    # Discretized Running Variable
    n, sd, rmin, rmax = descriptive_stats(Zs)

    #### NEED TO ADAPT THIS
    if method.bin === nothing
        b = bin_width(Zs) 
    else
        b = method.bin
    end
   
    J, Jl, Jr = bin_number(method, Zs)
    
    DRV = DiscretizedRunningVariable(Zs, J + 1, b)
    X = collect(DRV)
    Y = DRV.weights / (n * b)

    # Left part
    ldf = DataFrame(Y=Y[1:Jl], X=X[1:Jl])
    lols = lm(@formula(Y ~ X + X^2 + X^3 + X^4 ), ldf)
    lcoef = coef(lols)
    lmse =  deviance(lols) / dof_residual(lols)

    lf = 2 * lcoef[3] .+ 6 * lcoef[4] * ldf.X + 12 * lcoef[5] * ldf.X.^2 
    lh = κ * (lmse * (c - ldf.X[1]) / sum(lf'lf))^(0.2)
            
    # Right part
    rdf = DataFrame(Y=Y[Jl + 1:end], X=X[Jl + 1:end])
    rols = lm(@formula(Y ~ X + X^2 + X^3 + X^4 ), rdf)
    rcoef = coef(rols)
    rmse =  deviance(rols) / dof_residual(rols)
            
            
    rf = 2 * rcoef[3] .+ 6 * rcoef[4] * rdf.X + 12 * rcoef[5] * rdf.X.^2 
    rh = κ * (rmse * (rdf.X[end] - c) / sum(rf'rf))^(0.2)
            
    # Taking the average
    h = 0.5 * (rh + lh)
end

# 6. McCraryTest
"""
    fit(method::McCraryTest, data::RunningVariable)

Test for manipulation in the running variable following McCrary (2008).

# Arguments

- `data::RunningVariable`: The running variable.

# Returns
Returns  a FittedMcCraryTest object with fields `b` and `h` which are the bin size and the bandwidth used, respectively. Also, it returns `θhat`, `σθ`, `pval`, `Js`, and `fitted_res` which are the estimated difference in the intercepts, the standard error of the estimate, the pval of the test H0: θhat = 0, the number of bins used, and the fitted values for the support of the running variable (as a DataFrame). Finally, `coeftable` provides an object of type CoefTable for the estimator. 
"""
function fit(method::McCraryTest, Zs::RunningVariable)
    
    # Cutoff selection
    c = Zs.cutoff
    R = Zs
    kernel(t) = pdf(method.kernel, t) 

    # Call the functions 
    # 1.
    n, sd, rmax, rmin = descriptive_stats(Zs)

    # 2.
    if method.bin === nothing
        b = bin_width(Zs)
    else
        b = method.bin
    end

    # 3.
    J, Jl, Jr = bin_number(method, Zs)
    Js = [J; Jl; Jr]

    # 4.
    DRV = DiscretizedRunningVariable(Zs, J + 1, b)
    X = collect(DRV)
    Y = DRV.weights / (n * b)

    # 5.
    if method.bw === nothing
        h = bandwidth(method, Zs)
    else
        h = method.bw
    end
    
    ##############################################################
    # Smooth histograms to the left and to the right of the cutoff
    ldf = DataFrame(Y=Y[1:Jl], X=X[1:Jl], pred=zeros(Jl))
    rdf = DataFrame(Y=Y[Jl + 1:end], X=X[Jl + 1:end], pred=zeros(Jr))

    # Left side 
    Sminus(k) = sum(kernel.((ldf.X .- c) ./ h) .* (ldf.X .- c).^k)
    lfhat = sum(kernel.((ldf.X .- c) ./ h) .* ((Sminus(2) .- Sminus(1) .* (ldf.X .- c)) ./ (Sminus(2) * Sminus(0) - (Sminus(1))^2 )) .* ldf.Y)
 
    # Right side
    Splus(k) = sum(kernel.((rdf.X .- c) ./ h) .* (rdf.X .- c).^k)
    rfhat = sum(kernel.((rdf.X .- c) ./ h) .* ((Splus(2) .- Splus(1) .* (rdf.X .- c)) ./ (Splus(2) * Splus(0) - (Splus(1))^2)) .* rdf.Y)
 
    # The estimates:
    θhat = log(rfhat) - log(lfhat)
    σθ = sqrt((1 / (n * h)) * (24 / 5) * ( (1 / rfhat) + (1 / lfhat)))
    z = θhat / σθ
    pval = 2(1 -  cdf(Normal(), abs(z)))

    # Creating the table
    res = [θhat σθ z pval]
    colnms = ["θ̂"; "σ̂"; "z"; "p-val"]
    rownms = ["McCrary Test"]
    coeftbl = CoefTable(res, colnms, rownms, 4, 3)

    # Fitted Values:
        # Left side
    for i in eachindex(ldf.X)
        ldf[!,:dist] = ldf.X  .-  ldf.X[i]
        lwght = kernel.(ldf.dist / h)
        laux = lm(@formula(Y ~ dist), ldf, wts=lwght)
        ldf.pred[i] = predict(laux)[i]
    end

    m = min.((c .- ldf.X) / h, 1)

    lVarf = (12 / (5 * n * h)) .* ldf.Y .* (2.0 .+ 3 * m.^11 .- 24 * m.^10 .+ 83 * m.^9 .- 72 * m.^8 .- 42 * m.^7 .+ 18 * m.^6 .+ 18 * m.^5 .+ 18 * m.^4 .+ 3 * m.^3 .+ 18 * m.^2 .+ 15 * m) ./ ((1.0 .+ m.^6 .- 6 * m.^5 .- 3 * m.^4 .+ 4  * m.^3 .+ 9 * m.^2 .+ 6 * m))

    ldf[!, :se_pred] = sqrt.(lVarf)

        # Right side
    for i in eachindex(rdf.X)
        rdf[!,:dist] = rdf.X  .- rdf.X[i] 
        rwght = kernel.(rdf.dist / h)
        raux = lm(@formula(Y ~ dist), rdf, wts=rwght)
        rdf.pred[i] = predict(raux)[i]
            # pred_val = predict(raux)
            # rdf.pred[i]  = pred_val[i]
    end

    m = max.(-1, (-rdf.X .+ c) / h)

    rVarf = (12 / (5 * n * h)) .* rdf.Y .* (2.0 .- 3 * m.^11 .- 24 * m.^10 .- 83 * m.^9 .- 72 * m.^8 .+ 42 * m.^7 .+ 18 * m.^6 .- 18 * m.^5 .+ 18 * m.^4 .- 3 * m.^3 .+ 18 * m.^2 .- 15 * m) ./ ((1.0 .+ m.^6 .+ 6 * m.^5 .- 3 * m.^4 .- 4  * m.^3 .+ 9 * m.^2 .- 6 * m).^2 )
    
    rdf[!, :se_pred] = sqrt.(rVarf)

fitted_res = DataFrame(Y=Y, X=X, fhat=vcat(ldf.pred, rdf.pred), se_fhat=vcat(ldf.se_pred, rdf.se_pred))
# fitted_res = DataFrame(Y=Y, X=X, fhat=vcat(ldf.pred, rdf.pred))

    FittedMcCraryTest(
        b=b,
        h=h,
        θhat=θhat,
        σθ=σθ,
        pval=pval,
        Js=Js,
        fitted_res=fitted_res,
        coeftable=coeftbl
    )

end
function Base.show(io::IO, rdd_fit::FittedMcCraryTest)
    println(io, "The McCrary (2008) test for manipulation in the")
    println(io, "running variable for RDD.")
    println(io, "          ⋅⋅⋅⋅ " * "Bin size: ", string(round(rdd_fit.b, digits=4)))
    println(io, "          ⋅⋅⋅⋅ " * "Bandwidth size: ", string(round(rdd_fit.h, digits=4)))
    println(io, "          ⋅⋅⋅⋅ " * "Number of iterations: ", string(rdd_fit.Js[1]))
    Base.show(io, rdd_fit.coeftable)
end


# 7. Plot

@recipe function f(rdd_fit::FittedMcCraryTest)
    # Recovering the data
    X = rdd_fit.fitted_res[!,:X]
    Y = rdd_fit.fitted_res[!,:Y]
    f̂ = rdd_fit.fitted_res[!,:fhat]
    Jl = rdd_fit.Js[2]
    lX = X[1:Jl]
    rX = X[Jl + 1:end]
    lf̂ = f̂[1:Jl]
    rf̂ = f̂[Jl + 1:end]

    # set up
    legend := :false
    grid := false

    # scatter
    @series begin
        seriestype := :scatter
        seriescolor := :lightgray
        linecolor --> :lightgray
        seriesalpha := 0.99
        X, Y
    end

    # common to both series
    seriestype := :line
    linecolor --> :firebrick
    linewidth --> 2
    label --> "McCrary Test"

    # left side
    @series begin
        lX, lf̂
    end

    # left side
    @series begin
        rX, rf̂
end
end

