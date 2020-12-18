# Density test for manipulation in the running variable. 
# This work is based on McCrary, J. (2008). Manipulation of the running variable in the regression discontinuity design: A density test. Journal of econometrics, 142(2), 698-714.

"""
density_test(runvar,
            c=0.0;  
            bin::Union{Real,Nothing}=nothing,
            bw::Union{Real,Nothing}=nothing, 
            plot=true,
            verbose=true,
            generate=true)

Test for manipulation in the running variable following McCrary (2008).

# Arguments

- `runvar` Running variable.
- `c` Cutoff. The default is c = 0.0
- `bin` The width of each bin. The defaults is 2*sd(runvar)*length(runvar)^(-0.5), following McCrary 2008.
- `bw` The bandwidth. The defaults uses the calculation from McCrary (2008), pg. 705.
- `plot` Boolean indicating to print or not the density estimation. Default is `true`.
- `verbose` Boolean indicating wheter to print the information to the terminal. Default is `true`
- `generate` Boolean indicating wheter return the extended output. When `generate = false` only the pvalue is returned. If `generate = true` the function will return θ, σθ, the binsize used, the bandwidth used, the grid X, the normalized cell size Y, fhat, and σ_fhat. 


# Returns
If `plots = true` and `generate = true`
- `(pval, plot_final, θhat, σθ,  b, h, gen)` tuple consisting of pvalue of the test H0: θhat = 0, the plot, the estimated log difference in height at the cutpoint, the standard error of θ, the binsize used, the bandwidth used, and a data frame containing: the grid X, the normalized cell size Y, fhat, and σ_fhat.
If `plots = false` and `generate = true`
- `(pval, θhat, σθ,  b, h, gen)` tuple consisting of pvalue of the test H0: θhat = 0, the estimated log difference in height at the cutpoint, the standard error of θ, the binsize used, the bandwidth used, and a data frame containing: the grid X, the normalized cell size Y, fhat, and σ_fhat.
If `plots = false` and `generate = false`
- `(pvalue)` float consisting of the pvalue of the test H0: θhat = 0.
If `plots = true` and `generate = false`
- `(pval, plot_final)` tuple consisting of pvalue and the plot.
"""
function density_test(runvar,
            c=0.0;  
            bin::Union{Real,Nothing}=nothing,
            bw::Union{Real,Nothing}=nothing, 
            plot=true,
            verbose=true,
            generate=true)
    # Transforming the running variable into a dataframe.
    df = DataFrame(R=runvar)

    # Descriptive statistics of the running variable
    n = length(df.R)
    sd = std(df.R)
    rmax = max(df.R...)
    rmin = min(df.R...)
    
    if (rmin > c) || (rmax < c)
        error("The cutoff must be inside the support of the running variable.")
    end
    # Creating the triangular kernel used:
    tri_kernel(t) = max(0, (1 - abs(t)))

    # Bin size.
    if bin === nothing
        b = 2 * sd * n^(-.5)
    else
        b = bin
    end

    # Number of bins
    Jl = convert(Int64, round(floor((c - rmin) / b) + 1))
    J = convert(Int64, floor((rmax - rmin) / b) + 2)
    Jr = J - Jl

    # Creating the grid X
    l  = floor((rmin - c) / b) * b + 0.5 * b + c
    X = zeros(J)
    for j = 1:J
        X[j] = l + (j - 1) * b
    end

    # Creating the frequency table of the dicretetized version of the running variable    
    gR = floor.((df.R .- c) ./ b) * b .+ 0.5 * b .+ c;

    # Creating the smooth histogram Yi
    Y = zeros(J)
    for j = 1:J
        Y[j] = sum(1.0 * (gR .≈ X[j])) / (n * b)
    end

    # Bandwidth
    if bw === nothing
        # I will proceed to estimate the optimal bandwidth using McCrary (2008) optimal method.
        κ = 3.348
            
        # Left part
        ldf = DataFrame(lY=Y[1:Jl], lX=X[1:Jl])
        lols = lm(@formula(lY ~ lX + lX^2 + lX^3 + lX^4 ), ldf)
        lcoef = coef(lols)
        lmse =  deviance(lols) / dof_residual(lols)
        lf = 2 * lcoef[3] .+ 6 * lcoef[4] * ldf.lX + 12 * lcoef[5] * ldf.lX.^2 
            
        lh = κ * (lmse * (c - ldf.lX[1]) / sum(lf'lf))^(0.2)
            
        # Right part
        rdf = DataFrame(rY=Y[Jl + 1:end], rX=X[Jl + 1:end])
        rols = lm(@formula(rY ~ rX + rX^2 + rX^3 + rX^4 ), rdf)
        rcoef = coef(rols)
        rmse =  deviance(rols) / dof_residual(rols)
            
            
        rf = 2 * rcoef[3] .+ 6 * rcoef[4] * rdf.rX + 12 * rcoef[5] * rdf.rX.^2 
        rh = κ * (rmse * (rdf.rX[end] - c) / sum(rf'rf))^(0.2)
            
        # Taking the average
        h = 0.5 * (rh + lh)
    else
        h = bw
    end
    
    #######################
    # Plot
    #######################
    # Creating the dataframes to the left and to the right
    ldf = DataFrame(Y=Y[1:Jl], X=X[1:Jl], pred=zeros(Jl))
    rdf = DataFrame(Y=Y[Jl + 1:end], X=X[Jl + 1:end], pred=zeros(Jr))

    if plot
        #########################################
        # To eliminate the outliers at the end of the tails I will restrict the support to +- 2 sd from the cuttoff 
        llim = c - 2 * sd
        rlim = c + 2 * sd

        # Left side
        # Plotting the test
        for i in eachindex(ldf.X)
            ldf[!,:dist] = ldf.X  .-  ldf.X[i]
            wght = tri_kernel.(ldf.dist / h)
            aux = lm(@formula(Y ~ dist), ldf, wts=wght)
            pred_val = predict(aux)
            pred_se  = stderror(aux)[2]
            ldf.pred[i]  = pred_val[i]
        end

        m = min.((c .- ldf.X) / h, 1)

        lVarf = (12 / (5 * n * h)) .* ldf.Y .* (2.0 .+ 3 * m.^11 .- 24 * m.^10 .+ 83 * m.^9 .- 72 * m.^8 .- 42 * m.^7 .+ 18 * m.^6 .+ 18 * m.^5 .+ 18 * m.^4 .+ 3 * m.^3 .+ 18 * m.^2 .+ 15 * m) ./ ((1.0 .+ m.^6 .- 6 * m.^5 .- 3 * m.^4 .+ 4  * m.^3 .+ 9 * m.^2 .+ 6 * m))

        ldf[!, :se_pred] = sqrt.(lVarf)
        ldf[!, :CIup]  = ldf.pred .+ 1.96 .* ldf.se_pred
        ldf[!, :CIlow] = ldf.pred .- 1.96 .* ldf.se_pred
        #########################################

        #########################################
        # Right side
        for i in eachindex(rdf.X)
            rdf[!,:dist] = rdf.X  .- rdf.X[i] 
            wght = tri_kernel.(rdf.dist / h)
            aux = lm(@formula(Y ~ dist), rdf, wts=wght)
            pred_val = predict(aux)
            pred_se  = stderror(aux)[2]
            rdf.pred[i]  = pred_val[i]
        end

        # The variance estimator
        m = max.(-1, (-rdf.X .+ c) / h)

        rVarf = (12 / (5 * n * h)) .* rdf.Y .* (2.0 .- 3 * m.^11 .- 24 * m.^10 .- 83 * m.^9 .- 72 * m.^8 .+ 42 * m.^7 .+ 18 * m.^6 .- 18 * m.^5 .+ 18 * m.^4 .- 3 * m.^3 .+ 18 * m.^2 .- 15 * m) ./ ((1.0 .+ m.^6 .+ 6 * m.^5 .- 3 * m.^4 .- 4  * m.^3 .+ 9 * m.^2 .- 6 * m).^2 )
        
        rdf[!, :se_pred] = sqrt.(rVarf)
        rdf[!, :CIup]  = rdf.pred .+ 1.96 .* rdf.se_pred
        rdf[!, :CIlow] = rdf.pred .- 1.96 .* rdf.se_pred
        #########################################

        # Creating the plot
        plot_final = scatter(X, Y, leg=false, palette=:cyclic_grey_15_85_c0_n256,   alpha=0.5,  title="Density Test", background_color="#f3f6f9")
        plot!(ldf.X, ldf.pred, lw=2, linecolor=:black, alpha=0.55, palette=:cyclic_grey_15_85_c0_n256, ribbon=(ldf.CIup .- ldf.pred, ldf.pred .- ldf.CIlow))
        plot!(rdf.X, rdf.pred, lw=2, linecolor=:black, alpha=0.55, palette=:cyclic_grey_15_85_c0_n256, ribbon=(rdf.CIup .- rdf.pred, rdf.pred .- rdf.CIlow))

    end

    # Estimating the (log) difference in height
    auxdf = DataFrame(dist=0.0) 

    # Left side 
    Sminus(k) = sum(tri_kernel.((ldf.X .- c) ./ h) .* (ldf.X .- c).^k)
    lfhat = sum(tri_kernel.((ldf.X .- c) ./ h) .* ((Sminus(2) .- Sminus(1) .* (ldf.X .- c)) ./ (Sminus(2) * Sminus(0) - (Sminus(1))^2 )) .* ldf.Y)

    # Right side
    Splus(k) = sum(tri_kernel.((rdf.X .- c) ./ h) .* (rdf.X .- c).^k)
    rfhat = sum(tri_kernel.((rdf.X .- c) ./ h) .* ((Splus(2) .- Splus(1) .* (rdf.X .- c)) ./ (Splus(2) * Splus(0) - (Splus(1))^2 )) .* rdf.Y)


    θhat = log(rfhat) - log(lfhat)
    σθ = sqrt((1 / (n * h)) * (24 / 5) * ( (1 / rfhat) + (1 / lfhat)))
    z = θhat / σθ
    # pval = 1 - cdf(Normal(), abs(z))
    pval = 2(1 -  cdf(Normal(), abs(z)))

    #### Text

    if verbose
        println("The number of iterations (J):   ", J)
    end

    if verbose
        if bin === nothing
            println("Using default bin size calculation, bin size:   ", round(b, digits=4))
            else
            println("Using bin size given, bin size:                 ", round(b, digits=4))
    end
    end

    if verbose
        if bw === nothing
            println("Using default bandwidth calculation, bandwidth: ", round(h, digits=4))
            else
            println("Using bandwidth given, bandwidth:               ", round(h, digits=4))
        end
    end    

    if verbose
        println("\n")
        println("The (log) difference in height: ", round(θhat, digits=4))
        println("                               ", "($(round(σθ, digits=4)))")
        println("\n")
        println("The z-stat (θ/σ_θ) is :", round(z, digits=4))
        println("The p-value is        :", round(pval, digits=4))
        println("\n")
    end
    
    if generate
        gen = DataFrame(Y=Y, X=X, fhat=vcat(ldf.pred, rdf.pred), se_fhat=vcat(ldf.se_pred, rdf.se_pred))
        if plot 
            return pval, plot_final, θhat, σθ,  b, h, gen
            else
            return pval, b, h, gen
        end
    else
        if plot
            return pval, plot_final
            else
            return pval
        end
    end
end
        
        