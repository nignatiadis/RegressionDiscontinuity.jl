abstract type AbstractRunningVariable{T,C,VT} <: AbstractVector{T} end


"""
    RunningVariable(Zs; cutoff = 0.0, treated = :≥)

Represents the running variable values for data in a regression
discontinuity setting. The discontinuity is at `cutoff`, and `treated` is one of
`[:>; :>=; :≥; :≧; :<; :<=; :≤; :≦ ]' and determines the treatment based on the
running variable value compared to the cutoff.
"""
struct RunningVariable{T,C,VT} <: AbstractRunningVariable{T,C,VT}
    Zs::VT
    cutoff::C
    treated::Symbol
    Ws::BitArray{1}
    ZsC::VT
    function RunningVariable{T,C,VT}(Zs::VT, cutoff::C, treated) where {T,C,VT}
        treated = Symbol(treated)
        if treated ∉ [:>; :>=; :≥; :≧; :<; :<=; :≤; :≦]
            error("treated should be one of [:>; :>=; :≥; :≧; :<; :<=; :≤; :≦ ]")
        elseif treated ∈ [:>=; :≥; :≧]
            treated = :≥
        elseif treated ∈ [:<=; :≤; :≦]
            treated = :≤
        end
        Ws = broadcast(getfield(Base, treated), Zs, cutoff)
        ZsC = Zs .- cutoff
        new(Zs, cutoff, treated, Ws, ZsC)
    end
end

function RunningVariable(Zs::VT, cutoff::C, treated) where {C,VT}
    RunningVariable{eltype(VT),C,VT}(Zs, cutoff, treated)
end

RunningVariable(Zs; cutoff=0.0, treated=:≥) = RunningVariable(Zs, cutoff, treated)

Base.size(ZsR::AbstractRunningVariable) = Base.size(ZsR.Zs)
Base.maximum(ZsR::AbstractRunningVariable) = Base.maximum(ZsR.Zs)
Base.minimum(ZsR::AbstractRunningVariable) = Base.minimum(ZsR.Zs)
StatsBase.nobs(ZsR::AbstractRunningVariable) = length(ZsR)

Base.@propagate_inbounds function Base.getindex(ZsR::AbstractRunningVariable, x::Int)
    @boundscheck checkbounds(ZsR.Zs, x)
    @inbounds ret = getindex(ZsR.Zs, x)
    return ret
end

Base.@propagate_inbounds function Base.getindex(
    ZsR::AbstractRunningVariable,
    i::AbstractArray,
)
    @boundscheck checkbounds(ZsR, i)
    @inbounds Zs = ZsR.Zs[i]
    RunningVariable(Zs, ZsR.cutoff, ZsR.treated)
end


struct DiscretizedRunningVariable{T,C,VT} <: AbstractRunningVariable{T,C,VT}
    Zs::VT
    cutoff::C
    treated::Symbol
    Ws::BitArray{1}
    ZsC::VT
    weights::Array{Int,1}
    h::Array{Float64,1}
    binmap::Array{Int,1}

    function DiscretizedRunningVariable{T,C,VT}(
        ZsR::RunningVariable{T,C,VT},
        nbins::Int,
        bin_width::AbstractFloat
    ) where {T,C,VT}

        hist = fit(Histogram, ZsR; nbins=nbins, bin_width=bin_width)
        Zs = midpoints(hist.edges...)
        Ws = broadcast(getfield(Base, ZsR.treated), Zs, ZsR.cutoff)
        weights = hist.weights
        h = Zs[2:length(Zs)] .- Zs[1:(length(Zs) - 1)]
        binmap = StatsBase.binindex.(Ref(hist), ZsR.Zs)
        new(Zs, ZsR.cutoff, ZsR.treated, Ws, Zs .- ZsR.cutoff, weights, h, binmap)
    end
end

function DiscretizedRunningVariable(ZsR::RunningVariable{T,C,VT}, nbins::Int, bin_width::AbstractFloat) where {T,C,VT}
   DiscretizedRunningVariable{T,C,VT}(ZsR, nbins, bin_width)
end


# Tables interface

Tables.istable(ZsR::AbstractRunningVariable) = true
Tables.columnaccess(ZsR::AbstractRunningVariable) = true
Tables.columns(ZsR::AbstractRunningVariable) = (Ws = ZsR.Ws, Zs = ZsR.Zs, ZsC = ZsR.ZsC)
function Tables.schema(ZsR::AbstractRunningVariable)
    Tables.Schema((:Ws, :Zs, :ZsC), (eltype(ZsR.Ws), eltype(ZsR.Zs), eltype(ZsR.ZsC)))
end


function fit(
    ::Type{Histogram{T}},
    ZsR::AbstractRunningVariable;
    nbins=StatsBase.sturges(length(ZsR)),
    bin_width=bin_width(ZsR)
) where {T}
    @unpack cutoff, Zs, treated, ZsC = ZsR
    if treated in [:<; :≥]
        closed = :left
    else
        closed = :right
    end
    # nbins = iseven(nbins) ? nbins : nbins + 1

    min_Z, max_Z = extrema(Zs)


    # l  = floor((min_Z - cutoff) / bin_width) * bin_width + 0.5 * bin_width + cutoff
    
    l  = floor((min_Z - cutoff) / bin_width) * bin_width  + cutoff

    breaks = collect(range(l; step=bin_width, length=nbins))

    fit(Histogram{T}, ZsR, breaks; closed=closed)
end


@recipe function f(ZsR::AbstractRunningVariable)

    nbins = get(plotattributes, :bins, StatsBase.sturges(length(ZsR)))

    fitted_hist = fit(Histogram, ZsR; nbins=nbins)

    yguide --> "Frequency"
    xguide --> "Running variable"
    grid --> false
    label --> nothing
    fillcolor --> :lightgray
    thickness_scaling --> 1.7
    linewidth --> 0.3
    ylims --> (0, 1.5 * maximum(fitted_hist.weights))

    @series begin
        fitted_hist
    end

    @series begin
        seriestype := :vline
        linecolor := :purple
        linestyle := :dash
        linewidth := 1.7
        [ZsR.cutoff]
    end
end

"""
    RDData(Ys, ZsR::RunningVariable)

A dataset in the regression discontinuity setting. `Ys` is a vector of outcomes. 
"""
struct RDData{V,R <: AbstractRunningVariable}
    Ys::V
    ZsR::R
end

StatsBase.nobs(rdd_data::RDData) = nobs(rdd_data.ZsR)

Base.@propagate_inbounds function Base.getindex(rdd_data::RDData, i::AbstractArray)
    @boundscheck checkbounds(rdd_data.Ys, i)
    @boundscheck checkbounds(rdd_data.ZsR, i)

    @inbounds Ys = rdd_data.Ys[i]
    @inbounds ZsR = rdd_data.ZsR[i]
    RDData(Ys, ZsR)
end

# Tables interface

Tables.istable(::RDData) = true
Tables.columnaccess(::RDData) = true
Tables.columns(rdd_data::RDData) = merge((Ys = rdd_data.Ys,), Tables.columns(rdd_data.ZsR))
function Tables.schema(rdd_data::RDData)
    Tables.Schema(
        (:Ys, :Ws, :Zs, :cutoff, :ZsC),
        (eltype(rdd_data.Ys),
            eltype(rdd_data.ZsR.Ws),
            eltype(rdd_data.ZsR.Zs),
            typeof(cutoff),
            eltype(rdd_data.ZsR.ZsC),),
    )
end


function Base.getproperty(obj::RDData, sym::Symbol)
    if sym in (:Ys, :ZsR)
        return getfield(obj, sym)
    else
        return getfield(obj.ZsR, sym)
    end
end

function Base.propertynames(obj::RDData)
    (Base.fieldnames(typeof(obj))..., Base.fieldnames(typeof(obj.ZsR))...)
end



abstract type RDDIndexing end

struct Treated <: RDDIndexing end
struct Untreated <: RDDIndexing end

function Base.getindex(ZsR::R, i::Interval) where {R <: Union{RunningVariable,RDData}}
    idx = in.(ZsR.Zs, i)
    Base.getindex(ZsR, idx)
end





@recipe function f(ZsR::AbstractRunningVariable, Ys::AbstractVector)
    RDData(Ys, ZsR)
end

# -------------------------------------------------------
# include this temporarily, from Plots.jl codebase
# until issue 2360 is resolved
error_tuple(x) = x, x
error_tuple(x::Tuple) = x
_cycle(v::AbstractVector, idx::Int) = v[mod(idx, axes(v, 1))]
_cycle(v, idx::Int) = v

nanappend!(a::AbstractVector, b) = (push!(a, NaN); append!(a, b))

function error_coords(errorbar, errordata, otherdata...)
    ed = Vector{Float64}(undef, 0)
    od = [Vector{Float64}(undef, 0) for odi in otherdata]
    for (i, edi) in enumerate(errordata)
        for (j, odj) in enumerate(otherdata)
            odi = _cycle(odj, i)
            nanappend!(od[j], [odi, odi])
        end
        e1, e2 = error_tuple(_cycle(errorbar, i))
        nanappend!(ed, [edi - e1, edi + e2])
    end
    return (ed, od...)
end
# ----------------------------------------------------------
@recipe function f(rdd_data::RDData)

    ZsR = rdd_data.ZsR
    nbins = get(plotattributes, :bins, StatsBase.sturges(length(ZsR)))

    fitted_hist = fit(Histogram, ZsR; nbins=nbins)
    zs = StatsBase.midpoints(fitted_hist.edges[1])
    err_length = (zs[2] - zs[1]) / 2
    binidx = StatsBase.binindex.(Ref(fitted_hist), ZsR)

    tmp_df = rdd_data |> DataFrame
    tmp_df.binidx = binidx
    tmp_df = combine(groupby(tmp_df, :binidx), [:Ys => mean, :Ys => length])

    zs = zs[tmp_df.binidx]
    ys = tmp_df.Ys_mean

    yguide --> "Response"
    xguide --> "Running variable"
    grid --> false
    linecolor --> :grey
    linewidth --> 1.2
    thickness_scaling --> 1.7
    legend --> :outertop
    background_color_legend --> :transparent
    foreground_color_legend --> :transparent

    zs_scatter, ys_scatter = error_coords(err_length, zs, ys)

    # ylims --> extrema(ys)

    @series begin
        label --> "Regressogram"
        seriestype := :path
        zs_scatter, ys_scatter
    end

    @series begin
        label := nothing
        seriestype := :vline
        linecolor := :purple
        linestyle := :dash
        linewidth := 1.7
        [ZsR.cutoff]
    end
end
