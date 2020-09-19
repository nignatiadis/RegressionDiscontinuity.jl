struct RunningVariable{T, C, VT} <: AbstractVector{T}
	Zs::VT
	cutoff::C
	treated::Symbol
	Ws::BitArray{1}
	function RunningVariable{T, C, VT}(Zs::VT, cutoff::C, treated) where {T, C, VT}
		treated = Symbol(treated)
		if treated ∉ [:>; :>=; :≥; :≧; :<; :<=; :≤; :≦]
			error("treated should be one of [:>; :>=; :≥; :≧; :<; :<=; :≤; :≦ ]")
		elseif treated ∈ [:>=; :≥;  :≧]
			treated = :≥
		elseif treated ∈ [:<=; :≤; :≦]
			treated = :≤
		end
		Ws = broadcast(getfield(Base, treated), Zs, cutoff)
		new(Zs, cutoff, treated, Ws)
	end
end

function RunningVariable(Zs::VT, cutoff::C, treated) where {C,VT}
	RunningVariable{eltype(VT), C, VT}(Zs, cutoff, treated)
end

RunningVariable(Zs; cutoff=0.0, treated = :≥) = RunningVariable(Zs, cutoff, treated)

Base.size(ZsR::RunningVariable) = Base.size(ZsR.Zs)

Base.@propagate_inbounds function Base.getindex(ZsR::RunningVariable, x::Int)
    @boundscheck checkbounds(ZsR.Zs, x)
    @inbounds ret = getindex(ZsR.Zs, x)
    return ret
end


function fit(::Type{Histogram{T}}, ZsR::RunningVariable; nbins=StatsBase.sturges(length(ZsR))) where {T}
   @unpack cutoff, Zs, treated = ZsR
   if treated in [:<; :≥]
      closed = :left
   else
      closed = :right
   end 
   nbins =  iseven(nbins) ? nbins : nbins + 1
   
   min_Z, max_Z = extrema(Zs)

   bin_width = (max_Z - min_Z)*1.01/nbins

   prop_right = (max_Z - cutoff)*1.01/(max_Z - min_Z)
   prop_left = (cutoff - min_Z)*1.01/(max_Z - min_Z)
   nbins_right = ceil(Int, nbins * prop_right)
   nbins_left = ceil(Int, nbins * prop_left)

   breaks_left = reverse(range(cutoff; step=-bin_width, length=nbins_left))
   breaks_right = range(cutoff; step=bin_width, length=nbins_right)
   
   breaks = collect(sort(unique([breaks_left; breaks_right])))
      
   fit(Histogram{T}, ZsR, breaks; closed=closed)
end


@recipe function f(ZsR::RunningVariable; nbins = StatsBase.sturges(length(ZsR)))
   n = length(ZsR)
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