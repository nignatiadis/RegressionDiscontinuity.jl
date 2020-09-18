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

