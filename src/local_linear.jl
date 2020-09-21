abstract type BandwidthSelector end 

abstract type SharpRD end 

abstract type VarianceEstimator end

struct EickerHuberWhite <: VarianceEstimator end 
_string(::EickerHuberWhite) = "Eicker White Huber variance"

Base.@kwdef struct NaiveLocalLinearRD{K, B, V<:VarianceEstimator} <: SharpRD
	kernel::K
	bandwidth::B
	variance::V = EickerHuberWhite()
end



Base.@kwdef struct FittedLocalLinearRD{R, F, K, B, S, T, C}
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

function linearweights(fitted_lm::RegressionModel; idx=2)
	wts = fitted_lm.model.rr.wts
	γs = (fitted_lm.model.pp.chol \ fitted_lm.model.pp.X' * Diagonal(wts))[idx,:]
	γs
end 

function var(::EickerHuberWhite, fitted_lm::RegressionModel) 
	γs = linearweights(fitted_lm)
	dot(abs2.(γs), abs2.(residuals(fitted_lm)))
end 


fit(method::SharpRD, ZsR::RunningVariable, Y) = fit(method, RDData(ZsR, Y))

function fit(method::NaiveLocalLinearRD, rddata::RDData)
	@unpack kernel, variance = method
	h = bandwidth(method.bandwidth, kernel, rddata)
	fitted_kernel = setbandwidth(kernel, h)
	
	rddata_filt = rddata[support(fitted_kernel)]
	wts = weights(fitted_kernel, rddata_filt.ZsR)
	fitted_lm = fit(LinearModel, @formula(Ys ~ Ws*Zs), rddata_filt, wts=wts)
	
	tau_est = coef(fitted_lm)[2]
	se_est = sqrt(var(variance, fitted_lm))
	γs = linearweights(fitted_lm)
	
	res = [h tau_est se_est "unaccounted"]
	colnms = ["h"; "τ̂"; "se"; "bias"]
	rownms = ["Sharp RD estimand"]
	coeftbl = CoefTable(res,colnms,rownms)
	
	FittedLocalLinearRD(rdd_setting = method, 
	                     fitted_lm = fitted_lm,
						 fitted_kernel = fitted_kernel,
						 fitted_bandwidth = h,
						 data_subset = rddata_filt,
						 tau_est = tau_est,
						 se_est = se_est,
						 coeftable = coeftbl)
end





function Base.show(io::IO, rdd_fit::RegressionDiscontinuity.FittedLocalLinearRD)
   println(io, "Local linear regression for regression discontinuity design")
   println(io, "       ⋅⋅⋅⋅ "*"Naive inference (not accounting for bias)")
   println(io, "       ⋅⋅⋅⋅ "*_string(rdd_fit.rdd_setting.kernel))
   println(io, "       ⋅⋅⋅⋅ "*_string(rdd_fit.rdd_setting.bandwidth))
   println(io, "       ⋅⋅⋅⋅ "*_string(rdd_fit.rdd_setting.variance))
   Base.show(io, rdd_fit.coeftable)
end