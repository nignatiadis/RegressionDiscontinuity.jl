abstract type SharpRD end

abstract type VarianceEstimator end

struct EickerHuberWhite <: VarianceEstimator end
"""
    Homoskedastic()

Variance estimator for the RD estimator that assumes homoskedasticity.
"""
struct Homoskedastic <: VarianceEstimator end


abstract type BandwidthSelector end

bandwidth(h::Number, args...) = h
_string(::Number) = ""

abstract type BinwidthSelector end

binwidth(b::Number, args...) = b

abstract type WorstCaseCurvatureSelector end

curvature(b::Number, args...) = b
