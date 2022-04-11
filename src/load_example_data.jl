abstract type Dataset end

struct Lee08 <: Dataset end

function raw_table(::Lee08)
    _path = joinpath(dirname(@__FILE__), "..", "data", "lee08.csv")
    matrix, header = readdlm(_path, ',', Float64; header=true)
    columntable( (voteshare=matrix[:,1], margin = matrix[:,2]))
end

function RDData(dataset::Lee08)
    lee08 = raw_table(dataset)
    ZsR = RunningVariable(lee08.margin ./ 100; cutoff = 0.0, treated = :≥)
    Ys = lee08.voteshare ./ 100
    RDData(Ys, ZsR)
end




"""
    ECLS_EIWW

A regression discontinuity design artificially constructed from test scores
in early childhood using data from the Early Childhood Longitudinal Study (ECLS).

The construction of this synthetic RDD is described in:
> Eckles, Dean, Nikolaos Ignatiadis, Stefan Wager, and Han Wu.
> "Noise-induced randomization in regression discontinuity designs."
> arXiv preprint arXiv:2004.09458 (2022).


The following technical report provides more details regarding the ECLS:

> K. Tourangeau, C. Nord, T. Le, A.G. Sorongon, M.C. Hagedorn, P. Daly, and M. Najarian.
> Early Childhood Longitudinal Study, Kindergarten Class of 2010-11 (ECLS-K:2011),
> User’s Manual for the ECLS-K:2011 Kindergarten Data File and Electronic Codebook,
> Public Version (NCES 2015-074).
> Technical report, U.S. Department of Education. Washington, DC
> National Center for Education Statistics, 2015.
"""
struct ECLS_EIWW <: Dataset end

function raw_table(::ECLS_EIWW)
    _path = joinpath(dirname(@__FILE__), "..", "data", "ecls_RD.csv")
    matrix, header = readdlm(_path, ',', Float64; header=true)
    columntable( (Z=matrix[:,1], Y = matrix[:,2], SE = matrix[:,3]))
end
