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





#oreopoulos_matrix, orepoulos_header = readdlm(oreopoulos_path, ',', Float64; header=true)
# bla =
