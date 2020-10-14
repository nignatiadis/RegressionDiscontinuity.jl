function load_rdd_data(dataset::Symbol)
    if dataset == :lee08
        lee08_path = joinpath(dirname(@__FILE__), "..", "data", "lee08.feather")

        lee08 = Feather.read(lee08_path)
        ZsR = RunningVariable(lee08.margin ./ 100; cutoff = 0.0, treated = :≥)
        Ys = lee08.voteshare ./ 100
        dataset = RDData(Ys, ZsR)
    end

    return dataset
end
