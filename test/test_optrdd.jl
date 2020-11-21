using MosekTools, CSV, DataFrames
data = DataFrame(CSV.File("/Users/evanmunro/Documents/Github/rdd-gan/data/cleaned/m_math.csv"))
ZsR = RunningVariable(data.x[abs.(data.x).<10^6]; cutoff = 0.0, treated = :≥)
Ys = data.y[abs.(data.x).<10^6]
data = RDData(Ys, ZsR)

result = fit(MinMaxOptRD(B=0.01, solver=Mosek.Optimizer), data)
