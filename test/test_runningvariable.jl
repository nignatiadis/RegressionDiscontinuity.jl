using DataFrames
using GLM
#using Plots
using RegressionDiscontinuity
using Test

Zs = randn(1000)

ZsR = RunningVariable(Zs, cutoff = 0.0)
Zsgeg0 = RunningVariable(Zs, cutoff = 1.0)
Zsless0 = RunningVariable(Zs, cutoff = 1.0, treated = :<)
@test all(xor.(Zsgeg0.Ws + Zsless0.Ws) .== 1)

tmp_df = DataFrame(Ws = ZsR.Ws, Zs = ZsR.Zs)
@test Tables.schema(tmp_df) == Tables.schema(ZsR)



Ys = rand(1000)
rdd_data = RegressionDiscontinuity.RDData(Ys, ZsR)
myfit = fit(LinearModel, @formula(Ys ~ Zs + Ws), rdd_data)
