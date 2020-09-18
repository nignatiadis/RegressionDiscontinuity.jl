using Test

Zs = randn(1000)

Zsgeg0 = RunningVariable(Zs, cutoff=1.0)
Zsless0 = RunningVariable(Zs, cutoff=1.0, treated = :<)
@test all(xor.(Zsgeg0.Ws  + Zsless0.Ws) .== 1)