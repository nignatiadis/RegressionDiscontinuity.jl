
struct McCraryBinwidth <: BinwidthSelector end


function binwidth(::McCraryBinwidth, Zs::RunningVariable)
    2 * std(Zs) / sqrt(length(Zs))
end

function StatsDiscretizations.samplehull_discretizer(Zs::RunningVariable, b)
    b = binwidth(b, Zs)
    rmin, rmax = extrema(Zs)
    cutoff = Zs.cutoff

    leftend = floor(rmin/b - eps(rmin/b))*b
    leftgrid_length =  round(Int,(cutoff-leftend)/b) + 1

    rightend = (1+ceil(rmax/b + eps(rmax/b)))*b
    rightgrid_length =  round(Int,(rightend-cutoff)/b)

    if Zs.treated ∈ (:≥, :<)
        L, R = :open, :closed
    elseif Zs.treated ∈ (:≤, :>)
        L, R = :closed, :open
    end

    discr_left =  BoundedIntervalDiscretizer{L,R}(
        range(leftend; stop=cutoff, length=leftgrid_length)
    )
    discr_right = BoundedIntervalDiscretizer{L,R}(
        range(cutoff; stop=rightend, length=rightgrid_length + 1)
    )

    if Zs.treated ∈ (:≥, :>)
        discr = (untreated = discr_left, treated = discr_right)
    elseif Zs.treated ∈ (:<, :≤)
        discr = (untreated = discr_rightt, treated = discr_left)
    end
    discr
end
