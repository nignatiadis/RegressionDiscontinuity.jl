abstract type BandwidthSelector end


bandwidth(h::Number, kernel, rddata) = h
_string(::Number) = ""

struct ImbensKalyanaraman <: BandwidthSelector end

_string(::ImbensKalyanaraman) = "Imbens Kalyanaraman bandwidth"

function kernel_constant(::ImbensKalyanaraman, kernel)
    kernel = EquivalentKernel(kernel)
    νs = [kernel_moment(kernel, Val(j)) for j = 0:3]
    πs = [squared_kernel_moment(kernel, Val(j)) for j = 0:3]
    C1 = 1 / 4 * abs2((abs2(νs[3]) - νs[2] * νs[4]) / (νs[3] * νs[1] - abs2(νs[2])))
    C2_num = abs2(νs[3]) * πs[1] - 2 * νs[2] * νs[3] * πs[2] + abs2(νs[2]) * πs[3]
    C2_denom = abs2(νs[3] * νs[1] - abs2(νs[2]))
    C2 = C2_num / C2_denom
    (C2 / (4 * C1))^(1 / 5)
end


function bandwidth(ik::ImbensKalyanaraman, kernel::SupportedKernels, ZsR::RDData)
    ZsR_untreated = ZsR[Untreated()]
    ZsR_treated = ZsR[Treated()]

    cutoff = ZsR.cutoff

    N = nobs(ZsR)

    N_untreated = nobs(ZsR_untreated)
    N_treated = nobs(ZsR_treated)

    # Step 1: Density and conditional variance at 0
    h₁ = 1.84 * Statistics.std(ZsR.Zs) * N^(-1 / 5)

    interval_h₁ = Interval{:closed,:closed}(cutoff-h₁, cutoff+h₁)

    ZsR_untreated_h₁ = ZsR_untreated[interval_h₁]
    ZsR_treated_h₁ = ZsR_treated[interval_h₁]

    N_h₁_untreated = nobs(ZsR_untreated_h₁)
    N_h₁_treated = nobs(ZsR_treated_h₁)

    Ȳ_h₁_untreated = mean(ZsR_untreated_h₁.Ys)
    sd_Y_h₁_untreated = Statistics.std(ZsR_untreated_h₁.Ys)
    Ȳ_h₁_treated = mean(ZsR_treated_h₁.Ys)
    sd_Y_h₁_treated = Statistics.std(ZsR_treated_h₁.Ys)

    f̂₀ = (N_h₁_untreated + N_h₁_treated) / 2 / N / h₁

    # Step 2: Estimation of second derivatives
    global_cubic_lm = lm(@formula(Ys ~ 1 + (ZsC >= 0) + ZsC + ZsC^2 + ZsC^3), ZsR)

    m̂₀_triple_prime = 6 * coef(global_cubic_lm)[5]

    h₂_untreated = 3.56 * (sd_Y_h₁_untreated^2 / f̂₀ / m̂₀_triple_prime^2)^(1 / 7) * N_untreated^(-1 / 7)
    h₂_treated =
        3.56 * (sd_Y_h₁_treated^2 / f̂₀ / m̂₀_triple_prime^2)^(1 / 7) * N_treated^(-1 / 7)


    ZsR_untreated_h₂ = ZsR_untreated[Interval{:closed,:closed}(cutoff-h₂_untreated, cutoff+h₂_untreated)]
    ZsR_treated_h₂ = ZsR_treated[Interval{:closed,:closed}(cutoff-h₂_treated, cutoff+h₂_treated)]

    N_h₂_untreated = nobs(ZsR_untreated_h₂)
    N_h₂_treated = nobs(ZsR_treated_h₂)


    quadratic_fit_untreated = lm(@formula(Ys ~ 1 + ZsC + ZsC^2), ZsR_untreated_h₂)
    m̂₀_double_prime_untreated = 2 * last(coef(quadratic_fit_untreated))

    quadratic_fit_treated = lm(@formula(Ys~ 1 + ZsC + ZsC^2), ZsR_treated_h₂)
    m̂₀_double_prime_treated  = 2 * last(coef(quadratic_fit_treated ))

    # Step 3: Calculation of regularization terms

    r̂_untreated = 2160 * sd_Y_h₁_untreated^2 / N_h₂_untreated/ h₂_untreated^4
    r̂_treated = 2160 * sd_Y_h₁_treated^2 / N_h₂_treated / h₂_treated^4

    Ckernel = kernel_constant(ik, kernel)# 3.4375

    regularized_squared_diff =
        abs2(m̂₀_double_prime_treated - m̂₀_double_prime_untreated) + r̂_untreated + r̂_treated
    ĥ_IK =
        Ckernel *
        ((sd_Y_h₁_untreated^2 + sd_Y_h₁_treated^2) / f̂₀ / regularized_squared_diff)^(1 / 5) *
        N^(-1 / 5)
    ĥ_IK
end
