import CodecBzip2

using DataFrames

using GLM

using UnPack

using RData
using RegressionDiscontinuity

using Statistics
using StatsBase
using StatsModels

using Test

lee08_path = joinpath(dirname(@__FILE__), "..", "data", "lee08.rda")

lee08 = load(lee08_path)["lee08"]

lee08_rdd = load_rdd_data(:lee08)


Z = lee08.margin ./ 100
Y = lee08.voteshare ./ 100

rdd_df = DataFrame(Z = Z, Y = Y)

N = length(Z)
left_idx = findall(Z .< 0)
right_idx = findall(Z .>= 0)
N_left = length(left_idx)
N_right = length(right_idx)

# Step 1: Density and conditional variance at 0
h₁ = 1.84 * std(Z) * N^(-1 / 5)

h₁_left_idx = findall(-h₁ .<= Z .< 0)
h₁_right_idx = findall(+h₁ .>= Z .>= 0)

N_h₁_left = length(h₁_left_idx)
N_h₁_right = length(h₁_right_idx)

Ȳ_h₁_left = mean(Y[h₁_left_idx])
sd_Y_h₁_left = std(Y[h₁_left_idx])
Ȳ_h₁_right = mean(Y[h₁_right_idx])
sd_Y_h₁_right = std(Y[h₁_right_idx])

f̂₀ = (N_h₁_left + N_h₁_right) / 2 / N / h₁

# Step 2: Estimation of second derivatives
global_cubic_lm = lm(@formula(Y ~ 1 + (Z >= 0) + Z + Z^2 + Z^3), rdd_df)

m̂₀_triple_prime = 6 * coef(global_cubic_lm)[5]

h₂_left = 3.56 * (sd_Y_h₁_left^2 / f̂₀ / m̂₀_triple_prime^2)^(1 / 7) * N_left^(-1 / 7)
h₂_right = 3.56 * (sd_Y_h₁_right^2 / f̂₀ / m̂₀_triple_prime^2)^(1 / 7) * N_right^(-1 / 7)

h₂_left_idx = findall(-h₂_left .<= Z .< 0)
h₂_right_idx = findall(+h₂_right .>= Z .>= 0)

N_h₂_left = length(h₂_left_idx)
N_h₂_right = length(h₂_right_idx)


quadratic_fit_left = lm(@formula(Y ~ 1 + Z + Z^2), rdd_df[h₂_left_idx, :])
m̂₀_double_prime_left = 2 * last(coef(quadratic_fit_left))

quadratic_fit_right = lm(@formula(Y ~ 1 + Z + Z^2), rdd_df[h₂_right_idx, :])
m̂₀_double_prime_right = 2 * last(coef(quadratic_fit_right))

# Step 3: Calculation of regularization terms

r̂_left = 2160 * sd_Y_h₁_left^2 / N_h₂_left / h₂_left^4
r̂_right = 2160 * sd_Y_h₁_right^2 / N_h₂_right / h₂_right^4

Ckernel = 3.4375

regularized_squared_diff =
    abs2(m̂₀_double_prime_right - m̂₀_double_prime_left) + r̂_left + r̂_right
ĥ_IK =
    Ckernel *
    ((sd_Y_h₁_left^2 + sd_Y_h₁_right^2) / f̂₀ / regularized_squared_diff)^(1 / 5) *
    N^(-1 / 5)

ik_triang_paper = 0.2939
point_est_triang_paper = 0.0799
se_triang_paper = 0.0083

bw_ik_triang = bandwidth(ImbensKalyanaraman(), SymTriangularDist(), lee08_rdd)
fit_triang = fit(
    NaiveLocalLinearRD(kernel = SymTriangularDist(), bandwidth = ImbensKalyanaraman()),
    lee08_rdd,
)

@test bw_ik_triang ≈ ik_triang_paper atol = 0.001
@test fit_triang.fitted_bandwidth == bw_ik_triang
@test fit_triang.tau_est ≈ point_est_triang_paper atol = 0.001
@test fit_triang.se_est ≈ se_triang_paper atol = 0.001



@test bw_ik_triang ≈ ĥ_IK atol = 0.001

ik_rect_paper = 0.4617
point_est_rect_paper = 0.0806
se_rect_paper = 0.0087

bw_ik_rect = bandwidth(ImbensKalyanaraman(), Rectangular(), lee08_rdd)
fit_rect = fit(
    NaiveLocalLinearRD(kernel = Rectangular(), bandwidth = ImbensKalyanaraman()),
    lee08_rdd,
)

@test bw_ik_rect ≈ ik_rect_paper atol = 0.001
@test fit_rect.fitted_bandwidth == bw_ik_rect
@test fit_rect.tau_est ≈ point_est_rect_paper atol = 0.001
@test fit_rect.se_est ≈ se_rect_paper atol = 0.001


Ys = lee08_rdd.Ys
ZsR = lee08_rdd.ZsR

bw_kernel = SymTriangularDist(0, bw_ik_triang)

support(bw_kernel)

ZsR[support(bw_kernel)]





propertynames(lee08_rdd)
propertynames(lee08_rdd)



RegressionDiscontinuity.kernel_constant(ImbensKalyanaraman(), Rectangular())
RegressionDiscontinuity.kernel_constant(ImbensKalyanaraman(), SymTriangularDist())
