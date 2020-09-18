function ik_bandwidth(Z, Y)
	rdd_df = DataFrame(Z=Z,Y=Y)

	N = length(Z)
	left_idx  = findall(Z .< 0)
	right_idx = findall(Z .>= 0)
	N_left = length(left_idx)
	N_right = length(right_idx)

	# Step 1: Density and conditional variance at 0
	h₁ = 1.84 * std(Z) * N^(-1/5) 

	h₁_left_idx  = findall(-h₁ .<= Z .< 0)
	h₁_right_idx = findall(+h₁ .>= Z .>= 0)

	N_h₁_left  = length(h₁_left_idx)
	N_h₁_right = length(h₁_right_idx)

	Ȳ_h₁_left  = mean(Y[h₁_left_idx])
	sd_Y_h₁_left = std(Y[h₁_left_idx])
	Ȳ_h₁_right = mean(Y[h₁_right_idx])
	sd_Y_h₁_right = std(Y[h₁_right_idx])

	f̂₀ = (N_h₁_left + N_h₁_right)/2/N/h₁ 

	# Step 2: Estimation of second derivatives
	global_cubic_lm = lm(@formula(Y~1 + (Z>=0) + Z + Z^2 + Z^3), rdd_df)

	m̂₀_triple_prime =  6*coef(global_cubic_lm)[5]

	h₂_left  = 3.56 * (sd_Y_h₁_left^2/f̂₀/m̂₀_triple_prime^2)^(1/7) * N_left^(-1/7) 
	h₂_right = 3.56 * (sd_Y_h₁_right^2/f̂₀/m̂₀_triple_prime^2)^(1/7) * N_right^(-1/7) 

	h₂_left_idx  = findall(-h₂_left .<= Z .< 0)
	h₂_right_idx = findall(+h₂_right .>= Z .>= 0)

	N_h₂_left  = length(h₂_left_idx)
	N_h₂_right = length(h₂_right_idx)


	quadratic_fit_left = lm(@formula(Y~1+Z+Z^2),  rdd_df[h₂_left_idx,:])
	m̂₀_double_prime_left = 2*last(coef(quadratic_fit_left))

	quadratic_fit_right = lm(@formula(Y~1+Z+Z^2), rdd_df[h₂_right_idx,:])
	m̂₀_double_prime_right = 2*last(coef(quadratic_fit_right))

	# Step 3: Calculation of regularization terms

	r̂_left = 2160*sd_Y_h₁_left^2/N_h₂_left/h₂_left^4
	r̂_right = 2160*sd_Y_h₁_right^2/N_h₂_right/h₂_right^4

	Ckernel = 3.4375

	regularized_squared_diff = abs2(m̂₀_double_prime_right - m̂₀_double_prime_left) + r̂_left + r̂_right
	ĥ_IK = Ckernel*( (sd_Y_h₁_left^2 + sd_Y_h₁_right^2)/f̂₀/regularized_squared_diff)^(1/5)*N^(-1/5)
	ĥ_IK
end

