Rectangular() = Uniform(-0.5,+0.5)

const LocationScaleDists = Union{Cosine,
							     Epanechnikov,
                                 SymTriangularDist,
							     Triweight}
							 
const SupportedKernels = Union{LocationScaleDists, Uniform}



function setbandwidth(kernel::LocationScaleDists, h::Number)
	 typeof(kernel)(zero(kernel.μ), h * kernel.σ)
end

function setbandwidth(kernel::Uniform, h::Number) 
	@unpack a, b =  kernel
	if a + b != 0
		error("Use symmetric Uniform/Rectangular kernel")
	end 
	Uniform(a*h, b*h)
end  
	

weights(::Uniform, ZsR::RunningVariable) = uweights(length(ZsR))
weights(D::Distribution, ZsR::RunningVariable) = pdf.(D, ZsR)





struct EquivalentKernel{K, ET, T}
   kernel::K #only order 1 for now
   E::ET
   kernel_mult::T
end

function _expectation(ub, n)
   rawNodes, rawWeights = gausslegendre(n)
   _nodes = map(x -> (0.5*ub)*x + ub/2, rawNodes)
   _weights = map(x -> x * 1/2  * ub, rawWeights) # (result of doing 1/(b-a) * (b-a)/2)
   f -> dot(f.(_nodes), _weights)
end

function EquivalentKernel(kernel::SupportedKernels)
	@unpack ub = support(kernel)
	E = _expectation(ub, 10)
	cov_x = [x -> 1 x->x; x-> x x->x^2]
	kernel_mult = [1 0]*inv(map(f->E(u -> f(u)*pdf(kernel, u)), cov_x))
	EquivalentKernel(kernel, E, kernel_mult)
end

Distributions.support(eq::EquivalentKernel) = eq.kernel

function Distributions.pdf(eq::EquivalentKernel, u)
	@unpack kernel_mult, kernel = eq
	dot(kernel_mult,[1;u])*pdf(kernel, u)
end 


function kernel_moment(kernel, ::Val{j}) where j
	@unpack ub = support(kernel)
	quadgk(x-> x^j*pdf(kernel, x), 0, ub)[1]
end

kernel_moment(::SupportedKernels, ::Val{0}) = 0.5
kernel_moment(::SupportedKernels, ::Val{1}) = 0
kernel_moment(::SupportedKernels, ::Val{3}) = 0

function kernel_moment(eq::EquivalentKernel, ::Val{j}) where j
	@unpack E = eq
	E(x-> x^j*pdf(eq, x))
end

function squared_kernel_moment(kernel, ::Val{j}) where j
	@unpack ub = support(kernel)
	quadgk(x-> x^j*pdf(kernel, x)^2, 0, ub)[1]
end

squared_kernel_moment(::SupportedKernels, ::Val{1}) = 0
squared_kernel_moment(::SupportedKernels, ::Val{3}) = 0

function squared_kernel_moment(eq::EquivalentKernel, ::Val{j}) where j
	@unpack E = eq
	E(x-> x^j*pdf(eq, x)^2)
end




