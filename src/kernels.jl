Rectangular() = Uniform(-0.5, +0.5)

function _string(kernel::Uniform)
    a = kernel.a
    b = kernel.b
    "Rectangular kernel (U[$a,$b])"
end

const LocationScaleDists = Union{Cosine,Epanechnikov,SymTriangularDist,Triweight}

_string(kernel::SymTriangularDist) = "Triangular kernel"
_string(kernel::LocationScaleDists) = "$(Base.typename(typeof(kernel))) kernel"

const SupportedKernels = Union{LocationScaleDists,Uniform}

export Uniform, Epanechnikov, SymTriangularDist

function setbandwidth(kernel::LocationScaleDists, h::Number)
    typeof(kernel)(zero(kernel.μ), h * kernel.σ)
end

function setbandwidth(kernel::Uniform, h::Number)
    a = kernel.a
    b = kernel.b

    if a + b != 0
        error("Use symmetric Uniform/Rectangular kernel")
    end
    Uniform(a * h, b * h)
end


weights(::Uniform, ZsR::RunningVariable) = uweights(length(ZsR))
weights(D::Distribution, ZsR::RunningVariable) = pdf.(D, ZsR)





struct EquivalentKernel{K,T}
    kernel::K #only order 1 for now
    kernel_mult::T
end

function EquivalentKernel(kernel::SupportedKernels)
    ub = support(kernel).ub
    E(f) = quadgk(f, 0, ub)[1]
    cov_x = [x->1 x->x; x->x x->x^2]
    kernel_mult = [1 0] * inv(map(f -> E(u -> f(u) * pdf(kernel, u)), cov_x))
    EquivalentKernel(kernel, kernel_mult)
end

Distributions.support(eq::EquivalentKernel) = Distributions.support(eq.kernel)

function Distributions.pdf(eq::EquivalentKernel, u)
    kernel_mult = eq.kernel_mult
    kernel = eq.kernel
    dot(kernel_mult, [1; u]) * pdf(kernel, u)
end


function kernel_moment(kernel, ::Val{j}) where {j}
    ub = support(kernel).ub
    quadgk(x -> x^j * pdf(kernel, x), 0, ub)[1]
end

kernel_moment(::SupportedKernels, ::Val{0}) = 0.5
kernel_moment(::SupportedKernels, ::Val{1}) = 0
kernel_moment(::SupportedKernels, ::Val{3}) = 0

function kernel_moment(eq::EquivalentKernel, ::Val{j}) where {j}
    ub = support(eq).ub
    E(f) = quadgk(f, 0, ub)[1]
    E(x -> x^j * pdf(eq, x))
end

function squared_kernel_moment(kernel, ::Val{j}) where {j}
    ub = support(kernel).ub
    quadgk(x -> x^j * pdf(kernel, x)^2, 0, ub)[1]
end

squared_kernel_moment(::SupportedKernels, ::Val{1}) = 0
squared_kernel_moment(::SupportedKernels, ::Val{3}) = 0

function squared_kernel_moment(eq::EquivalentKernel, ::Val{j}) where {j}
    ub = support(eq).ub
    E(f) = quadgk(f, 0, ub)[1]
    E(x -> x^j * pdf(eq, x)^2)
end
