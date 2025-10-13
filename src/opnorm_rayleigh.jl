using LinearAlgebra

function opnorm_Rayleigh_v1(B)
    # generate x = [1, -1, 1, -1, ...]
    n = size(B, 1)
    x = zeros(n)
    for i in 1:n
        x[i] = isodd(i) ? 1.0 : -1.0
    end
    # normalize x
    # compute Rayleigh quotient 
    λ = abs(dot(x, B * x)) / dot(x, x)
    return λ
end

function opnorm_Rayleigh_v2(B)
    # generate x = [1, -1, 1, -1, ...]
    n = size(B, 1)
    x = randn(n)
    # normalize x
    # compute Rayleigh quotient 
    λ = abs(dot(x, B * x)) / dot(x, x)
    return λ
end

using LinearAlgebra, Random

"""
power_svmax(A; tol=1e-8, maxiter=500, v0=nothing)

Returns (σ, u, v, iters), where σ≈largest singular value of A,
u≈left singular vector, v≈right singular vector.
"""
function power_svmax(A; tol=1e-8, maxiter=500, v0=nothing)
    m, n = size(A)
    v = v0 === nothing ? randn(n) : copy(v0)
    v ./= norm(v)
    σ_old = 0.0
    u = similar(v, m)
    for k in 1:maxiter
        u = A * v
        σ = norm(u)
        u ./= σ
        v = A' * u
        v ./= norm(v)
        if abs(σ - σ_old) ≤ tol * max(1.0, σ)
            return σ, u, v, k
        end
        σ_old = σ
    end
    return σ_old, u, v, maxiter
end


function power_iteration(A::AbstractMatrix, num_iterations::Integer)
    @assert size(A,1) == size(A,2) "A must be square"
    n = size(A, 2)
    b = rand(n)              # random initial vector
    b ./= norm(b)            # normalize

    for _ in 1:num_iterations
        b1 = A * b           # matrix–vector product
        b  = b1 / norm(b1)   # re-normalize
    end

    λ = dot(b, A * b) / dot(b, b)  # Rayleigh quotient

    return λ                 # approx. dominant eigenvector
end
