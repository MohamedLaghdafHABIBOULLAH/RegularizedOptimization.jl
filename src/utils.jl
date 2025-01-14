export default_prox_callback
export default_prox_callback_v2
# use Arpack to obtain largest eigenvalue in magnitude with a minimum of robustness
function LinearAlgebra.opnorm(B; kwargs...)
  m, n = size(B)
  opnorm_fcn = m == n ? opnorm_eig : opnorm_svd
  return opnorm_fcn(B; kwargs...)
end
function opnorm_eig(B; max_attempts::Int = 3)
  have_eig = false
  attempt = 0
  λ = zero(eltype(B))
  n = size(B, 1)
  nev = 1
  ncv = max(20, 2 * nev + 1)

  while !(have_eig || attempt >= max_attempts)
      attempt += 1
      try
          # Perform eigendecomposition
          d, nconv, niter, nmult, resid = eigs(B; nev = nev, ncv = ncv, which = :LM, ritzvec = false, check = 1)
          
          # Check if eigenvalue has converged
          have_eig = nconv == 1
          if have_eig
              λ = abs(d[1])  # Take absolute value of the largest eigenvalue
              break  # Exit loop if successful
          else
              # Increase NCV for the next attempt if convergence wasn't achieved
              ncv = min(2 * ncv, n)
          end
      catch e
          if occursin("XYAUPD_Exception", string(e))
              @warn "Arpack error: $e. Increasing NCV to $ncv and retrying."
              ncv = min(2 * ncv, n)  # Increase NCV but don't exceed matrix size
          else
              rethrow(e)  # Re-raise if it's a different error
          end
      end
  end

  return λ, have_eig
end

function opnorm_svd(J; max_attempts::Int = 3)
  have_svd = false
  attempt = 0
  σ = zero(eltype(J))
  n = min(size(J)...)  # Minimum dimension of the matrix
  nsv = 1
  ncv = 10

  while !(have_svd || attempt >= max_attempts)
      attempt += 1
      try
          # Perform singular value decomposition
          s, nconv, niter, nmult, resid = svds(J; nsv = nsv, ncv = ncv, ritzvec = false, check = 1)
          
          # Check if singular value has converged
          have_svd = nconv >= 1
          if have_svd
              σ = maximum(s.S)  # Take the largest singular value
              break  # Exit loop if successful
          else
              # Increase NCV for the next attempt if convergence wasn't achieved
              ncv = min(2 * ncv, n)
          end
      catch e
          if occursin("XYAUPD_Exception", string(e))
              @warn "Arpack error: $e. Increasing NCV to $ncv and retrying."
              ncv = min(2 * ncv, n)  # Increase NCV but don't exceed matrix size
          else
              rethrow(e)  # Re-raise if it's a different error
          end
      end
  end

  return σ, have_svd
end

function default_prox_callback(
    s_ptr::Ptr{Cdouble},
    s_length::Csize_t,
    delta_k::Cdouble,
    ctx_ptr::Ptr{Cvoid}
)::Cint
    s_k = unsafe_wrap(Vector{Float64}, s_ptr, s_length; own = false)
    context = unsafe_pointer_to_objref(ctx_ptr)::AlgorithmContextCallback

    # In-place operation to avoid memory allocations
    @. context.s_k_unshifted = s_k - context.shift

    # Computations without allocations
    ξk = context.hk - context.mk(context.s_k_unshifted) + max(1, abs(context.hk)) * 10 * eps()
    condition = delta_k ≤ (1 - context.κξ) / context.κξ * ξk
    return condition ? Int32(1) : Int32(0)
end

function default_prox_callback_v2(
    s_ptr::Ptr{Cdouble},
    s_length::Csize_t,
    delta_k::Cdouble,
    ctx_ptr::Ptr{Cvoid}
)::Cint

    s_k = unsafe_wrap(Vector{Float64}, s_ptr, s_length; own = false)
    context = unsafe_pointer_to_objref(ctx_ptr)::AlgorithmContextCallback

    # In-place operation to avoid memory allocations
    @. context.s_k_unshifted = s_k - context.shift

    # Computations without allocations
    ξk = context.hk - context.mk(context.s_k_unshifted) + max(1, abs(context.hk)) * 10 * eps()

    condition = (delta_k ≤ context.dualGap) && (ξk ≥ 0)

    return condition ? Int32(1) : Int32(0)
end

function default_prox_callback_v3(
    s_ptr::Ptr{Cdouble},
    s_length::Csize_t,
    delta_k::Cdouble,
    ctx_ptr::Ptr{Cvoid}
)::Cint

    s_k = unsafe_wrap(Vector{Float64}, s_ptr, s_length; own = false)
    context = unsafe_pointer_to_objref(ctx_ptr)::AlgorithmContextCallback

    # In-place operation to avoid memory allocations
    @. context.s_k_unshifted = s_k - context.shift

    # Computations without allocations
    ξk = context.hk - context.mk(context.s_k_unshifted) + max(1, abs(context.hk)) * 10 * eps()

    aux = (1 - context.κξ) / context.κξ * ξk

    if aux < context.dualGap && aux ≥ 0
        context.dualGap = aux
    end

    condition = (delta_k ≤ context.dualGap) && (ξk ≥ 0)

    return condition ? Int32(1) : Int32(0)
end

ShiftedProximalOperators.iprox!(
  y::AbstractVector,
  ψ::ShiftedProximableFunction,
  g::AbstractVector,
  D::AbstractDiagonalQuasiNewtonOperator,
) = iprox!(y, ψ, g, D.d)

ShiftedProximalOperators.iprox!(
  y::AbstractVector,
  ψ::ShiftedProximableFunction,
  g::AbstractVector,
  D::SpectralGradient,
) = iprox!(y, ψ, g, fill!(similar(g), D.d[1]))

LinearAlgebra.diag(op::AbstractDiagonalQuasiNewtonOperator) = copy(op.d)
LinearAlgebra.diag(op::SpectralGradient{T}) where {T} = zeros(T, op.nrow) .* op.d[1]