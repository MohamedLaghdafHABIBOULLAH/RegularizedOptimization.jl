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