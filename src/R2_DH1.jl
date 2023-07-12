export R2_DH1

"""
    R2_DH1(nlp, h, options)
    R2_DH1(f, ∇f!, h, options, x0)

A first-order quadratic regularization method for the problem

    min f(x) + h(x)

where f: ℝⁿ → ℝ has a Lipschitz-continuous gradient, and h: ℝⁿ → ℝ is
lower semi-continuous, proper and prox-bounded.

About each iterate xₖ, a step sₖ is computed as a solution of

    min  φ(s; xₖ) + ½ σₖ ‖s‖² + ψ(s; xₖ)

where φ(s ; xₖ) = f(xₖ) + ∇f(xₖ)ᵀs + ½ sᵀ Dₖ s  is a quadratic approximation of f about xₖ,
ψ(s; xₖ) = h(xₖ + s), ‖⋅‖ is a user-defined norm, Dₖ is a diagonal Hessian approximation
and σₖ > 0 is the regularization parameter.

### Arguments

* `nlp::AbstractNLPModel`: a smooth optimization problem
* `h`: a regularizer such as those defined in ProximalOperators
* `options::ROSolverOptions`: a structure containing algorithmic parameters
* `x0::AbstractVector`: an initial guess (in the second calling form)

### Keyword Arguments

* `x0::AbstractVector`: an initial guess (in the first calling form: default = `nlp.meta.x0`)
* `selected::AbstractVector{<:Integer}`: (default `1:length(x0)`).
* `Bk`: initial diagonal Hessian approximation (default: `(one(R) / options.ν) * I`).

The objective and gradient of `nlp` will be accessed.

In the second form, instead of `nlp`, the user may pass in

* `f` a function such that `f(x)` returns the value of f at x
* `∇f!` a function to evaluate the gradient in place, i.e., such that `∇f!(g, x)` store ∇f(x) in `g`

### Return values

* `xk`: the final iterate
* `Fobj_hist`: an array with the history of values of the smooth objective
* `Hobj_hist`: an array with the history of values of the nonsmooth objective
* `Complex_hist`: an array with the history of number of inner iterations.
"""
function R2_DH1(nlp::AbstractNLPModel, args...; kwargs...)
  kwargs_dict = Dict(kwargs...)
  x0 = pop!(kwargs_dict, :x0, nlp.meta.x0)
  xk, k, outdict = R2_DH1(
    x -> obj(nlp, x),
    (g, x) -> grad!(nlp, x, g),
    args...,
    x0;
    l_bound = nlp.meta.lvar,
    u_bound = nlp.meta.uvar,
    kwargs_dict...,
  )
  ξ = outdict[:ξ]
  stats = GenericExecutionStats(nlp)
  set_status!(stats, outdict[:status])
  set_solution!(stats, xk)
  set_objective!(stats, outdict[:fk] + outdict[:hk])
  set_residuals!(stats, zero(eltype(xk)), ξ ≥ 0 ? sqrt(ξ) : ξ)
  set_iter!(stats, k)
  set_time!(stats, outdict[:elapsed_time])
  set_solver_specific!(stats, :Fhist, outdict[:Fhist])
  set_solver_specific!(stats, :Hhist, outdict[:Hhist])
  set_solver_specific!(stats, :NonSmooth, outdict[:NonSmooth])
  set_solver_specific!(stats, :SubsolverCounter, outdict[:Chist])
  return stats
end

function R2_DH1(
  f::F,
  ∇f!::G,
  h::H,
  options::ROSolverOptions{R},
  x0::AbstractVector{R};
  selected::AbstractVector{<:Integer} = 1:length(x0),
  Bk = (one(R) / options.ν) * I,
  kwargs...,
) where {F <: Function, G <: Function, H, R <: Real}
  start_time = time()
  elapsed_time = 0.0
  ϵ = options.ϵa
  ϵr = options.ϵr
  neg_tol = options.neg_tol
  verbose = options.verbose
  maxIter = options.maxIter
  maxTime = options.maxTime
  σmin = options.σmin
  η1 = options.η1
  η2 = options.η2
  ν = options.ν
  γ = options.γ
  spectral = options.spectral
  psb = options.psb
  andrei = options.andrei
  hess_init_val = (Bk isa UniformScaling) ? Bk.λ : (one(R) / options.ν)

  local l_bound, u_bound
  l_bound = R(-Inf) * ones(size(x0,1))
  u_bound = R(Inf) * ones(size(x0,1))
  has_bnds = false

  if verbose == 0
    ptf = Inf
  elseif verbose == 1
    ptf = round(maxIter / 10)
  elseif verbose == 2
    ptf = round(maxIter / 100)
  else
    ptf = 1
  end

  # initialize parameters
  xk = copy(x0)
  hk = h(xk[selected])
  if hk == Inf
    verbose > 0 && @info "R2_DH1: finding initial guess where nonsmooth term is finite"
    prox!(xk, h, x0, one(eltype(x0)))
    hk = h(xk[selected])
    hk < Inf || error("prox computation must be erroneous")
    verbose > 0 && @debug "R2_DH1: found point where h has value" hk
  end
  hk == -Inf && error("nonsmooth term is not proper")

  xkn = similar(xk)
  s = zero(xk)
  ψ = shifted(h, xk, l_bound - xk, u_bound - xk, selected)# : shifted(h, xk)

  Fobj_hist = zeros(maxIter)
  Hobj_hist = zeros(maxIter)
  Complex_hist = zeros(Int, maxIter)
  if verbose > 0
    #! format: off
    @info @sprintf "%6s %8s %8s %7s %8s %7s %7s %7s %7s %1s" "iter" "f(x)" "h(x)" "√ξ" "ρ" "σ" "‖x‖" "‖D‖" "‖s‖" ""
    #! format: off
  end

  local ξ
  k = 0
  σk = σmin

  fk = f(xk)
  ∇fk = similar(xk)
  ∇f!(∇fk, xk)
  ∇fk⁻ = copy(∇fk)
  Dk = spectral ? SpectralGradient(hess_init_val, length(xk)) :
    ((Bk isa UniformScaling) ? DiagonalQN(fill!(similar(xk), hess_init_val), psb, andrei) : DiagonalQN(diag(Bk), psb, andrei))
  DkNorm = norm(Dk.d, Inf)
  σkdk = Dk.d  .+ σk  

  optimal = false
  tired = maxIter > 0 && k ≥ maxIter || elapsed_time > maxTime

  while !(optimal || tired)
    k = k + 1
    elapsed_time = time() - start_time
    Fobj_hist[k] = fk
    Hobj_hist[k] = hk
    σkdk = max.(σkdk, eps())

    # model with diagonal hessian 
    φ(d) = ∇fk' * d + (d' * (σkdk .* d)) / 2
    mk(d) = φ(d) + ψ(d)

    if spectral
        iprox!(s, ψ, ∇fk, fill!(similar(∇fk), σkdk[1]))
     else
       iprox!(s, ψ, ∇fk, σkdk)
    end

    Complex_hist[k] += 1
    xkn .= xk .+ s
    fkn = f(xkn)
    hkn = h(xkn[selected])
    hkn == -Inf && error("nonsmooth term is not proper")

    Δobj = fk + hk - (fkn + hkn) + max(1, abs(fk + hk)) * 10 * eps()
    ξ = hk - mk(s) + max(1, abs(hk)) * 10 * eps()

    if ξ ≥ 0 && k == 1
      ϵ += ϵr * sqrt(ξ)  # make stopping test absolute and relative
    end
    
    if (ξ < 0 && sqrt(-ξ) ≤ neg_tol) || (ξ ≥ 0 && sqrt(ξ) < ϵ)
        # the current xk is approximately first-order stationary
      optimal = true
      continue
    end

    ξ > 0 || error("R2_DH1: prox-gradient step should produce a decrease but ξ = $(ξ)")
    
    ρk = Δobj / ξ

    σ_stat = (η2 ≤ ρk < Inf) ? "↘" : (ρk < η1 ? "↗" : "=")

    if (verbose > 0) && (k % ptf == 0)
      #! format: off
      @info @sprintf "%6d %8.1e %8.1e %7.1e %8.1e %7.1e %7.1e %7.1e %7.1e %1s" k fk hk sqrt(ξ) ρk σk norm(xk) norm(Dk.d) norm(s) σ_stat
      #! format: on
    end

    if η2 ≤ ρk < Inf
      σk = max(σk / γ, σmin)
    end

    if η1 ≤ ρk < Inf
      xk .= xkn
      fk = fkn
      hk = hkn
      shift!(ψ, xk)
      ∇f!(∇fk, xk)
      
      push!(Dk, s, ∇fk - ∇fk⁻) # update QN operator
      DkNorm = norm(Dk.d, Inf) 
      ∇fk⁻ .= ∇fk
    end

    if ρk < η1 || ρk == Inf
      σk = σk * γ
    end

    σkdk = Dk.d  .+ σk  

    tired = k ≥ maxIter || elapsed_time > maxTime
  end

  if verbose > 0
    if k == 1
      @info @sprintf "%6d %8.1e %8.1e" k fk hk
    elseif optimal
      #! format: off
      @info @sprintf "%6d %8.1e %8.1e %7.1e %8s %7.1e %7.1e %7.1e %7.1e" k fk hk sqrt(ξ) "" σk norm(xk) norm(Dk.d) norm(s)
      #! format: on
      @info "R2_DH1: terminating with √ξ = $(sqrt(ξ))"
    end
  end

  status = if optimal
    :first_order
  elseif elapsed_time > maxTime
    :max_time
  elseif tired
    :max_iter
  else
    :exception
  end
  outdict = Dict(
    :Fhist => Fobj_hist[1:k],
    :Hhist => Hobj_hist[1:k],
    :Chist => Complex_hist[1:k],
    :NonSmooth => h,
    :status => status,
    :fk => fk,
    :hk => hk,
    :ξ => ξ,
    :elapsed_time => elapsed_time,
  )

  return xk, k, outdict
end