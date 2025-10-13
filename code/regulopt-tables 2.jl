using PrettyTables, LaTeXStrings
using Random
using LinearAlgebra
using ProximalOperators
using NLPModels,
  NLPModelsModifiers,
  RegularizedProblems,
  RegularizedOptimization,
  ShiftedProximalOperators
using Printf
random_seed = 1234
Random.seed!(random_seed)
include("format.jl")

# utils for extracting stats / display table
modelname(op::LSR1Operator) = "LSR1"
modelname(op::LBFGSOperator) = "LBFGS"
modelname(op::SpectralGradient) = "SpectralGradient"
modelname(op::DiagonalPSB) = "DiagonalPSB"
modelname(op::DiagonalAndrei) = "DiagonalAndrei"
subsolvername(subsolver::Symbol) = subsolver == :None ? "" : string("-", subsolver)
function options_str(
  options::ROSolverOptions,
  solver::Symbol,
  subsolver_options::ROSolverOptions,
  subsolver::Symbol,
)
  if solver == :TRDH
    out_str = !options.spectral ? (options.psb ? "-PSB" : "-Andrei") : "-Spec"
    out_str = (options.reduce_TR) ? out_str : string(out_str, "-noredTR")
  elseif solver == :TR && subsolver == :TRDH
    out_str = !subsolver_options.spectral ? (subsolver_options.psb ? "-PSB" : "-Andrei") : "-Spec"
    out_str = (subsolver_options.reduce_TR) ? out_str : string(out_str, "-noredTR")
  else
    out_str = ""
  end
  return out_str
end
grad_evals(nlp::AbstractNLPModel) = neval_grad(nlp)
grad_evals(nls::AbstractNLSModel) = neval_jtprod_residual(nls) + neval_jprod_residual(nls)
obj_evals(nlp::AbstractNLPModel) = neval_obj(nlp)
obj_evals(nls::AbstractNLSModel) = neval_residual(nls)
function nb_prox_evals(stats, solver)
  prox_evals = sum(stats.solver_specific[:SubsolverCounter])
  return prox_evals
end

acc = vec -> length(findall(x -> x < 1, vec)) / length(vec) * 100 # for SVM

function benchmark_table(
  f::AbstractNLPModel,
  nls,
  selected,
  sol,
  h,
  λ,
  solvers,
  solver_names,
  subsolvers,
  random_seed::Int;
  pb_name = "bpdn",
  tex::Bool = false,
  nls_train::Union{Nothing, AbstractNLSModel} = nothing, # for SVM
  nls_test::Union{Nothing, AbstractNLSModel} = nothing, # for SVM
  nonlinear = true,
)

  nf_evals = []
  n∇f_evals = []
  nprox_evals = []
  solver_stats = []
  obj_min = Inf
  err = 0.0

  for (solver, subsolver) in
    zip(solvers, subsolvers)
    @info " using $solver with subsolver = $subsolver"
    args = solver == :R2 ? () : (NormLinf(1.0),)
    Random.seed!(random_seed)
      if pb_name == "denoise"
        x0 = f.meta.x0
      elseif pb_name == "fh"
        x0 = [1., 1., 1., 1., 1.]
      else
        x0 = randn(f.meta.nvar)
      end
      if solver == LM_R2DH_Spec || solver == LM_R2
      # x0 = nls.meta.x0
        solver_out = solver(nls, h, x0, nonlinear = nonlinear)#, args..., opt, x0 = f.meta.x0, selected = selected)
        push!(nf_evals, neval_residual(nls))
        push!(n∇f_evals, neval_jtprod_residual(nls) + neval_jprod_residual(nls))
      else
    #  x0 = f.meta.x0
        solver_out = solver(f, h, x0) 
        push!(nf_evals, neval_obj(f))
        push!(n∇f_evals, neval_grad(f))
      end
    push!(nprox_evals, nb_prox_evals(solver_out, solver))
    push!(solver_stats, solver_out)
    fx = solver_out.solver_specific[:Fhist][end]
    hx = solver_out.solver_specific[:Hhist][end]
    if obj_min > fx + hx
      obj_min = fx + hx
    end
    reset!(f)
    reset!(nls)
  end

  if tex && nonlinear
    if length(sol) == 0
      header = [
        "solver",
        L"$f(x)$",
        L"$h(x)/\lambda$",
        L"\Delta(f+h)",
        L"$\sqrt{\xi/\nu}$",
        L"$\#f$",
        L"$\#\nabla f$",
        L"$\#prox$",
        L"$t$($s$)",
      ]
    else
      header = [
        "Solver",
        L"$f$",
        L"$h/\lambda$",
        L"\Delta(f+h)",
        L"$\sqrt{\xi/\nu}$",
        L"$\#f$",
        L"$\#\nabla f$",
        L"$\#prox$",
        L"$t$($s$)",
      ]
    end
  elseif tex && nonlinear == false
    if length(sol) == 0
      header = [
        "solver",
        L"$f(x)$",
        L"$h(x)/\lambda$",
        L"\Delta(f+h)",
        L"$\sqrt{\xi/\nu}$",
        L"$\#f$",
        L"$\#J$",
        L"$\#prox$",
        L"$t$($s$)",
      ]
    else
      header = [
        "Solver",
        L"$f$",
        L"$h/\lambda$",
        L"\Delta(f+h)",
        L"$\sqrt{\xi/\nu}$",
        L"$\#f$",
        L"$\#J$",
        L"$\#prox$",
        L"$t$($s$)",
      ]
    end
  else
    if length(sol) == 0
      header = ["solver", "f(x)", "h(x)/λ", "Δ(f + h)","√(ξ/ν)", "# f", "# ∇f", "# prox", "t (s)"]
    else
      header = [
        "solver",
        "f(x)",
        "h(x)/λ",
        "Δ(f + h)",
        "√ξ/√ν",
        "# f",
        "# ∇f",
        "# prox",
        "t(s)",
      ]
    end
  end

  nh = length(header)
  n_solvers = length(solver_names)
  data = Matrix{Any}(undef, n_solvers, nh)
  for i = 1:n_solvers
    sname = solver_names[i]
    solver_out = solver_stats[i]
    x = solver_out.solution
    fx = solver_out.solver_specific[:Fhist][end]
    hx = solver_out.solver_specific[:Hhist][end]
    objx = fx + hx
    if solvers[i] == R2_None
      ξ = solver_out.dual_feas
    else
      ξ = solver_out.dual_feas
    end
    nf = nf_evals[i]
    n∇f = n∇f_evals[i]
    nprox = nprox_evals[i]
    t = solver_out.elapsed_time
    if length(sol) == 0
      data[i, :] .= [sname, fx, hx / λ, objx - obj_min, ξ, nf, n∇f, nprox, t]
    else
      if pb_name == "svm"
        string(round(t, digits = 2))
        err = "($(
          round(acc(residual(nls_train, solver_out.solution)), digits=1)), $(
            round(acc(residual(nls_test, solver_out.solution)), digits = 1)))"
      else
        err = norm(x - sol)
      end
      data[i, :] .= [sname, fx, hx / λ, objx - obj_min, ξ, nf, n∇f, nprox, t]
    end
  end

  if h isa NormL0 || h isa Rank
    h_format = "%i"
  else
    h_format = "%7.1e"
  end
  #h_format = h isa (NormL0 || Rank) ? "%i" : "%7.1e"
  if length(sol) == 0
    print_formats = ft_printf(["%s", "%7.2e", h_format, "%7.2e", "%7.1e", "%i", "%i", "%i", "%.2f"], 1:nh)
  else
    if pb_name == "svm"
      print_formats =
        ft_printf(["%s", "%7.2e", h_format, "%7.2e", "%7.1e" , "%i", "%i", "%i", "%.2f"], 1:nh)
    else
      print_formats =
        ft_printf(["%s", "%7.2e", h_format, "%7.2e", "%7.1e", "%i", "%i", "%i", "%.2f"], 1:nh)
    end
  end

  title = "$pb_name $(modelname(f.op)) $(typeof(h).name.name)"
  if (length(sol) != 0) && pb_name != "svm"
    title = string(title, " \$f(x_T) = $(@sprintf("%.2e", obj(model, sol)))\$")
  end
  if tex
    println("problem: $pb_name")
    println("obj minimum: $obj_min")
    println("error: ", err)
    open("Tables/$pb_name-unformatted.tex", "w") do f
      pretty_table(f,
      data;
      header = header,
      title = title,
      backend = Val(:latex),
      formatters = (
        print_formats,
        (v, i, j) -> (j == 1 ? v : v),
      ),
      )
    end
    # Reformater le fichier .tex
    input_file = "Tables/$pb_name-unformatted.tex"
    output_file = "Tables/$pb_name.tex"
    reformat_tex_file(input_file, output_file)
  else
    open("Tables/$pb_name.txt", "w") do f
      pretty_table(f,
      data;
      header = header,
      title = title,
      formatters = (print_formats,),
      )
    end
    #pretty_table(data; header = header, title = title, formatters = (print_formats,))
  end
  return solver_names, solver_stats
end

# λ = norm(grad(model, rand(model.meta.nvar)), Inf) / 100000
# h = NormL1(λ)
# benchmark_table(f, selected, [], h, λ, solvers, subsolvers, solver_options, subsolver_options,
#                 "NNMF with m = $m, n = $n, k = $k, ν = 1.0e-3,")

# header = ["TR LSR1 L0Box", "R2 LSR1 L0Box", "LM L0Box", "LMTR L0Box"]
# TR_out = TR(f, h, χ, options, x0 = f.meta.x0)
# n∇f_TR = neval_grad(f)
# prox_evals_TR = sum(TR_out.solver_specific[:SubsolverCounter])
# reset!(f)
# R2_out = R2(f, h, options, x0 = f.meta.x0)
# n∇f_R2 = neval_grad(f)
# prox_evals_R2 = R2_out.iter
# reset!(f)
# LM_out = LM(nls_model, h, options, x0 = nls_model.meta.x0)
# n∇f_LM = neval_jtprod_residual(nls_model) + neval_jprod_residual(nls_model)
# prox_evals_LM = sum(LM_out.solver_specific[:SubsolverCounter])
# reset!(nls_model)
# LMTR_out = LMTR(nls_model, h, χ, options, x0 = nls_model.meta.x0)
# n∇f_LMTR = neval_jtprod_residual(nls_model) + neval_jprod_residual(nls_model)
# prox_evals_LMTR = sum(LMTR_out.solver_specific[:SubsolverCounter])
# reset!(nls_model)
# n∇f_evals = [n∇f_TR, n∇f_R2, n∇f_LM, n∇f_LMTR]
# nprox_evals = [prox_evals_TR, prox_evals_R2, prox_evals_LM, prox_evals_LMTR]

# solver_stats = [TR_out, R2_out, LM_out, LMTR_out]

