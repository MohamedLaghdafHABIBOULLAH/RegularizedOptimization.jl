using PGFPlotsX
using Colors
using LaTeXStrings
using PrettyTables, LaTeXStrings
using Random
using LinearAlgebra
using ProximalOperators
using LinearOperators
using NLPModels,
  NLPModelsModifiers,
  RegularizedProblems,
  RegularizedOptimization,
  ShiftedProximalOperators
using Printf

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

function benchmark_plot(
  f::AbstractNLPModel,
  nls,
  selected,
  h,
  solvers,
  colors,
  styles,
  solver_names,
  subsolvers,
  random_seed::Int;
  measured::Symbol = :obj, # set to :grad to eval grad
  xmode::String = "log",
  pb_name = "BPDN",
  ymode::String = "log",
  criterion::Symbol = :iter,
  file_name::String = "bpdn-iter",
  nonlinear::Bool = true
)
  n_solvers = length(solver_names)
  objdecs = Vector{Float64}[]
  coords = Coordinates{2}[]
  obj_min = Float64(Inf)

  reset!(f)
  reset!(nls)
  for (solver, subsolver) in
      zip(solvers, subsolvers)
    @info " using $solver with subsolver = $subsolver"
    args = solver == :R2 ? () : (NormLinf(1.0),)
    Random.seed!(random_seed)
    if pb_name == "Denoise" #|| pb_name == "FH"
      x0 = f.meta.x0
    elseif pb_name == "FH"
      x0 = [1., 1., 1., 1., 1.]
    else
      x0 = randn(f.meta.nvar)
    end
    if solver == LM_R2DH_Spec || solver == LM_R2
     # x0 = nls.meta.x0
      solver_out = solver(nls, h, x0, nonlinear = nonlinear)
    else
    #  x0 = f.meta.x0
      solver_out, obj_history = solver(f, h, x0) 
    #  println("nfevals $solver", neval_obj(f))
    end
    if solver == LM_R2DH_Spec || solver == LM_R2
      obj_history = solver_out.solver_specific[:Fhist] + solver_out.solver_specific[:Hhist]
    end
    #   objdec = solver_out.solver_specific[:Fhist] + solver_out.solver_specific[:Hhist]
    #   pushfirst!(objdec, obj(f, x0) + h(x0[selected])) 
    # else
    #   objdec = solver_out.solver_specific[:smooth_obj] + solver_out.solver_specific[:nonsmooth_obj]
    # end
    # measured == :grad && (objdec = objdec[solver_out.solver_specific[:IterSucc]])
    obj_min = min(minimum(obj_history), obj_min)
    # objdec[end] = obj(f, solver_out.solution) + h(solver_out.solution[selected])
    push!(objdecs, obj_history)
   # println(obj_history)
    reset!(f)
   # reset!(nls)
    if pb_name == "FH"
        println("initial $solver", x0)
        println("solution $solver", solver_out.solution)
    end
  end
  for i in 1:length(objdecs)
    objdec = objdecs[i]
   # time_ = time_hist[i]
    println(length(objdec))
    push!(
      coords,
      Coordinates([(k, objdec[k] - obj_min) for k in 1:length(objdec)]),
    )
  end
  
  l_plots = [@pgf PGFPlotsX.Plot({color = colors[i],  style = styles[i]}, coords[i]) for i in 1:n_solvers]
  
  a = @pgf PGFPlotsX.Axis(
    {
      xlabel = "Iterations",
      ylabel = L"$\Delta(f + h)(x_k)$",
      ymode = ymode,
      xmode = xmode,
      no_markers,
      style = "very thick",
      legend_style = {
        nodes={scale=0.8},
        font = "\\small",
      },
      legend_pos="south west",
      title = "\\bf $pb_name",
      # legend_pos="south east",
    },
    Tuple(l_plots)...,
    Legend(solver_names),
  )
  pgfsave("Résultats/" * file_name * ".pdf", a)
  pgfsave("Résultats/" *file_name * ".tikz", a)

end

function benchmark_plot_time(
  f::AbstractNLPModel,
  nls,
  selected,
  h,
  solvers,
  colors,
  styles,
  solver_names,
  subsolvers,
  random_seed::Int;
  measured::Symbol = :obj, # set to :grad to eval grad
  xmode::String = "log",
  pb_name = "BPDN",
  ymode::String = "log",
  criterion::Symbol = :iter,
  file_name = "bpdn-time",
)
  n_solvers = length(solver_names)
  objdecs = Vector{Float64}[] 
  time_hist = Vector{Float64}[]
  coords = Coordinates{2}[]
  obj_min = Float64(Inf)

  reset!(f)
  for (solver, subsolver) in
      zip(solvers, subsolvers)
    @info " using $solver with subsolver = $subsolver"
    args = solver == :R2 ? () : (NormLinf(1.0),)
    Random.seed!(random_seed)
    if pb_name == "Denoise"
      x0 = f.meta.x0
    else
      x0 = randn(f.meta.nvar)
      x1 = copy(x0)
    end
    if solver == LM_R2DH_Spec || solver == LM_R2
      solver_out = solver(nls, h, x0)
    else
      solver_out = solver(f, h, x0) 
    end
    objdec = solver_out.solver_specific[:Fhist] + solver_out.solver_specific[:Hhist]
    time_ = solver_out.solver_specific[:Time_hist]
    # push 0. at front of time_
    pushfirst!(time_, 0.)
    measured == :grad && (objdec = objdec[solver_out.solver_specific[:IterSucc]])
    obj_min = min(minimum(objdec), obj_min)
    objdec[end] = obj(f, solver_out.solution) + h(solver_out.solution[selected])
    push!(objdecs, objdec)
    push!(time_hist, time_)
    reset!(f)
  end
  for i in 1:length(objdecs)
    objdec = objdecs[i]
    time_ = time_hist[i]
    println(length(objdec))
    push!(
      coords,
      Coordinates([(time_[k], objdec[k] - obj_min) for k in 1:min(length(objdec), length(time_))]),
    )
  end
  l_plots = [@pgf PGFPlotsX.Plot({color = colors[i],  style = styles[i]}, coords[i]) for i in 1:n_solvers]
  
  a = @pgf PGFPlotsX.Axis(
    {
      xlabel = "CPU time",
      ylabel = L"$\Delta(f + h)(x_k)$",
      ymode = ymode,
      xmode = xmode,
      no_markers,
      style = "very thick",
      legend_style = {
        nodes={scale=0.8},
        font = "\\small",
      },
      legend_pos="south west",
      title = "\\bf $pb_name",
      # legend_pos="south east",
    },
    Tuple(l_plots)...,
    Legend(solver_names),
  )
  pgfsave("Résultats/" * file_name * ".pdf", a)
  pgfsave("Résultats/" * file_name * ".tikz", a)

end

colors = [
    RGB{}(1., 0., 0.),  # Rouge R2
    RGB{}(0, 0.5, 0.5),  # Bleu-vert R2DH-spec
    RGB{}(0, 0, 1),  # Bleu dashed R2N-R2DH-Spec
    RGB{}(0, 0, 0),  # Noir R2N
    RGB{}(0.8, 0.3, 0),  # Orange R2DH
    RGB{}(0.5, 0, 0.5),  # Violet R2DH_Spec
    RGB{}(0, 0, 0),  # Noir dashed LM R2DH Spec
    RGB{}(0.8, 0.3, 0),  # Orange dashed LM R2
    RGB{}(0.5, 0, 0.5),  # Violet R2DH_Spec
    RGB{}(0.5, 0.5, 0),  # Olive R2DH_Andrei
    RGB{}(0, 0, 1),  # Bleu R2DH_PSB
    RGB{}(0, 0, 0),  # Orange foncé R2DH_DBFGS
]

file_names = [
    "mc-rank-iter",
    "mc-nn-iter",
    "bpdn-time",
    "bpdn-iter",
    "bpdn-3-2-iter",
    "svm-iter",
    "denoising-iter",
    "fh-iter",
]

styles = [
  "solid",
  "solid",
  "dotted",
  "dotted",
  "dotted",
  "dotted",
  "loosely dashed",
  "loosely dashed",
  "solid",
  "solid",
  "solid",
  "solid"
]


# Options for the solvers
ν = 1.0
ϵ = eps()^(3/10)
ϵi = eps()^(3/10)
ϵri = 0.
maxIter = 1000
maxIter_inner = 200
verbose_R2 = 0
verbose = 0
σmin = 0.
σmin_R2 = eps()^(1/3)
σ0 = eps()^(1/3)

function R2_None(f, h, x0; ϵ = ϵ, ϵi = ϵi, ϵri = ϵri, maxIter = maxIter, maxIter_inner = maxIter_inner, verbose = verbose_R2, σmin = σmin_R2, σ0 = σ0)
  options = ROSolverOptions(ν = ν, ϵa = ϵ, ϵr = ϵ, maxIter = maxIter, reduce_TR = false, verbose = verbose, σmin = σmin)
  #R2(f, h, opt; x0 = x0)#, selected = selected)
  selected = 1:length(x0)
  obj_history = Float64[]
  time_history = Float64[]
  cb = (nlp, solver, output) -> begin
         push!(obj_history, output.objective)
         push!(time_history, output.elapsed_time)
        # push!(dual_history, output.dual_feas)
  end
  reg_nlp = RegularizedNLPModel(f, h, selected)
  solver = R2Solver(reg_nlp)
  stats = RegularizedExecutionStats(reg_nlp)
  solve!(solver, reg_nlp, stats, 
  x = x0,
  atol = options.ϵa,
  rtol = options.ϵr,
  neg_tol = options.neg_tol,
  verbose = options.verbose,
  max_iter = options.maxIter,
  max_time = options.maxTime,
  σmin = options.σmin,
  η1 = options.η1,
  η2 = options.η2,
  ν = options.ν,
  γ = options.γ, callback = cb)
  return stats, obj_history, time_history
end

#### Variantes diagonales

function R2DH_DBFGS(f, h, x0; ϵ = ϵ, ϵi = ϵi, ϵri = ϵri, maxIter = maxIter, maxIter_inner = maxIter_inner, verbose = verbose, σmin = σmin, σ0 = σ0)
  options = ROSolverOptions(ν = ν, ϵa = ϵ, ϵr = ϵ, maxIter = maxIter, reduce_TR = false, verbose = verbose, σmin = σmin, σk = σ0)
  selected = 1:length(x0)
  obj_history = Float64[]
  time_history = Float64[]
  cb = (nlp, solver, output) -> begin
         push!(obj_history, output.objective)
         push!(time_history, output.elapsed_time)
        # push!(dual_history, output.dual_feas)
  end
  reg_nlp = RegularizedNLPModel(DiagonalBFGSModel(f), h, selected)
  solver = R2DHSolver(reg_nlp, m_monotone= 1)
  stats = RegularizedExecutionStats(reg_nlp)
  solve!(solver, reg_nlp, stats,  
  x = x0,
  atol = options.ϵa,
  rtol = options.ϵr,
  neg_tol = options.neg_tol,
  verbose = options.verbose,
  max_iter = options.maxIter,
  max_time = options.maxTime,
  σk = options.σk,
  σmin = options.σmin,
  η1 = options.η1,
  η2 = options.η2,
  γ = options.γ,
  θ = options.θ, callback = cb)
  return stats, obj_history, time_history
end

function R2DH_Spec(f, h, x0; ϵ = ϵ, ϵi = ϵi, ϵri = ϵri, maxIter = maxIter, maxIter_inner = maxIter_inner, verbose = verbose, σmin = σmin, σ0 = σ0)
  options = ROSolverOptions(ν = ν, ϵa = ϵ, ϵr = ϵ, maxIter = maxIter, reduce_TR = false, verbose = verbose, σmin = σmin, σk = σ0)
  selected = 1:length(x0)
  obj_history = Float64[]
  time_history = Float64[]
  cb = (nlp, solver, output) -> begin
         push!(obj_history, output.objective)
         push!(time_history, output.elapsed_time)
        # push!(dual_history, output.dual_feas)
  end
  reg_nlp = RegularizedNLPModel(SpectralGradientModel(f), h, selected)
  solver = R2DHSolver(reg_nlp, m_monotone= 1)
  stats = RegularizedExecutionStats(reg_nlp)
  solve!(solver, reg_nlp, stats, 
  x = x0,
  atol = options.ϵa,
  rtol = options.ϵr,
  neg_tol = options.neg_tol,
  verbose = options.verbose,
  max_iter = options.maxIter,
  max_time = options.maxTime,
  σk = options.σk,
  σmin = options.σmin,
  η1 = options.η1,
  η2 = options.η2,
  γ = options.γ,
  θ = options.θ, callback = cb)
  return stats, obj_history, time_history
  #R2DH(SpectralGradientModel(f), h, opt; x0 = x0, Mmonotone = 0)#, selected = selected)
end

function R2DH_Andrei(f, h, x0; ϵ = ϵ, ϵi = ϵi, ϵri = ϵri, maxIter = maxIter, maxIter_inner = maxIter_inner, verbose = verbose, σmin = σmin, σ0 = σ0)
  options = ROSolverOptions(ν = ν, ϵa = ϵ, ϵr = ϵ, maxIter = maxIter, reduce_TR = false, verbose = verbose, σmin = σmin, σk = σ0)
  selected = 1:length(x0)
  obj_history = Float64[]
  time_history = Float64[]
  cb = (nlp, solver, output) -> begin
         push!(obj_history, output.objective)
         push!(time_history, output.elapsed_time)
        # push!(dual_history, output.dual_feas)
  end
  reg_nlp = RegularizedNLPModel(DiagonalAndreiModel(f), h, selected)
  solver = R2DHSolver(reg_nlp, m_monotone= 1)
  stats = RegularizedExecutionStats(reg_nlp)
  solve!(solver, reg_nlp, stats, 
  x = x0,
  atol = options.ϵa,
  rtol = options.ϵr,
  neg_tol = options.neg_tol,
  verbose = options.verbose,
  max_iter = options.maxIter,
  max_time = options.maxTime,
  σk = options.σk,
  σmin = options.σmin,
  η1 = options.η1,
  η2 = options.η2,
  γ = options.γ,
  θ = options.θ, callback = cb)
  return stats, obj_history, time_history
end

function R2DH_PSB(f, h, x0; ϵ = ϵ, ϵi = ϵi, ϵri = ϵri, maxIter = maxIter, maxIter_inner = maxIter_inner, verbose = verbose, σmin = σmin, σ0 = σ0)
  options = ROSolverOptions(ν = ν, ϵa = ϵ, ϵr = ϵ, maxIter = maxIter, reduce_TR = false, verbose = verbose, σmin = σmin, σk = σ0)
  selected = 1:length(x0)
  obj_history = Float64[]
  time_history = Float64[]
  cb = (nlp, solver, output) -> begin
         push!(obj_history, output.objective)
         push!(time_history, output.elapsed_time)
        # push!(dual_history, output.dual_feas)
  end
  reg_nlp = RegularizedNLPModel(DiagonalPSBModel(f), h, selected)
  solver = R2DHSolver(reg_nlp, m_monotone= 1)
  stats = RegularizedExecutionStats(reg_nlp)
  
  solve!(solver, reg_nlp, stats, 
  x = x0,
  atol = options.ϵa,
  rtol = options.ϵr,
  neg_tol = options.neg_tol,
  verbose = options.verbose,
  max_iter = options.maxIter,
  max_time = options.maxTime,
  σk = options.σk,
  σmin = options.σmin,
  η1 = options.η1,
  η2 = options.η2,
  γ = options.γ,
  θ = options.θ, callback = cb)
  return stats, obj_history, time_history
end

function R2DH_Spec_NM(f, h, x0; ϵ = ϵ, ϵi = ϵi, ϵri = ϵri, maxIter = maxIter, maxIter_inner = maxIter_inner, verbose = verbose, σmin = σmin, σ0 = σ0)
  options = ROSolverOptions(ν = ν, ϵa = ϵ, ϵr = ϵ, maxIter = maxIter, reduce_TR = false, verbose = verbose, σmin = σmin, σk = σ0)
  selected = 1:length(x0)
  obj_history = Float64[]
  time_history = Float64[]
  cb = (nlp, solver, output) -> begin
         push!(obj_history, output.objective)
         push!(time_history, output.elapsed_time)
        # push!(dual_history, output.dual_feas)
  end
  reg_nlp = RegularizedNLPModel(SpectralGradientModel(f), h, selected)
  solver = R2DHSolver(reg_nlp, m_monotone= 6)
  stats = RegularizedExecutionStats(reg_nlp)
  
  solve!(solver, reg_nlp, stats, 
  x = x0,
  atol = options.ϵa,
  rtol = options.ϵr,
  neg_tol = options.neg_tol,
  verbose = options.verbose,
  max_iter = options.maxIter,
  max_time = options.maxTime,
  σk = options.σk,
  σmin = options.σmin,
  η1 = options.η1,
  η2 = options.η2,
  γ = options.γ,
  θ = options.θ, callback = cb)
  return stats, obj_history, time_history
end

function R2N_R2(f, h, x0; ϵ = ϵ, ϵi = ϵi, ϵri = ϵri, maxIter = maxIter, maxIter_inner = maxIter_inner, verbose = verbose, σmin = σmin, σ0 = σ0)
  options = ROSolverOptions(ν = ν, ϵa = ϵ, ϵr = ϵ,  maxIter = maxIter, verbose = verbose, σmin = σmin, σk = σ0)
  selected = 1:length(x0)
  sub_kwargs = Dict(:ν => options.ν, :ϵa => ϵi, :ϵr => ϵri, :max_iter => maxIter_inner, :σk => options.σk)
  obj_history = Float64[]
  time_history = Float64[]
  cb = (nlp, solver, output) -> begin
         push!(obj_history, output.objective)
         push!(time_history, output.elapsed_time)
        # push!(dual_history, output.dual_feas)
  end
  reg_nlp = RegularizedNLPModel(f, h, selected)
  solver = R2NSolver(reg_nlp)
  stats = RegularizedExecutionStats(reg_nlp)
  
  solve!(solver, reg_nlp, stats, 
  x = x0,
  atol = options.ϵa,
  rtol = options.ϵr,
  neg_tol = options.neg_tol,
  verbose = options.verbose,
  max_iter = options.maxIter,
  max_time = options.maxTime,
  σmin = options.σmin,
  σk = options.σk,
  η1 = options.η1,
  η2 = options.η2,
  γ = options.γ, sub_kwargs = sub_kwargs, callback = cb)
  return stats, obj_history, time_history
end

function R2N_R2DH_Spec(f, h, x0; ϵ = ϵ, ϵi = ϵi, ϵri = ϵri, maxIter = maxIter, maxIter_inner = maxIter_inner, verbose = verbose, σmin = σmin, σ0 = σ0)
  options = ROSolverOptions(ν = ν, ϵa = ϵ, ϵr = ϵ, maxIter = maxIter, verbose = verbose, σmin = σmin, σk = σ0)
  selected = 1:length(x0)
  sub_kwargs = Dict(:ν => options.ν, :ϵa => ϵi, :ϵr => ϵri, :max_iter => maxIter_inner, :σk => options.σk)
  obj_history = Float64[]
  time_history = Float64[]
  cb = (nlp, solver, output) -> begin
         push!(obj_history, output.objective)
         push!(time_history, output.elapsed_time)
        # push!(dual_history, output.dual_feas)
  end
  reg_nlp = RegularizedNLPModel(f, h, selected)
  solver = R2NSolver(reg_nlp, subsolver = R2DHSolver)
  stats = RegularizedExecutionStats(reg_nlp)
  
  solve!(solver, reg_nlp, stats, 
  x = x0,
  atol = options.ϵa,
  rtol = options.ϵr,
  neg_tol = options.neg_tol,
  verbose = options.verbose,
  max_iter = options.maxIter,
  max_time = options.maxTime,
  σmin = options.σmin,
  σk = options.σk,
  η1 = options.η1,
  η2 = options.η2,
  γ = options.γ, sub_kwargs = sub_kwargs, callback = cb)
  return stats, obj_history, time_history
end

function R2N_R2_NM(f, h, x0; ϵ = ϵ, ϵi = ϵi, ϵri = ϵri, maxIter = maxIter, maxIter_inner = maxIter_inner, verbose = verbose, σmin = σmin, σ0 = σ0)
  opt = ROSolverOptions(ν = ν, ϵa = ϵ, ϵr = ϵ,  maxIter = maxIter, verbose = verbose, σmin = σmin, σk = σ0)
  sub_opt = ROSolverOptions(ϵa = ϵi, ϵr = ϵri, maxIter = maxIter_inner)
  R2N(f, h, opt; x0 = x0, subsolver_options = sub_opt, Mmonotone = 5)#, selected = selected)
end

function R2N_R2DH_Spec_NM(f, h, x0; ϵ = ϵ, ϵi = ϵi, ϵri = ϵri, maxIter = maxIter, maxIter_inner = maxIter_inner, verbose = verbose, σmin = σmin, σ0 = σ0)
  opt = ROSolverOptions(ν = ν, ϵa = ϵ, ϵr = ϵ, maxIter = maxIter, verbose = verbose, σmin = σmin, σk = σ0)
  sub_opt = ROSolverOptions(ϵa = ϵi, ϵr = ϵri, maxIter = maxIter_inner)
  R2N(f, h, opt; x0 = x0, subsolver_options = sub_opt, subsolver = R2DH, Mmonotone = 5)#, selected = selected)
end

#### Levenberg Marquardt

function LM_R2(nls_model, h, x0; ϵ = ϵ, ϵi = ϵi, ϵri = ϵri, maxIter = maxIter, maxIter_inner = maxIter_inner, verbose = verbose, σmin = σmin, nonlinear = true, σ0 = σ0)
  opt = ROSolverOptions(ν = ν, ϵa = ϵ, ϵr = ϵ, maxIter = maxIter, verbose = verbose, σmin = σmin, σk = σ0)
  sub_opt = ROSolverOptions(ϵa = ϵi, ϵr = ϵri, maxIter = maxIter_inner)
  LM(nls_model, h, opt; x0 = x0, subsolver_options = sub_opt, nonlinear = nonlinear)#, selected = selected)
end

function LM_R2DH_Spec(nls_model, h, x0; ϵ = ϵ, ϵi = ϵi, ϵri = ϵri, maxIter = maxIter, maxIter_inner = maxIter_inner, verbose = verbose, σmin = σmin, nonlinear = true, σ0 = σ0)
  opt = ROSolverOptions(ν = ν, ϵa = ϵ, ϵr = ϵ, maxIter = maxIter, verbose = verbose, σmin = σmin, σk = σ0)
  sub_opt = ROSolverOptions(ϵa = ϵi, ϵr = ϵri, maxIter = maxIter_inner)
  LM(nls_model, h, opt; x0 = x0, subsolver_options = sub_opt, subsolver = R2DH, nonlinear = nonlinear)#, selected = selected)
end
