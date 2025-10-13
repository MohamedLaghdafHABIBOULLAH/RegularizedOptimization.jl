include("regulopt-plots.jl")
include("regulopt-tables.jl")

random_seed = 1234
Random.seed!(random_seed)
compound = 10
model, sol = bpdn_model_three_half(compound, bounds = false)

# parameters
f = LBFGSModel(model)
λ = norm(grad(model, zeros(model.meta.nvar)), Inf) / 10
h = NormL0(λ)

solvers = [R2_None, R2DH_Spec_NM, R2DH_Spec, R2DH_DBFGS, R2N_R2, R2N_R2DH_Spec]#, R2DH_Andrei, R2DH_PSB, R2DH_DBFGS]
solver_names = ["R2", "R2DH-Spec-NM", "R2DH-Spec", "R2DH-DBFGS", "R2N-R2", "R2N-R2DH"]#, "R2DH-Andrei", "R2DH-PSB", "R2DH-DBFGS"]
file_name = file_names[3]
colors = [colors[1:2] ; colors[7]; colors[10]; colors[3:4]]
styles = [styles[1:2] ; styles[7]; styles[10]; styles[3:4]]
subsolvers =
  [:None, :None, :None, :None, :None, :None, :None]


# benchmark_plot_time(
#   f,
#     nls_model,
#   1:(f.meta.nvar),
#   h,
#   solvers,
#   colors,
#   styles,
#   solver_names,
#   subsolvers,
#   random_seed,
#   file_name = file_name,
# )
# file_name = file_names[5]
# benchmark_plot(
#   f,
#     model,
#   1:(f.meta.nvar),
#   h,
#   solvers,
#   colors,
#   styles,
#   solver_names,
#   subsolvers,
#   random_seed,
#   file_name = file_name,
#   pb_name = "BPDN-3-2",
# )

benchmark_table(
  f,
  model,
  1:(f.meta.nvar),
  sol,
  h,
  λ,
  solvers,
  solver_names,
  subsolvers,
  colors,
  styles,
  random_seed,
  tex = true,
  legend_name = "BPDN-3-2",
  pb_name = "bpdn-3-2",
  file_name = file_name,
  time_plot = true,
);



# #helper_plot_PFG(coords, colors, solver_names)