include("regulopt-plots.jl")
include("regulopt-tables.jl")

random_seed = 1234
Random.seed!(random_seed)
compound = 10
model, nls_model, sol = bpdn_model(compound, bounds = false)

# parameters
f = LSR1Model(model)
λ = norm(grad(model, zeros(model.meta.nvar)), Inf) / 10
h = NormL0(λ)

solvers = [R2_None, R2DH_Spec_NM, R2DH_Spec,  R2DH_Andrei, R2DH_PSB, R2DH_DBFGS]
solver_names = ["R2", "R2DH-Spec-NM", "R2DH-Spec", "R2DH-Andrei", "R2DH-PSB", "R2DH-DBFGS"]

file_name = file_names[3]
colors = [colors[1:2] ; colors[7:10]]
styles = [styles[1:2] ; styles[7:10]]
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
file_name = file_names[4]
# benchmark_plot(
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

# benchmark_table(
#   f,
#   model,
#   1:(f.meta.nvar),
#     sol,
#   h,
#     λ,
#   solvers,
#   solver_names,
#   subsolvers,
#   random_seed,
#   tex = false,
# );

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
  legend_name = "BPDN",
  pb_name = "bpdn",
  file_name = file_name,
  time_plot = true,
);

# #helper_plot_PFG(coords, colors, solver_names)