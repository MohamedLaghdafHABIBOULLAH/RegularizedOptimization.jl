include("regulopt-plots.jl")
include("regulopt-tables.jl")

# choose random init
random_seed = 1234
Random.seed!(random_seed)
compound = 1
dim = 120
λ = 1.e-1
model, nls_model, sol = random_matrix_completion_model(m = dim, n = dim, r = 40, sr = 0.9, va = 0.0001, vb = 0.1, c = 0.2)
f = LSR1Model(model)
F = psvd_workspace_dd(zeros(dim, dim), full = false)
h = Nuclearnorm(λ, ones(dim, dim), F)

solvers = [R2_None, R2DH_Spec_NM, LM_R2DH_Spec, LM_R2]
solver_names = ["R2", "R2DH", "LM-R2DH", "LM-R2"]
subsolvers =
  [:None, :None, :R2DH_Spec_NM, :R2_None]
colors = [colors[1:2] ; colors[5:6]]
styles = [styles[1:2] ; styles[5:6]]
file_name = file_names[2]

# benchmark_plot(
#   f,
#     nls_model,
#   1:(f.meta.nvar),
#   h,
#   solvers,
#     colors,
#   styles,
#   solver_names,
#   subsolvers,
#   random_seed,
#   pb_name = "MC with nuclear norm",
#   file_name = file_name,
#   nonlinear = false
# )

benchmark_table(
  f,
  nls_model,
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
  nls_train = nls_train,
  nls_test = nls_test,
  legend_name = "MC with nuclear norm",
  pb_name = "mc-nn",
  file_name = file_name,
  nonlinear = false
);

#helper_plot_PFG(coords, colors, solver_names)