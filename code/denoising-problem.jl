#include("Model_denoising/denoising_model.jl")
include("regulopt-plots.jl")
include("regulopt-tables.jl")

using FFTW
using Wavelets
using Images

random_seed = 1234
Random.seed!(random_seed)
n, m = 256, 256
n_p, m_p = 260, 260
kernel_size = 9
model, sol = denoising_model((n, m), (n_p, m_p), kernel_size)
f = LBFGSModel(model)
λ = 1.e-4
h = NormL1(λ)

solvers = [R2_None, R2DH_Spec_NM, R2N_R2, R2N_R2DH_Spec]
solver_names = ["R2", "R2DH", "R2N-R2", "R2N-R2DH"]
colors = colors[1:4] # then add colors(7) and colors(8)
styles = styles[1:4]
file_name = file_names[7]

subsolvers =
  [:None, :None, :R2_None, :R2DH_Spec_NM]


# benchmark_plot(
#   f,
#     f,
#   1:(f.meta.nvar),
#   h,
#   solvers,
#     colors,
#   styles,
#   solver_names,
#   subsolvers,
#   random_seed,
#   pb_name = "Denoise",
#   file_name = file_name
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
  legend_name = "Denoise",
  pb_name = "denoise",
  file_name = file_name,
);

#helper_plot_PFG(coords, colors, solver_names)