include("regulopt-plots.jl")
include("regulopt-tables.jl")

using MLDatasets

random_seed = 1234
Random.seed!(random_seed)
nlp_train, nls_train, sol_train  = RegularizedProblems.svm_train_model(); #
nlp_test, nls_test, sol_test = RegularizedProblems.svm_train_model();
#  f_test = LBFGSModel(nlp_test)
acc = vec -> length(findall(x -> x < 1, vec)) / length(vec) * 100; # for SVM
model = nlp_train;
nls_model = nls_train;
f = LBFGSModel(model);
λ = 1.0e-1
h = NormL0(λ);


solvers = [R2_None, R2DH_Spec_NM, R2N_R2, R2N_R2DH_Spec]
solver_names = ["R2", "R2DH", "R2N-R2", "R2N-R2DH"]
colors = colors[1:4] # then add colors(7) and colors(8)
styles = styles[1:4]
subsolvers =
  [:None, :None, :R2_None, :R2DH_Spec_NM]
file_name = file_names[6]


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
#   pb_name = "SVM",
#   file_name = file_name
# )

benchmark_table(
  f,
    nls_model,
  1:(f.meta.nvar),
  (sol_train, sol_test),
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
  legend_name = "SVM",
  pb_name = "svm",
  file_name = file_name
);

#helper_plot_PFG(coords, colors, solver_names)