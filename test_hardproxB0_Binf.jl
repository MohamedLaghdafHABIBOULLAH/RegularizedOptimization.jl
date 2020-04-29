using TRNC,Printf, Convex,SCS, Random, LinearAlgebra, IterativeSolvers, Roots


function hardproxtestB0Binf(n)

A,_ = qr(5*randn(n,n))
B = Array(A)'
A = Array(B)
# rng(2)
# vectors
k = compound*20
p = randperm(n)
#initialize x
x = zeros(n,)
x[p[1:k]]=sign.(randn(k))
g = 5*randn(n)

# scalars
ν = 1/norm(A'*A)^2
λ = 10*rand()
τ = 3*rand()

# This constructs q = ν∇qᵢ(sⱼ) = Bksⱼ + gᵢ (note that i = k in paper notation)
#but it's first order tho so sⱼ = 0 and it's just ∇f(x_k)
q = g #doesn't really matter tho in the example

fval(y, bq, bx, νi) = (y-(bx-bq)).^2/(2*νi)+λ*abs.(y)
projbox(w, bx, τi) = min.(max.(w,bx.-τi), bx.+τi)

Doptions=s_options(1/ν; maxIter=10, λ=λ, ∇fk = g, Bk = A'*A, xk=x, Δ = τ)
# n=10

# (s,f) = hardproxBinf(q, x, ν,λ, τ)
(s,s⁻, f, funEvals) = hardproxBinf(fval, zeros(size(x)), projbox, Doptions)


s_cvx = Variable(n)
problem = minimize(sumsquares(s_cvx+q)/(2*ν) + λ*norm(s_cvx+x,1), norm(s_cvx, Inf)<=τ);
solve!(problem, SCSSolver())


return norm(s_cvx.value .- s)

end
