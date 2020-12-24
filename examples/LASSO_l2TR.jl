using Random, LinearAlgebra, TRNC, Printf,Roots

# min_x 1/2||Ax - b||^2 + λ||x||₁
compound = 1
m,n = compound*200,compound*512 #if you want to rapidly change problem size 
k = compound*10 #10 signals 
α = .01 #noise level 

#start bpdn stuff 
x0 = zeros(n)
p   = randperm(n)[1:k]
x0 = zeros(n,)
x0[p[1:k]]=sign.(randn(k)) #create sparse signal 

A,_ = qr(randn(n,m))
B = Array(A)'
A = Array(B)

b0 = A*x0
b = b0 + α*randn(m,)


λ = norm(A'*b, Inf)/10 #this can change around 

#define your smooth objective function
function f_obj(x) #gradient and hessian info are smooth parts, m also includes nonsmooth part
    r = A*x - b
    g = A'*r
    return norm(r)^2/2, g, A'*A
end

function h_obj(x)
    return norm(x,1) #without  lambda
end

#combination l1 and B2 prox. This is a little wonky sometimes with s gets really small 
function prox(q, σ, xk, Δ) #q = s - ν*g, ν*λ, xk, Δ - > basically inputs the value you need

    ProjB(y) = min.(max.(y, q.-σ), q.+σ)
    froot(η) = η - norm(ProjB((-xk).*(η/Δ)))

    # %do the 2 norm projection
    y1 = ProjB(-xk) #start with eta = tau

    if (norm(y1)<= Δ)
        y = y1  # easy case
    else
        η = fzero(froot, 1e-10, Inf)
        y = ProjB((-xk).*(η/Δ))
    end
    if (norm(y)<=Δ)
        snew = y
    else
        snew = Δ.*y./norm(y)
    end
    return snew
end 

#set options for inner algorithm - only requires ||Bk|| norm guess to start (and λ but that is updated in IP_alg)
#verbosity is levels: 0 = nothing, 1 -> maxIter % 10, 2 = maxIter % 100, 3+ -> print all 
β = eigmax(A'*A) #1/||Bk|| for exact Bk = A'*A
Doptions=s_options(1/β; maxIter=10000, verbose=0, λ = λ, optTol=1e-16)



#define parameters - must feed in smooth, nonsmooth, and λ
#first order options default ||Bk|| = 1.0, no printing. PG is default inner, Rkprox is inner prox loop - defaults to 2-norm ball projection (not accurate if h=0)
parameters = IP_struct(f_obj, h_obj, λ; FO_options = Doptions, s_alg=PG, Rkprox=prox)
options = IP_options(; ϵD=1e-7, verbose = 10) #options, such as printing (same as above), tolerance, γ, σ, τ, w/e
#put in your initial guesses
xi = ones(n,)/2

#input initial guess, parameters, options 
x_pr, k, Fhist, Hhist, Comp_pg = IntPt_TR(xi, parameters, options)
#final value, kth iteration, smooth history, nonsmooth history (with λ), # of evaluations in the inner PG loop 




#If you want to test PG 
function funcF(x)
    r = A*x - b
    g = A'*r
    return norm(r)^2, g
end
function proxp(z, α)
    return sign.(z).*max.(abs.(z).-(λ*α)*ones(size(z)), zeros(size(z)))
end

Doptions.verbose = 2 #print every 100 
Doptions.ν = 1/β #guess exact step size 
xpg, xpg⁻, histpg, fevals = PG(funcF, h_obj, xi, proxp, Doptions) #takes in smooth, nonsmooth, initial guess, prox, options (with λ)
#output final, secont to last, total function history, number of evals 