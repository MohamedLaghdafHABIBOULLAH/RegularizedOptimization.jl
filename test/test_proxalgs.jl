@testset "Lasso ($T) Descent Methods" for T in [Float32, Float64]#, ComplexF32, ComplexF64]
    using LinearAlgebra
    using TRNC 
    using Plots, Printf

    # compound = 1
    compound = 10
    m,n = compound*25, compound*64
    p = randperm(n)
    k = compound*2

    #initialize x 
    x0 = zeros(T,n)
    p = randperm(n)[1:k]
    x0[p[1:k]]=sign.(randn(T, k))

    A,_ = qr(randn(T, (n,m)))
    B = Array(A)'
    A = Array(B)

    b0 = A*x0
    

    R = real(T)
    b = b0 + R(.001)*randn(T,m)

    λ = R(0.1)*norm(A'*b, Inf)
    @test typeof(λ) == R


    β = eigmax(A'*A)


    function proxp!(z, α)
        n = length(z)
        for i = 1:n
            abs(z[i]) > α ? z[i]-= sign(z[i])*α : z[i] = T(0.0)
        end
    end

    function funcF!(z, g)
        r = copy(b)
        BLAS.gemv!('N', T(1.0), A, z, T(-1.0), r)
        BLAS.gemv!('T', T(1.0), A, r, T(0.0), g)
        return r'*r
    end

    function funcF(x)
        r = A*x - b
        g = A'*r
        return norm(r)^2, g
    end
    function proxp(z, α)
        # y = copy(z)
        # for i in eachindex(z)
        #     aux = max(abs(z[i]) - α,0)
        #     y[i] = aux/(aux+α)*z[i]
        # end
        # return y
        return sign.(z).*max.(abs.(z).-(α)*ones(T, size(z)), zeros(T, size(z)))
    end

    function funcH(x)
        return λ*norm(x,1)
    end

    TOL = R(1e-6)

    @testset "PG" begin

        ## PG and PG! (in place)
        pg_options=s_options(β; maxIter=5000, verbose=0, λ=λ, optTol=TOL)
        x = zeros(T, n)

        x_out, x⁻_out, hispg_out, fevalpg_out = PG(funcF,funcH, x, proxp, pg_options)
        x_d, x⁻_d, hispg_d, fevalpg_d = PGLnsch(funcF, funcH, x, proxp, fista_options)
        x⁻, hispg, fevalpg = PG!(funcF!,funcH, x, proxp!,pg_options)

        #check types
        @test eltype(x) == T
        @test eltype(x_out) == T
        @test eltype(x⁻_out) == T
        @test eltype(x⁻) == T
        @test eltype(x_d) == T
        @test eltype(x⁻_d) == T

        #check func evals less than maxIter 
        @test fevalpg_out <= 5000
        @test fevalpg <= 5000
        @test fevalpg_d <= 5000

        #check overall accuracy
        @test norm(x - x0)/norm(x0) <= .15
        @test norm(x_out - x0)/norm(x0) <= .15 
        @test norm(x_d - x0)/norm(x0) <= .15

        #check relative accuracy 
        @test norm(x_out - x⁻_out, Inf) <= TOL
        @test norm(x - x⁻, Inf) <= TOL
        @test norm(x⁻_d - x_d, Inf) <=TOL

        @test norm(x_out - x, Inf) <= TOL
        @test norm(x⁻_out - x⁻, Inf) <= TOL
        @test norm(x_d - x_out, Inf) <= TOL 
        @test norm(x_d - x, Inf) <= TOL 
        

        #test monotonicity
        @test sum(diff(hispg_out)>.0)==length(diff(hispg_out))
        @test sum(diff(hispg_d)>.0)==length(diff(hispg_d))
        @test sum(diff(hispg)>.0)==length(diff(hispg))
        


    end

    @testset "FISTA" begin
        ## FISTA and FISTA! (in place)
        fista_options=s_options(β; maxIter=5000, verbose=0, λ=λ, optTol=TOL)
        x = zeros(T, n)

        x_out, x⁻_out, hisf_out, fevalf_out = FISTA(funcF,funcH, x, proxp, fista_options)
        x_d, x⁻_d, hisf_d, fevalf_d = FISTAD(funcF, funcH, x, proxp, fista_options)
        x⁻, hispf, fevalf = FISTA!(funcF!,funchH, x, proxp!,fista_options)
        


        #check types
        @test eltype(x) == T
        @test eltype(x_out) == T
        @test eltype(x⁻_out) == T
        @test eltype(x⁻) == T
        @test eltype(x_d) == T
        @test eltype(x⁻_d) == T

        #check func evals less than maxIter 
        @test fevalf_out <= 5000
        @test fevalf <= 5000
        @test fevalf_d <= 5000

        #check overall accuracy
        @test norm(x - x0)/norm(x0) <= .15
        @test norm(x_out - x0)/norm(x0) <= .15
        @test norm(x_d - x0)/norm(x0) <= .15 


        #check relative accuracy 
        @test norm(x_out - x⁻_out, Inf) <= TOL
        @test norm(x - x⁻, Inf) <= TOL
        @test norm(x_d - x⁻_d, Inf) <= TOL


        @test norm(x_out - x, Inf) <= TOL
        @test norm(x⁻_out - x⁻, Inf) <= TOL
        @test norm(x_d - x_out, Inf) <= TOL 
        @test norm(x_d - x, Inf) <= TOL
        
        #test monotonicity
        temp = diff(hisf_d)
        @test sum(temp>.0)==length(temp)

    end


end