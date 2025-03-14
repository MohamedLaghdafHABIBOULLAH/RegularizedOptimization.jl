"""
    @wrappedallocs(expr)

Given an expression, this macro wraps that expression inside a new function
which will evaluate that expression and measure the amount of memory allocated
by the expression. Wrapping the expression in a new function allows for more
accurate memory allocation detection when using global variables (e.g. when
at the REPL).

This code is based on that of https://github.com/JuliaAlgebra/TypedPolynomials.jl/blob/master/test/runtests.jl

For example, `@wrappedallocs(x + y)` produces:

```julia
function g(x1, x2)
    @allocated x1 + x2
end
g(x, y)
```

You can use this macro in a unit test to verify that a function does not
allocate:

```
@test @wrappedallocs(x + y) == 0
```
"""
macro wrappedallocs(expr)
  argnames = [gensym() for a in expr.args]
  quote
    function g($(argnames...))
      @allocated $(Expr(expr.head, argnames...))
    end
    $(Expr(:call, :g, [esc(a) for a in expr.args]...))
  end
end

# Test non allocating solve!
@testset "allocs" begin
  for (h, h_name) ∈ ((NormL0(λ), "l0"), (NormL1(λ), "l1"))
    for solver ∈ (:R2Solver,)
      reg_nlp = RegularizedNLPModel(bpdn, h)
      solver = eval(solver)(reg_nlp)
      stats = RegularizedExecutionStats(reg_nlp)
      @test @wrappedallocs(solve!(solver, reg_nlp, stats)) == 0
    end
  end
end
