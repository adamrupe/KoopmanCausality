struct Koopman{T<:Number} <: AbstractDMDModel
    dict::Dictionary
    M::Matrix{T}
end

struct PerronFrobenius{T<:Number} <: AbstractDMDModel
    dict::Dictionary
    M::Matrix{T}
end

LinearAlgebra.eigen(model::AbstractDMDModel) = sorted_eigen(model.M')
modes(model::AbstractDMDModel) = sorted_eigen(model.M)
Base.size(model::AbstractDMDModel) = Base.size(model.M)

function eigenfunc_eval(x::AbstractVector, eigenvec::AbstractVector, dict::AbstractDictionary)
    ψx = dict(x)
    return eigenvec'*ψx
end

function DMD(X::AbstractMatrix, Y::AbstractMatrix, dict::AbstractDictionary; koopman::Bool=true)
    ΨX = dict(X)
    ΨY = dict(Y)
    A = ΨY * transpose(ΨX)
    G = ΨX * transpose(ΨX)
    if koopman
        M = A*LinearAlgebra.pinv(G) # Koopman approximation 
        return Koopman(dict, M)
    else
        M = transpose(A)*LinearAlgebra.pinv(G) # Perron-Frobenius approximation
        return PerronFrobenius(dict, M)
    end
end

function DMD(X::AbstractMatrix, Y::AbstractMatrix; koopman::Bool=true)
    A = Y * transpose(X)
    G = X * transpose(X)
    if koopman
        M = A*LinearAlgebra.pinv(G) # Koopman approximation 
        return Koopman(missing, M)
    else
        M = transpose(A)*LinearAlgebra.pinv(G) # Perron-Frobenius approximation
        return PerronFrobenius(missing, M)
    end
end

"""
Computes the 'exact' DMD Koopman approximation using the pseudoinverse as 
K = Y/X
where Y is the unit time-shift of X (i.e. Y[:, t] = X[:t+1])
"""
function DMDpinv(X::AbstractMatrix, Y::AbstractMatrix)
    M = Y/X
    return Koopman(missing, M)
end

"""
Computes the 'exact' DMD Koopman approximation using the pseudoinverse as 
K = Y/X
where Y is the unit time-shift of X (i.e. Y[:, t] = X[:t+1])
"""
function DMDpinv(X::AbstractMatrix, Y::AbstractMatrix, dict::AbstractDictionary)
    # ΨX = apply_dict(X, dict)
    # ΨY = apply_dict(Y, dict)
    ΨX = dict(X)
    ΨY = dict(Y)
    M = ΨY/ΨX
    return Koopman(dict, M)
end

"""
Constructs deyal / Hankel matrix and its unit time-shift from input series
for direct use with HankelDMD. Input series should be one dimensional 
(a Vector). d is embedding dimension (length of 
delay embedding vector), and τ is the lag.
In this formulation, time moves forward as you go down the delay 
embedding vector (rows of the Hankel matrices).
"""
function delay_matrices(series::AbstractVector, d::Int; τ::Int=1)
    delays = embed(series, d, τ)
    delays = permutedims(Matrix(delays))
    T = size(delays, 2)
    X = delays[:, 1:T-1]
    Y = delays[:, 2:T]
    return (X, Y)
end

function delayDMD(series::AbstractVector, d::Int; τ::Int=1)
    X, Y = delay_matrices(series, d; τ)
    K = Y / X
end

function pseudospectrum(A::Matrix{<:Real}, grid_range)
    L = Base.length(grid_range)
    pspec = Matrix{Float64}(undef, L, L)
    for (i, x) in Base.enumerate(grid_range)
        for (j, y) in Base.enumerate(grid_range)
            z = x + y*im
            pspec[j,i] = LinearAlgebra.norm( (A - z*LinearAlgebra.I)^(-1) )
        end
    end
    return pspec
end

pseudospectrum(model::AbstractDMDModel, grid_range) = pseudospectrum(model.M, grid_range)

"""
```julia
    evolve(K::Koopman, x::AbstractVector, Tsteps::Int)
```

Uses the given Koopman approximation K to evolve the state vector x forward Tsteps.
If a dict is provided, evolves the lifted state vector ψ forward Tsteps. 

The first column of the output is the given initial state vector, preds[:, 1] = ψ(x)
"""
function evolve(K::Koopman, x::AbstractVector, Tsteps::Int)
    Nvars = Base.length(x)
    preds = Array{eltype(x)}(undef, Nvars, Tsteps+1)
    preds[:, 1] = x
    for t in 1:Tsteps
        preds[:, t+1] = K.M*preds[:, t]
    end
    return preds
end    

"""
```julia
    evolve(K::Koopman, x::AbstractVector, Tsteps::Int, dict::AbstractDictionary)
```

Uses the given Koopman approximation K to evolve the state vector x forward Tsteps.
If a dict is provided, evolves the lifted state vector ψ forward Tsteps. 

The first column of the output is the given initial state vector, preds[:, 1] = ψ(x)
"""
function evolve(K::Koopman, x::AbstractVector, Tsteps::Int, dict::AbstractDictionary)
    Nfuncs = length(dict)
    ψx = dict(x)
    preds = Array{eltype(x)}(undef, Nfuncs, Tsteps+1)
    preds[:, 1] = ψx
    for t in 1:Tsteps
        preds[:, t+1] = K.M*preds[:, t]
    end
    return preds
end