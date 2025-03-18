# struct Dictionary{T} <: AbstractDictionary where T <:Number
#     Ψ::Vector{Function}
#     eltype::T
# end
struct Dictionary <: AbstractDictionary 
    Ψ::Vector{Function}
    eltype::Type
end

Base.length(d::Dictionary) = Base.length(d.Ψ)
Base.enumerate(d::Dictionary) = Base.enumerate(d.Ψ)

# """
# ```julia
#     evaluate(x::AbstractVector, dict::Dictionary)
# ```
# Evaluate the dictionary functions on the given input vector. 
# """
# function evaluate(x::AbstractVector, dict::Dictionary)
#     Nfuncs = length(dict)
#     ψ = Vector{dict.eltype}(undef, Nfuncs)
#     for (i, f) in enumerate(dict)
#         ψ[i] = f(x)
#     end
#     return ψ
# end

# (d::Dictionary)(x::AbstractVector) = evaluate(x, d)
(dict::Dictionary)(x::AbstractVector) = map(func->func(x), dict.Ψ)

# function evaluate(X::AbstractArray, dict::Dictionary)
#     T = size(X, 2)
#     Nfuncs = length(dict)
#     ΨX = Array{dict.eltype}(undef, Nfuncs, T)
#     @views for t in 1:T
#         xₜ = X[:, t]
#         # xₜ = view(X, :, t)
#         # ΨX[:, t] = evaluate(xₜ, dict)
#         ΨX[:, t] = dict(xₜ)
#     end
#     return ΨX
# end

function evaluate(X::AbstractMatrix, dict::Dictionary)
    Nfuncs = length(dict)
    Nsamps = size(X, 2)
    evals = Array{dict.eltype}(undef, Nfuncs, Nsamps)
    for (i, f) in enumerate(dict)
        evals[i, :] .= f.(eachcol(X))
    end
    return evals
end

(d::Dictionary)(X::AbstractMatrix) = evaluate(X, d)


"""
```julia
    evaluate(x::AbstractVector, y::AbstractVector, dict::Dictionary) 
```
Apply the dictionary functions to the given input vector. 
"""
function evaluate(x::AbstractVector, y::AbstractVector, dict::Dictionary) 
    Nfuncs = length(dict)
    ψx = Vector{dict.eltype}(undef, Nfuncs)
    ψy = Vector{dict.eltype}(undef, Nfuncs)
    for (i, f) in enumerate(dict)
        ψx[i] = f(x)
        ψy[i] = f(y)
    end
    return (ψx, ψy)
end

(d::Dictionary)(x::AbstractVector, y::AbstractVector) = evaluate(x, y, d)

"""
```julia
    identity_dict(Nvars::Int)
```
Create the component-wise identity dictionary elements. Useful for forecasting
by ensuring the identity is in the span of a larger concatenated dictionary.
"""
function identity_dict(Nvars::Int; eltype=Float64)
    ψ_I = Vector{Function}(undef, Nvars)
    for i in 1:Nvars
        ψ_I[i] = u->u[i]
    end
    return Dictionary(ψ_I, eltype)
end

"""
```julia
    RFF_dict(Nfeatures::Int, Nvars::Int, Dist::Distribution)
```
Create a dictionary of Random Fourier Features (RFFs), 
sampled from the given Distribution.
"""
function RFF_dict(Nfeatures::Int, Nvars::Int, Dist::Distribution)
    Ψ = Vector{Function}(undef, Nfeatures)
    ωs = rand(Dist, (Nvars, Nfeatures))
    bs = 2π*rand(Nfeatures)
    
    for i in 1:Nfeatures
        ω = ωs[:, i]
        b = bs[i]
        Ψ[i] = x->cos(ω'*x + b)
    end
    return Dictionary(Ψ, Float64)
end

"""
```julia
    RFFI_dict(Nfeatures::Int, Nvars::Int, Dist::Distribution)
```
Create a dictionary of Random Fourier Features (RFFs), 
sampled from the given Distribution, with the identity dictionary concatenated
at the top of the dictionary.
"""
function RFFI_dict(Nfeatures::Int, Nvars::Int, Dist::Distribution)
    Ψrff = RFF_dict(Nfeatures, Nvars, Dist).Ψ
    Ψ_I = identity_dict(Nvars).Ψ
    return Dictionary(vcat(Ψ_I, Ψrff), Float64)
end

"""
Create a multivariate random Fourier feature diction with 
multivariate Gaussian (i.e. with covariance matrix specified).
"""
function mvRFF_dict(Nfeatures::Int, Nvars::Int, Dist::AbstractMvNormal)
    ψ = Vector{Function}(undef, Nfeatures)
    ωs = rand(Dist, (Nvars, Nfeatures))
    bs = 2π*rand(Nfeatures)
    
    for i in 1:Nfeatures
        ω = ωs[:, i]
        b = bs[i]
        ψ[i] = u->cos(ω'*u + b)
    end
    return Dictionary(ψ, Float64)
end

gauss_rbf(x::AbstractVector,z::AbstractVector; α=1.0) = exp(-0.5*α * Distances.euclidean(x,z)^2)

"""
Gaussian RBF dictionary
"""
function RBF_dict(centers::AbstractArray; α=1)
    Nvars, N = size(centers)
    ψ = Vector{Function}(undef, N)
    for i in 1:N
        c = centers[:, i]
        ψ[i] = u->gauss_rbf(c, u; α=α)
    end
    return (ψ, Float64) 
end

function indicator(x::Float64, a, b)
    if a <= x < b 
        return 1.0
    else
        return 0.0
    end
end

function indicator(x::Vector{Float64}, a, b)
    return indicator(x[1], a, b)
end

function Ulam_dict(Nboxes, low_bound, high_bound)
    Ψ = Vector{Function}(undef, Nboxes)
    boundaries = collect(LinRange(low_bound, high_bound, Nboxes+1))
    for i in 1:Nboxes
        low = boundaries[i]
        high = boundaries[i+1]
        Ψ[i] = u -> indicator(u, low, high)
    end
    return Dictionary(Ψ, Float64)
end

function indicator2D(
        u::Vector{<:Real}, 
        (xl, xh)::Tuple{<:Real, <:Real}, 
        (yl, yh)::Tuple{<:Real, <:Real}
    ) 
    x,y = u
    if (xl <= x < xh) && (yl <= y < yh)
        return 1.0
    else
        return 0.0
    end
end

function Ulam_dict_2D(Nboxes_x, Nboxes_y, (low_bound_x, high_bound_x), (low_bound_y, high_bound_y))
    Ψ = Vector{Function}(undef, Nboxes_x * Nboxes_y)
    x_boundaries = collect(LinRange(low_bound_x, high_bound_x, Nboxes_x + 1))
    y_boundaries = collect(LinRange(low_bound_y, high_bound_y, Nboxes_y + 1))
    
    idx = 1
    for i in 1:Nboxes_x
        for j in 1:Nboxes_y
            xl = x_boundaries[i]
            xh = x_boundaries[i + 1]
            yl = y_boundaries[j]
            yh = y_boundaries[j + 1]
            Ψ[idx] = u -> indicator2D(u, (xl, xh), (yl, yh))
            idx += 1
        end
    end
    
    return Dictionary(Ψ, Float64)
end

function monomial_dict_2D(order::Int)
    ψ = []
    for i in 0:order
        x_power = i
        y_power = 0
        for j in 0:i
            local xp, yp = x_power, y_power
            push!(ψ, u::AbstractArray{<:Real} -> (u[1]^xp)*(u[2]^yp))
            x_power -= 1
            y_power += 1
        end
    end
    ψ = convert(Vector{Function}, ψ)
    return Dictionary(ψ, Float64)
end