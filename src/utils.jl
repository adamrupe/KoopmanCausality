"""
Takes input trajectory (as an Array) and creates input-output data matrices for Koopman approximation.
If a dictionary is supplied, the columns of the data matrices are dictionary function
evaluations. 
input_data must have size = (n variables, t time steps), dict must be a vector of functions

`data_matrices(data::AbstractArray{Ty, 2}) where Ty <: Real`
"""
function data_matrices(data::AbstractArray{Ty, 2}) where Ty <: Real
    M = data
    Nvars, T = size(M)
    X = Array{eltype(M)}(undef, Nvars, T-1)
    Y = Array{eltype(M)}(undef, Nvars, T-1)
    for t in 1:(T-1)
        X[:,t] = M[:, t]
        Y[:,t] = M[:, t+1]
    end
    return(X,Y)
end    

"""
Takes input trajectory (as an Array) and creates input-output data matrices for Koopman approximation.
If a dictionary is supplied, the columns of the data matrices are dictionary function
evaluations. 
input_data must have size = (n variables, t time steps), dict must be a vector of functions

`data_matrices(data::AbstractArray{Ty, 2}, dict::Dictionary) where Ty <: Real`
"""
function data_matrices(data::AbstractArray{Ty, 2}, dict::Dictionary) where Ty <: Real
    M = data
    T = size(M, 2)
    Nfuncs = length(dict)
    X = Array{eltype(M)}(undef, Nfuncs, T-1)
    Y = Array{eltype(M)}(undef, Nfuncs, T-1)
    for t in 1:(T-1)
        x = M[:, t]
        y = M[:, t+1]
        ψx, ψy = dict(x, y)
        X[:, t] = ψx
        Y[:, t] = ψy
    end
    return(X,Y)
end

# """
# `data_matrices(dataset::Dataset)`

# Takes input trajectory (as a Dataset) and creates input-output data matrices for Koopman approximation.
# If a dictionary is supplied, the columns of the data matrices are dictionary function
# evaluations. 
# """
# function data_matrices(dataset::Dataset)
#     T, Nvars = size(dataset)
#     X = Array{eltype(dataset)}(undef, Nvars, T-1)
#     Y = Array{eltype(dataset)}(undef, Nvars, T-1)
#     for t in 1:(T-1)
#         X[:, t] = dataset[t]
#         Y[:, t] = dataset[t+1]
#     end
#     return (X,Y)
# end

# """
# `data_matrices(dataset::Dataset, dict::Vector{Any})`

# Takes input trajectory (as a Dataset) and creates input-output data matrices for Koopman approximation.
# If a dictionary is supplied, the columns of the data matrices are dictionary function
# evaluations. 
# """
# function data_matrices(dataset::Dataset, dict::AbstractVector)
#     T = size(dataset, 1)
#     Nfuncs = length(dict)
#     X = Array{eltype(dataset)}(undef, Nfuncs, T-1)
#     Y = Array{eltype(dataset)}(undef, Nfuncs, T-1)
#     for t in 1:(T-1)
#         x = dataset[t]
#         y = dataset[t+1]
#         ψx, ψy = apply_dict(x, y, dict)
#         X[:, t] = ψx
#         Y[:, t] = ψy
#     end
#     return (X,Y)
# end


Xₜ(t) = sin(t)
Yₜ(t) = cos(t)
function eigen_plot(evals::AbstractVector)
    scatter(real.(evals), imag.(evals); legend=false, aspectratio=1)
    plot!(Xₜ, Yₜ, 0, 2π; leg=false, aspectratio=1)
end
function eigen_plot(model::AbstractDMDModel)
    evals, evecs = eigen(model)
    eigen_plot(evals)
end

function truncated_svd(A; rank=20)
    U, Σ, V = svd(A)
    Ũ = U[:, 1:rank]
    Σ̃ = Σ[1:rank]
    Ṽ = V[:, 1:rank]
    return (Ũ, Σ̃, Ṽ)
end

"""
Sorts and returns eigenvectors and eigenvalues of M;
in order of largest to smallest absolute value of eigenvalues.
"""
function sorted_eigen(M)
    evals, evecs = LinearAlgebra.eigen(M)
    ind_sort = reverse(sortperm(abs.(evals)))
    sorted_evals = evals[ind_sort]
    sorted_evecs = evecs[:, ind_sort]
    return (sorted_evals, sorted_evecs)
end

function normalize_modes(modes)
    normed_modes = Array{Float64}(undef, size(modes)...)
    skip = false
    for i in 1:size(modes,2)
        if skip
            skip = false
            continue
        else
            mode = modes[:, i]
            rm = real.(mode)
            im = imag.(mode)
            normed_modes[:, i] = rm ./ norm(rm) 
            if sum(abs.(im)) > 1e-6 
                normed_modes[:, i+1] = im ./ norm(im)
                skip = true
            end
        end
    end
    return normed_modes
end

function eigen_periods(evals)
    T = zeros(size(evals)...)
    for (i, val) in enumerate(evals)
        if abs.(imag(val)) > 1e-6
            θ = angle(val)
            T[i] = abs(2π / θ)
        end
    end
    return T
end

"""
Compute the mean squared error. 
"""
function mse(Y::AbstractArray, Ŷ::AbstractArray)
    sqr_err = euclidean.(Y, Ŷ).^2
    mse = sum(sqr_err) / length(sqr_err)
end