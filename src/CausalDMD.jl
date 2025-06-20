struct MarginalCausalKoopman{T<:Number} <: AbstractCausalDMDModel
    dict::Dictionary
    M::Matrix{T}
end

struct JointCausalKoopman{T<:Number} <: AbstractCausalDMDModel
    dict::Dictionary
    M::Matrix{T}
end

"""
    causal_DMD(X::AbstractMatrix, Y::AbstractMatrix, dict::AbstractDictionary)

Create a marginal CausalKoopman model using (extended) Dynamic Mode Decomposition.
"""
function causal_DMD(X::AbstractMatrix, Y::AbstractMatrix, dict::AbstractDictionary)
    ΨX = [X; dict(X)]
    K = Y / ΨX
    return MarginalCausalKoopman(dict, K)
end

"""
    causal_DMD(X1::AbstractMatrix, X2::AbstractMatrix, Y::AbstractMatrix, dict::AbstractDictionary)

Create a joint CasualKoopman model for causal effect of X2->X1 (i.e. Y should be time-shift of X1)
"""
function causal_DMD(X1::AbstractMatrix, X2::AbstractMatrix, Y::AbstractMatrix, dict::AbstractDictionary)
    Xjoint = [X1; X2]
    ΨX = [X1; dict(Xjoint)]
    K = Y / ΨX
    return JointCausalKoopman(dict, K)
end

"""
    causal_eval(Xtest::AbstractMatrix, Ytest::AbstractMatrix, marginal_model::MarginalCausalKoopman)

Evaluate the marginal CausalKoopman model on the test data using mean squared error.
"""
function causal_eval(Xtest::AbstractMatrix, Ytest::AbstractMatrix, marginal_model::MarginalCausalKoopman)
    K = marginal_model
    ΨXtest = [Xtest; K.dict(Xtest)]
    predictions = K.M * ΨXtest
    return mse(predictions, Ytest)
end

"""
    causal_eval(X1test::AbstractMatrix, X2test::AbstractMatrix, Ytest::AbstractMatrix, joint_model::JointCausalKoopman)

Evaluate the joint CausalKoopman model on the test data using mean squared error.
"""
function causal_eval(X1test::AbstractMatrix, X2test::AbstractMatrix, Ytest::AbstractMatrix, joint_model::JointCausalKoopman)
    K = joint_model
    Xjoint = [X1test; X2test]
    ΨXtest = [X1test; K.dict(Xjoint)]
    predictions = K.M * ΨXtest
    return mse(predictions, Ytest)
end

# function koopman_causality(
#     X1::AbstractMatrix,
#     X2::AbstractMatrix,
#     Y::AbstractMatrix,
#     X1test::AbstractMatrix,
#     X2test::AbstractMatrix,
#     Ytest::AbstractMatrix,
#     marginal_dict::AbstractDictionary,
#     joint_dict::AbstractDictionary;
#     rectify::Bool = true
# )
#     marginal_model = causal_DMD(X1, Y, marginal_dict)
#     joint_model = causal_DMD(X1, X2, Y, joint_dict)
#     marg_err = causal_eval(X1test, Ytest, marginal_model)
#     joint_err = causal_eval(X1test, X2test, Ytest, joint_model)
#     cause = marg_err - joint_err
#     if cause < 0 && rectify
#         return 0.0
#     else
#         return cause
#     end
# end


function RFF_koopman_causality_base(Xe, Xc, Ye, Xteste, Xtestc, Yteste, N_features, σs, N_samples)
    N_vars_marg = size(Xe, 1)
    N_vars_joint = size(Xc, 1) + N_vars_marg
    marg_err = Inf
    joint_err = Inf

    j = 1
    for σ in σs
        for _ in 1:N_samples
            marg_dist = Normal(0.0, σ)
            marg_dict = RFF_dict(N_features, N_vars_marg, marg_dist)
            marg_model = causal_DMD(Xe, Ye, marg_dict)
            marg = causal_eval(Xteste, Yteste, marg_model)
            marg_err = min(marg_err, marg)
    
            joint_dist = Normal(0.0, σ)
            joint_dict = RFF_dict(N_features, N_vars_joint, joint_dist)
            joint_model = causal_DMD(Xe, Xc, Ye, joint_dict)
            joint = causal_eval(Xteste, Xtestc, Yteste, joint_model)
            joint_err = min(joint_err, joint)
            j += 1
        end
    end    
    return (marg_err, joint_err)
end

function RFF_koopman_causality(Xe, Xc, Ye, Xteste, Xtestc, Yteste, N_features, σs, N_samples)
    marg_err, joint_err = RFF_koopman_causality_base(Xe, Xc, Ye, Xteste, Xtestc, Yteste, N_features, σs, N_samples)
    return marg_err - joint_err
end

"""
Trying different ways of multithread processing
"""
function RFF_koopman_causality_p1(Xe, Xc, Ye, Xteste, Xtestc, Yteste, N_features, σs, N_samples; parallel=false)
    if parallel
        batches = Iterators.partition(σs, div(length(σs), Threads.nthreads(), RoundUp))
        tasks = map(batches) do batch
            Threads.@spawn RFF_koopman_causality_base(Xe, Xc, Ye, Xteste, Xtestc, Yteste, N_features, batch, N_samples)
        end
        results = fetch.(tasks)
        marg_err = mapreduce(r->r[1], min, results)
        joint_err = mapreduce(r->r[2], min, results)
    else
        marg_err, joint_err = RFF_koopman_causality_base(Xe, Xc, Ye, Xteste, Xtestc, Yteste, N_features, σs, N_samples)
    end
    return marg_err - joint_err
end

"""
Trying different ways of multithread processing
"""
function RFF_koopman_causality_p2(Xe, Xc, Ye, Xteste, Xtestc, Yteste, N_features, σs, N_samples; parallel=false)
    if parallel
        marg_err = Threads.Atomic{Float64}(Inf)
        joint_err = Threads.Atomic{Float64}(Inf)
        Threads.@threads for σ in σs
            local_marg_err, local_joint_err = RFF_koopman_causality_base(Xe, Xc, Ye, Xteste, Xtestc, Yteste, N_features, [σ], N_samples)
            Threads.atomic_min!(marg_err, local_marg_err)
            Threads.atomic_min!(joint_err, local_joint_err)
        end
        marg_err = marg_err[]
        joint_err = joint_err[]
    else
        marg_err, joint_err = RFF_koopman_causality_base(Xe, Xc, Ye, Xteste, Xtestc, Yteste, N_features, σs, N_samples)
    end
    return marg_err - joint_err
end