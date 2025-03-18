module KoopmanCausality
import Distributions: Distribution, AbstractMvNormal
import LinearAlgebra
import DynamicalSystems: embed
import Distances: euclidean
import Plots: scatter, plot!

abstract type AbstractDMDModel end
abstract type AbstractDictionary end
export AbstractDictionary
abstract type AbstractCausalDMDModel end

include("Dictionaries.jl")
export 
    Dictionary,
    evaluate,
    identity_dict,
    RFF_dict,
    RFFI_dict,
    mvRFF_dict,
    RBF_dict,
    Ulam_dict, 
    Ulam_dict_2D,
    monomial_dict_2D

include("DMD.jl")
export
    Koopman,
    PerronFrobenius,
    eigenfunc_eval,
    modes,
    delay_matrices,
    DMD,
    DMDpinv,
    delayDMD,
    pseudospectrum,
    evolve

include("utils.jl")
export
    data_matrices,
    eigen_plot,
    truncated_svd,
    sorted_eigen,
    normalize_modes,
    eigen_periods,
    mse

include("CausalDMD.jl")
export 
    MarginalCausalKoopman,
    JointCausalKoopman,
    causal_DMD,
    causal_eval,
    koopman_causality,
    RFF_koopman_causality
end