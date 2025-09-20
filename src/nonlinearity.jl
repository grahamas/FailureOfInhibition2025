# Simple sigmoid nonlinearity implementation
# Based on NeuralModels.jl with minimal complexity

############## Basic Sigmoid Functions ##############

"""
    simple_sigmoid(x, a, theta)

The sigmoid function: 1/(1 + exp(-a*(x - theta)))
where a describes the slope's steepness and theta describes translation of the slope's center.
"""
function simple_sigmoid(x, a, theta)
    1.0 / (1 + exp(-a * (x - theta)))
end

"""
    rectified_zeroed_sigmoid(x, a, theta)

A rectified version of the zeroed sigmoid function.
In practice, we use rectified sigmoid functions because firing rates cannot be negative.
"""
function rectified_zeroed_sigmoid(x, a, theta)
    zeroed = simple_sigmoid(x, a, theta) - simple_sigmoid(0.0, a, theta)
    return max(0.0, zeroed)
end

############## Sigmoid Parameter Types ##############

"""
    SigmoidNonlinearity{T}

Simple sigmoid nonlinearity parameter type with slope a and threshold theta.
"""
struct SigmoidNonlinearity{T}
    a::T
    θ::T
end

# Constructor with keyword arguments
SigmoidNonlinearity(; a, θ) = SigmoidNonlinearity(a, θ)

"""
    (s::SigmoidNonlinearity)(inplace::AbstractArray, source, t)

Apply sigmoid nonlinearity in-place to the array.
"""
function (s::SigmoidNonlinearity)(inplace::AbstractArray, source=nothing, t=nothing)
    inplace .= simple_sigmoid.(inplace, s.a, s.θ)
end

############## Apply Nonlinearity Interface ##############

"""
    apply_nonlinearity(dA, A, nonlinearity, t)

Apply the specified nonlinearity to the activation array A, modifying dA.
This is the main interface function called by the model.
"""
function apply_nonlinearity(dA, A, nonlinearity, t)
    # Apply nonlinearity to A and add to dA
    # Create a temporary copy to avoid modifying A
    temp = copy(A)
    nonlinearity(temp, A, t)
    dA .+= temp .- A  # Add the nonlinear transformation
end