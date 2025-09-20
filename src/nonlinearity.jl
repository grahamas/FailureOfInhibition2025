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
    RectifiedZeroedSigmoidNonlinearity{T}

Rectified zeroed sigmoid nonlinearity parameter type with slope a and threshold theta.
In practice, we use rectified sigmoid functions because firing rates cannot be negative.
"""
struct RectifiedZeroedSigmoidNonlinearity{T}
    a::T
    θ::T
end

# Constructor with keyword arguments
RectifiedZeroedSigmoidNonlinearity(; a, θ) = RectifiedZeroedSigmoidNonlinearity(a, θ)

############## Apply Nonlinearity Interface ##############

"""
    apply_nonlinearity(dA, A, nonlinearity::SigmoidNonlinearity, t)

Apply simple sigmoid nonlinearity to the activation array A, modifying dA.
Implements sigmoid logic directly without unnecessary copies.
"""
function apply_nonlinearity(dA, A, nonlinearity::SigmoidNonlinearity, t)
    # Apply sigmoid nonlinearity directly: dA += sigmoid(A) - A
    @. dA += simple_sigmoid(A, nonlinearity.a, nonlinearity.θ) - A
end

"""
    apply_nonlinearity(dA, A, nonlinearity::RectifiedZeroedSigmoidNonlinearity, t)

Apply rectified zeroed sigmoid nonlinearity to the activation array A, modifying dA.
Implements rectified zeroed sigmoid logic directly without unnecessary copies.
"""
function apply_nonlinearity(dA, A, nonlinearity::RectifiedZeroedSigmoidNonlinearity, t)
    # Apply rectified zeroed sigmoid nonlinearity directly: dA += rectified_zeroed_sigmoid(A) - A
    @. dA += rectified_zeroed_sigmoid(A, nonlinearity.a, nonlinearity.θ) - A
end