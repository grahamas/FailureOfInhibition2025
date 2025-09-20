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

"""
    difference_of_simple_sigmoids(x, a_up, θ_up, a_down, θ_down)

The difference of two sigmoid functions: sigmoid_up(x) - sigmoid_down(x).
This can create bump-like or other complex nonlinear shapes.
"""
function difference_of_simple_sigmoids(x, a_up, θ_up, a_down, θ_down)
    simple_sigmoid(x, a_up, θ_up) - simple_sigmoid(x, a_down, θ_down)
end

"""
    difference_of_rectified_zeroed_sigmoids(x, a_up, θ_up, a_down, θ_down)

The difference of two rectified zeroed sigmoid functions: rectified_zeroed_sigmoid_up(x) - rectified_zeroed_sigmoid_down(x).
This ensures the result cannot be negative and creates more biologically realistic bump-like functions.
"""
function difference_of_rectified_zeroed_sigmoids(x, a_up, θ_up, a_down, θ_down)
    rectified_zeroed_sigmoid(x, a_up, θ_up) - rectified_zeroed_sigmoid(x, a_down, θ_down)
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

"""
    DifferenceOfSigmoidsNonlinearity{T}

Difference of sigmoids nonlinearity parameter type with parameters for two sigmoids:
- a_up, θ_up: parameters for the "up" sigmoid
- a_down, θ_down: parameters for the "down" sigmoid
The result is rectified_zeroed_sigmoid_up(x) - rectified_zeroed_sigmoid_down(x).
This defaults to using rectified zeroed sigmoids for biological realism.
"""
struct DifferenceOfSigmoidsNonlinearity{T}
    a_up::T
    θ_up::T
    a_down::T
    θ_down::T
end

# Constructor with keyword arguments
DifferenceOfSigmoidsNonlinearity(; a_up, θ_up, a_down, θ_down) = DifferenceOfSigmoidsNonlinearity(a_up, θ_up, a_down, θ_down)

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

"""
    apply_nonlinearity(dA, A, nonlinearity::DifferenceOfSigmoidsNonlinearity, t)

Apply difference of sigmoids nonlinearity to the activation array A, modifying dA.
Implements difference of rectified zeroed sigmoids logic directly without unnecessary copies.
"""
function apply_nonlinearity(dA, A, nonlinearity::DifferenceOfSigmoidsNonlinearity, t)
    # Apply difference of rectified zeroed sigmoids nonlinearity directly: dA += difference_of_rectified_zeroed_sigmoids(A) - A
    @. dA += difference_of_rectified_zeroed_sigmoids(A, nonlinearity.a_up, nonlinearity.θ_up, nonlinearity.a_down, nonlinearity.θ_down) - A
end