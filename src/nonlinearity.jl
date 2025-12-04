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
    difference_of_simple_sigmoids(x, a_activating, θ_activating, a_failing, θ_failing)

The difference of two sigmoid functions: sigmoid_activating(x) - sigmoid_failing(x).
This can create bump-like or other complex nonlinear shapes characteristic of failure of inhibition.
"""
function difference_of_simple_sigmoids(x, a_activating, θ_activating, a_failing, θ_failing)
    simple_sigmoid(x, a_activating, θ_activating) - simple_sigmoid(x, a_failing, θ_failing)
end

"""
    difference_of_rectified_zeroed_sigmoids(x, a_activating, θ_activating, a_failing, θ_failing)

The difference of two rectified zeroed sigmoid functions: rectified_zeroed_sigmoid_activating(x) - rectified_zeroed_sigmoid_failing(x).
This ensures the result cannot be negative and creates more biologically realistic bump-like functions
characteristic of failure of inhibition dynamics.
"""
function difference_of_rectified_zeroed_sigmoids(x, a_activating, θ_activating, a_failing, θ_failing)
    rectified_zeroed_sigmoid(x, a_activating, θ_activating) - rectified_zeroed_sigmoid(x, a_failing, θ_failing)
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

Difference of sigmoids nonlinearity parameter type for Failure of Inhibition (FoI) models.
This creates a non-monotonic nonlinearity using two sigmoids:
- a_activating, θ_activating: slope and threshold for the activating sigmoid
- a_failing, θ_failing: slope and threshold for the failing (inhibitory) sigmoid
The result is rectified_zeroed_sigmoid_activating(x) - rectified_zeroed_sigmoid_failing(x).
This defaults to using rectified zeroed sigmoids for biological realism.

In an FoI model, the "failing" sigmoid represents inhibition that fails at higher activity levels,
creating a bump-like or non-monotonic response function characteristic of failure of inhibition dynamics.
"""
struct DifferenceOfSigmoidsNonlinearity{T}
    a_activating::T
    θ_activating::T
    a_failing::T
    θ_failing::T
end

# Constructor with keyword arguments
DifferenceOfSigmoidsNonlinearity(; a_activating, θ_activating, a_failing, θ_failing) = DifferenceOfSigmoidsNonlinearity(a_activating, θ_activating, a_failing, θ_failing)

############## Apply Nonlinearity Interface ##############

"""
    apply_nonlinearity!(dA, A, nonlinearity::SigmoidNonlinearity, t)

Apply simple sigmoid nonlinearity to the activation array A, modifying dA.
Implements sigmoid logic directly without unnecessary copies.
"""
function apply_nonlinearity!(dA, A, nonlinearity::SigmoidNonlinearity, t)
    # Apply sigmoid nonlinearity directly: dA += sigmoid(A) - A
    @. dA += simple_sigmoid(A, nonlinearity.a, nonlinearity.θ) - A
end

"""
    apply_nonlinearity!(dA, A, nonlinearity::RectifiedZeroedSigmoidNonlinearity, t)

Apply rectified zeroed sigmoid nonlinearity to the activation array A, modifying dA.
Implements rectified zeroed sigmoid logic directly without unnecessary copies.
"""
function apply_nonlinearity!(dA, A, nonlinearity::RectifiedZeroedSigmoidNonlinearity, t)
    # Apply rectified zeroed sigmoid nonlinearity directly: dA += rectified_zeroed_sigmoid(A) - A
    @. dA += rectified_zeroed_sigmoid(A, nonlinearity.a, nonlinearity.θ) - A
end

"""
    apply_nonlinearity!(dA, A, nonlinearity::DifferenceOfSigmoidsNonlinearity, t)

Apply difference of sigmoids nonlinearity to the activation array A, modifying dA.
Implements difference of rectified zeroed sigmoids logic for Failure of Inhibition (FoI) models
directly without unnecessary copies.
"""
function apply_nonlinearity!(dA, A, nonlinearity::DifferenceOfSigmoidsNonlinearity, t)
    # Apply difference of rectified zeroed sigmoids nonlinearity directly: dA += difference_of_rectified_zeroed_sigmoids(A) - A
    @. dA += difference_of_rectified_zeroed_sigmoids(A, nonlinearity.a_activating, nonlinearity.θ_activating, nonlinearity.a_failing, nonlinearity.θ_failing) - A
end

"""
    apply_nonlinearity!(dA, A, nonlinearity::Tuple, t)

Apply per-population nonlinearities when nonlinearity is a tuple.
Each element of the tuple is applied to the corresponding population.
"""
function apply_nonlinearity!(dA, A, nonlinearity::Tuple, t)
    P = length(nonlinearity)
    for i in 1:P
        # Extract population i
        if ndims(dA) == 1
            dAi = dA
            Ai = A
        elseif ndims(dA) == 2
            dAi = view(dA, :, i)
            Ai = view(A, :, i)
        else
            error("Unsupported array dimensionality")
        end
        
        # Apply nonlinearity for this population
        nl_i = nonlinearity[i]
        if nl_i isa SigmoidNonlinearity
            @. dAi += simple_sigmoid(Ai, nl_i.a, nl_i.θ) - Ai
        elseif nl_i isa RectifiedZeroedSigmoidNonlinearity
            @. dAi += rectified_zeroed_sigmoid(Ai, nl_i.a, nl_i.θ) - Ai
        elseif nl_i isa DifferenceOfSigmoidsNonlinearity
            @. dAi += difference_of_rectified_zeroed_sigmoids(Ai, nl_i.a_activating, nl_i.θ_activating, nl_i.a_failing, nl_i.θ_failing) - Ai
        else
            error("Unsupported nonlinearity type: $(typeof(nl_i))")
        end
    end
end
