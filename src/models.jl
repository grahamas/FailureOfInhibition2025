
"""
    population(array, index)

Extract population `index` from a multi-population array.
For simplicity, assumes populations are stored along the last dimension.
"""
function population(array, index)
    # For a 2D array, assume populations are along the second dimension
    # This is a simplified implementation - may need adjustment based on actual data structure
    if ndims(array) == 1
        return array  # Single population case
    elseif ndims(array) == 2
        return view(array, :, index)
    else
        # For higher dimensions, assume last dimension is populations
        indices = ntuple(i -> i == ndims(array) ? index : Colon(), ndims(array))
        return view(array, indices...)
    end
end

"""
    stimulate!(dA, A, ::Nothing, t)

No-op stimulation when stimulus is nothing.
"""
function stimulate!(dA, A, ::Nothing, t)
    # Do nothing when no stimulus is present
    return nothing
end

############## Wilson-Cowan Model ##############

"""
    WilsonCowanParameters{T,P}

Parameters for the Wilson-Cowan model (Wilson & Cowan 1972, 1973).

# Fields
- `α::NTuple{P,T}`: Decay rates for each population
- `β::NTuple{P,T}`: Saturation coefficients for each population (typically 1.0)
- `τ::NTuple{P,T}`: Time constants for each population
- `connectivity`: Connectivity parameter (defines how populations interact)
- `nonlinearity`: Nonlinearity parameter (defines activation functions)
- `stimulus`: Stimulus parameter (defines external inputs)
- `pop_names::NTuple{P,String}`: Names of populations (e.g., ("E", "I") for excitatory and inhibitory)

# Implementation Differences from WilsonCowanModel.jl

This implementation differs from the reference WilsonCowanModel.jl repository in the following ways:

1. **No callable objects**: The original uses callable structs (functors) where the model parameter 
   struct itself can be called as a function. This implementation uses plain structs with separate
   functions, following a more functional programming style.

2. **Direct function dispatch**: Instead of `model(space)` creating an "Action" object, we pass
   parameters directly to the differential equation function `wcm1973!(dA, A, p, t)`.

3. **Simplified parameter structure**: The original separates "Parameter" and "Action" types.
   This implementation combines them into a single parameter struct that's used directly.

4. **No intermediate Action layer**: The original creates `WCMSpatialAction` from `WCMSpatial`.
   This implementation passes `WilsonCowanParameters` directly to the ODE function.

# Example

```julia
using FailureOfInhibition2025

# Create spatial lattice
lattice = CompactLattice(extent=(10.0,), n_points=(21,))

# Create connectivity, stimulus, and nonlinearity objects
connectivity = GaussianConnectivityParameter(1.0, (2.0,))
stimulus = CircleStimulus(
    radius=2.0, strength=0.5,
    time_windows=[(0.0, 10.0)],
    lattice=lattice
)
nonlinearity = SigmoidNonlinearity(a=2.0, θ=0.5)

# Create parameters for a 2-population Wilson-Cowan model
params = WilsonCowanParameters{2}(
    α = (1.0, 1.0),           # Decay rates
    β = (1.0, 1.0),           # Saturation coefficients
    τ = (1.0, 1.0),           # Time constants
    connectivity = connectivity,
    nonlinearity = nonlinearity,
    stimulus = stimulus,
    pop_names = ("E", "I")    # Excitatory and Inhibitory populations
)

# Use with ODE solver
# dA/dt = wcm1973!(dA, A, params, t)
```
"""
struct WilsonCowanParameters{T,P}
    α::NTuple{P,T}
    β::NTuple{P,T}
    τ::NTuple{P,T}
    connectivity
    nonlinearity
    stimulus
    pop_names::NTuple{P,String}
end

# Constructor with keyword arguments for convenience
function WilsonCowanParameters{P}(; α, β, τ, connectivity, nonlinearity, stimulus, 
                                   pop_names=ntuple(i -> "Pop$i", P)) where {P}
    T = eltype(α)
    WilsonCowanParameters{T,P}(α, β, τ, connectivity, nonlinearity, stimulus, pop_names)
end

"""
    wcm1973!(dA, A, p::WilsonCowanParameters, t)

Wilson-Cowan model differential equation (Wilson & Cowan 1973).

Implements the classic Wilson-Cowan equations for neural population dynamics:
```
τᵢ dAᵢ/dt = -αᵢ Aᵢ + βᵢ (1 - Aᵢ) f(Sᵢ(t) + Cᵢ(A))
```

where:
- `Aᵢ` is the activity of population i
- `αᵢ` is the decay rate
- `βᵢ` is the saturation coefficient
- `τᵢ` is the time constant
- `f` is the nonlinearity (firing rate function)
- `Sᵢ(t)` is external stimulus (function of time)
- `Cᵢ(A)` is recurrent input from connectivity (function of activity)

# Arguments
- `dA`: Output array for derivatives (modified in-place)
- `A`: Current activity state
- `p`: WilsonCowanParameters containing model parameters
- `t`: Current time

# Implementation Note
This implementation follows the functional programming style used in this repository,
avoiding callable objects. The equation structure matches the original Wilson-Cowan
model but uses direct function calls instead of functor dispatch.
"""
function wcm1973!(dA, A, p::WilsonCowanParameters{T,P}, t) where {T,P}
    # Apply stimulus, connectivity, and nonlinearity
    stimulate!(dA, A, p.stimulus, t)
    propagate_activation(dA, A, p.connectivity, t)
    apply_nonlinearity!(dA, A, p.nonlinearity, t)
    
    # Apply Wilson-Cowan dynamics for each population
    for i in 1:P
        dAi = population(dA, i)
        Ai = population(A, i)
        
        # Wilson-Cowan equations:
        # dA/dt = (-α*A + β*(1-A)*f(S+I)) / τ
        # where f(S+I) is captured in dA from nonlinearity
        dAi .*= p.β[i] .* (1 .- Ai)
        dAi .+= -p.α[i] .* Ai
        dAi ./= p.τ[i]
    end
end
