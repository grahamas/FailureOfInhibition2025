# Wilson-Cowan Model Implementation

This document details the Wilson-Cowan model implementation in FailureOfInhibition2025 and how it differs from the reference [WilsonCowanModel.jl](https://github.com/grahamas/WilsonCowanModel) repository.

## Overview

The Wilson-Cowan model (Wilson & Cowan 1972, 1973) describes the dynamics of neural populations using coupled differential equations:

```
τᵢ dAᵢ/dt = -αᵢ Aᵢ + βᵢ (1 - Aᵢ) f(Sᵢ + Iᵢ)
```

where:
- `Aᵢ` is the activity of population i
- `αᵢ` is the decay rate
- `βᵢ` is the saturation coefficient
- `τᵢ` is the time constant
- `f` is the nonlinearity (firing rate function)
- `Sᵢ` is external stimulus
- `Iᵢ` is recurrent input from connectivity

## Mathematical Equivalence

Both implementations solve the same Wilson-Cowan equations and produce identical results. The differences are purely architectural.

## Key Implementation Differences

### 1. No Callable Objects (Functors)

**Reference Implementation (WilsonCowanModel.jl)**:
```julia
# Model struct is callable
struct WCMSpatial{...} <: AbstractWilsonCowanModel
    α, β, τ, connectivity, nonlinearity, stimulus, pop_names
end

# Calling the model on a space creates an Action object
function (wcm::WCMSpatial)(space::AbstractSpace)
    # Returns WCMSpatialAction
    ...
end

# The Action object is also callable
function (wcm::WCMSpatialAction)(dA, A, p, t)
    # Computes derivatives
    ...
end
```

**This Implementation**:
```julia
# Plain parameter struct (not callable)
struct WilsonCowanParameters{T,P}
    α, β, τ, connectivity, nonlinearity, stimulus, pop_names
end

# Separate function for computing derivatives
function wcm1973!(dA, A, params::WilsonCowanParameters, t)
    # Computes derivatives
    ...
end
```

### 2. Direct Function Dispatch vs. Action Objects

**Reference Implementation**:
- Two-stage process: `Parameter → Action → Computation`
- First, call `model(space)` to create an "Action" object
- Then, call the Action object to compute derivatives
- Separates parameter definition from space-specific computation

**This Implementation**:
- Single-stage process: `Parameter → Computation`
- Pass parameters directly to `wcm1973!()` function
- No intermediate Action layer
- Simpler, more direct approach

### 3. Parameter Structure

**Reference Implementation**:
```julia
# Separate Parameter and Action types
struct WCMSpatial{...}  # Parameter type
    ...
end

struct WCMSpatialAction{...}  # Action type
    ...
end
```

**This Implementation**:
```julia
# Single parameter type
struct WilsonCowanParameters{T,P}
    ...
end
```

### 4. Functional vs. Object-Oriented Style

**Reference Implementation**:
- Object-oriented: Methods defined on structs using functor pattern
- Model objects encapsulate behavior
- Follows "objects with behavior" paradigm

**This Implementation**:
- Functional: Functions operate on data structures
- Clear separation between data (WilsonCowanParameters) and behavior (wcm1973!)
- Follows "data and functions" paradigm

## Code Comparison

### Creating a Model

**Reference Implementation**:
```julia
using WilsonCowanModel

model = WCMSpatial{N_CDT,2}(
    α = (1.0, 1.5),
    β = (1.0, 1.0),
    τ = (1.0, 0.8),
    connectivity = ...,
    nonlinearity = ...,
    stimulus = ...,
    pop_names = ("E", "I")
)

# Create Action object
action = model(space)

# Use in ODE solver
prob = ODEProblem(action, A₀, tspan, nothing)
```

**This Implementation**:
```julia
using FailureOfInhibition2025

params = WilsonCowanParameters{2}(
    α = (1.0, 1.5),
    β = (1.0, 1.0),
    τ = (1.0, 0.8),
    connectivity = nothing,
    nonlinearity = SigmoidNonlinearity(a=2.0, θ=0.5),
    stimulus = nothing,
    pop_names = ("E", "I")
)

# Use directly in ODE solver
prob = ODEProblem(wcm1973!, A₀, tspan, params)
```

### Computing Derivatives

**Reference Implementation**:
```julia
# Action object is callable
action(dA, A, p, t)
```

**This Implementation**:
```julia
# Call function with parameters
wcm1973!(dA, A, params, t)
```

## Advantages of This Implementation

1. **Simplicity**: Fewer types and concepts to understand
2. **Composability**: Functions can be easily combined and reused
3. **Clarity**: Clear separation between data and operations
4. **Testing**: Easier to test individual functions
5. **Debugging**: Stack traces are more straightforward
6. **Learning curve**: More familiar to functional programmers

## Advantages of Reference Implementation

1. **Encapsulation**: Behavior bundled with data
2. **Type safety**: Action objects ensure space compatibility
3. **Performance**: Can cache space-specific computations
4. **Flexibility**: Easier to create model variants
5. **Abstraction**: Hides implementation details

## When to Use Each

**Use this implementation** when:
- You prefer functional programming style
- You want simpler, more direct code
- You're prototyping or learning
- You need easy testing and debugging

**Use reference implementation** when:
- You need advanced performance optimizations
- You're building complex model hierarchies
- You want strong type guarantees
- You prefer object-oriented patterns

## Migration Guide

If migrating from WilsonCowanModel.jl to this implementation:

1. Replace `WCMSpatial{N_CDT,P}(...)` with `WilsonCowanParameters{P}(...)`
2. Remove the `model(space)` step
3. Replace `action(dA, A, p, t)` with `wcm1973!(dA, A, params, t)`
4. Pass `params` directly to ODE solvers

Example:
```julia
# Before (WilsonCowanModel.jl)
model = WCMSpatial{1,2}(...)
action = model(space)
prob = ODEProblem(action, A₀, tspan, nothing)

# After (FailureOfInhibition2025)
params = WilsonCowanParameters{2}(...)
prob = ODEProblem(wcm1973!, A₀, tspan, params)
```

## References

- Wilson, H. R., & Cowan, J. D. (1972). Excitatory and inhibitory interactions in localized populations of model neurons. *Biophysical journal*, 12(1), 1-24.
- Wilson, H. R., & Cowan, J. D. (1973). A mathematical theory of the functional dynamics of cortical and thalamic nervous tissue. *Kybernetik*, 13(2), 55-80.
- WilsonCowanModel.jl: https://github.com/grahamas/WilsonCowanModel
