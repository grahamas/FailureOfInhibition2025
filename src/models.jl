
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
Initializes dA to zero since there is no stimulus.
"""
function stimulate!(dA, A, ::Nothing, t)
    # Initialize to zero when no stimulus is present
    # This ensures propagate_activation starts with clean input
    dA .= 0
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
  - Must be a ConnectivityMatrix{P} for per-population-pair connectivity
- `nonlinearity`: Nonlinearity parameter (defines activation functions)
- `stimulus`: Stimulus parameter (defines external inputs)
- `lattice`: Spatial lattice for the model (required for connectivity propagation)
- `pop_names::NTuple{P,String}`: Names of populations (e.g., ("E", "I") for excitatory and inhibitory)

# Connectivity Matrix Convention

When using ConnectivityMatrix{P}, the indexing follows matrix multiplication conventions:
`connectivity[i,j]` maps the activity of population j into population i.

For a 2-population (E=1, I=2) model:
- connectivity[1,1]: E → E (excitatory self-connection)
- connectivity[1,2]: I → E (inhibitory to excitatory)
- connectivity[2,1]: E → I (excitatory to inhibitory) 
- connectivity[2,2]: I → I (inhibitory self-connection)

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

# Per-population-pair connectivity (2x2 for 2 populations)
# connectivity[i,j] maps population j → i
conn_ee = GaussianConnectivityParameter(1.0, (2.0,))   # E → E
conn_ei = GaussianConnectivityParameter(-0.5, (1.5,))  # I → E
conn_ie = GaussianConnectivityParameter(0.8, (2.5,))   # E → I
conn_ii = GaussianConnectivityParameter(-0.3, (1.0,))  # I → I
connectivity = ConnectivityMatrix{2}([
    conn_ee conn_ei;
    conn_ie conn_ii
])

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
    lattice = lattice,
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
    lattice
    pop_names::NTuple{P,String}
    
    # Inner constructor that ensures connectivity is always prepared
    # This makes both positional and keyword constructors safe
    function WilsonCowanParameters{T,P}(α::NTuple{P}, β::NTuple{P}, τ::NTuple{P},
                                        connectivity, nonlinearity, stimulus, lattice,
                                        pop_names::NTuple{P,String}) where {T,P}
        # Prepare connectivity to ensure kernels are pre-computed
        prepared_connectivity = prepare_connectivity(connectivity, lattice)
        new{T,P}(α, β, τ, prepared_connectivity, nonlinearity, stimulus, lattice, pop_names)
    end
end

# Constructor with keyword arguments for convenience
function WilsonCowanParameters{P}(; α, β, τ, connectivity, nonlinearity, stimulus, lattice,
                                   pop_names=ntuple(i -> "Pop$i", P)) where {P}
    T = eltype(α)
    # Call the positional constructor which now handles prepare_connectivity
    WilsonCowanParameters{T,P}(α, β, τ, connectivity, nonlinearity, stimulus, lattice, pop_names)
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
    propagate_activation(dA, A, p.connectivity, t, p.lattice)
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

############## Failure of Inhibition (FoI) Model ##############

"""
    FailureOfInhibitionParameters(; α, β, τ, connectivity, nonlinearity_E, nonlinearity_I, stimulus, lattice)

Construct WilsonCowanParameters for a Failure of Inhibition (FoI) model.

An FoI model is a 2-population Wilson-Cowan model where:
- The excitatory population uses a standard sigmoid nonlinearity
- The inhibitory population uses a non-monotonic difference of sigmoids nonlinearity

This creates dynamics where inhibition can fail at higher activity levels, leading to
characteristic FoI behaviors such as traveling waves and sustained activity patterns.

# Arguments
- `α`: Decay rates for [E, I] populations
- `β`: Saturation coefficients for [E, I] populations (typically 1.0)
- `τ`: Time constants for [E, I] populations
- `connectivity`: Connectivity parameter (defines how populations interact)
- `nonlinearity_E`: Nonlinearity for excitatory population (SigmoidNonlinearity or RectifiedZeroedSigmoidNonlinearity)
- `nonlinearity_I`: Nonlinearity for inhibitory population (DifferenceOfSigmoidsNonlinearity)
- `stimulus`: Stimulus parameter (defines external inputs)
- `lattice`: Spatial lattice for the model

# Returns
WilsonCowanParameters{T,2} with per-population nonlinearities as a tuple (nonlinearity_E, nonlinearity_I)

# Example

```julia
using FailureOfInhibition2025

# Create spatial lattice
lattice = CompactLattice(extent=(10.0,), n_points=(21,))

# Define connectivity
conn_ee = GaussianConnectivityParameter(1.0, (2.0,))
conn_ei = GaussianConnectivityParameter(-0.5, (1.5,))
conn_ie = GaussianConnectivityParameter(0.8, (2.5,))
conn_ii = GaussianConnectivityParameter(-0.3, (1.0,))
connectivity = ConnectivityMatrix{2}([
    conn_ee conn_ei;
    conn_ie conn_ii
])

# Create FoI parameters
params = FailureOfInhibitionParameters(
    α = (1.0, 1.5),
    β = (1.0, 1.0),
    τ = (10.0, 8.0),
    connectivity = connectivity,
    nonlinearity_E = RectifiedZeroedSigmoidNonlinearity(a=2.0, θ=0.5),
    nonlinearity_I = DifferenceOfSigmoidsNonlinearity(
        a_activating=5.0, θ_activating=0.3,
        a_failing=3.0, θ_failing=0.7
    ),
    stimulus = nothing,
    lattice = lattice
)

# Use with wcm1973! (or foi! which is an alias)
wcm1973!(dA, A, params, 0.0)
```
"""
function FailureOfInhibitionParameters(; α, β, τ, connectivity, nonlinearity_E, nonlinearity_I, stimulus, lattice)
    # Create tuple of nonlinearities: (E, I)
    nonlinearity = (nonlinearity_E, nonlinearity_I)
    
    # Use WilsonCowanParameters constructor which will prepare connectivity
    return WilsonCowanParameters{2}(
        α = α,
        β = β,
        τ = τ,
        connectivity = connectivity,
        nonlinearity = nonlinearity,
        stimulus = stimulus,
        lattice = lattice,
        pop_names = ("E", "I")
    )
end

"""
    foi!(dA, A, p::WilsonCowanParameters, t)

Failure of Inhibition model differential equation.

This is an alias for wcm1973! that can be used for clarity when working with FoI models.

# Expected Parameter Structure
For proper FoI dynamics, the WilsonCowanParameters should be configured with:
- Two populations (E and I)
- `p.nonlinearity` as a tuple: `(nonlinearity_E, nonlinearity_I)` where:
  - First element (E): Standard nonlinearity (SigmoidNonlinearity or RectifiedZeroedSigmoidNonlinearity)
  - Second element (I): DifferenceOfSigmoidsNonlinearity (non-monotonic, characteristic of FoI)

Use `FailureOfInhibitionParameters()` constructor to ensure correct configuration.

# Note
There is no separate FoI parameter type. FoI models use WilsonCowanParameters directly to
avoid object construction in the ODE solver inner loop. This function is purely an alias 
to wcm1973! with zero overhead.
"""
foi!(dA, A, p::WilsonCowanParameters, t) = wcm1973!(dA, A, p, t)

############## Analytical Jacobian ##############

"""
    wcm1973_jacobian!(J, A, p::WilsonCowanParameters{T,P}, t) where {T,P}

Compute the analytical Jacobian ∂(dA/dt)/∂A for the Wilson-Cowan model.

The Wilson-Cowan equations are:
```
τᵢ dAᵢ/dt = -αᵢ Aᵢ + βᵢ (1 - Aᵢ) fᵢ(Sᵢ(t) + ∑ⱼ Cᵢⱼ Aⱼ)
```

The Jacobian elements are:
```
∂(dAᵢ/dt)/∂Aₖ = 1/τᵢ * [
    -αᵢ δᵢₖ                                    (decay term)
    - βᵢ fᵢ(Input) δᵢₖ                         (saturation term)
    + βᵢ (1 - Aᵢ) f'ᵢ(Input) Cᵢₖ              (connectivity term)
]
```

where:
- δᵢₖ is the Kronecker delta (1 if i=k, 0 otherwise)
- Input = Sᵢ(t) + ∑ⱼ Cᵢⱼ Aⱼ (total input to population i)
- f'ᵢ is the derivative of the nonlinearity function

# Arguments
- `J`: Output Jacobian matrix (modified in-place). For point models: PxP matrix.
       For spatial models: (N*P)x(N*P) block matrix where N is number of spatial points.
- `A`: Current activity state (same format as in wcm1973!)
- `p`: WilsonCowanParameters containing model parameters
- `t`: Current time

# Implementation Notes
- For point models (PointLattice), J is a simple PxP matrix
- For spatial models with ScalarConnectivity, J has a block-diagonal structure
  with PxP blocks repeated for each spatial point
- For spatial models with GaussianConnectivity, J has off-diagonal blocks
  representing spatial coupling through convolution

# Example
```julia
using FailureOfInhibition2025

# Point model example
params = create_point_model_wcm1973(:oscillatory)
A = reshape([0.1, 0.1], 1, 2)  # 1 spatial point, 2 populations
J = zeros(2, 2)
wcm1973_jacobian!(J, A, params, 0.0)

# Spatial model example
lattice = CompactLattice(extent=(10.0,), n_points=(11,))
# ... create spatial params ...
A_spatial = rand(11, 2)  # 11 spatial points, 2 populations
J_spatial = zeros(22, 22)  # Flattened: 11*2 = 22 state variables
A_flat = vec(A_spatial)
wcm1973_jacobian!(J_spatial, A_flat, params_spatial, 0.0)
```
"""
function wcm1973_jacobian!(J, A, p::WilsonCowanParameters{T,P}, t) where {T,P}
    # Determine if this is a point model or spatial model
    is_point_model = p.lattice isa PointLattice
    
    if is_point_model
        # Point model: J is PxP, A is vector or (1, P) matrix
        wcm1973_jacobian_point!(J, A, p, t)
    else
        # Spatial model: more complex
        # For now, implement only the point model case
        error("Analytical Jacobian for spatial models is not yet implemented. " *
              "Use point models (PointLattice) or numerical differentiation for spatial cases.")
    end
end

"""
    wcm1973_jacobian_point!(J, A, p::WilsonCowanParameters{T,P}, t) where {T,P}

Compute analytical Jacobian for point models (zero-dimensional, no spatial structure).

For point models, the state A is (1, P) or a P-vector, and the Jacobian J is a PxP matrix.
Each element J[i,k] = ∂(dAᵢ/dt)/∂Aₖ.
"""
function wcm1973_jacobian_point!(J, A, p::WilsonCowanParameters{T,P}, t) where {T,P}
    # Ensure A is in the right shape - convert to matrix if needed
    if A isa AbstractVector
        if length(A) == P
            A_mat = reshape(A, 1, P)
        else
            error("For point models, A must be a vector of length P=$P or a (1, P) matrix")
        end
    else
        A_mat = A
        if size(A_mat) != (1, P)
            error("For point models, A must have shape (1, P), got $(size(A_mat))")
        end
    end
    
    # Compute inputs to each population: Input_i = S_i + sum_j C_ij * A_j
    inputs = zeros(T, P)
    
    # Add stimulus (if present)
    if p.stimulus !== nothing
        stim_temp = zeros(T, 1, P)
        stimulate!(stim_temp, A_mat, p.stimulus, t)
        for i in 1:P
            inputs[i] = stim_temp[1, i]
        end
    end
    
    # Add connectivity: sum_j C_ij * A_j
    if p.connectivity !== nothing && p.connectivity isa ConnectivityMatrix
        for i in 1:P
            for j in 1:P
                conn_ij = p.connectivity[i, j]
                if conn_ij isa ScalarConnectivity
                    inputs[i] += conn_ij.weight * A_mat[1, j]
                elseif conn_ij !== nothing
                    error("Jacobian only supports ScalarConnectivity for point models, got $(typeof(conn_ij))")
                end
            end
        end
    end
    
    # Compute nonlinearity values f(Input) and derivatives f'(Input) for each population
    f_values = zeros(T, P)
    df_values = zeros(T, P)
    
    if p.nonlinearity isa Tuple
        # Per-population nonlinearities
        for i in 1:P
            nl_i = p.nonlinearity[i]
            # Compute f(input)
            if nl_i isa SigmoidNonlinearity
                f_values[i] = simple_sigmoid(inputs[i], nl_i.a, nl_i.θ)
            elseif nl_i isa RectifiedZeroedSigmoidNonlinearity
                f_values[i] = rectified_zeroed_sigmoid(inputs[i], nl_i.a, nl_i.θ)
            elseif nl_i isa DifferenceOfSigmoidsNonlinearity
                f_values[i] = difference_of_rectified_zeroed_sigmoids(
                    inputs[i], nl_i.a_activating, nl_i.θ_activating,
                    nl_i.a_failing, nl_i.θ_failing
                )
            else
                error("Unsupported nonlinearity type: $(typeof(nl_i))")
            end
            # Compute f'(input)
            df_values[i] = nonlinearity_derivative(inputs[i], nl_i)
        end
    else
        # Same nonlinearity for all populations
        for i in 1:P
            if p.nonlinearity isa SigmoidNonlinearity
                f_values[i] = simple_sigmoid(inputs[i], p.nonlinearity.a, p.nonlinearity.θ)
            elseif p.nonlinearity isa RectifiedZeroedSigmoidNonlinearity
                f_values[i] = rectified_zeroed_sigmoid(inputs[i], p.nonlinearity.a, p.nonlinearity.θ)
            elseif p.nonlinearity isa DifferenceOfSigmoidsNonlinearity
                f_values[i] = difference_of_rectified_zeroed_sigmoids(
                    inputs[i], p.nonlinearity.a_activating, p.nonlinearity.θ_activating,
                    p.nonlinearity.a_failing, p.nonlinearity.θ_failing
                )
            else
                error("Unsupported nonlinearity type: $(typeof(p.nonlinearity))")
            end
            df_values[i] = nonlinearity_derivative(inputs[i], p.nonlinearity)
        end
    end
    
    # Compute Jacobian elements
    # J[i,k] = ∂(dAᵢ/dt)/∂Aₖ
    for i in 1:P
        for k in 1:P
            # Get connectivity C_ik
            C_ik = zero(T)
            if p.connectivity !== nothing && p.connectivity isa ConnectivityMatrix
                conn_ik = p.connectivity[i, k]
                if conn_ik isa ScalarConnectivity
                    C_ik = conn_ik.weight
                end
            end
            
            # Compute Jacobian element
            if i == k
                # Diagonal: includes decay and saturation terms
                J[i, k] = (1/p.τ[i]) * (
                    -p.α[i]                           # decay term
                    - p.β[i] * f_values[i]           # saturation term  
                    + p.β[i] * (1 - A_mat[1, i]) * df_values[i] * C_ik  # connectivity term
                )
            else
                # Off-diagonal: only connectivity term
                J[i, k] = (1/p.τ[i]) * (
                    p.β[i] * (1 - A_mat[1, i]) * df_values[i] * C_ik
                )
            end
        end
    end
    
    return J
end
