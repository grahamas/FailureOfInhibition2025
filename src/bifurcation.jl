"""
Bifurcation analysis utilities for Wilson-Cowan models using BifurcationKit.

This module provides tools for analyzing how system dynamics change as parameters vary,
using continuation methods from BifurcationKit for accurate bifurcation detection.

The module provides an ergonomic interface to BifurcationKit by offering:
- Helper functions to create parameter lenses for common parameters
- Utilities to find good initial conditions for continuation
- Simplified continuation setup with sensible defaults
"""

using BifurcationKit

"""
    wcm_rhs!(dA, A, params::WilsonCowanParameters, t=0.0)

Right-hand side function for Wilson-Cowan model suitable for BifurcationKit.

This wrapper function adapts wcm1973! to work with BifurcationKit's conventions,
allowing the Wilson-Cowan model to be used with continuation methods.

# Arguments
- `dA`: Output array for derivatives (modified in-place)
- `A`: Current activity state
- `params`: WilsonCowanParameters containing model parameters
- `t`: Current time (default: 0.0 for autonomous systems)

# Returns
- `dA`: The derivative array (for BifurcationKit compatibility)

# Example
```julia
using FailureOfInhibition2025

params = create_point_model_wcm1973(:active_transient)
A = reshape([0.1, 0.1], 1, 2)
dA = zeros(size(A))
wcm_rhs!(dA, A, params, 0.0)
```
"""
function wcm_rhs!(dA, A, params, t=0.0)
    wcm1973!(dA, A, params, t)
    return dA
end

"""
    create_bifurcation_problem(params::WilsonCowanParameters, param_lens; u0=nothing)

Create a BifurcationProblem for the Wilson-Cowan model.

This function creates a BifurcationKit-compatible problem object that can be used
with continuation methods to trace bifurcation curves, detect bifurcation points
(Hopf, fold, etc.), and analyze stability.

# Arguments
- `params`: WilsonCowanParameters for the model  
- `param_lens`: Lens specifying which parameter to vary (from BifurcationKit's @lens macro)
- `u0`: Initial guess for steady state (if nothing, uses small random values)

# Returns
- `BifurcationProblem`: Problem object suitable for BifurcationKit continuation analysis

# Example
```julia
using FailureOfInhibition2025
using BifurcationKit

# Create Wilson-Cowan parameters
params = create_point_model_wcm1973(:active_transient)

# Initial guess for steady state
u0 = reshape([0.1, 0.1], 1, 2)

# Create bifurcation problem
# Note: param_lens requires using BifurcationKit's lens system
# For example, to vary a parameter, you would use @lens
prob = create_bifurcation_problem(params, param_lens, u0=u0)

# Now use with BifurcationKit continuation:
# opts = ContinuationPar(dsmax = 0.1, dsmin = 1e-4, ds = -0.01, maxSteps = 100)
# br = continuation(prob, PALC(), opts)
```

# Notes
- For point models (PointLattice), u0 can be either a flat vector or reshaped matrix (1, P)
- For spatial models, u0 can be either a flat vector or matrix (N_points, P)
- The parameter lens must be compatible with the WilsonCowanParameters structure
- Continuation can be numerically sensitive; good initial conditions near a steady state help
- Use BifurcationKit's continuation methods (continuation, bifurcationdiagram) with the returned problem
"""
function create_bifurcation_problem(params::WilsonCowanParameters{T,P}, param_lens; u0=nothing) where {T,P}
    # Determine initial condition if not provided
    local u0_matrix
    if u0 === nothing
        if params.lattice isa PointLattice
            # Point model: (1, P) shape
            u0_matrix = reshape(0.1 .+ 0.05 .* rand(P), 1, P)
        else
            # Spatial model: (N_points, P) shape
            n_points = size(params.lattice)[1]
            u0_matrix = 0.1 .+ 0.05 .* rand(n_points, P)
        end
    else
        # Handle both vector and matrix input
        if u0 isa AbstractVector
            # If u0 is already a vector, reshape it appropriately
            if params.lattice isa PointLattice
                u0_matrix = reshape(u0, 1, P)
            else
                n_points = size(params.lattice)[1]
                u0_matrix = reshape(u0, n_points, P)
            end
        else
            # u0 is already a matrix
            u0_matrix = u0
        end
    end
    
    # Store the shape information for reshaping
    state_shape = size(u0_matrix)
    
    # Create a wrapper function that handles reshaping
    # BifurcationKit works with flat vectors, but wcm_rhs! expects matrices
    function F!(dz_flat, z_flat, p)
        # Reshape flat vector to matrix
        z_matrix = reshape(z_flat, state_shape)
        dz_matrix = reshape(dz_flat, state_shape)
        
        # Call the original function
        wcm_rhs!(dz_matrix, z_matrix, p, 0.0)
        
        # Copy reshaped result back to flat vector
        dz_flat .= vec(dz_matrix)
        return dz_flat
    end
    
    # Convert u0 to flat vector for BifurcationKit
    u0_flat = vec(u0_matrix)
    
    # Create bifurcation problem
    prob = BifurcationProblem(F!, u0_flat, params, param_lens)
    
    return prob
end

"""
    create_connectivity_lens(i::Int, j::Int)

Create a lens for accessing connectivity weight between populations i and j.

This is a convenience function for creating parameter lenses to vary connectivity
in bifurcation analysis. The lens accesses `params.connectivity.matrix[i,j].weight`
for ScalarConnectivity (point models). For spatial models with GaussianConnectivityParameter,
users should create custom lenses using `@optic _.connectivity.matrix[i,j].amplitude`.

# Arguments
- `i`: Target population index (row in connectivity matrix)
- `j`: Source population index (column in connectivity matrix)

# Returns
- A lens (optic) that can be used with `create_bifurcation_problem`

# Example
```julia
using FailureOfInhibition2025
using BifurcationKit

params = create_point_model_wcm1973(:oscillatory)
u0 = reshape([0.1, 0.1], 1, 2)

# Create lens for E→E connectivity (i=1, j=1)
lens = create_connectivity_lens(1, 1)
prob = create_bifurcation_problem(params, lens, u0=u0)

# Run continuation
opts = ContinuationPar(dsmax=0.05, dsmin=1e-4, ds=0.01, max_steps=100)
br = continuation(prob, PALC(), opts)
```
"""
function create_connectivity_lens(i::Int, j::Int)
    return @optic _.connectivity.matrix[i,j].weight
end

"""
    create_nonlinearity_lens(pop_index::Int, param::Symbol)

Create a lens for accessing nonlinearity parameters for a specific population.

This is a convenience function for creating parameter lenses to vary nonlinearity
parameters (`:a` for slope, `:θ` for threshold) in bifurcation analysis.

# Arguments
- `pop_index`: Population index (1 for E, 2 for I in 2-population models)
- `param`: Either `:a` (sigmoid slope) or `:θ` (sigmoid threshold)

# Returns
- A lens (optic) that can be used with `create_bifurcation_problem`

# Example
```julia
using FailureOfInhibition2025
using BifurcationKit

params = create_point_model_wcm1973(:active_transient)
u0 = reshape([0.1, 0.1], 1, 2)

# Create lens for E population threshold
lens = create_nonlinearity_lens(1, :θ)
prob = create_bifurcation_problem(params, lens, u0=u0)

# Run continuation
opts = ContinuationPar(dsmax=0.1, dsmin=1e-4, ds=0.05, max_steps=150)
br = continuation(prob, PALC(), opts)
```
"""
function create_nonlinearity_lens(pop_index::Int, param::Symbol)
    if param == :a
        return @optic _.nonlinearity[pop_index].a
    elseif param == :θ
        return @optic _.nonlinearity[pop_index].θ
    else
        error("param must be :a or :θ, got $param")
    end
end

"""
    create_default_continuation_opts(; p_min=0.1, p_max=5.0, max_steps=200, kwargs...)

Create ContinuationPar with sensible defaults for Wilson-Cowan models.

This provides an ergonomic way to set up continuation parameters without needing
to specify all the details. The defaults are tuned for point models and typical
parameter ranges.

# Arguments
- `p_min`: Minimum parameter value (default: 0.1)
- `p_max`: Maximum parameter value (default: 5.0)
- `max_steps`: Maximum number of continuation steps (default: 200)
- `kwargs...`: Additional keyword arguments passed to ContinuationPar

# Returns
- `ContinuationPar` object ready for use with continuation methods

# Example
```julia
using FailureOfInhibition2025
using BifurcationKit

params = create_point_model_wcm1973(:oscillatory)
lens = create_connectivity_lens(1, 1)  # E→E connectivity
prob = create_bifurcation_problem(params, lens)

# Use default continuation options
opts = create_default_continuation_opts(p_min=0.5, p_max=3.0)
br = continuation(prob, PALC(), opts)
```
"""
function create_default_continuation_opts(; p_min=0.1, p_max=5.0, max_steps=200, kwargs...)
    return ContinuationPar(
        dsmax = 0.05,
        dsmin = 1e-4,
        ds = 0.01,
        max_steps = max_steps,
        p_min = p_min,
        p_max = p_max,
        detect_bifurcation = 3,
        n_inversion = 6;
        kwargs...
    )
end
