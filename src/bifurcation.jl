"""
Bifurcation analysis utilities for Wilson-Cowan models using BifurcationKit.

This module provides tools for analyzing how system dynamics change as parameters vary,
using continuation methods from BifurcationKit for accurate bifurcation detection.
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
- For point models (PointLattice), u0 should have shape (1, P) where P is number of populations
- For spatial models, u0 should have shape (N_points, P)
- The parameter lens must be compatible with the WilsonCowanParameters structure
- Use BifurcationKit's continuation methods (continuation, bifurcationdiagram) with the returned problem
"""
function create_bifurcation_problem(params::WilsonCowanParameters{T,P}, param_lens; u0=nothing) where {T,P}
    # Determine initial condition if not provided
    if u0 === nothing
        if params.lattice isa PointLattice
            # Point model: (1, P) shape
            u0 = reshape(0.1 .+ 0.05 .* rand(P), 1, P)
        else
            # Spatial model: (N_points, P) shape
            n_points = size(params.lattice)[1]
            u0 = 0.1 .+ 0.05 .* rand(n_points, P)
        end
    end
    
    # Create the right-hand side function for BifurcationKit
    # BifurcationKit expects F!(dz, z, p) signature
    F! = (dz, z, p) -> wcm_rhs!(dz, z, p, 0.0)
    
    # Create bifurcation problem
    prob = BifurcationProblem(F!, u0, params, param_lens)
    
    return prob
end
