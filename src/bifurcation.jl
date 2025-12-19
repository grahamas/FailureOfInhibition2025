"""
Bifurcation analysis utilities for Wilson-Cowan models using BifurcationKit.

This module provides tools for analyzing how system dynamics change as parameters vary,
using continuation methods from BifurcationKit for accurate bifurcation detection.

The module provides an ergonomic interface to BifurcationKit by offering:
- Helper functions to create parameter lenses for common parameters
- Utilities to find good initial conditions for continuation
- Simplified continuation setup with sensible defaults

Additionally, this module provides tools for finding and analyzing fixed points:
- Finding fixed points (equilibria) of Wilson-Cowan models
- Computing stability of fixed points via eigenvalue analysis
- Optimizing parameters to achieve a target number of stable fixed points
"""

using BifurcationKit
using LinearAlgebra
using Optim

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
    u0_matrix = if u0 === nothing
        if params.lattice isa PointLattice
            # Point model: (1, P) shape
            reshape(0.1 .+ 0.05 .* rand(P), 1, P)
        else
            # Spatial model: (N_points, P) shape
            n_points = size(params.lattice)[1]
            0.1 .+ 0.05 .* rand(n_points, P)
        end
    else
        # Handle both vector and matrix input
        if u0 isa AbstractVector
            # If u0 is already a vector, reshape it appropriately
            if params.lattice isa PointLattice
                reshape(u0, 1, P)
            else
                n_points = size(params.lattice)[1]
                reshape(u0, n_points, P)
            end
        else
            # u0 is already a matrix
            u0
        end
    end
    
    # Store the shape information for reshaping
    state_shape = size(u0_matrix)
    
    # Create a wrapper function that handles reshaping
    # BifurcationKit works with flat vectors, but wcm_rhs! expects matrices
    # Use views to avoid allocations during continuation
    function F!(dz_flat, z_flat, p)
        # Create reshaped views (no allocation)
        z_matrix = reshape(view(z_flat, :), state_shape)
        dz_matrix = reshape(view(dz_flat, :), state_shape)
        
        # Call the original function
        wcm_rhs!(dz_matrix, z_matrix, p, 0.0)
        
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

"""
    find_fixed_points(params::WilsonCowanParameters{T,P}; 
                      n_trials=10, 
                      u0_range=(0.0, 0.5),
                      tol=1e-6) where {T,P}

Find fixed points (equilibria) of the Wilson-Cowan model.

A fixed point is a state where dA/dt = 0, meaning the system is at rest.
This function searches for fixed points by starting from multiple random 
initial conditions and using optimization to find zeros of the dynamics.

# Arguments
- `params`: WilsonCowanParameters containing model parameters
- `n_trials`: Number of random initial guesses to try (default: 10)
- `u0_range`: Tuple of (min, max) values for random initial conditions (default: (0.0, 0.5))
- `tol`: Tolerance for fixed point convergence (default: 1e-6)

# Returns
- `fixed_points::Vector{Vector{Float64}}`: List of unique fixed points found
- `converged::Vector{Bool}`: Whether each search converged

# Example
```julia
using FailureOfInhibition2025

params = create_point_model_wcm1973(:steady_state)
fps, converged = find_fixed_points(params, n_trials=20)

println("Found \$(length(fps)) fixed points")
for (i, fp) in enumerate(fps)
    println("  FP \$i: E=\$(fp[1]), I=\$(fp[2])")
end
```

# Notes
- For point models (PointLattice), fixed points are scalar values per population
- For spatial models, this finds spatially uniform fixed points
- Multiple trials help find all fixed points in systems with multiple equilibria
- Uses optimization (Optim.jl) to minimize ||dA/dt||²
"""
function find_fixed_points(params::WilsonCowanParameters{T,P}; 
                          n_trials=10, 
                          u0_range=(0.0, 0.5),
                          tol=1e-6) where {T,P}
    
    fixed_points = Vector{Float64}[]
    converged_list = Bool[]
    
    # Determine state shape
    if params.lattice isa PointLattice
        state_size = (1, P)
    else
        n_points = size(params.lattice)[1]
        state_size = (n_points, P)
    end
    
    # Define objective: minimize ||dA/dt||²
    function objective(u_flat)
        u_matrix = reshape(u_flat, state_size)
        du_matrix = zeros(size(u_matrix))
        
        # Compute dynamics
        wcm_rhs!(du_matrix, u_matrix, params, 0.0)
        
        # Return squared norm (we want dA/dt ≈ 0)
        return sum(abs2, du_matrix)
    end
    
    # Try multiple random initial conditions
    for trial in 1:n_trials
        # Random initial guess
        u0 = u0_range[1] .+ (u0_range[2] - u0_range[1]) .* rand(prod(state_size))
        
        # Optimize using Nelder-Mead (doesn't require gradients)
        result = Optim.optimize(objective, u0, Optim.NelderMead(), 
                               Optim.Options(g_tol=tol, iterations=1000))
        
        # Check if converged to a fixed point
        if Optim.minimum(result) < tol
            fp = Optim.minimizer(result)
            
            # Check if this is a new fixed point (not already found)
            is_new = true
            for existing_fp in fixed_points
                if norm(fp - existing_fp) < 0.01  # Consider points within 0.01 as same
                    is_new = false
                    break
                end
            end
            
            if is_new
                push!(fixed_points, fp)
                push!(converged_list, true)
            end
        end
    end
    
    return fixed_points, converged_list
end

"""
    compute_stability(fixed_point, params::WilsonCowanParameters{T,P}; 
                     perturbation=1e-6) where {T,P}

Compute the stability of a fixed point by analyzing eigenvalues of the Jacobian.

A fixed point is:
- Stable if all eigenvalues have negative real parts
- Unstable if any eigenvalue has positive real part
- Marginally stable if eigenvalues have zero real parts

# Arguments
- `fixed_point`: Fixed point as a flat vector
- `params`: WilsonCowanParameters containing model parameters
- `perturbation`: Size of perturbation for numerical Jacobian (default: 1e-6)

# Returns
- `eigenvalues::Vector{ComplexF64}`: Eigenvalues of the Jacobian at the fixed point
- `is_stable::Bool`: True if all eigenvalues have negative real parts

# Example
```julia
using FailureOfInhibition2025

params = create_point_model_wcm1973(:steady_state)
fps, _ = find_fixed_points(params)

for (i, fp) in enumerate(fps)
    eigvals, stable = compute_stability(fp, params)
    println("FP \$i: \$(stable ? "Stable" : "Unstable")")
    println("  Eigenvalues: ", eigvals)
end
```

# Notes
- Uses finite differences to compute the Jacobian matrix
- For point models, Jacobian is P×P (e.g., 2×2 for E-I system)
- For spatial models, Jacobian can be large (N_points×P × N_points×P)
- Computation time scales with system size
"""
function compute_stability(fixed_point, params::WilsonCowanParameters{T,P}; 
                          perturbation=1e-6) where {T,P}
    
    # Determine state shape
    if params.lattice isa PointLattice
        state_size = (1, P)
    else
        n_points = size(params.lattice)[1]
        state_size = (n_points, P)
    end
    
    n = length(fixed_point)
    jacobian = zeros(n, n)
    
    # Compute Jacobian using finite differences
    u_matrix = reshape(fixed_point, state_size)
    du_base = zeros(state_size)
    wcm_rhs!(du_base, u_matrix, params, 0.0)
    
    for j in 1:n
        # Perturb j-th component
        u_pert = copy(fixed_point)
        u_pert[j] += perturbation
        
        # Compute perturbed dynamics
        u_pert_matrix = reshape(u_pert, state_size)
        du_pert = zeros(state_size)
        wcm_rhs!(du_pert, u_pert_matrix, params, 0.0)
        
        # Finite difference approximation of derivative
        jacobian[:, j] = vec(du_pert - du_base) / perturbation
    end
    
    # Compute eigenvalues
    eigenvalues = eigvals(jacobian)
    
    # Check stability: all real parts must be negative
    is_stable = all(real(λ) < 0 for λ in eigenvalues)
    
    return eigenvalues, is_stable
end

"""
    count_stable_fixed_points(params::WilsonCowanParameters{T,P}; 
                             n_trials=20, 
                             kwargs...) where {T,P}

Count the number of stable fixed points for given parameters.

This is a convenience function that combines fixed point finding and 
stability analysis to count how many stable equilibria exist.

# Arguments
- `params`: WilsonCowanParameters containing model parameters
- `n_trials`: Number of random initial guesses for fixed point search (default: 20)
- `kwargs...`: Additional arguments passed to `find_fixed_points`

# Returns
- `n_stable::Int`: Number of stable fixed points found
- `fixed_points::Vector{Vector{Float64}}`: All fixed points found
- `stabilities::Vector{Bool}`: Stability of each fixed point

# Example
```julia
using FailureOfInhibition2025

params = create_point_model_wcm1973(:steady_state)
n_stable, fps, stabilities = count_stable_fixed_points(params)

println("Found \$n_stable stable fixed points out of \$(length(fps)) total")
```
"""
function count_stable_fixed_points(params::WilsonCowanParameters{T,P}; 
                                  n_trials=20, 
                                  kwargs...) where {T,P}
    
    # Find all fixed points
    fixed_points, _ = find_fixed_points(params; n_trials=n_trials, kwargs...)
    
    # Compute stability for each
    stabilities = Bool[]
    for fp in fixed_points
        _, is_stable = compute_stability(fp, params)
        push!(stabilities, is_stable)
    end
    
    n_stable = sum(stabilities)
    
    return n_stable, fixed_points, stabilities
end

"""
    optimize_for_stable_fixed_points(base_params::WilsonCowanParameters{T,P},
                                    param_ranges::NamedTuple,
                                    target_n_stable::Int;
                                    n_trials_per_eval=20,
                                    maxiter=100,
                                    population_size=20,
                                    tolerance=0) where {T,P}

Find parameter values that produce a target number of stable fixed points.

This function uses optimization to search parameter space for configurations
that yield exactly (or approximately) the desired number of stable equilibria.
This is useful for studying bifurcations and understanding how parameter changes
affect system dynamics.

# Arguments
- `base_params`: Starting WilsonCowanParameters (serves as template)
- `param_ranges`: NamedTuple specifying parameter ranges to search
  - Supported keys: `:connectivity_ee`, `:connectivity_ei`, `:connectivity_ie`, `:connectivity_ii`
  - `:sigmoid_a_e`, `:sigmoid_theta_e`, `:sigmoid_a_i`, `:sigmoid_theta_i`
  - `:tau_e`, `:tau_i`, `:alpha_e`, `:alpha_i`
  - Each value is a tuple (min, max)
- `target_n_stable`: Desired number of stable fixed points
- `n_trials_per_eval`: Number of random starts for fixed point search per evaluation (default: 20)
- `maxiter`: Maximum optimization iterations (default: 100)
- `population_size`: Size of particle swarm for global optimization (default: 20)
- `tolerance`: Allow ±tolerance deviation from target (default: 0, meaning exact match)

# Returns
- `result`: Optim.jl optimization result
- `best_params`: WilsonCowanParameters with optimized parameter values

# Example
```julia
using FailureOfInhibition2025

# Start with steady state parameters
base_params = create_point_model_wcm1973(:steady_state)

# Search for parameters giving exactly 2 stable fixed points
param_ranges = (
    connectivity_ee = (1.0, 3.0),
    connectivity_ei = (0.5, 2.0),
    sigmoid_theta_e = (5.0, 15.0)
)

result, best_params = optimize_for_stable_fixed_points(
    base_params,
    param_ranges,
    2,  # Target: 2 stable fixed points
    maxiter=50
)

# Verify the result
n_stable, fps, stab = count_stable_fixed_points(best_params)
println("Optimized parameters have \$n_stable stable fixed points")
```

# Notes
- This is a global optimization problem and may not always find the exact target
- The objective minimizes |n_stable - target|, so it finds the closest match
- Uses particle swarm optimization for global search
- Computation time depends on system size and number of evaluations
- For complex systems, consider increasing `maxiter` and `population_size`
"""
function optimize_for_stable_fixed_points(base_params::WilsonCowanParameters{T,P},
                                         param_ranges::NamedTuple,
                                         target_n_stable::Int;
                                         n_trials_per_eval=20,
                                         maxiter=100,
                                         population_size=20,
                                         tolerance=0) where {T,P}
    
    # Extract parameter names and bounds
    param_names = collect(keys(param_ranges))
    lower_bounds = [param_ranges[k][1] for k in param_names]
    upper_bounds = [param_ranges[k][2] for k in param_names]
    
    # Helper function to update parameters
    function update_parameters(params, param_values)
        new_params = params
        
        for (i, name) in enumerate(param_names)
            val = param_values[i]
            
            # Update connectivity parameters
            if name == :connectivity_ee
                conn = new_params.connectivity.matrix[1,1]
                if conn isa ScalarConnectivity
                    new_conn = ScalarConnectivity(val)
                else
                    new_conn = GaussianConnectivityParameter(val, conn.spread)
                end
                new_matrix = copy(new_params.connectivity.matrix)
                new_matrix[1,1] = new_conn
                new_connectivity = ConnectivityMatrix{P}(new_matrix)
                new_params = WilsonCowanParameters{P}(
                    α=new_params.α, β=new_params.β, τ=new_params.τ,
                    connectivity=new_connectivity, nonlinearity=new_params.nonlinearity,
                    stimulus=new_params.stimulus, lattice=new_params.lattice, pop_names=new_params.pop_names
                )
            elseif name == :connectivity_ei
                conn = new_params.connectivity.matrix[1,2]
                if conn isa ScalarConnectivity
                    new_conn = ScalarConnectivity(-abs(val))  # Keep negative for inhibition
                else
                    new_conn = GaussianConnectivityParameter(-abs(val), conn.spread)
                end
                new_matrix = copy(new_params.connectivity.matrix)
                new_matrix[1,2] = new_conn
                new_connectivity = ConnectivityMatrix{P}(new_matrix)
                new_params = WilsonCowanParameters{P}(
                    α=new_params.α, β=new_params.β, τ=new_params.τ,
                    connectivity=new_connectivity, nonlinearity=new_params.nonlinearity,
                    stimulus=new_params.stimulus, lattice=new_params.lattice, pop_names=new_params.pop_names
                )
            elseif name == :connectivity_ie
                conn = new_params.connectivity.matrix[2,1]
                if conn isa ScalarConnectivity
                    new_conn = ScalarConnectivity(val)
                else
                    new_conn = GaussianConnectivityParameter(val, conn.spread)
                end
                new_matrix = copy(new_params.connectivity.matrix)
                new_matrix[2,1] = new_conn
                new_connectivity = ConnectivityMatrix{P}(new_matrix)
                new_params = WilsonCowanParameters{P}(
                    α=new_params.α, β=new_params.β, τ=new_params.τ,
                    connectivity=new_connectivity, nonlinearity=new_params.nonlinearity,
                    stimulus=new_params.stimulus, lattice=new_params.lattice, pop_names=new_params.pop_names
                )
            elseif name == :connectivity_ii
                conn = new_params.connectivity.matrix[2,2]
                if conn isa ScalarConnectivity
                    new_conn = ScalarConnectivity(-abs(val))  # Keep negative for inhibition
                else
                    new_conn = GaussianConnectivityParameter(-abs(val), conn.spread)
                end
                new_matrix = copy(new_params.connectivity.matrix)
                new_matrix[2,2] = new_conn
                new_connectivity = ConnectivityMatrix{P}(new_matrix)
                new_params = WilsonCowanParameters{P}(
                    α=new_params.α, β=new_params.β, τ=new_params.τ,
                    connectivity=new_connectivity, nonlinearity=new_params.nonlinearity,
                    stimulus=new_params.stimulus, lattice=new_params.lattice, pop_names=new_params.pop_names
                )
                
            # Update nonlinearity parameters
            elseif name == :sigmoid_a_e
                nl_e = new_params.nonlinearity[1]
                new_nl_e = typeof(nl_e)(a=val, θ=nl_e.θ)
                new_nonlinearity = (new_nl_e, new_params.nonlinearity[2])
                new_params = WilsonCowanParameters{P}(
                    α=new_params.α, β=new_params.β, τ=new_params.τ,
                    connectivity=new_params.connectivity, nonlinearity=new_nonlinearity,
                    stimulus=new_params.stimulus, lattice=new_params.lattice, pop_names=new_params.pop_names
                )
            elseif name == :sigmoid_theta_e
                nl_e = new_params.nonlinearity[1]
                new_nl_e = typeof(nl_e)(a=nl_e.a, θ=val)
                new_nonlinearity = (new_nl_e, new_params.nonlinearity[2])
                new_params = WilsonCowanParameters{P}(
                    α=new_params.α, β=new_params.β, τ=new_params.τ,
                    connectivity=new_params.connectivity, nonlinearity=new_nonlinearity,
                    stimulus=new_params.stimulus, lattice=new_params.lattice, pop_names=new_params.pop_names
                )
            elseif name == :sigmoid_a_i
                nl_i = new_params.nonlinearity[2]
                new_nl_i = typeof(nl_i)(a=val, θ=nl_i.θ)
                new_nonlinearity = (new_params.nonlinearity[1], new_nl_i)
                new_params = WilsonCowanParameters{P}(
                    α=new_params.α, β=new_params.β, τ=new_params.τ,
                    connectivity=new_params.connectivity, nonlinearity=new_nonlinearity,
                    stimulus=new_params.stimulus, lattice=new_params.lattice, pop_names=new_params.pop_names
                )
            elseif name == :sigmoid_theta_i
                nl_i = new_params.nonlinearity[2]
                new_nl_i = typeof(nl_i)(a=nl_i.a, θ=val)
                new_nonlinearity = (new_params.nonlinearity[1], new_nl_i)
                new_params = WilsonCowanParameters{P}(
                    α=new_params.α, β=new_params.β, τ=new_params.τ,
                    connectivity=new_params.connectivity, nonlinearity=new_nonlinearity,
                    stimulus=new_params.stimulus, lattice=new_params.lattice, pop_names=new_params.pop_names
                )
                
            # Update time constants and decay rates
            elseif name == :tau_e
                new_tau = (val, new_params.τ[2])
                new_params = WilsonCowanParameters{P}(
                    α=new_params.α, β=new_params.β, τ=new_tau,
                    connectivity=new_params.connectivity, nonlinearity=new_params.nonlinearity,
                    stimulus=new_params.stimulus, lattice=new_params.lattice, pop_names=new_params.pop_names
                )
            elseif name == :tau_i
                new_tau = (new_params.τ[1], val)
                new_params = WilsonCowanParameters{P}(
                    α=new_params.α, β=new_params.β, τ=new_tau,
                    connectivity=new_params.connectivity, nonlinearity=new_params.nonlinearity,
                    stimulus=new_params.stimulus, lattice=new_params.lattice, pop_names=new_params.pop_names
                )
            elseif name == :alpha_e
                new_alpha = (val, new_params.α[2])
                new_params = WilsonCowanParameters{P}(
                    α=new_alpha, β=new_params.β, τ=new_params.τ,
                    connectivity=new_params.connectivity, nonlinearity=new_params.nonlinearity,
                    stimulus=new_params.stimulus, lattice=new_params.lattice, pop_names=new_params.pop_names
                )
            elseif name == :alpha_i
                new_alpha = (new_params.α[1], val)
                new_params = WilsonCowanParameters{P}(
                    α=new_alpha, β=new_params.β, τ=new_params.τ,
                    connectivity=new_params.connectivity, nonlinearity=new_params.nonlinearity,
                    stimulus=new_params.stimulus, lattice=new_params.lattice, pop_names=new_params.pop_names
                )
            end
        end
        
        return new_params
    end
    
    # Objective function: minimize deviation from target number of stable FPs
    function objective(param_values)
        try
            # Update parameters
            params = update_parameters(base_params, param_values)
            
            # Count stable fixed points
            n_stable, _, _ = count_stable_fixed_points(params; n_trials=n_trials_per_eval)
            
            # Return deviation from target
            # Allow some tolerance
            deviation = abs(n_stable - target_n_stable)
            if deviation <= tolerance
                return 0.0
            else
                return Float64(deviation)
            end
        catch e
            # If there's an error, return a large penalty
            return 1e6
        end
    end
    
    # Use particle swarm optimization for global search
    # Start from middle of parameter ranges
    x0 = [(lower_bounds[i] + upper_bounds[i]) / 2 for i in 1:length(param_names)]
    
    # Optimize using Optim.ParticleSwarm with bounds
    result = Optim.optimize(
        objective,
        lower_bounds,
        upper_bounds,
        x0,
        Optim.ParticleSwarm(; n_particles=population_size),
        Optim.Options(iterations=maxiter, show_trace=false)
    )
    
    # Get best parameters
    best_param_values = Optim.minimizer(result)
    best_params = update_parameters(base_params, best_param_values)
    
    return result, best_params
end
