"""
Global sensitivity analysis for Wilson-Cowan models.

This module provides functions for performing global sensitivity analysis (GSA)
on Wilson-Cowan model parameters using various methods:

1. **Sobol sensitivity analysis**: Variance-based method that decomposes variance
   into contributions from individual parameters and their interactions.
2. **Morris screening**: Efficient screening method for identifying most important
   parameters with fewer model evaluations.

# Example

```julia
using FailureOfInhibition2025

# Create a point model configuration
lattice = PointLattice()
connectivity = ConnectivityMatrix{2}([
    ScalarConnectivity(0.5) ScalarConnectivity(-0.3);
    ScalarConnectivity(0.4) ScalarConnectivity(-0.2)
])

# Define base parameters
base_params = WilsonCowanParameters{2}(
    α = (1.0, 1.5),
    β = (1.0, 1.0),
    τ = (10.0, 8.0),
    connectivity = connectivity,
    nonlinearity = SigmoidNonlinearity(a=1.5, θ=0.3),
    stimulus = nothing,
    lattice = lattice,
    pop_names = ("E", "I")
)

# Define parameter ranges for sensitivity analysis
# Each parameter is given as (name, lower_bound, upper_bound)
param_ranges = [
    ("α_E", 0.5, 2.0),
    ("α_I", 0.5, 2.0),
    ("τ_E", 5.0, 15.0),
    ("τ_I", 4.0, 12.0),
    ("conn_EE", 0.2, 0.8),
    ("conn_EI", -0.6, -0.1),
    ("conn_IE", 0.2, 0.6),
    ("conn_II", -0.4, -0.1)
]

# Run Sobol sensitivity analysis
result = sobol_sensitivity_analysis(
    base_params,
    param_ranges,
    n_samples=1000,
    tspan=(0.0, 100.0),
    output_metric=:final_mean
)

# Run Morris screening
result = morris_sensitivity_analysis(
    base_params,
    param_ranges,
    n_trajectories=100,
    tspan=(0.0, 100.0),
    output_metric=:final_mean
)
```
"""

using GlobalSensitivity
using QuasiMonteCarlo
using Statistics
using LinearAlgebra

"""
    create_parameter_builder(base_params::WilsonCowanParameters{T,P}, param_ranges) where {T,P}

Create a function that builds WilsonCowanParameters from a parameter vector.

# Arguments
- `base_params`: Base WilsonCowanParameters to use as template
- `param_ranges`: Vector of tuples (param_name, lower_bound, upper_bound)

# Returns
A function `f(p_vector) -> WilsonCowanParameters` that creates parameters from a vector.

# Parameter Naming Convention
Parameters are named as:
- `α_<pop>`: Decay rate for population (e.g., "α_E", "α_I")
- `β_<pop>`: Saturation coefficient for population
- `τ_<pop>`: Time constant for population  
- `conn_<dst><src>`: Connectivity from src to dst (e.g., "conn_EE", "conn_EI")
- `a_<pop>`: Nonlinearity slope for population
- `θ_<pop>`: Nonlinearity threshold for population
"""
function create_parameter_builder(base_params::WilsonCowanParameters{T,P}, param_ranges) where {T,P}
    param_names = [pr[1] for pr in param_ranges]
    
    # Build index maps for parameters
    α_indices = Dict{Int,Int}()
    β_indices = Dict{Int,Int}()
    τ_indices = Dict{Int,Int}()
    conn_indices = Dict{Tuple{Int,Int},Int}()
    a_indices = Dict{Int,Int}()
    θ_indices = Dict{Int,Int}()
    
    for (idx, name) in enumerate(param_names)
        # Parse parameter name
        parts = split(name, "_")
        if length(parts) == 2
            param_type, pop_spec = parts
            
            if param_type == "α"
                pop_idx = findfirst(==(pop_spec), base_params.pop_names)
                if !isnothing(pop_idx)
                    α_indices[pop_idx] = idx
                end
            elseif param_type == "β"
                pop_idx = findfirst(==(pop_spec), base_params.pop_names)
                if !isnothing(pop_idx)
                    β_indices[pop_idx] = idx
                end
            elseif param_type == "τ"
                pop_idx = findfirst(==(pop_spec), base_params.pop_names)
                if !isnothing(pop_idx)
                    τ_indices[pop_idx] = idx
                end
            elseif param_type == "a"
                pop_idx = findfirst(==(pop_spec), base_params.pop_names)
                if !isnothing(pop_idx)
                    a_indices[pop_idx] = idx
                end
            elseif param_type == "θ"
                pop_idx = findfirst(==(pop_spec), base_params.pop_names)
                if !isnothing(pop_idx)
                    θ_indices[pop_idx] = idx
                end
            elseif param_type == "conn"
                # Parse connectivity: e.g., "EE" -> (1, 1), "EI" -> (1, 2)
                if length(pop_spec) == 2
                    dst_idx = findfirst(==(string(pop_spec[1])), base_params.pop_names)
                    src_idx = findfirst(==(string(pop_spec[2])), base_params.pop_names)
                    if !isnothing(dst_idx) && !isnothing(src_idx)
                        conn_indices[(dst_idx, src_idx)] = idx
                    end
                end
            end
        end
    end
    
    # Return builder function
    function build_params(p_vector)
        # Start with base values
        α = collect(base_params.α)
        β = collect(base_params.β)
        τ = collect(base_params.τ)
        
        # Update α, β, τ from parameter vector
        for (pop_idx, param_idx) in α_indices
            α[pop_idx] = p_vector[param_idx]
        end
        for (pop_idx, param_idx) in β_indices
            β[pop_idx] = p_vector[param_idx]
        end
        for (pop_idx, param_idx) in τ_indices
            τ[pop_idx] = p_vector[param_idx]
        end
        
        # Handle connectivity
        connectivity = base_params.connectivity
        if !isempty(conn_indices) && connectivity isa ConnectivityMatrix
            # Create new connectivity matrix with updated values
            new_conn_matrix = Matrix{Any}(undef, P, P)
            for i in 1:P, j in 1:P
                old_conn = connectivity.matrix[i, j]
                if haskey(conn_indices, (i, j))
                    # Update this connectivity parameter
                    new_weight = p_vector[conn_indices[(i, j)]]
                    if old_conn isa ScalarConnectivity
                        new_conn_matrix[i, j] = ScalarConnectivity(new_weight)
                    elseif old_conn isa GaussianConnectivityParameter
                        # Keep same length scale, update amplitude
                        new_conn_matrix[i, j] = GaussianConnectivityParameter(
                            new_weight, old_conn.spread
                        )
                    else
                        new_conn_matrix[i, j] = old_conn
                    end
                else
                    new_conn_matrix[i, j] = old_conn
                end
            end
            connectivity = ConnectivityMatrix{P}(new_conn_matrix)
        end
        
        # Handle nonlinearity
        nonlinearity = base_params.nonlinearity
        if !isempty(a_indices) || !isempty(θ_indices)
            if nonlinearity isa Tuple
                # Multiple nonlinearities (one per population)
                new_nonlinearity = []
                for pop_idx in 1:P
                    old_nl = nonlinearity[pop_idx]
                    if old_nl isa SigmoidNonlinearity
                        a_val = haskey(a_indices, pop_idx) ? p_vector[a_indices[pop_idx]] : old_nl.a
                        θ_val = haskey(θ_indices, pop_idx) ? p_vector[θ_indices[pop_idx]] : old_nl.θ
                        push!(new_nonlinearity, SigmoidNonlinearity(a=a_val, θ=θ_val))
                    else
                        push!(new_nonlinearity, old_nl)
                    end
                end
                nonlinearity = Tuple(new_nonlinearity)
            elseif nonlinearity isa SigmoidNonlinearity
                # Single nonlinearity for all populations
                a_val = haskey(a_indices, 1) ? p_vector[a_indices[1]] : nonlinearity.a
                θ_val = haskey(θ_indices, 1) ? p_vector[θ_indices[1]] : nonlinearity.θ
                nonlinearity = SigmoidNonlinearity(a=a_val, θ=θ_val)
            end
        end
        
        # Create new parameters
        WilsonCowanParameters{T,P}(
            Tuple(α), Tuple(β), Tuple(τ),
            connectivity, nonlinearity,
            base_params.stimulus, base_params.lattice,
            base_params.pop_names
        )
    end
    
    return build_params
end

"""
    create_output_function(tspan, output_metric::Symbol, saveat=nothing)

Create a function that runs a simulation and computes an output metric.

# Arguments
- `tspan`: Time span for simulation (start_time, end_time)
- `output_metric`: Symbol specifying which metric to compute:
  - `:final_mean`: Mean activity across all populations at final time
  - `:final_E`: Mean activity of first population (E) at final time
  - `:final_I`: Mean activity of second population (I) at final time (if exists)
  - `:max_mean`: Maximum mean activity over time
  - `:variance`: Variance of mean activity over time
  - `:oscillation_amplitude`: Peak-to-trough amplitude in final portion
- `saveat`: Time points to save (default: adaptive)

# Returns
A function `f(params) -> scalar` that simulates and returns the metric.
"""
function create_output_function(tspan, output_metric::Symbol, saveat=nothing)
    function compute_output(params::WilsonCowanParameters{T,P}) where {T,P}
        # Create initial condition
        if params.lattice isa PointLattice
            A₀ = reshape(fill(0.1, P), 1, P)
        else
            n_points = size(coordinates(params.lattice), 1)
            A₀ = fill(0.1, n_points, P)
        end
        
        # Solve model
        try
            if saveat === nothing
                sol = solve_model(A₀, tspan, params)
            else
                sol = solve_model(A₀, tspan, params, saveat=saveat)
            end
            
            # Compute requested metric
            if output_metric == :final_mean
                # Mean activity across all populations at final time
                final_state = sol.u[end]
                return mean(final_state)
                
            elseif output_metric == :final_E
                # Mean activity of first population at final time
                final_state = sol.u[end]
                if ndims(final_state) == 2
                    return mean(final_state[:, 1])
                else
                    return final_state[1]
                end
                
            elseif output_metric == :final_I && P >= 2
                # Mean activity of second population at final time
                final_state = sol.u[end]
                if ndims(final_state) == 2
                    return mean(final_state[:, 2])
                else
                    return final_state[2]
                end
                
            elseif output_metric == :max_mean
                # Maximum mean activity over time
                max_activity = 0.0
                for u in sol.u
                    max_activity = max(max_activity, mean(u))
                end
                return max_activity
                
            elseif output_metric == :variance
                # Variance of mean activity over time
                mean_activities = [mean(u) for u in sol.u]
                return var(mean_activities)
                
            elseif output_metric == :oscillation_amplitude
                # Peak-to-trough amplitude in final 20% of simulation
                n_points = length(sol.u)
                start_idx = max(1, div(4 * n_points, 5))
                final_activities = [mean(u) for u in sol.u[start_idx:end]]
                return maximum(final_activities) - minimum(final_activities)
                
            else
                error("Unknown output metric: $output_metric")
            end
        catch e
            # Return NaN if simulation fails
            @warn "Simulation failed: $e"
            return NaN
        end
    end
    
    return compute_output
end

"""
    sobol_sensitivity_analysis(
        base_params::WilsonCowanParameters,
        param_ranges,
        n_samples=1000;
        tspan=(0.0, 100.0),
        output_metric=:final_mean,
        saveat=nothing
    )

Perform Sobol global sensitivity analysis on Wilson-Cowan model parameters.

Sobol analysis decomposes the variance of the output into contributions from
individual parameters (first-order indices) and their interactions (total-order indices).

# Arguments
- `base_params`: Base WilsonCowanParameters to use as template
- `param_ranges`: Vector of tuples (param_name, lower_bound, upper_bound)
- `n_samples`: Number of samples for Monte Carlo estimation (default: 1000)
- `tspan`: Time span for simulation (default: (0.0, 100.0))
- `output_metric`: Which output metric to analyze (default: :final_mean)
- `saveat`: Time points to save (default: adaptive)

# Returns
A dictionary containing:
- `:S1`: First-order Sobol indices (individual parameter effects)
- `:ST`: Total-order Sobol indices (including interactions)
- `:param_names`: Names of parameters analyzed

# Example

```julia
result = sobol_sensitivity_analysis(
    base_params,
    [("α_E", 0.5, 2.0), ("α_I", 0.5, 2.0)],
    n_samples=500
)
println("First-order indices: ", result[:S1])
println("Total-order indices: ", result[:ST])
```
"""
function sobol_sensitivity_analysis(
    base_params::WilsonCowanParameters,
    param_ranges,
    n_samples=1000;
    tspan=(0.0, 100.0),
    output_metric=:final_mean,
    saveat=nothing
)
    # Extract bounds
    n_params = length(param_ranges)
    param_names = [pr[1] for pr in param_ranges]
    lb = [pr[2] for pr in param_ranges]
    ub = [pr[3] for pr in param_ranges]
    
    # Create parameter builder and output function
    build_params = create_parameter_builder(base_params, param_ranges)
    compute_output = create_output_function(tspan, output_metric, saveat)
    
    # Create wrapper function for GlobalSensitivity.jl
    function f(p_matrix)
        # p_matrix has parameters in columns
        n_samples = size(p_matrix, 2)
        outputs = zeros(n_samples)
        
        for i in 1:n_samples
            p_vector = p_matrix[:, i]
            params = build_params(p_vector)
            outputs[i] = compute_output(params)
        end
        
        return outputs
    end
    
    # Run Sobol analysis
    println("Running Sobol sensitivity analysis with $n_samples samples...")
    println("Parameters: ", param_names)
    println("Output metric: $output_metric")
    
    result = gsa(f, Sobol(), [[l, u] for (l, u) in zip(lb, ub)], samples=n_samples)
    
    return Dict(
        :S1 => result.S1,
        :ST => result.ST,
        :param_names => param_names,
        :method => "Sobol",
        :n_samples => n_samples,
        :output_metric => output_metric
    )
end

"""
    morris_sensitivity_analysis(
        base_params::WilsonCowanParameters,
        param_ranges,
        n_trajectories=100;
        tspan=(0.0, 100.0),
        output_metric=:final_mean,
        saveat=nothing,
        num_levels=10
    )

Perform Morris screening (Elementary Effects) sensitivity analysis.

Morris screening is a computationally efficient method for identifying the most
important parameters. It computes the mean (μ) and standard deviation (σ) of
elementary effects for each parameter.

# Arguments
- `base_params`: Base WilsonCowanParameters to use as template
- `param_ranges`: Vector of tuples (param_name, lower_bound, upper_bound)
- `n_trajectories`: Number of trajectories to sample (default: 100)
- `tspan`: Time span for simulation (default: (0.0, 100.0))
- `output_metric`: Which output metric to analyze (default: :final_mean)
- `saveat`: Time points to save (default: adaptive)
- `num_levels`: Number of grid levels for Morris sampling (default: 10)

# Returns
A dictionary containing:
- `:means`: Mean of elementary effects (μ) for each parameter
- `:means_star`: Mean of absolute elementary effects (μ*) for each parameter
- `:variances`: Variance of elementary effects (σ²) for each parameter
- `:param_names`: Names of parameters analyzed

High μ* indicates important parameter, high σ indicates interactions or nonlinearity.

# Example

```julia
result = morris_sensitivity_analysis(
    base_params,
    [("α_E", 0.5, 2.0), ("α_I", 0.5, 2.0)],
    n_trajectories=50
)
println("Mean effects: ", result[:means_star])
println("Variances: ", result[:variances])
```
"""
function morris_sensitivity_analysis(
    base_params::WilsonCowanParameters,
    param_ranges,
    n_trajectories=100;
    tspan=(0.0, 100.0),
    output_metric=:final_mean,
    saveat=nothing,
    num_levels=10
)
    # Extract bounds
    n_params = length(param_ranges)
    param_names = [pr[1] for pr in param_ranges]
    lb = [pr[2] for pr in param_ranges]
    ub = [pr[3] for pr in param_ranges]
    
    # Create parameter builder and output function
    build_params = create_parameter_builder(base_params, param_ranges)
    compute_output = create_output_function(tspan, output_metric, saveat)
    
    # Create wrapper function for GlobalSensitivity.jl
    function f(p_matrix)
        # p_matrix has parameters in columns
        n_samples = size(p_matrix, 2)
        outputs = zeros(n_samples)
        
        for i in 1:n_samples
            p_vector = p_matrix[:, i]
            params = build_params(p_vector)
            outputs[i] = compute_output(params)
        end
        
        return outputs
    end
    
    # Run Morris analysis
    println("Running Morris sensitivity analysis with $n_trajectories trajectories...")
    println("Parameters: ", param_names)
    println("Output metric: $output_metric")
    
    result = gsa(f, Morris(), [[l, u] for (l, u) in zip(lb, ub)], 
                 num_trajectory=n_trajectories, total_num_trajectory=n_trajectories,
                 num_levels=num_levels)
    
    return Dict(
        :means => result.means,
        :means_star => result.means_star,
        :variances => result.variances,
        :param_names => param_names,
        :method => "Morris",
        :n_trajectories => n_trajectories,
        :output_metric => output_metric
    )
end
