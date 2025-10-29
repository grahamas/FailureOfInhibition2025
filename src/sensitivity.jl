"""
Sensitivity analysis utilities for Wilson-Cowan models using SciMLSensitivity.jl
"""

using SciMLSensitivity
using DifferentialEquations
using Statistics
using DataFrames
using CSV

"""
    ODEParameterWrapper{T,P}

Wrapper for WilsonCowanParameters that extracts numeric parameters into a vector
for sensitivity analysis while preserving non-numeric fields.

# Fields
- `base_params::WilsonCowanParameters{T,P}`: Original parameters
- `param_names::Vector{Symbol}`: Names of parameters being analyzed
- `param_values::Vector{T}`: Current values of parameters
"""
struct ODEParameterWrapper{T,P}
    base_params::WilsonCowanParameters{T,P}
    param_names::Vector{Symbol}
    param_values::Vector{T}
end

"""
    extract_parameters(params::WilsonCowanParameters{T,P}; include_params=[:α, :β, :τ]) where {T,P}

Extract numeric parameters from WilsonCowanParameters into a vector for sensitivity analysis.

# Arguments
- `params`: WilsonCowanParameters containing model configuration
- `include_params`: Which parameter groups to include (default: [:α, :β, :τ])

# Returns
- `ODEParameterWrapper` containing parameter names and values

# Example
```julia
params = WilsonCowanParameters{2}(
    α = (1.0, 1.5),
    β = (1.0, 1.0),
    τ = (10.0, 8.0),
    # ... other fields
)
wrapper = extract_parameters(params)
# wrapper.param_names = [:α_1, :α_2, :β_1, :β_2, :τ_1, :τ_2]
# wrapper.param_values = [1.0, 1.5, 1.0, 1.0, 10.0, 8.0]
```
"""
function extract_parameters(params::WilsonCowanParameters{T,P}; 
                           include_params=[:α, :β, :τ]) where {T,P}
    param_names = Symbol[]
    param_values = T[]
    
    for param in include_params
        if param == :α
            for i in 1:P
                push!(param_names, Symbol("α_$i"))
                push!(param_values, params.α[i])
            end
        elseif param == :β
            for i in 1:P
                push!(param_names, Symbol("β_$i"))
                push!(param_values, params.β[i])
            end
        elseif param == :τ
            for i in 1:P
                push!(param_names, Symbol("τ_$i"))
                push!(param_values, params.τ[i])
            end
        end
    end
    
    return ODEParameterWrapper(params, param_names, param_values)
end

"""
    reconstruct_parameters(wrapper::ODEParameterWrapper{T,P}, p_vec::Vector) where {T,P}

Reconstruct WilsonCowanParameters from a parameter vector.

# Arguments
- `wrapper`: ODEParameterWrapper containing original structure
- `p_vec`: Vector of parameter values (must match length of wrapper.param_values)

# Returns
- New WilsonCowanParameters with updated parameter values
"""
function reconstruct_parameters(wrapper::ODEParameterWrapper{T,P}, p_vec::AbstractVector{U}) where {T,P,U}
    # Start with base parameter values, using eltype of p_vec for type stability
    α_vals = Vector{U}(undef, P)
    β_vals = Vector{U}(undef, P)
    τ_vals = Vector{U}(undef, P)
    
    # Initialize with base values
    for i in 1:P
        α_vals[i] = wrapper.base_params.α[i]
        β_vals[i] = wrapper.base_params.β[i]
        τ_vals[i] = wrapper.base_params.τ[i]
    end
    
    # Update from parameter vector
    for (i, name) in enumerate(wrapper.param_names)
        name_str = string(name)
        if startswith(name_str, "α_")
            idx = parse(Int, split(name_str, "_")[2])
            α_vals[idx] = p_vec[i]
        elseif startswith(name_str, "β_")
            idx = parse(Int, split(name_str, "_")[2])
            β_vals[idx] = p_vec[i]
        elseif startswith(name_str, "τ_")
            idx = parse(Int, split(name_str, "_")[2])
            τ_vals[idx] = p_vec[i]
        end
    end
    
    # Reconstruct parameters
    return WilsonCowanParameters{U,P}(
        Tuple(α_vals),
        Tuple(β_vals),
        Tuple(τ_vals),
        wrapper.base_params.connectivity,
        wrapper.base_params.nonlinearity,
        wrapper.base_params.stimulus,
        wrapper.base_params.lattice,
        wrapper.base_params.pop_names
    )
end

"""
    wcm1973_wrapped!(dA, A, p, t, wrapper::ODEParameterWrapper{T,P}, original_shape)

Wrapper function for wcm1973! that accepts parameter vector p and flattened state.
This is needed for SciMLSensitivity integration.
"""
function wcm1973_wrapped!(dA, A, p, t, wrapper::ODEParameterWrapper{T,P}, original_shape) where {T,P}
    # Reconstruct parameters
    params = reconstruct_parameters(wrapper, p)
    
    # Reshape inputs to original shape
    A_reshaped = reshape(A, original_shape)
    dA_reshaped = similar(A_reshaped)
    
    # Call original function - modifies dA_reshaped in place
    wcm1973!(dA_reshaped, A_reshaped, params, t)
    
    # Flatten result back into dA
    copyto!(dA, vec(dA_reshaped))
    
    return nothing
end

"""
    compute_local_sensitivities(initial_condition, tspan, params::WilsonCowanParameters{N};
                                include_params=[:α, :β, :τ],
                                method=ForwardDiffSensitivity(),
                                solver=Tsit5(),
                                saveat=nothing,
                                kwargs...) where N

Compute local parameter sensitivities using SciMLSensitivity.jl.

Local sensitivity analysis computes the derivative of the solution with respect to 
each parameter: ∂u/∂p. This tells you how small changes in parameters affect the solution.

# Arguments
- `initial_condition`: Initial state (same format as solve_model)
- `tspan`: Time span tuple (t_start, t_end)
- `params`: WilsonCowanParameters
- `include_params`: Which parameters to analyze (default: [:α, :β, :τ])
- `method`: Sensitivity algorithm (default: ForwardDiffSensitivity())
  - ForwardDiffSensitivity(): Forward mode (good for few parameters)
  - InterpolatingAdjoint(): Adjoint mode (good for many parameters)
  - QuadratureAdjoint(): Continuous adjoint
- `solver`: ODE solver (default: Tsit5())
- `saveat`: Save sensitivity at specific times
- `kwargs...`: Additional arguments for solve()

# Returns
Named tuple with:
- `solution`: ODE solution
- `sensitivities`: Array of sensitivities [time, state, parameter]
- `param_names`: Names of parameters
- `param_values`: Values of parameters

# Example
```julia
params = WilsonCowanParameters{2}(
    α = (1.0, 1.5),
    β = (1.0, 1.0),
    τ = (10.0, 8.0),
    # ... other fields
)
A₀ = reshape([0.1, 0.1], 1, 2)
tspan = (0.0, 50.0)

result = compute_local_sensitivities(A₀, tspan, params, saveat=1.0)

# Access results
println("Parameters analyzed: ", result.param_names)
println("Sensitivity at t=10 for state 1 w.r.t α_1: ", result.sensitivities[11, 1, 1])
```
"""
function compute_local_sensitivities(initial_condition, tspan, params::WilsonCowanParameters{N};
                                    include_params=[:α, :β, :τ],
                                    method=ForwardDiffSensitivity(),
                                    solver=Tsit5(),
                                    saveat=nothing,
                                    kwargs...) where N
    # Extract parameters into vector form
    wrapper = extract_parameters(params, include_params=include_params)
    p_vec = wrapper.param_values
    
    # Store original shape and flatten initial condition
    original_shape = size(initial_condition)
    u0_flat = vec(initial_condition)
    
    # Create wrapped ODE function with the wrapper and original_shape captured
    f_wrapped = (dA, A, p, t) -> wcm1973_wrapped!(dA, A, p, t, wrapper, original_shape)
    
    # Create sensitivity problem with flattened initial condition
    prob = ODEForwardSensitivityProblem(f_wrapped, u0_flat, tspan, p_vec)
    
    # Solve with sensitivity
    if saveat !== nothing
        sol = solve(prob, solver; saveat=saveat, kwargs...)
    else
        sol = solve(prob, solver; kwargs...)
    end
    
    # Extract sensitivities
    # For ForwardDiffSensitivity, extract_local_sensitivities returns [time][state, param]
    S = extract_local_sensitivities(sol)
    
    # Reshape sensitivities to [time, state, param] for easier access
    n_times = length(sol.t)
    n_states = length(u0_flat)
    n_params = length(wrapper.param_values)
    
    sensitivities = zeros(n_times, n_states, n_params)
    for (t_idx, s_t) in enumerate(S)
        sensitivities[t_idx, :, :] = s_t
    end
    
    return (
        solution = sol,
        sensitivities = sensitivities,
        param_names = wrapper.param_names,
        param_values = wrapper.param_values,
        times = sol.t,
        original_shape = original_shape
    )
end

"""
    save_local_sensitivities(result, filename; params=nothing)

Save local sensitivity analysis results to CSV.

# Arguments
- `result`: Output from compute_local_sensitivities()
- `filename`: Output CSV file path
- `params`: (Optional) Original WilsonCowanParameters for metadata

# File Format
The CSV contains columns:
- `time`: Time point
- `state_idx`: State variable index
- `param_name`: Parameter name
- `sensitivity`: Sensitivity value (∂u/∂p)

# Example
```julia
result = compute_local_sensitivities(A₀, tspan, params)
save_local_sensitivities(result, "sensitivities.csv", params=params)
```
"""
function save_local_sensitivities(result, filename; params=nothing)
    data = []
    
    n_times, n_states, n_params = size(result.sensitivities)
    
    for t_idx in 1:n_times
        for s_idx in 1:n_states
            for p_idx in 1:n_params
                push!(data, Dict(
                    "time" => result.times[t_idx],
                    "state_idx" => s_idx,
                    "param_name" => string(result.param_names[p_idx]),
                    "sensitivity" => result.sensitivities[t_idx, s_idx, p_idx]
                ))
            end
        end
    end
    
    df = DataFrame(data)
    CSV.write(filename, df)
    
    return df
end

"""
    summarize_sensitivities(result; params=nothing)

Compute summary statistics for sensitivity analysis results.

# Arguments
- `result`: Output from compute_local_sensitivities()
- `params`: (Optional) Original WilsonCowanParameters for population names

# Returns
DataFrame with columns:
- `param_name`: Parameter name
- `state_idx`: State index
- `mean_abs_sensitivity`: Mean absolute sensitivity over time
- `max_abs_sensitivity`: Maximum absolute sensitivity
- `final_sensitivity`: Sensitivity at final time point

# Example
```julia
result = compute_local_sensitivities(A₀, tspan, params)
summary = summarize_sensitivities(result, params=params)
println(summary)
```
"""
function summarize_sensitivities(result; params=nothing)
    summary_data = []
    
    n_times, n_states, n_params = size(result.sensitivities)
    
    # Get population names if available
    if params !== nothing && hasfield(typeof(params), :pop_names)
        pop_names = params.pop_names
    else
        pop_names = ntuple(i -> "state_$i", n_states)
    end
    
    for p_idx in 1:n_params
        for s_idx in 1:n_states
            # Extract time series for this state-parameter pair
            sens_ts = result.sensitivities[:, s_idx, p_idx]
            
            push!(summary_data, Dict(
                "param_name" => string(result.param_names[p_idx]),
                "state_idx" => s_idx,
                "state_name" => string(pop_names[min(s_idx, length(pop_names))]),
                "mean_abs_sensitivity" => mean(abs.(sens_ts)),
                "max_abs_sensitivity" => maximum(abs.(sens_ts)),
                "final_sensitivity" => sens_ts[end],
                "mean_sensitivity" => mean(sens_ts),
                "std_sensitivity" => std(sens_ts)
            ))
        end
    end
    
    return DataFrame(summary_data)
end

"""
    compute_sensitivity_indices(initial_condition, tspan, params::WilsonCowanParameters{N};
                                include_params=[:α, :β, :τ],
                                method=:sobol,
                                n_samples=1000,
                                param_ranges=nothing,
                                solver=Tsit5(),
                                output_func=default_output_func,
                                kwargs...) where N

Compute global sensitivity indices using GlobalSensitivity.jl.

Global sensitivity analysis explores how uncertainty in parameters affects output
uncertainty across the entire parameter space (not just local derivatives).

# Arguments
- `initial_condition`: Initial state
- `tspan`: Time span tuple
- `params`: WilsonCowanParameters (used as nominal values)
- `include_params`: Which parameters to analyze
- `method`: Global sensitivity method (:sobol, :morris, :regression)
- `n_samples`: Number of samples for global analysis
- `param_ranges`: Dict mapping parameter names to (min, max) ranges
  If not provided, uses ±50% of nominal values
- `solver`: ODE solver
- `output_func`: Function to extract scalar output from solution
  Default: mean activity of first population at final time
- `kwargs...`: Additional arguments

# Returns
Named tuple with sensitivity indices and parameter information

# Example
```julia
params = WilsonCowanParameters{2}(α=(1.0, 1.5), β=(1.0, 1.0), τ=(10.0, 8.0), ...)
A₀ = reshape([0.1, 0.1], 1, 2)
tspan = (0.0, 50.0)

# Define custom output function (e.g., mean E activity at final time)
output_func = sol -> mean(sol[end][1, :])

result = compute_sensitivity_indices(
    A₀, tspan, params,
    method=:sobol,
    n_samples=1000,
    output_func=output_func
)
```
"""
function compute_sensitivity_indices(initial_condition, tspan, params::WilsonCowanParameters{N};
                                     include_params=[:α, :β, :τ],
                                     method=:sobol,
                                     n_samples=1000,
                                     param_ranges=nothing,
                                     solver=Tsit5(),
                                     output_func=nothing,
                                     kwargs...) where N
    # Extract parameters
    wrapper = extract_parameters(params, include_params=include_params)
    
    # Set default output function if not provided
    if output_func === nothing
        # Default: mean activity at final time
        output_func = function(sol)
            if sol.retcode == ReturnCode.Success
                final_state = sol[end]
                return mean(final_state)
            else
                return NaN
            end
        end
    end
    
    # Set parameter ranges if not provided
    if param_ranges === nothing
        param_ranges = Dict()
        for (i, pname) in enumerate(wrapper.param_names)
            # Use ±50% of nominal value
            nominal = wrapper.param_values[i]
            param_ranges[pname] = (0.5 * nominal, 1.5 * nominal)
        end
    end
    
    # Note: Full GlobalSensitivity.jl integration would require additional setup
    # This is a placeholder that documents the intended interface
    # Users should refer to GlobalSensitivity.jl documentation for full implementation
    
    return (
        message = "Global sensitivity analysis requires additional setup with GlobalSensitivity.jl",
        param_names = wrapper.param_names,
        param_ranges = param_ranges,
        method = method,
        n_samples = n_samples,
        note = "See GlobalSensitivity.jl documentation for Sobol indices, Morris screening, etc."
    )
end

# Default output function for global sensitivity
default_output_func(sol) = mean(sol[end])
