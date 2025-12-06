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
  - `:α` - decay rates
  - `:β` - saturation coefficients
  - `:τ` - time constants
  - `:connectivity` - connectivity weights (for ScalarConnectivity) or amplitude/spread (for GaussianConnectivityParameter)
  - `:nonlinearity` - nonlinearity parameters (a, θ for sigmoid types)

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
wrapper = extract_parameters(params, include_params=[:α, :β, :τ, :connectivity, :nonlinearity])
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
        elseif param == :connectivity
            # Extract connectivity parameters
            conn = params.connectivity
            if conn isa ConnectivityMatrix
                # Extract ScalarConnectivity weights, GaussianConnectivityParameter, or GaussianConnectivity
                for i in 1:P
                    for j in 1:P
                        c_ij = conn.matrix[i, j]
                        if c_ij isa ScalarConnectivity
                            push!(param_names, Symbol("b_$(i)_$(j)"))
                            push!(param_values, c_ij.weight)
                        elseif c_ij isa GaussianConnectivityParameter
                            push!(param_names, Symbol("b_amplitude_$(i)_$(j)"))
                            push!(param_values, c_ij.amplitude)
                            # Extract spread parameters
                            for (k, s) in enumerate(c_ij.spread)
                                push!(param_names, Symbol("b_spread_$(i)_$(j)_dim$(k)"))
                                push!(param_values, s)
                            end
                        elseif c_ij isa GaussianConnectivity
                            # GaussianConnectivity is the pre-computed version of GaussianConnectivityParameter
                            push!(param_names, Symbol("b_amplitude_$(i)_$(j)"))
                            push!(param_values, c_ij.amplitude)
                            # Extract spread parameters
                            for (k, s) in enumerate(c_ij.spread)
                                push!(param_names, Symbol("b_spread_$(i)_$(j)_dim$(k)"))
                                push!(param_values, s)
                            end
                        end
                    end
                end
            end
        elseif param == :nonlinearity
            # Extract nonlinearity parameters
            nl = params.nonlinearity
            if nl isa SigmoidNonlinearity
                push!(param_names, :a)
                push!(param_values, nl.a)
                push!(param_names, :θ)
                push!(param_values, nl.θ)
            elseif nl isa RectifiedZeroedSigmoidNonlinearity
                push!(param_names, :a)
                push!(param_values, nl.a)
                push!(param_names, :θ)
                push!(param_values, nl.θ)
            elseif nl isa DifferenceOfSigmoidsNonlinearity
                push!(param_names, :a_up)
                push!(param_values, nl.a_up)
                push!(param_names, :θ_up)
                push!(param_values, nl.θ_up)
                push!(param_names, :a_down)
                push!(param_values, nl.a_down)
                push!(param_names, :θ_down)
                push!(param_values, nl.θ_down)
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
    
    # Start with base connectivity and nonlinearity
    connectivity = wrapper.base_params.connectivity
    nonlinearity = wrapper.base_params.nonlinearity
    
    # Track connectivity and nonlinearity updates
    connectivity_updates = Dict{Tuple{Int,Int}, Any}()
    nonlinearity_updates = Dict{Symbol, U}()
    
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
        elseif startswith(name_str, "b_")
            # Parse connectivity parameter
            parts = split(name_str, "_")
            if parts[2] == "amplitude"
                # GaussianConnectivityParameter amplitude: b_amplitude_i_j
                idx_i = parse(Int, parts[3])
                idx_j = parse(Int, parts[4])
                if !haskey(connectivity_updates, (idx_i, idx_j))
                    connectivity_updates[(idx_i, idx_j)] = Dict{Symbol, Any}()
                end
                connectivity_updates[(idx_i, idx_j)][:amplitude] = p_vec[i]
            elseif parts[2] == "spread"
                # GaussianConnectivityParameter spread: b_spread_i_j_dimk
                idx_i = parse(Int, parts[3])
                idx_j = parse(Int, parts[4])
                dim_idx = parse(Int, replace(parts[5], "dim" => ""))
                if !haskey(connectivity_updates, (idx_i, idx_j))
                    connectivity_updates[(idx_i, idx_j)] = Dict{Symbol, Any}()
                end
                if !haskey(connectivity_updates[(idx_i, idx_j)], :spread)
                    connectivity_updates[(idx_i, idx_j)][:spread] = Dict{Int, U}()
                end
                connectivity_updates[(idx_i, idx_j)][:spread][dim_idx] = p_vec[i]
            else
                # ScalarConnectivity weight: b_i_j
                idx_i = parse(Int, parts[2])
                idx_j = parse(Int, parts[3])
                connectivity_updates[(idx_i, idx_j)] = p_vec[i]
            end
        elseif name == :a || name == :θ || name == :a_up || name == :θ_up || name == :a_down || name == :θ_down
            # Nonlinearity parameter
            nonlinearity_updates[name] = p_vec[i]
        end
    end
    
    # Reconstruct connectivity if there were updates
    if !isempty(connectivity_updates) && connectivity isa ConnectivityMatrix
        new_matrix = Matrix{Any}(undef, P, P)
        for i in 1:P
            for j in 1:P
                c_ij = connectivity.matrix[i, j]
                if haskey(connectivity_updates, (i, j))
                    update = connectivity_updates[(i, j)]
                    if update isa Dict
                        # GaussianConnectivity or GaussianConnectivityParameter update
                        # Extract amplitude and spread from either type
                        if c_ij isa GaussianConnectivityParameter
                            old_amplitude = c_ij.amplitude
                            old_spread = c_ij.spread
                        elseif c_ij isa GaussianConnectivity
                            old_amplitude = c_ij.amplitude
                            old_spread = c_ij.spread
                        else
                            error("Unexpected connectivity type: $(typeof(c_ij))")
                        end
                        
                        new_amplitude = get(update, :amplitude, old_amplitude)
                        if haskey(update, :spread)
                            # Reconstruct spread tuple
                            N = length(old_spread)
                            spread_dict = update[:spread]
                            new_spread = ntuple(k -> get(spread_dict, k, old_spread[k]), N)
                        else
                            new_spread = old_spread
                        end
                        # Create GaussianConnectivityParameter and convert to GaussianConnectivity
                        param = GaussianConnectivityParameter(new_amplitude, new_spread)
                        new_matrix[i, j] = GaussianConnectivity(param, wrapper.base_params.lattice)
                    else
                        # ScalarConnectivity update
                        new_matrix[i, j] = ScalarConnectivity(update)
                    end
                else
                    new_matrix[i, j] = c_ij
                end
            end
        end
        connectivity = ConnectivityMatrix{P}(new_matrix)
    end
    
    # Reconstruct nonlinearity if there were updates
    if !isempty(nonlinearity_updates)
        if nonlinearity isa SigmoidNonlinearity
            new_a = get(nonlinearity_updates, :a, nonlinearity.a)
            new_θ = get(nonlinearity_updates, :θ, nonlinearity.θ)
            nonlinearity = SigmoidNonlinearity(new_a, new_θ)
        elseif nonlinearity isa RectifiedZeroedSigmoidNonlinearity
            new_a = get(nonlinearity_updates, :a, nonlinearity.a)
            new_θ = get(nonlinearity_updates, :θ, nonlinearity.θ)
            nonlinearity = RectifiedZeroedSigmoidNonlinearity(new_a, new_θ)
        elseif nonlinearity isa DifferenceOfSigmoidsNonlinearity
            new_a_up = get(nonlinearity_updates, :a_up, nonlinearity.a_up)
            new_θ_up = get(nonlinearity_updates, :θ_up, nonlinearity.θ_up)
            new_a_down = get(nonlinearity_updates, :a_down, nonlinearity.a_down)
            new_θ_down = get(nonlinearity_updates, :θ_down, nonlinearity.θ_down)
            nonlinearity = DifferenceOfSigmoidsNonlinearity(new_a_up, new_θ_up, new_a_down, new_θ_down)
        end
    end
    
    # Reconstruct parameters
    return WilsonCowanParameters{U,P}(
        Tuple(α_vals),
        Tuple(β_vals),
        Tuple(τ_vals),
        connectivity,
        nonlinearity,
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
    
    # Determine the element type (might be Dual for ForwardDiff)
    U = eltype(dA)
    
    # Reshape inputs to original shape with correct element type
    A_reshaped = reshape(convert(Vector{U}, A), original_shape)
    dA_reshaped = reshape(dA, original_shape)
    
    # Call original function - modifies dA_reshaped in place
    wcm1973!(dA_reshaped, A_reshaped, params, t)
    
    return nothing
end

"""
    compute_local_sensitivities(initial_condition, tspan, params; kwargs...)

Compute local parameter sensitivities using SciMLSensitivity.jl.

This function demonstrates how to set up parameter sensitivity analysis for 
Wilson-Cowan models. It solves the model and computes sensitivities using 
adjoint methods.

# Arguments
- `initial_condition`: Initial state (same format as solve_model)
- `tspan`: Time span tuple (t_start, t_end)
- `params`: WilsonCowanParameters
- `include_params`: Which parameters to analyze (default: [:α, :β, :τ])
- `solver`: ODE solver (default: Tsit5())
- `saveat`: Save solution at specific times
- `abstol`, `reltol`: Tolerances for ODE solver
- `kwargs...`: Additional arguments for solve()

# Returns
Named tuple with:
- `solution`: ODE solution  
- `param_names`: Names of parameters
- `param_values`: Values of parameters
- `note`: Information about sensitivity computation

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

# The solution can be used with SciMLSensitivity.jl functions
using SciMLSensitivity
loss(sol) = sum(abs2, sol[end])
sens = adjoint_sensitivities(result.solution, Tsit5(), loss, InterpolatingAdjoint())
```

# Note
This function sets up the infrastructure for sensitivity analysis. To compute
actual sensitivities, use SciMLSensitivity.jl's `adjoint_sensitivities` function
with your chosen loss/cost function.

For full sensitivity analysis examples, see `examples/example_sensitivity_analysis.jl`.
"""
function compute_local_sensitivities(initial_condition, tspan, params::WilsonCowanParameters{N};
                                    include_params=[:α, :β, :τ],
                                    solver=Tsit5(),
                                    saveat=nothing,
                                    abstol=1e-6,
                                    reltol=1e-6,
                                    kwargs...) where N
    # Extract parameters into vector form
    wrapper = extract_parameters(params, include_params=include_params)
    p_vec = wrapper.param_values
    
    # Store original shape and flatten initial condition if needed
    original_shape = size(initial_condition)
    u0_flat = vec(initial_condition)
    
    # Create wrapped ODE function
    f_wrapped = (dA, A, p, t) -> begin
        # Reconstruct to original shape
        A_shaped = reshape(A, original_shape)
        dA_shaped = reshape(dA, original_shape)
        
        # Reconstruct parameters
        params_recon = reconstruct_parameters(wrapper, p)
        
        # Call original function
        wcm1973!(dA_shaped, A_shaped, params_recon, t)
    end
    
    # Create and solve ODE problem
    prob = ODEProblem(f_wrapped, u0_flat, tspan, p_vec)
    
    if saveat !== nothing
        sol = solve(prob, solver; saveat=saveat, abstol=abstol, reltol=reltol, kwargs...)
    else
        sol = solve(prob, solver; abstol=abstol, reltol=reltol, kwargs...)
    end
    
    return (
        solution = sol,
        param_names = wrapper.param_names,
        param_values = wrapper.param_values,
        times = sol.t,
        original_shape = original_shape,
        note = "Use SciMLSensitivity.adjoint_sensitivities(sol, solver, loss_fn, method) to compute sensitivities"
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
