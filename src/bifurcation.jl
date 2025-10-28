"""
Bifurcation analysis utilities for Wilson-Cowan models.

This module provides tools for analyzing how system dynamics change as parameters vary,
including parameter sweeps, steady-state detection, and oscillation analysis.
"""

using DifferentialEquations
using Statistics

"""
    BifurcationPoint

Represents the dynamical state at a specific parameter value.

# Fields
- `param_values::NamedTuple`: Parameter values at this point
- `steady_state::Union{Nothing, Array}`: Steady state activity (if system reaches steady state)
- `is_oscillatory::Bool`: Whether the system exhibits sustained oscillations
- `oscillation_amplitude::Union{Nothing, Float64}`: Amplitude of oscillations (if oscillatory)
- `oscillation_period::Union{Nothing, Float64}`: Period of oscillations (if oscillatory)
- `mean_activity::Array`: Mean activity over the simulation for each population
- `max_activity::Array`: Maximum activity over the simulation for each population
- `min_activity::Array`: Minimum activity over the simulation for each population
"""
struct BifurcationPoint
    param_values::NamedTuple
    steady_state::Union{Nothing, Array}
    is_oscillatory::Bool
    oscillation_amplitude::Union{Nothing, Float64}
    oscillation_period::Union{Nothing, Float64}
    mean_activity::Array
    max_activity::Array
    min_activity::Array
end

"""
    BifurcationDiagram

Contains results from a bifurcation analysis across parameter space.

# Fields
- `param_names::Tuple{Symbol, Symbol}`: Names of the two parameters being varied
- `param1_values::Vector`: Values of first parameter
- `param2_values::Vector`: Values of second parameter
- `points::Matrix{BifurcationPoint}`: Grid of bifurcation points
"""
struct BifurcationDiagram
    param_names::Tuple{Symbol, Symbol}
    param1_values::Vector
    param2_values::Vector
    points::Matrix{BifurcationPoint}
end

"""
    detect_steady_state(sol; transient_fraction=0.5, threshold=1e-4)

Detect if the solution has reached a steady state.

# Arguments
- `sol`: ODE solution from solve_model
- `transient_fraction`: Fraction of time to discard as transient (default: 0.5)
- `threshold`: Maximum change in activity to consider steady state (default: 1e-4)

# Returns
- `steady_state::Union{Nothing, Array}`: Steady state values if detected, nothing otherwise
"""
function detect_steady_state(sol; transient_fraction=0.5, threshold=1e-4)
    # Discard initial transient
    n_points = length(sol.t)
    transient_idx = max(1, floor(Int, n_points * transient_fraction))
    
    # Check if the activity is approximately constant in the latter half
    steady_period = sol.u[transient_idx:end]
    
    if isempty(steady_period)
        return nothing
    end
    
    # Calculate variation in the steady period
    # For spatial models, check spatial average
    mean_activity = mean(steady_period)
    max_variation = maximum([maximum(abs.(u .- mean_activity)) for u in steady_period])
    
    if max_variation < threshold
        return mean_activity
    else
        return nothing
    end
end

"""
    detect_oscillations(sol; transient_fraction=0.5, min_peaks=3)

Detect if the solution exhibits sustained oscillations.

# Arguments
- `sol`: ODE solution from solve_model
- `transient_fraction`: Fraction of time to discard as transient (default: 0.5)
- `min_peaks`: Minimum number of peaks to confirm oscillation (default: 3)

# Returns
- `is_oscillatory::Bool`: Whether sustained oscillations were detected
- `amplitude::Union{Nothing, Float64}`: Mean amplitude of oscillations
- `period::Union{Nothing, Float64}`: Mean period of oscillations
"""
function detect_oscillations(sol; transient_fraction=0.5, min_peaks=3)
    # Discard initial transient
    n_points = length(sol.t)
    transient_idx = max(1, floor(Int, n_points * transient_fraction))
    
    # Extract time series after transient (use first population, first spatial point if spatial)
    times = sol.t[transient_idx:end]
    if ndims(sol.u[1]) == 1
        # Point model without connectivity
        activity = [u[1] for u in sol.u[transient_idx:end]]
    else
        # Spatial model or point model with connectivity - use spatial average or first point
        if size(sol.u[1], 1) == 1
            # Point model with connectivity
            activity = [u[1, 1] for u in sol.u[transient_idx:end]]
        else
            # Spatial model - use spatial average
            activity = [mean(u[:, 1]) for u in sol.u[transient_idx:end]]
        end
    end
    
    if length(activity) < min_peaks + 2
        return false, nothing, nothing
    end
    
    # Simple peak detection: find local maxima
    peaks_idx = Int[]
    for i in 2:(length(activity)-1)
        if activity[i] > activity[i-1] && activity[i] > activity[i+1]
            push!(peaks_idx, i)
        end
    end
    
    if length(peaks_idx) < min_peaks
        return false, nothing, nothing
    end
    
    # Calculate mean amplitude (difference between peaks and troughs)
    peak_values = activity[peaks_idx]
    mean_amplitude = (maximum(activity) - minimum(activity)) / 2.0
    
    # Calculate mean period
    if length(peaks_idx) >= 2
        peak_times = times[peaks_idx]
        periods = diff(peak_times)
        mean_period = mean(periods)
    else
        mean_period = nothing
    end
    
    return true, mean_amplitude, mean_period
end

"""
    analyze_dynamics(sol; transient_fraction=0.5)

Analyze the dynamics of a solution to classify its behavior.

# Arguments
- `sol`: ODE solution from solve_model
- `transient_fraction`: Fraction of time to discard as transient (default: 0.5)

# Returns
- `BifurcationPoint` containing analysis results
"""
function analyze_dynamics(sol, param_values::NamedTuple; transient_fraction=0.5)
    # Detect steady state
    steady_state = detect_steady_state(sol; transient_fraction=transient_fraction)
    
    # Detect oscillations
    is_osc, osc_amp, osc_period = detect_oscillations(sol; transient_fraction=transient_fraction)
    
    # Calculate overall statistics
    # Discard transient for statistics
    n_points = length(sol.t)
    transient_idx = max(1, floor(Int, n_points * transient_fraction))
    steady_period = sol.u[transient_idx:end]
    
    # Get dimensions
    first_state = sol.u[1]
    if ndims(first_state) == 1
        n_pops = length(first_state)
        # Extract statistics for each population
        mean_act = [mean([u[i] for u in steady_period]) for i in 1:n_pops]
        max_act = [maximum([u[i] for u in steady_period]) for i in 1:n_pops]
        min_act = [minimum([u[i] for u in steady_period]) for i in 1:n_pops]
    else
        n_pops = size(first_state, 2)
        # For spatial models, use spatial average
        mean_act = [mean([mean(u[:, i]) for u in steady_period]) for i in 1:n_pops]
        max_act = [maximum([mean(u[:, i]) for u in steady_period]) for i in 1:n_pops]
        min_act = [minimum([mean(u[:, i]) for u in steady_period]) for i in 1:n_pops]
    end
    
    return BifurcationPoint(
        param_values,
        steady_state,
        is_osc,
        osc_amp,
        osc_period,
        mean_act,
        max_act,
        min_act
    )
end

"""
    parameter_sweep_2d(base_params, param1_name, param1_range, param2_name, param2_range;
                       initial_condition=nothing, tspan=(0.0, 500.0), transient_fraction=0.5,
                       solver=Tsit5(), saveat=0.5)

Perform a 2D parameter sweep to generate a bifurcation diagram.

# Arguments
- `base_params`: Base WilsonCowanParameters to use as template
- `param1_name::Symbol`: Name of first parameter to vary (e.g., :bₑₑ)
- `param1_range`: Range of values for first parameter
- `param2_name::Symbol`: Name of second parameter to vary
- `param2_range`: Range of values for second parameter
- `initial_condition`: Initial state (if nothing, uses small random values)
- `tspan`: Time span for simulation (default: (0.0, 500.0))
- `transient_fraction`: Fraction of time to discard as transient (default: 0.5)
- `solver`: ODE solver to use (default: Tsit5())
- `saveat`: Sampling interval for solution (default: 0.5)

# Returns
- `BifurcationDiagram` containing results across the 2D parameter space

# Example
```julia
using FailureOfInhibition2025

# Create base parameters
params = create_point_model_wcm1973(:active_transient)

# Sweep E-E and I-E coupling strengths
diagram = parameter_sweep_2d(
    params, :bₑₑ, 0.5:0.1:3.0, :bᵢₑ, 0.5:0.1:3.0
)
```
"""
function parameter_sweep_2d(base_params, param1_name, param1_range, param2_name, param2_range;
                            initial_condition=nothing, tspan=(0.0, 500.0), transient_fraction=0.5,
                            solver=Tsit5(), saveat=0.5)
    # Convert ranges to vectors
    p1_vals = collect(param1_range)
    p2_vals = collect(param2_range)
    
    # Determine initial condition if not provided
    if initial_condition === nothing
        # Use small random initial conditions
        if base_params.lattice isa PointLattice
            # Point model
            P = length(base_params.pop_names)
            initial_condition = reshape(0.1 .+ 0.05 .* rand(P), 1, P)
        else
            # Spatial model
            n_points = size(base_params.lattice)[1]
            P = length(base_params.pop_names)
            initial_condition = 0.1 .+ 0.05 .* rand(n_points, P)
        end
    end
    
    # Initialize results grid
    points = Matrix{BifurcationPoint}(undef, length(p1_vals), length(p2_vals))
    
    # Sweep over parameter space
    for (i, p1) in enumerate(p1_vals)
        for (j, p2) in enumerate(p2_vals)
            # Update parameters
            updated_params = update_parameter(base_params, param1_name, p1)
            updated_params = update_parameter(updated_params, param2_name, p2)
            
            # Solve model
            sol = solve_model(initial_condition, tspan, updated_params; solver=solver, saveat=saveat)
            
            # Analyze dynamics
            param_vals = NamedTuple{(param1_name, param2_name)}((p1, p2))
            points[i, j] = analyze_dynamics(sol, param_vals; transient_fraction=transient_fraction)
        end
    end
    
    return BifurcationDiagram((param1_name, param2_name), p1_vals, p2_vals, points)
end

"""
    update_parameter(params::WilsonCowanParameters{T,P}, param_name::Symbol, value) where {T,P}

Create new WilsonCowanParameters with a specific parameter updated.

This function handles updating connectivity parameters (e.g., bₑₑ, bᵢₑ) and other parameters.
"""
function update_parameter(params::WilsonCowanParameters{T,P}, param_name::Symbol, value) where {T,P}
    # Map parameter names to connectivity matrix indices
    # For a 2-population model: E=1, I=2
    # connectivity[i,j] maps j → i
    param_map = Dict(
        :bₑₑ => (1, 1),  # E → E
        :bᵢₑ => (1, 2),  # I → E (inhibitory, so use negative)
        :bₑᵢ => (2, 1),  # E → I
        :bᵢᵢ => (2, 2),  # I → I (inhibitory, so use negative)
        :aₑₑ => (1, 1),  # length scale E → E
        :aᵢₑ => (1, 2),  # length scale I → E
        :aₑᵢ => (2, 1),  # length scale E → I
        :aᵢᵢ => (2, 2),  # length scale I → I
        :vₑ => 1,        # sigmoid steepness for E
        :vᵢ => 2,        # sigmoid steepness for I
        :θₑ => 1,        # sigmoid threshold for E
        :θᵢ => 2         # sigmoid threshold for I
    )
    
    if haskey(param_map, param_name)
        if param_name in [:bₑₑ, :bᵢₑ, :bₑᵢ, :bᵢᵢ, :aₑₑ, :aᵢₑ, :aₑᵢ, :aᵢᵢ]
            # Update connectivity parameter
            i, j = param_map[param_name]
            new_conn = update_connectivity(params.connectivity, i, j, param_name, value)
            return WilsonCowanParameters{T,P}(
                params.α, params.β, params.τ, new_conn, params.nonlinearity,
                params.stimulus, params.lattice, params.pop_names
            )
        elseif param_name in [:vₑ, :vᵢ, :θₑ, :θᵢ]
            # Update nonlinearity parameter
            pop_idx = param_map[param_name]
            new_nonlin = update_nonlinearity(params.nonlinearity, pop_idx, param_name, value)
            return WilsonCowanParameters{T,P}(
                params.α, params.β, params.τ, params.connectivity, new_nonlin,
                params.stimulus, params.lattice, params.pop_names
            )
        end
    elseif param_name == :τ
        # Update time constant (assuming same for both populations for simplicity)
        new_τ = ntuple(i -> value, P)
        return WilsonCowanParameters{T,P}(
            params.α, params.β, new_τ, params.connectivity, params.nonlinearity,
            params.stimulus, params.lattice, params.pop_names
        )
    end
    
    error("Parameter $param_name not recognized or not supported for updating")
end

"""
    update_connectivity(connectivity::ConnectivityMatrix{P}, i, j, param_name, value) where P

Update a specific connectivity parameter in the connectivity matrix.
"""
function update_connectivity(connectivity::ConnectivityMatrix{P}, i, j, param_name, value) where P
    # Get the current connectivity kernel
    old_kernel = connectivity.matrix[i, j]
    
    # Create new kernel based on parameter type
    if param_name in [:bₑₑ, :bₑᵢ]
        # Excitatory connection (positive)
        if old_kernel isa ScalarConnectivity
            new_kernel = ScalarConnectivity(value)
        else
            # GaussianConnectivityParameter
            new_kernel = GaussianConnectivityParameter(value, old_kernel.spread)
        end
    elseif param_name in [:bᵢₑ, :bᵢᵢ]
        # Inhibitory connection (negative)
        if old_kernel isa ScalarConnectivity
            new_kernel = ScalarConnectivity(-value)
        else
            # GaussianConnectivityParameter
            new_kernel = GaussianConnectivityParameter(-value, old_kernel.spread)
        end
    elseif param_name in [:aₑₑ, :aᵢₑ, :aₑᵢ, :aᵢᵢ]
        # Length scale parameter (spread)
        if old_kernel isa ScalarConnectivity
            error("Cannot update length scale for ScalarConnectivity")
        else
            # Keep amplitude, update spread
            new_kernel = GaussianConnectivityParameter(old_kernel.amplitude, (value,))
        end
    else
        error("Unknown connectivity parameter: $param_name")
    end
    
    # Create new connectivity matrix with updated kernel
    new_matrix = copy(connectivity.matrix)
    new_matrix[i, j] = new_kernel
    
    return ConnectivityMatrix{P}(new_matrix)
end

"""
    update_nonlinearity(nonlinearity, pop_idx, param_name, value)

Update a specific nonlinearity parameter.
"""
function update_nonlinearity(nonlinearity, pop_idx, param_name, value)
    # Nonlinearity is a tuple of SigmoidNonlinearity objects
    nonlin_array = collect(nonlinearity)
    old_nonlin = nonlin_array[pop_idx]
    
    if param_name in [:vₑ, :vᵢ]
        # Update steepness (a parameter)
        new_nonlin = SigmoidNonlinearity(a=value, θ=old_nonlin.θ)
    elseif param_name in [:θₑ, :θᵢ]
        # Update threshold
        new_nonlin = SigmoidNonlinearity(a=old_nonlin.a, θ=value)
    else
        error("Unknown nonlinearity parameter: $param_name")
    end
    
    nonlin_array[pop_idx] = new_nonlin
    return tuple(nonlin_array...)
end
