"""
Traveling wave analysis and metrics for neural field simulations.

This module provides functions to detect and measure properties of traveling waves
in spatiotemporal activity patterns from Wilson-Cowan model simulations.
"""

using Statistics
using LinearAlgebra

"""
    detect_traveling_peak(sol, pop_idx=1; threshold=0.1, min_distance=2)

Detect if a traveling peak exists in the spatiotemporal activity pattern.

A traveling peak is detected by tracking the location of maximum activity over time
and checking if it moves consistently in space.

# Arguments
- `sol`: ODE solution object from `solve_model()`
- `pop_idx`: Population index to analyze (default: 1)
- `threshold`: Minimum activity threshold to consider as a peak (default: 0.1)
- `min_distance`: Minimum spatial distance the peak must travel to be considered traveling (default: 2)

# Returns
- `has_traveling_peak::Bool`: True if a traveling peak is detected
- `peak_trajectory::Vector{Int}`: Indices of peak locations over time
- `peak_times::Vector{Float64}`: Time points where peaks are detected

# Examples
```julia
sol = solve_model(A₀, tspan, params, saveat=0.1)
has_peak, trajectory, times = detect_traveling_peak(sol, 1, threshold=0.15)
```
"""
function detect_traveling_peak(sol, pop_idx=1; threshold=0.1, min_distance=2)
    # Extract spatial dimension
    first_state = sol.u[1]
    if ndims(first_state) == 1
        error("detect_traveling_peak requires spatial model (2D state array)")
    end
    
    n_points, n_pops = size(first_state)
    if n_points == 1
        error("detect_traveling_peak requires spatial model with multiple points")
    end
    
    # Track peak locations over time
    peak_trajectory = Int[]
    peak_times = Float64[]
    
    for (t_idx, (t, state)) in enumerate(zip(sol.t, sol.u))
        activity = state[:, pop_idx]
        max_val = maximum(activity)
        
        # Only track if above threshold
        if max_val >= threshold
            max_idx = argmax(activity)
            push!(peak_trajectory, max_idx)
            push!(peak_times, t)
        end
    end
    
    # Check if peak traveled significant distance
    has_traveling_peak = false
    if length(peak_trajectory) > 1
        # Calculate total distance traveled
        distance = abs(peak_trajectory[end] - peak_trajectory[1])
        has_traveling_peak = (distance >= min_distance)
    end
    
    return has_traveling_peak, peak_trajectory, peak_times
end

"""
    compute_decay_rate(sol, pop_idx=1; fit_window=nothing)

Compute the decay rate of activity over time.

The decay rate is estimated by fitting an exponential decay to the maximum activity
over time: A(t) ≈ A₀ * exp(-λt), where λ is the decay rate.

# Arguments
- `sol`: ODE solution object from `solve_model()`
- `pop_idx`: Population index to analyze (default: 1)
- `fit_window`: Time window (t_start, t_end) for fitting. If nothing, uses entire simulation.

# Returns
- `decay_rate::Union{Float64, Nothing}`: Decay rate λ, or nothing if no decay is detected
- `amplitude_over_time::Vector{Float64}`: Maximum activity at each time point

# Examples
```julia
sol = solve_model(A₀, tspan, params, saveat=0.1)
decay_rate, amplitudes = compute_decay_rate(sol, 1)
```
"""
function compute_decay_rate(sol, pop_idx=1; fit_window=nothing)
    first_state = sol.u[1]
    
    # Extract maximum activity over time
    amplitude_over_time = Float64[]
    times = Float64[]
    
    for (t, state) in zip(sol.t, sol.u)
        # Skip times outside fit window if specified
        if fit_window !== nothing
            if t < fit_window[1] || t > fit_window[2]
                continue
            end
        end
        
        # Get activity for this population
        if ndims(state) == 1
            activity = state[pop_idx]
        elseif size(state, 1) == 1
            # Point model with connectivity
            activity = state[1, pop_idx]
        else
            # Spatial model - use maximum
            activity = maximum(state[:, pop_idx])
        end
        
        push!(amplitude_over_time, activity)
        push!(times, t)
    end
    
    # Check if we have enough data and if there's decay
    if length(amplitude_over_time) < 3
        return nothing, amplitude_over_time
    end
    
    # Simple decay detection: check if amplitude generally decreases
    if amplitude_over_time[end] >= amplitude_over_time[1] * 0.9
        # No significant decay
        return nothing, amplitude_over_time
    end
    
    # Estimate decay rate using log-linear fit
    # A(t) = A₀ * exp(-λt) => log(A(t)) = log(A₀) - λt
    # Avoid log of zero or negative values
    valid_idx = findall(x -> x > 0, amplitude_over_time)
    if length(valid_idx) < 2
        return nothing, amplitude_over_time
    end
    
    valid_times = times[valid_idx]
    valid_amplitudes = amplitude_over_time[valid_idx]
    log_amplitudes = log.(valid_amplitudes)
    
    # Linear regression: log(A) = a + b*t, where b = -λ
    n = length(valid_times)
    mean_t = mean(valid_times)
    mean_log_A = mean(log_amplitudes)
    
    numerator = sum((valid_times .- mean_t) .* (log_amplitudes .- mean_log_A))
    denominator = sum((valid_times .- mean_t) .^ 2)
    
    if abs(denominator) < 1e-10
        return nothing, amplitude_over_time
    end
    
    slope = numerator / denominator
    decay_rate = -slope  # Since we want positive decay rate
    
    # Return positive decay rate only if it's significant
    if decay_rate > 1e-6
        return decay_rate, amplitude_over_time
    else
        return nothing, amplitude_over_time
    end
end

"""
    compute_amplitude(sol, pop_idx=1; method=:max)

Compute the amplitude of activity in the simulation.

# Arguments
- `sol`: ODE solution object from `solve_model()`
- `pop_idx`: Population index to analyze (default: 1)
- `method`: Method for computing amplitude
  - `:max` - Maximum activity across all space and time (default)
  - `:peak` - Maximum of the peak values over time
  - `:mean_max` - Mean of maximum values over time

# Returns
- `amplitude::Float64`: Computed amplitude value

# Examples
```julia
sol = solve_model(A₀, tspan, params, saveat=0.1)
amp_max = compute_amplitude(sol, 1, method=:max)
amp_peak = compute_amplitude(sol, 1, method=:peak)
```
"""
function compute_amplitude(sol, pop_idx=1; method=:max)
    first_state = sol.u[1]
    
    # Collect all activity values
    max_values = Float64[]
    
    for state in sol.u
        if ndims(state) == 1
            # Point model without connectivity
            push!(max_values, state[pop_idx])
        elseif size(state, 1) == 1
            # Point model with connectivity
            push!(max_values, state[1, pop_idx])
        else
            # Spatial model - get maximum across space
            push!(max_values, maximum(state[:, pop_idx]))
        end
    end
    
    if method == :max
        return maximum(max_values)
    elseif method == :peak
        return maximum(max_values)
    elseif method == :mean_max
        return mean(max_values)
    else
        error("Unknown method: $method. Use :max, :peak, or :mean_max")
    end
end

"""
    compute_distance_traveled(sol, pop_idx=1, lattice=nothing; threshold=0.1)

Compute the spatial distance traveled by the activity peak.

# Arguments
- `sol`: ODE solution object from `solve_model()`
- `pop_idx`: Population index to analyze (default: 1)
- `lattice`: Lattice object to convert indices to physical distances (optional)
- `threshold`: Minimum activity threshold to track peak (default: 0.1)

# Returns
- `distance::Float64`: Distance traveled by the peak (in lattice units if no lattice provided, physical units otherwise)
- `peak_trajectory::Vector{Int}`: Indices of peak locations over time

# Examples
```julia
lattice = CompactLattice(extent=(10.0,), n_points=(101,))
sol = solve_model(A₀, tspan, params, saveat=0.1)
distance, trajectory = compute_distance_traveled(sol, 1, lattice, threshold=0.15)
```
"""
function compute_distance_traveled(sol, pop_idx=1, lattice=nothing; threshold=0.1)
    first_state = sol.u[1]
    if ndims(first_state) == 1 || size(first_state, 1) == 1
        error("compute_distance_traveled requires spatial model")
    end
    
    # Track peak locations
    peak_trajectory = Int[]
    
    for state in sol.u
        activity = state[:, pop_idx]
        max_val = maximum(activity)
        
        if max_val >= threshold
            max_idx = argmax(activity)
            push!(peak_trajectory, max_idx)
        end
    end
    
    if length(peak_trajectory) < 2
        return 0.0, peak_trajectory
    end
    
    # Calculate distance in index space
    idx_distance = abs(peak_trajectory[end] - peak_trajectory[1])
    
    # Convert to physical distance if lattice is provided
    if lattice !== nothing
        # For 1D lattice, get the extent and number of points
        ext = extent(lattice)
        n_pts = size(first_state, 1)
        
        # Physical distance per index unit
        if length(ext) == 1
            spatial_scale = ext[1] / (n_pts - 1)
            physical_distance = idx_distance * spatial_scale
            return physical_distance, peak_trajectory
        else
            # Multi-dimensional: use index distance for now
            return Float64(idx_distance), peak_trajectory
        end
    else
        return Float64(idx_distance), peak_trajectory
    end
end

"""
    compute_half_max_width(sol, pop_idx=1, time_idx=nothing, lattice=nothing; baseline=:min)

Compute the half-maximum half-width (HMHW) of the activity profile.

The HMHW is a measure of the spatial extent of activation, calculated as the
width of the region where activity exceeds half of the maximum value.

# Arguments
- `sol`: ODE solution object from `solve_model()`
- `pop_idx`: Population index to analyze (default: 1)
- `time_idx`: Time index to analyze. If nothing, uses time of maximum activity (default: nothing)
- `lattice`: Lattice object to convert to physical units (optional)
- `baseline`: Method for baseline calculation
  - `:min` - Use minimum value as baseline (default)
  - `:zero` - Use zero as baseline
  - number - Use specific value as baseline

# Returns
- `width::Float64`: Half-maximum half-width (in lattice units or physical units)
- `half_max_level::Float64`: The half-maximum threshold used
- `profile::Vector{Float64}`: Activity profile at the analyzed time

# Examples
```julia
lattice = CompactLattice(extent=(10.0,), n_points=(101,))
sol = solve_model(A₀, tspan, params, saveat=0.1)
width, half_max, profile = compute_half_max_width(sol, 1, nothing, lattice)
```
"""
function compute_half_max_width(sol, pop_idx=1, time_idx=nothing, lattice=nothing; baseline=:min)
    first_state = sol.u[1]
    if ndims(first_state) == 1 || size(first_state, 1) == 1
        error("compute_half_max_width requires spatial model")
    end
    
    # Find time index with maximum activity if not specified
    if time_idx === nothing
        max_activity = -Inf
        time_idx = 1
        for (idx, state) in enumerate(sol.u)
            activity = maximum(state[:, pop_idx])
            if activity > max_activity
                max_activity = activity
                time_idx = idx
            end
        end
    end
    
    # Get activity profile at selected time
    profile = sol.u[time_idx][:, pop_idx]
    
    # Calculate baseline
    if baseline == :min
        baseline_val = minimum(profile)
    elseif baseline == :zero
        baseline_val = 0.0
    else
        baseline_val = Float64(baseline)
    end
    
    # Calculate half-maximum level
    max_val = maximum(profile)
    half_max_level = baseline_val + (max_val - baseline_val) / 2
    
    # Find regions above half-max
    above_half_max = profile .>= half_max_level
    
    if !any(above_half_max)
        return 0.0, half_max_level, profile
    end
    
    # Find contiguous regions above half-max
    # Get indices where activity is above half-max
    above_indices = findall(above_half_max)
    
    if isempty(above_indices)
        return 0.0, half_max_level, profile
    end
    
    # Calculate width as span of indices above half-max
    width_indices = maximum(above_indices) - minimum(above_indices) + 1
    
    # Convert to physical distance if lattice is provided
    if lattice !== nothing
        ext = extent(lattice)
        n_pts = length(profile)
        
        if length(ext) == 1
            spatial_scale = ext[1] / (n_pts - 1)
            physical_width = (width_indices - 1) * spatial_scale
            return physical_width, half_max_level, profile
        else
            return Float64(width_indices), half_max_level, profile
        end
    else
        return Float64(width_indices), half_max_level, profile
    end
end
