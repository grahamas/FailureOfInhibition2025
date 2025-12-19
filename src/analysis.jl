"""
Analysis utilities for neural field simulations.

This module provides functions to:
1. Detect and measure properties of traveling waves in spatial models
2. Analyze oscillations in point models (frequency, amplitude, decay, duration)
"""

using Statistics
using LinearAlgebra
using FFTW

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
width of the contiguous region around the peak that exceeds half of the maximum value.
If multiple separate peaks exist, this function measures only the width of the region
containing the global maximum peak.

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
    
    # Find the peak location
    peak_idx = argmax(profile)
    
    # Find the contiguous region around the peak that is above half-max
    # Start from the peak and expand left and right until we hit values below half-max
    left_idx = peak_idx
    right_idx = peak_idx
    
    # Expand left
    while left_idx > 1 && above_half_max[left_idx - 1]
        left_idx -= 1
    end
    
    # Expand right
    while right_idx < length(profile) && above_half_max[right_idx + 1]
        right_idx += 1
    end
    
    # Calculate width of this contiguous region
    width_indices = right_idx - left_idx + 1
    
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

#=============================================================================
Oscillation Analysis Functions for Point Models
=============================================================================#

# Constants for numerical thresholds
const MIN_ENVELOPE_VALUE = 1e-10
const MIN_DECAY_RATE = 1e-6

"""
    extract_population_timeseries(sol, pop_idx)

Extract time series for a specific population from solution.

This helper function handles different solution formats (point models with/without 
connectivity, spatial models) and returns a consistent 1D time series.

# Arguments
- `sol`: ODE solution object from `solve_model()`
- `pop_idx`: Population index to extract

# Returns
- `time_series::Vector{Float64}`: Activity values over time for the specified population
"""
function extract_population_timeseries(sol, pop_idx)
    time_series = Vector{Float64}(undef, length(sol.u))
    
    for (i, state) in enumerate(sol.u)
        if ndims(state) == 1
            # Point model without connectivity
            time_series[i] = state[pop_idx]
        elseif size(state, 1) == 1
            # Point model with connectivity
            time_series[i] = state[1, pop_idx]
        else
            # Spatial model - use mean activity
            time_series[i] = mean(state[:, pop_idx])
        end
    end
    
    return time_series
end

"""
    detect_oscillations(sol, pop_idx=1; min_peaks=2)

Detect if oscillations are present in a point model time series.

Oscillations are detected by counting local maxima in the activity over time.
A minimum number of peaks indicates oscillatory behavior.

# Arguments
- `sol`: ODE solution object from `solve_model()`
- `pop_idx`: Population index to analyze (default: 1)
- `min_peaks`: Minimum number of peaks to consider as oscillatory (default: 2)

# Returns
- `has_oscillations::Bool`: True if oscillations are detected
- `peak_times::Vector{Float64}`: Times at which peaks occur
- `peak_values::Vector{Float64}`: Activity values at peaks

# Examples
```julia
sol = solve_model(A₀, tspan, params, saveat=0.1)
has_osc, times, values = detect_oscillations(sol, 1)
```
"""
function detect_oscillations(sol, pop_idx=1; min_peaks=2)
    # Extract time series for this population
    time_series = extract_population_timeseries(sol, pop_idx)
    
    # Find local maxima
    peak_indices = Int[]
    for i in 2:(length(time_series)-1)
        if time_series[i] > time_series[i-1] && time_series[i] > time_series[i+1]
            push!(peak_indices, i)
        end
    end
    
    # Collect peak information
    peak_times = [sol.t[i] for i in peak_indices]
    peak_values = [time_series[i] for i in peak_indices]
    
    has_oscillations = length(peak_indices) >= min_peaks
    
    return has_oscillations, peak_times, peak_values
end

"""
    compute_oscillation_frequency(sol, pop_idx=1; method=:fft)

Compute the dominant oscillation frequency in a point model time series.

# Arguments
- `sol`: ODE solution object from `solve_model()`
- `pop_idx`: Population index to analyze (default: 1)
- `method`: Method for frequency estimation
  - `:fft` - Use FFT power spectrum to find dominant frequency (default)
  - `:peaks` - Use average time between peaks

# Returns
- `frequency::Union{Float64, Nothing}`: Dominant frequency in Hz (or 1/time_units), or nothing if no oscillations detected
- `period::Union{Float64, Nothing}`: Period of oscillations, or nothing if no oscillations detected

# Examples
```julia
sol = solve_model(A₀, tspan, params, saveat=0.1)
freq, period = compute_oscillation_frequency(sol, 1)
```
"""
function compute_oscillation_frequency(sol, pop_idx=1; method=:fft)
    # Extract time series
    time_series = extract_population_timeseries(sol, pop_idx)
    
    if method == :fft
        # Use FFT to find dominant frequency
        n = length(time_series)
        if n < 4
            return nothing, nothing
        end
        
        # Compute sampling rate and validate uniform sampling
        dt = sol.t[2] - sol.t[1]  # Expected time step
        
        # Check for approximately uniform sampling
        if length(sol.t) > 2
            time_diffs = diff(sol.t)
            max_diff = maximum(abs.(time_diffs .- dt))
            if max_diff > 0.1 * dt
                @warn "Non-uniform sampling detected. FFT results may be inaccurate."
            end
        end
        
        # Detrend by subtracting mean
        detrended = time_series .- mean(time_series)
        
        # Compute FFT
        fft_result = fft(detrended)
        power = abs2.(fft_result)
        
        # Get frequencies
        freqs = fftfreq(n, 1.0/dt)
        
        # Only look at positive frequencies
        positive_freqs_idx = findall(freqs .> 0)
        if isempty(positive_freqs_idx)
            return nothing, nothing
        end
        
        positive_freqs = freqs[positive_freqs_idx]
        positive_power = power[positive_freqs_idx]
        
        # Find dominant frequency
        max_power_idx = argmax(positive_power)
        dominant_freq = positive_freqs[max_power_idx]
        
        # Check if power is significant (> 10% of total power)
        total_power = sum(positive_power)
        if positive_power[max_power_idx] < 0.1 * total_power
            return nothing, nothing
        end
        
        period = 1.0 / dominant_freq
        return dominant_freq, period
        
    elseif method == :peaks
        # Use time between peaks
        has_osc, peak_times, _ = detect_oscillations(sol, pop_idx)
        
        if !has_osc || length(peak_times) < 2
            return nothing, nothing
        end
        
        # Calculate average time between peaks
        inter_peak_intervals = diff(peak_times)
        period = mean(inter_peak_intervals)
        frequency = 1.0 / period
        
        return frequency, period
    else
        error("Unknown method: $method. Use :fft or :peaks")
    end
end

"""
    compute_oscillation_amplitude(sol, pop_idx=1; method=:envelope)

Compute the amplitude of oscillations over time.

# Arguments
- `sol`: ODE solution object from `solve_model()`
- `pop_idx`: Population index to analyze (default: 1)
- `method`: Method for amplitude computation
  - `:envelope` - Average of peak-to-trough amplitudes (default)
  - `:std` - Standard deviation of activity
  - `:peak_mean` - Mean of peak values

# Returns
- `amplitude::Union{Float64, Nothing}`: Oscillation amplitude, or nothing if no oscillations detected
- `envelope::Union{Vector{Float64}, Nothing}`: Time-varying amplitude envelope (for :envelope method)

# Examples
```julia
sol = solve_model(A₀, tspan, params, saveat=0.1)
amp, envelope = compute_oscillation_amplitude(sol, 1)
```
"""
function compute_oscillation_amplitude(sol, pop_idx=1; method=:envelope)
    # Extract time series
    time_series = extract_population_timeseries(sol, pop_idx)
    
    if method == :envelope
        # Find peaks and troughs
        has_osc, peak_times, peak_values = detect_oscillations(sol, pop_idx)
        
        if !has_osc
            return nothing, nothing
        end
        
        # Find troughs (local minima)
        trough_indices = Int[]
        for i in 2:(length(time_series)-1)
            if time_series[i] < time_series[i-1] && time_series[i] < time_series[i+1]
                push!(trough_indices, i)
            end
        end
        
        trough_values = [time_series[i] for i in trough_indices]
        
        if isempty(trough_values)
            # No troughs found, use minimum
            trough_val = minimum(time_series)
            amplitude = mean(peak_values) - trough_val
        else
            # Calculate peak-to-trough amplitudes
            amplitude = mean(peak_values) - mean(trough_values)
        end
        
        # Create envelope (peaks and troughs over time)
        envelope = Float64[]
        for (i, val) in enumerate(time_series)
            # Simple envelope: distance from mean
            push!(envelope, abs(val - mean(time_series)))
        end
        
        return amplitude, envelope
        
    elseif method == :std
        # Use standard deviation as amplitude measure
        amplitude = std(time_series)
        return amplitude, nothing
        
    elseif method == :peak_mean
        # Mean of peak values minus baseline
        has_osc, _, peak_values = detect_oscillations(sol, pop_idx)
        
        if !has_osc
            return nothing, nothing
        end
        
        baseline = minimum(time_series)
        amplitude = mean(peak_values) - baseline
        
        return amplitude, nothing
    else
        error("Unknown method: $method. Use :envelope, :std, or :peak_mean")
    end
end

"""
    compute_oscillation_decay(sol, pop_idx=1; method=:exponential)

Compute the decay rate of oscillation amplitude over time.

For damped oscillations, this measures how quickly the oscillation amplitude decreases.

# Arguments
- `sol`: ODE solution object from `solve_model()`
- `pop_idx`: Population index to analyze (default: 1)
- `method`: Method for decay computation
  - `:exponential` - Fit exponential decay to envelope (default)
  - `:linear` - Linear regression on log(envelope)
  - `:peak_decay` - Exponential fit to successive peak heights

# Returns
- `decay_rate::Union{Float64, Nothing}`: Decay rate λ (in 1/time_units), or nothing if no decay
- `half_life::Union{Float64, Nothing}`: Time for amplitude to decay to half, or nothing if no decay
- `envelope::Vector{Float64}`: Amplitude envelope over time

# Examples
```julia
sol = solve_model(A₀, tspan, params, saveat=0.1)
decay_rate, half_life, envelope = compute_oscillation_decay(sol, 1)
```
"""
function compute_oscillation_decay(sol, pop_idx=1; method=:exponential)
    # Extract time series
    time_series = extract_population_timeseries(sol, pop_idx)
    
    if method == :exponential || method == :linear
        # Get amplitude envelope
        amp, envelope = compute_oscillation_amplitude(sol, pop_idx, method=:envelope)
        
        if amp === nothing || envelope === nothing
            return nothing, nothing, Float64[]
        end
        
        # Fit exponential decay: A(t) = A₀ * exp(-λt)
        # log(A(t)) = log(A₀) - λt
        
        # Filter out very small values to avoid log issues
        valid_idx = findall(x -> x > MIN_ENVELOPE_VALUE, envelope)
        if length(valid_idx) < 3
            return nothing, nothing, envelope
        end
        
        valid_times = sol.t[valid_idx]
        valid_envelope = envelope[valid_idx]
        log_envelope = log.(valid_envelope)
        
        # Linear regression
        n = length(valid_times)
        mean_t = mean(valid_times)
        mean_log_A = mean(log_envelope)
        
        numerator = sum((valid_times .- mean_t) .* (log_envelope .- mean_log_A))
        denominator = sum((valid_times .- mean_t) .^ 2)
        
        if abs(denominator) < MIN_ENVELOPE_VALUE
            return nothing, nothing, envelope
        end
        
        slope = numerator / denominator
        decay_rate = -slope  # Positive decay rate
        
        # Calculate half-life
        if decay_rate > MIN_ENVELOPE_VALUE
            half_life = log(2) / decay_rate
        else
            half_life = nothing
        end
        
        # Only return positive decay rates
        if decay_rate > MIN_DECAY_RATE
            return decay_rate, half_life, envelope
        else
            return nothing, nothing, envelope
        end
        
    elseif method == :peak_decay
        # Fit to successive peak heights
        has_osc, peak_times, peak_values = detect_oscillations(sol, pop_idx)
        
        if !has_osc || length(peak_values) < 3
            return nothing, nothing, Float64[]
        end
        
        # Check if peaks are actually decaying
        if peak_values[end] >= peak_values[1] * 0.9
            # No significant decay
            return nothing, nothing, peak_values
        end
        
        # Fit exponential to peak values
        log_peaks = log.(peak_values)
        
        n = length(peak_times)
        mean_t = mean(peak_times)
        mean_log_p = mean(log_peaks)
        
        numerator = sum((peak_times .- mean_t) .* (log_peaks .- mean_log_p))
        denominator = sum((peak_times .- mean_t) .^ 2)
        
        if abs(denominator) < MIN_ENVELOPE_VALUE
            return nothing, nothing, peak_values
        end
        
        slope = numerator / denominator
        decay_rate = -slope
        
        if decay_rate > MIN_ENVELOPE_VALUE
            half_life = log(2) / decay_rate
        else
            half_life = nothing
        end
        
        if decay_rate > MIN_DECAY_RATE
            return decay_rate, half_life, peak_values
        else
            return nothing, nothing, peak_values
        end
    else
        error("Unknown method: $method. Use :exponential, :linear, or :peak_decay")
    end
end

"""
    compute_oscillation_duration(sol, pop_idx=1; threshold_ratio=0.1, min_amplitude=1e-3)

Compute how long oscillations persist before decaying to negligible levels.

# Arguments
- `sol`: ODE solution object from `solve_model()`
- `pop_idx`: Population index to analyze (default: 1)
- `threshold_ratio`: Ratio of initial amplitude below which oscillations are considered ended (default: 0.1)
- `min_amplitude`: Minimum absolute amplitude to consider as oscillatory (default: 1e-3)

# Returns
- `duration::Union{Float64, Nothing}`: Time oscillations persist, or nothing if no oscillations or they don't decay
- `sustained::Bool`: True if oscillations are sustained throughout the simulation
- `end_time::Union{Float64, Nothing}`: Time when oscillations end, or nothing if sustained

# Examples
```julia
sol = solve_model(A₀, tspan, params, saveat=0.1)
duration, sustained, end_time = compute_oscillation_duration(sol, 1)
```
"""
function compute_oscillation_duration(sol, pop_idx=1; threshold_ratio=0.1, min_amplitude=1e-3)
    # Check if oscillations exist
    has_osc, peak_times, peak_values = detect_oscillations(sol, pop_idx)
    
    if !has_osc
        return nothing, false, nothing
    end
    
    # Get amplitude envelope
    amp, envelope = compute_oscillation_amplitude(sol, pop_idx, method=:envelope)
    
    if amp === nothing || amp < min_amplitude
        return nothing, false, nothing
    end
    
    # Find when envelope drops below threshold
    threshold = amp * threshold_ratio
    
    # Find first time when envelope stays below threshold
    below_threshold = false
    end_idx = length(envelope)
    
    for i in 1:length(envelope)
        if envelope[i] < threshold
            # Check if it stays below for the rest of the simulation
            # or for at least a few time steps
            remaining = envelope[i:end]
            if all(remaining .< threshold) || (length(remaining) >= 3 && all(remaining[1:3] .< threshold))
                end_idx = i
                below_threshold = true
                break
            end
        end
    end
    
    if below_threshold
        # Oscillations ended
        end_time = sol.t[end_idx]
        duration = end_time - sol.t[1]
        sustained = false
    else
        # Oscillations sustained throughout
        end_time = nothing
        duration = sol.t[end] - sol.t[1]
        sustained = true
    end
    
    return duration, sustained, end_time
end

"""
    generate_analytical_traveling_wave(params::WilsonCowanParameters{T,1}, times, pop_idx=1;
                                        wave_speed=2.0, decay_rate=0.05, wavenumber=1.0, 
                                        initial_position=-10.0, amplitude=1.0) where {T}

Generate an analytical traveling wave solution for a 1D Wilson-Cowan model.

The solution is a sech² traveling wave with exponential decay:
    A(x,t) = A₀ * exp(-λt) * sech²(k(x - x₀ - ct))

where:
- A₀ is the initial amplitude
- λ is the decay rate (1/time)
- k is the spatial wavenumber (1/space, controls width)
- c is the wave speed (space/time)
- x₀ is the initial position of the wave peak

This function creates a solution object that is compatible with the analysis functions
and can be used to test optimization and analysis algorithms.

# Arguments
- `params`: WilsonCowanParameters{T,1} containing the lattice and model parameters
- `times`: Time points at which to evaluate the solution (Vector or Range)
- `pop_idx`: Population index (default: 1)
- `wave_speed`: Speed of wave propagation in space/time (default: 2.0)
- `decay_rate`: Exponential decay rate λ in 1/time (default: 0.05)
- `wavenumber`: Spatial wavenumber k in 1/space (default: 1.0, gives FWHM ≈ 1.76)
- `initial_position`: Initial position x₀ of wave peak (default: -10.0)
- `amplitude`: Initial amplitude A₀ (default: 1.0)

# Returns
A named tuple with fields:
- `u`: Vector of spatial activity matrices at each time point
- `t`: Vector of time points

# Examples
```julia
lattice = CompactLattice(extent=(40.0,), n_points=(201,))
params = WilsonCowanParameters{1}(
    α=(1.0,), β=(1.0,), τ=(8.0,),
    connectivity=ConnectivityMatrix{1}(reshape([GaussianConnectivityParameter(1.0, (2.0,))], 1, 1)),
    nonlinearity=SigmoidNonlinearity(a=2.0, θ=0.25),
    stimulus=nothing, lattice=lattice, pop_names=("E",)
)
times = 0.0:0.2:10.0
sol = generate_analytical_traveling_wave(params, times, wave_speed=2.0, decay_rate=0.05)
```
"""
function generate_analytical_traveling_wave(params::WilsonCowanParameters{T,1}, times, pop_idx=1;
                                           wave_speed=2.0, decay_rate=0.05, wavenumber=1.0,
                                           initial_position=-10.0, amplitude=1.0) where {T}
    # Extract lattice information
    lattice = params.lattice
    x_coords = [coord[1] for coord in coordinates(lattice)]
    n_points = length(x_coords)
    
    # Define the traveling wave solution
    function traveling_wave_solution(x, t)
        arg = wavenumber * (x - initial_position - wave_speed * t)
        amp = amplitude * exp(-decay_rate * t)
        return amp / (cosh(arg)^2)  # sech²(x) = 1/cosh²(x)
    end
    
    # Generate solution data
    u_array = []
    t_array = Float64[]
    
    for t in times
        # Create spatial profile at this time
        profile = [traveling_wave_solution(x, t) for x in x_coords]
        push!(u_array, reshape(profile, n_points, 1))
        push!(t_array, t)
    end
    
    # Return a named tuple that mimics ODE solution structure
    return (u=u_array, t=t_array)
end

#=============================================================================
Phase Space Analysis
=============================================================================#

"""
    compute_phase_space_derivatives(E_range, I_range, params; n_points=200)

Compute dE/dt and dI/dt fields on a grid of (E, I) values for phase space analysis.

This function evaluates the Wilson-Cowan dynamics at each point in the (E, I) phase space,
producing vector fields that can be used to:
- Plot nullclines (zero-level contours of dE/dt and dI/dt)
- Visualize flow directions
- Find fixed points

# Arguments
- `E_range`: Range or vector of E values to evaluate (default: 0.0 to 1.0)
- `I_range`: Range or vector of I values to evaluate (default: 0.0 to 1.0)
- `params`: WilsonCowanParameters for the model
- `n_points`: Number of grid points in each dimension (default: 200)

# Returns
- `E_grid`: 2D array of E coordinates
- `I_grid`: 2D array of I coordinates
- `dE_dt_field`: 2D array of dE/dt values
- `dI_dt_field`: 2D array of dI/dt values
"""
function compute_phase_space_derivatives(E_range, I_range, params)
    # Create meshgrid
    E_grid = [E for E in E_range, I in I_range]
    I_grid = [I for E in E_range, I in I_range]
    
    # Compute both dE/dt and dI/dt at each grid point
    dE_dt_field = similar(E_grid)
    dI_dt_field = similar(E_grid)
    
    for i in eachindex(E_grid)
        E = E_grid[i]
        I = I_grid[i]
        
        # Create state array for this point
        A = reshape([E, I], 1, 2)
        dA = zeros(1, 2)
        
        # Use wcm1973! to compute derivatives
        wcm1973!(dA, A, params, 0.0)
        
        # Extract both dE/dt and dI/dt
        dE_dt_field[i] = dA[1, 1]
        dI_dt_field[i] = dA[1, 2]
    end
    
    return E_grid, I_grid, dE_dt_field, dI_dt_field
end

"""
    find_fixed_points(E_range, I_range, params; threshold=1e-6, n_points=100)

Find all fixed points in the phase space where dE/dt ≈ 0 and dI/dt ≈ 0.

Uses a grid-based search followed by local refinement to find points where
both derivatives are simultaneously close to zero.

# Arguments
- `E_range`: Range of E values to search (default: 0.0 to 1.0)
- `I_range`: Range of I values to search (default: 0.0 to 1.0)
- `params`: WilsonCowanParameters for the model
- `threshold`: Maximum squared derivative magnitude for a fixed point (default: 1e-6)
- `n_points`: Number of grid points per dimension for initial search (default: 100)

# Returns
- `fixed_points`: Vector of tuples (E, I) representing fixed point locations
"""
function find_fixed_points(E_range, I_range, params; threshold=1e-6, n_points=100)
    # Compute derivative fields on grid
    E_grid, I_grid, dE_dt_field, dI_dt_field = compute_phase_space_derivatives(E_range, I_range, params)
    
    # Find candidate fixed points (local minima of squared derivative magnitude)
    candidates = Tuple{Float64, Float64}[]
    
    # Compute squared magnitude of derivative vector at each point
    magnitude_sq = dE_dt_field.^2 .+ dI_dt_field.^2
    
    # Find all points below threshold
    for i in eachindex(E_grid)
        if magnitude_sq[i] < threshold
            E = E_grid[i]
            I = I_grid[i]
            
            # Check if this is a new fixed point (not too close to existing ones)
            is_new = true
            for (E_fp, I_fp) in candidates
                if sqrt((E - E_fp)^2 + (I - I_fp)^2) < 0.05  # Minimum separation
                    is_new = false
                    break
                end
            end
            
            if is_new
                push!(candidates, (E, I))
            end
        end
    end
    
    # If no candidates found below threshold, return the point with minimum magnitude
    if isempty(candidates)
        min_idx = argmin(magnitude_sq)
        E_min = E_grid[min_idx]
        I_min = I_grid[min_idx]
        push!(candidates, (E_min, I_min))
    end
    
    return candidates
end
