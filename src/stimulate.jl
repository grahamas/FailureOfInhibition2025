"""
    euclidean_distance(coord1::NTuple{N,T}, coord2::NTuple{N,T}) where {N,T}

Compute the Euclidean (L2) distance between two coordinate tuples.

# Arguments
- `coord1`: First coordinate as an N-tuple
- `coord2`: Second coordinate as an N-tuple

# Returns
The Euclidean distance between the two coordinates.

# Example
```julia
julia> euclidean_distance((0.0, 0.0), (3.0, 4.0))
5.0
```
"""
function euclidean_distance(coord1::NTuple{N,T}, coord2::NTuple{N,T}) where {N,T}
    sqrt(sum((coord1[i] - coord2[i])^2 for i in 1:N))
end

# Handle scalar case
euclidean_distance(coord1::T, coord2::T) where T<:Number = abs(coord1 - coord2)

struct CircleStimulus{T,L}
    radius::T
    strength::T
    time_windows::Array{Tuple{T,T},1}
    center::Union{NTuple,Nothing}
    baseline::T
    lattice::L
end

"""
    CircleStimulus(; radius, strength, time_windows, lattice, center=nothing, baseline=0.0)

Construct a CircleStimulus with keyword arguments.

# Arguments
- `radius`: Radius of the circular stimulus region
- `strength`: Strength of the stimulus to be added inside the circle
- `time_windows`: Array of (start, end) tuples defining when the stimulus is active
- `lattice`: The spatial lattice on which the stimulus will be applied
- `center`: Center coordinates of the stimulus (defaults to origin if nothing)
- `baseline`: Baseline value to set everywhere (default 0.0)
"""
function CircleStimulus(; radius::T, strength::T, time_windows::Array{Tuple{T,T},1}, 
                         lattice::L, center::Union{NTuple,Nothing}=nothing, 
                         baseline=zero(typeof(radius))) where {T,L}
    CircleStimulus{T,L}(radius, strength, time_windows, center, convert(T, baseline), lattice)
end

"""
    stimulate!(dA, A, stim::CircleStimulus, t)

Applies a circular stimulus to the field `dA`, setting values inside the stimulus region (a circle of given `radius` and `center`) to `stim.strength` added to `stim.baseline`. Operates in-place on `dA` for use at time `t`.

The stimulus is only applied during the time windows specified in `stim.time_windows`. If the current time `t` is not within any of the time windows, only the baseline is applied.
"""
function stimulate!(dA, A, stim::CircleStimulus{T,L}, t) where {T,L}
    coords = coordinates(stim.lattice)
    dA .= stim.baseline
    
    # Check if current time is within any time window
    in_time_window = any(window -> window[1] <= t <= window[2], stim.time_windows)
    
    if in_time_window
        # Determine center coordinates
        if stim.center === nothing
            # Use origin (all zeros) with correct dimensionality
            first_coord = first(coords)
            center_coordinates = typeof(first_coord)(zeros(T, length(first_coord)))
        else
            center_coordinates = stim.center
        end
        
        # Compute distances from center
        distances = euclidean_distance.(coords, Ref(center_coordinates))
        
        # Identify points within the stimulus radius
        on_center = (distances .< stim.radius) .| (distances .≈ stim.radius)
        
        # Apply stimulus strength to points within radius
        dA[on_center] .+= stim.strength
    end
end

struct ConstantStimulus{T,L}
    strength::T
    time_windows::Array{Tuple{T,T},1}
    baseline::T
    lattice::L
end

"""
    ConstantStimulus(; strength, time_windows, lattice, baseline=0.0)

Construct a ConstantStimulus with keyword arguments.

# Arguments
- `strength`: Strength of the stimulus to be applied uniformly
- `time_windows`: Array of (start, end) tuples defining when the stimulus is active
- `lattice`: The spatial lattice on which the stimulus will be applied
- `baseline`: Baseline value to set everywhere when stimulus is not active (default 0.0)

# Example
```julia
# Create a constant stimulus for a point model
lattice = PointLattice()
stim = ConstantStimulus(
    strength=5.0,
    time_windows=[(10.0, 50.0)],
    lattice=lattice
)
```
"""
function ConstantStimulus(; strength::T, time_windows::Array{Tuple{T,T},1}, 
                          lattice::L, baseline=zero(typeof(strength))) where {T,L}
    ConstantStimulus{T,L}(strength, time_windows, convert(T, baseline), lattice)
end

"""
    stimulate!(dA, A, stim::ConstantStimulus, t)

Applies a constant (uniform) stimulus to the entire field `dA`.

The stimulus is only applied during the time windows specified in `stim.time_windows`. 
If the current time `t` is not within any of the time windows, only the baseline is applied.

This is particularly useful for point models where you want a sustained, 
non-oscillatory external input.
"""
function stimulate!(dA, A, stim::ConstantStimulus{T,L}, t) where {T,L}
    dA .= stim.baseline
    
    # Check if current time is within any time window
    in_time_window = any(window -> window[1] <= t <= window[2], stim.time_windows)
    
    if in_time_window
        # Apply stimulus strength uniformly
        dA .+= stim.strength
    end
end

struct RampStimulus{T,L}
    ramp_up_time::T
    plateau_time::T
    ramp_down_time::T
    max_strength::T
    start_time::T
    baseline::T
    lattice::L
end

"""
    RampStimulus(; ramp_up_time, plateau_time, ramp_down_time, max_strength, start_time=0.0, lattice, baseline=0.0)

Construct a RampStimulus with keyword arguments.

This stimulus ramps up linearly from baseline to max_strength, holds at max_strength for a plateau period,
then ramps down linearly back to baseline. This is useful for demonstrating failure of inhibition dynamics
where gradual stimulus increase leads to activation followed by paradoxical inhibitory suppression.

# Arguments
- `ramp_up_time`: Duration of the ramp-up phase (time to reach max_strength from baseline)
- `plateau_time`: Duration of the plateau phase (time to hold at max_strength)
- `ramp_down_time`: Duration of the ramp-down phase (time to return to baseline from max_strength)
- `max_strength`: Maximum strength of the stimulus at plateau
- `start_time`: Time when the ramp-up begins (default 0.0)
- `lattice`: The spatial lattice on which the stimulus will be applied
- `baseline`: Baseline value to set everywhere (default 0.0)

# Example
```julia
# Create a ramping stimulus for a point model
lattice = PointLattice()
stim = RampStimulus(
    ramp_up_time=50.0,
    plateau_time=50.0,
    ramp_down_time=50.0,
    max_strength=10.0,
    start_time=10.0,
    lattice=lattice
)
```
"""
function RampStimulus(; ramp_up_time::T, plateau_time::T, ramp_down_time::T, 
                       max_strength::T, start_time::T=zero(typeof(ramp_up_time)),
                       lattice::L, baseline=zero(typeof(max_strength))) where {T,L}
    RampStimulus{T,L}(ramp_up_time, plateau_time, ramp_down_time, max_strength, 
                       start_time, convert(T, baseline), lattice)
end

"""
    stimulate!(dA, A, stim::RampStimulus, t)

Applies a ramping stimulus to the entire field `dA`.

The stimulus ramps up linearly from baseline to max_strength, holds at max_strength for a plateau period,
then ramps down linearly back to baseline. The time progression is:
- [start_time, start_time + ramp_up_time]: Linear ramp from baseline to max_strength
- [start_time + ramp_up_time, start_time + ramp_up_time + plateau_time]: Hold at max_strength
- [start_time + ramp_up_time + plateau_time, start_time + ramp_up_time + plateau_time + ramp_down_time]: Linear ramp from max_strength to baseline
- Outside these time ranges: baseline only
"""
function stimulate!(dA, A, stim::RampStimulus{T,L}, t) where {T,L}
    # Start with baseline
    dA .= stim.baseline
    
    # Calculate phase boundaries
    ramp_up_end = stim.start_time + stim.ramp_up_time
    plateau_end = ramp_up_end + stim.plateau_time
    ramp_down_end = plateau_end + stim.ramp_down_time
    
    if t < stim.start_time
        # Before stimulus starts: baseline only
        return
    elseif t <= ramp_up_end
        # Ramp-up phase: linear increase from baseline to max_strength
        progress = (t - stim.start_time) / stim.ramp_up_time
        current_strength = stim.baseline + progress * stim.max_strength
        dA .+= (current_strength - stim.baseline)
    elseif t <= plateau_end
        # Plateau phase: hold at max_strength
        dA .+= stim.max_strength
    elseif t <= ramp_down_end
        # Ramp-down phase: linear decrease from max_strength to baseline
        progress = (t - plateau_end) / stim.ramp_down_time
        current_strength = stim.max_strength * (1.0 - progress)
        dA .+= current_strength
    end
    # After ramp_down_end: baseline only (already set)
end
