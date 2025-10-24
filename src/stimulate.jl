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
