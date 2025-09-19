
struct CircleStimulus{T}
    radius::T
    strength::T
    time_windows::Array{Tuple{T,T},1}
    center::Union{NTuple,Nothing}
    baseline::T
end

"""
    stimulate!(dA, A, stim::CircleStimulus{T}, t) where T

Applies a circular stimulus to the field `dA`, setting values inside the stimulus region (a circle of given `radius` and `center`) to `stim.strength` added to `stim.baseline`. Operates in-place on `dA` for use at time `t`.
"""
function stimulate!(dA, A, stim::CircleStimulus{T}, t) where T
    coords = coordinates(dA)
    dA .= stim.baseline
    center_coordinates = stim.center == nothing ? Tuple(zeros(T,N_CDT)) : stim.center
    distances = distance.(coords, Ref(center_coordinates))
    on_center = (distances .< stim.radius) .| (distances .≈ stim.radius)
    dA[on_center] .+= stim.strength
end
