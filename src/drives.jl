abstract type AbstractPointDrive end

"""A point-model drive that contributes zero input to both populations."""
struct NoDrive <: AbstractPointDrive end

"""
    PiecewiseConstantDrive(; baseline=(0.0, 0.0), amplitude, windows)

Two-population drive with E/I `baseline` values and an E/I `amplitude` added
during any inclusive time window `(start, stop)`.
"""
struct PiecewiseConstantDrive{T<:Real} <: AbstractPointDrive
    baseline::NTuple{2,T}
    amplitude::NTuple{2,T}
    windows::Vector{Tuple{T,T}}
end

function PiecewiseConstantDrive(; baseline=(0.0, 0.0), amplitude, windows)
    baseline isa Tuple && length(baseline) == 2 ||
        throw(ArgumentError("baseline must be an (E, I) tuple"))
    amplitude isa Tuple && length(amplitude) == 2 ||
        throw(ArgumentError("amplitude must be an (E, I) tuple"))

    collected_windows = collect(windows)
    window_values = Any[]
    for window in collected_windows
        window isa Tuple && length(window) == 2 ||
            throw(ArgumentError("each drive window must be a (start, stop) tuple"))
        append!(window_values, window)
    end

    all_values = (baseline..., amplitude..., window_values...)
    all(value -> value isa Real, all_values) ||
        throw(ArgumentError("drive values and window bounds must be real"))
    T = isempty(all_values) ? Float64 : promote_type(map(typeof, all_values)...)

    promoted_baseline = (convert(T, baseline[1]), convert(T, baseline[2]))
    promoted_amplitude = (convert(T, amplitude[1]), convert(T, amplitude[2]))
    promoted_windows = Tuple{T,T}[]

    all(isfinite, promoted_baseline) || throw(ArgumentError("baseline values must be finite"))
    all(isfinite, promoted_amplitude) || throw(ArgumentError("amplitude values must be finite"))

    for window in collected_windows
        start_time, stop_time = convert.(T, window)
        isfinite(start_time) && isfinite(stop_time) ||
            throw(ArgumentError("drive window bounds must be finite"))
        start_time <= stop_time ||
            throw(ArgumentError("drive window start must not exceed its stop"))
        push!(promoted_windows, (start_time, stop_time))
    end

    return PiecewiseConstantDrive{T}(
        promoted_baseline,
        promoted_amplitude,
        promoted_windows,
    )
end

drive_value(::NoDrive, time) = (zero(time), zero(time))

function drive_value(drive::PiecewiseConstantDrive, time)
    active = any(window -> window[1] <= time <= window[2], drive.windows)
    active || return drive.baseline
    return (
        drive.baseline[1] + drive.amplitude[1],
        drive.baseline[2] + drive.amplitude[2],
    )
end
