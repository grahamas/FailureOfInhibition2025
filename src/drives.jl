abstract type AbstractPointDrive end

"""A point-model drive that contributes zero input to both populations."""
struct NoDrive <: AbstractPointDrive end

"""
    DriveInterpretation

Interpretation of the two components of a point-model drive. `AfferentExcitation`
requires both total drive components to remain nonnegative, while
`AbstractIntervention` permits signed totals.
"""
@enum DriveInterpretation begin
    AfferentExcitation
    AbstractIntervention
end

"""
    DrivePulse(; onset, offset, increment)

An E/I drive increment active on the half-open interval `[onset, offset)`.
"""
struct DrivePulse{T<:Real}
    onset::T
    offset::T
    increment::NTuple{2,T}

    function DrivePulse{T}(
        onset::T,
        offset::T,
        increment::NTuple{2,T},
    ) where {T<:Real}
        isfinite(onset) && isfinite(offset) ||
            throw(ArgumentError("drive pulse bounds must be finite"))
        onset < offset ||
            throw(ArgumentError("drive pulse onset must be less than its offset"))
        all(isfinite, increment) ||
            throw(ArgumentError("drive pulse increments must be finite"))
        return new{T}(onset, offset, increment)
    end
end

function DrivePulse(; onset, offset, increment)
    increment isa Tuple && length(increment) == 2 ||
        throw(ArgumentError("increment must be an (E, I) tuple"))
    values = (onset, offset, increment...)
    all(value -> value isa Real, values) ||
        throw(ArgumentError("drive pulse bounds and increments must be real"))

    T = promote_type(map(typeof, values)...)
    promoted_onset = convert(T, onset)
    promoted_offset = convert(T, offset)
    promoted_increment = (convert(T, increment[1]), convert(T, increment[2]))
    return DrivePulse{T}(promoted_onset, promoted_offset, promoted_increment)
end

"""
    PiecewiseConstantDrive(; baseline=(0, 0), pulses, interpretation)

Two-population drive formed from an E/I `baseline` and additive `DrivePulse`s.
All active pulses are summed, and each pulse is active on `[onset, offset)`.
`interpretation` must be either `AfferentExcitation` or `AbstractIntervention`.
"""
struct PiecewiseConstantDrive{T<:Real,N} <: AbstractPointDrive
    baseline::NTuple{2,T}
    pulses::NTuple{N,DrivePulse{T}}
    interpretation::DriveInterpretation

    function PiecewiseConstantDrive{T,N}(
        baseline::NTuple{2,T},
        pulses::NTuple{N,DrivePulse{T}},
        interpretation::DriveInterpretation,
    ) where {T<:Real,N}
        all(isfinite, baseline) ||
            throw(ArgumentError("drive baseline values must be finite"))

        drive = new{T,N}(baseline, pulses, interpretation)
        _validate_drive_segments(drive)
        return drive
    end
end

function PiecewiseConstantDrive(; baseline=(0, 0), pulses, interpretation)
    baseline isa Tuple && length(baseline) == 2 ||
        throw(ArgumentError("baseline must be an (E, I) tuple"))
    all(value -> value isa Real, baseline) ||
        throw(ArgumentError("drive baseline values must be real"))
    interpretation isa DriveInterpretation ||
        throw(ArgumentError("interpretation must be a DriveInterpretation"))

    applicable(iterate, pulses) ||
        throw(ArgumentError("pulses must be an iterable of DrivePulse values"))
    collected_pulses = collect(pulses)
    all(pulse -> pulse isa DrivePulse, collected_pulses) ||
        throw(ArgumentError("pulses must contain only DrivePulse values"))

    value_types = Type[typeof(baseline[1]), typeof(baseline[2])]
    for pulse in collected_pulses
        append!(
            value_types,
            (
                typeof(pulse.onset),
                typeof(pulse.offset),
                typeof(pulse.increment[1]),
                typeof(pulse.increment[2]),
            ),
        )
    end
    T = promote_type(value_types...)
    promoted_baseline = (convert(T, baseline[1]), convert(T, baseline[2]))
    promoted_pulses = DrivePulse{T}[
        DrivePulse{T}(
            convert(T, pulse.onset),
            convert(T, pulse.offset),
            (convert(T, pulse.increment[1]), convert(T, pulse.increment[2])),
        ) for pulse in collected_pulses
    ]
    sort!(
        promoted_pulses;
        by=pulse -> (pulse.onset, pulse.offset, pulse.increment...),
    )

    immutable_pulses = Tuple(promoted_pulses)
    return PiecewiseConstantDrive{T,length(immutable_pulses)}(
        promoted_baseline,
        immutable_pulses,
        interpretation,
    )
end

drive_value(::NoDrive, time) = (zero(time), zero(time))

function drive_value(drive::PiecewiseConstantDrive, time)
    excitatory, inhibitory = drive.baseline
    for pulse in drive.pulses
        if pulse.onset <= time < pulse.offset
            excitatory += pulse.increment[1]
            inhibitory += pulse.increment[2]
        end
    end
    return (excitatory, inhibitory)
end

"""Return the sorted unique pulse onset and offset times for `drive`."""
drive_transition_times(::NoDrive) = Float64[]

function drive_transition_times(drive::PiecewiseConstantDrive{T}) where {T}
    transitions = T[]
    sizehint!(transitions, 2length(drive.pulses))
    for pulse in drive.pulses
        push!(transitions, pulse.onset, pulse.offset)
    end
    sort!(transitions)
    unique!(transitions)
    return transitions
end

function _validate_drive_segments(drive::PiecewiseConstantDrive)
    function validate_total(total)
        all(isfinite, total) ||
            throw(ArgumentError("total drive must be finite on every constant segment"))
        if drive.interpretation == AfferentExcitation
            all(value -> value >= zero(value), total) ||
                throw(
                    ArgumentError(
                        "afferent-excitation drive must be nonnegative on every constant segment",
                    ),
                )
        end
        return nothing
    end

    validate_total(drive.baseline)
    for time in drive_transition_times(drive)
        validate_total(drive_value(drive, time))
    end
    return nothing
end
