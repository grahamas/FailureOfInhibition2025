function _finite_model_parameter(value, name)
    value isa Real || throw(ArgumentError("$name must be real"))
    isfinite(value) || throw(ArgumentError("$name must be finite"))
    return value
end

"""
    PopulationParameters(; timescale, response)

Parameters for one population in the supported point-model equation.
`timescale` must be finite and strictly positive. Population parameters are
constructed with keywords so that the time constant and response cannot be
confused positionally.
"""
struct PopulationParameters{T<:Real,R<:AbstractPopulationResponse}
    timescale::T
    response::R

    function PopulationParameters(; timescale, response::R) where {R<:AbstractPopulationResponse}
        _finite_model_parameter(timescale, "timescale")
        timescale > zero(timescale) || throw(ArgumentError("timescale must be positive"))
        return new{typeof(timescale),R}(timescale, response)
    end
end

"""
    PointCoupling(e_to_e, i_to_e, e_to_i, i_to_i)
    PointCoupling(; e_to_e, i_to_e, e_to_i, i_to_i)

Finite, nonnegative coupling magnitudes named as `source_to_target`.
Inhibitory-source magnitudes (`i_to_e` and `i_to_i`) are subtracted when the
population inputs are formed.
"""
struct PointCoupling{T<:Real}
    e_to_e::T
    i_to_e::T
    e_to_i::T
    i_to_i::T

    function PointCoupling(e_to_e, i_to_e, e_to_i, i_to_i)
        raw_weights = (e_to_e, i_to_e, e_to_i, i_to_i)
        names = ("e_to_e", "i_to_e", "e_to_i", "i_to_i")
        for (name, value) in zip(names, raw_weights)
            _finite_model_parameter(value, name)
            value >= zero(value) || throw(ArgumentError("$name must be nonnegative"))
        end

        weights = promote(raw_weights...)
        T = typeof(first(weights))
        return new{T}(weights...)
    end
end

function PointCoupling(; e_to_e, i_to_e, e_to_i, i_to_i)
    return PointCoupling(e_to_e, i_to_e, e_to_i, i_to_i)
end

function _validate_supported_responses(excitatory, inhibitory)
    excitatory.response isa LogisticResponse || throw(
        ArgumentError("the excitatory response must be LogisticResponse"),
    )
    inhibitory.response isa Union{LogisticResponse,FailureOfInhibitionResponse} || throw(
        ArgumentError(
            "the inhibitory response must be LogisticResponse or FailureOfInhibitionResponse",
        ),
    )
    return nothing
end

"""
    PointModelParameters(; excitatory, inhibitory, coupling, drive=NoDrive())

Typed parameters for the supported two-population point model. The state
order is always `[E, I]`. The excitatory response must be `LogisticResponse`;
the inhibitory response may be `LogisticResponse` or
`FailureOfInhibitionResponse`.
"""
struct PointModelParameters{
    E<:PopulationParameters,
    I<:PopulationParameters,
    C<:PointCoupling,
    D<:AbstractPointDrive,
}
    excitatory::E
    inhibitory::I
    coupling::C
    drive::D

    function PointModelParameters(
        excitatory::E,
        inhibitory::I,
        coupling::C,
        drive::D,
    ) where {
        E<:PopulationParameters,
        I<:PopulationParameters,
        C<:PointCoupling,
        D<:AbstractPointDrive,
    }
        _validate_supported_responses(excitatory, inhibitory)
        return new{E,I,C,D}(excitatory, inhibitory, coupling, drive)
    end
end

function PointModelParameters(;
    excitatory::PopulationParameters,
    inhibitory::PopulationParameters,
    coupling::PointCoupling,
    drive::AbstractPointDrive=NoDrive(),
)
    return PointModelParameters(excitatory, inhibitory, coupling, drive)
end

function _require_point_state(state, name)
    state isa AbstractVector || throw(ArgumentError("$name must be a vector ordered [E, I]"))
    length(state) == 2 || throw(ArgumentError("$name must contain exactly E and I"))
    return state
end

"""
    point_rhs!(derivative, state, parameters, time)

Evaluate the supported two-population point-model equation in place:
`dX/dt = (-X + (1 - X) F_X(u_X)) / tau_X`. Both `state` and
`derivative` must be two-element vectors ordered `[E, I]`. This low-level
kernel deliberately does not restrict or project the supplied state.
"""
function point_rhs!(derivative, state, parameters::PointModelParameters, time)
    _require_point_state(state, "state")
    _require_point_state(derivative, "derivative")

    excitatory_activity, inhibitory_activity = state
    excitatory_drive, inhibitory_drive = drive_value(parameters.drive, time)
    coupling = parameters.coupling

    excitatory_input = excitatory_drive +
                       coupling.e_to_e * excitatory_activity -
                       coupling.i_to_e * inhibitory_activity
    inhibitory_input = inhibitory_drive +
                       coupling.e_to_i * excitatory_activity -
                       coupling.i_to_i * inhibitory_activity

    excitatory = parameters.excitatory
    inhibitory = parameters.inhibitory
    excitatory_rate = response(excitatory.response, excitatory_input)
    inhibitory_rate = response(inhibitory.response, inhibitory_input)

    derivative[1] = (
        -excitatory_activity +
        (one(excitatory_activity) - excitatory_activity) * excitatory_rate
    ) / excitatory.timescale
    derivative[2] = (
        -inhibitory_activity +
        (one(inhibitory_activity) - inhibitory_activity) * inhibitory_rate
    ) / inhibitory.timescale

    return nothing
end

"""
    point_jacobian!(jacobian, state, parameters, time)

Evaluate the analytical Jacobian of `point_rhs!` in place. Like the RHS
kernel, this function evaluates the supplied state without clamping it.
"""
function point_jacobian!(jacobian, state, parameters::PointModelParameters, time)
    _require_point_state(state, "state")
    jacobian isa AbstractMatrix || throw(ArgumentError("jacobian must be a 2x2 matrix"))
    size(jacobian) == (2, 2) || throw(ArgumentError("jacobian must be a 2x2 matrix"))

    excitatory_activity, inhibitory_activity = state
    excitatory_drive, inhibitory_drive = drive_value(parameters.drive, time)
    coupling = parameters.coupling
    excitatory = parameters.excitatory
    inhibitory = parameters.inhibitory

    excitatory_input = excitatory_drive +
                       coupling.e_to_e * excitatory_activity -
                       coupling.i_to_e * inhibitory_activity
    inhibitory_input = inhibitory_drive +
                       coupling.e_to_i * excitatory_activity -
                       coupling.i_to_i * inhibitory_activity

    excitatory_rate = response(excitatory.response, excitatory_input)
    inhibitory_rate = response(inhibitory.response, inhibitory_input)
    excitatory_slope = response_derivative(excitatory.response, excitatory_input)
    inhibitory_slope = response_derivative(inhibitory.response, inhibitory_input)

    jacobian[1, 1] = (
        -one(excitatory_activity) - excitatory_rate +
        (one(excitatory_activity) - excitatory_activity) *
        excitatory_slope * coupling.e_to_e
    ) / excitatory.timescale
    jacobian[1, 2] = (
        -(one(excitatory_activity) - excitatory_activity) *
        excitatory_slope * coupling.i_to_e
    ) / excitatory.timescale
    jacobian[2, 1] = (
        (one(inhibitory_activity) - inhibitory_activity) *
        inhibitory_slope * coupling.e_to_i
    ) / inhibitory.timescale
    jacobian[2, 2] = (
        -one(inhibitory_activity) - inhibitory_rate -
        (one(inhibitory_activity) - inhibitory_activity) *
        inhibitory_slope * coupling.i_to_i
    ) / inhibitory.timescale

    return jacobian
end
