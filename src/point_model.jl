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

function _point_inputs(state, parameters::PointModelParameters, time)
    excitatory_activity, inhibitory_activity = state
    excitatory_drive, inhibitory_drive = drive_value(parameters.drive, time)
    coupling = parameters.coupling
    excitatory_input = excitatory_drive +
                       coupling.e_to_e * excitatory_activity -
                       coupling.i_to_e * inhibitory_activity
    inhibitory_input = inhibitory_drive +
                       coupling.e_to_i * excitatory_activity -
                       coupling.i_to_i * inhibitory_activity
    return excitatory_activity, inhibitory_activity, excitatory_input, inhibitory_input
end

function _point_balance_values(state, parameters::PointModelParameters, time)
    excitatory_activity, inhibitory_activity, excitatory_input, inhibitory_input =
        _point_inputs(state, parameters, time)
    excitatory_rate = response(parameters.excitatory.response, excitatory_input)
    inhibitory_rate = response(parameters.inhibitory.response, inhibitory_input)

    excitatory_balance = -excitatory_activity +
                        (one(excitatory_activity) - excitatory_activity) * excitatory_rate
    inhibitory_balance = -inhibitory_activity +
                        (one(inhibitory_activity) - inhibitory_activity) * inhibitory_rate
    return excitatory_balance, inhibitory_balance
end

"""
    point_balance!(residual, state, parameters, time)

Evaluate the dimensionless population-balance residual
`g_X = -X + (1 - X) F_X(u_X)` in place. Both vectors must contain exactly
`[E, I]`. The kernel permits trial states outside the physical domain and does
not clip or project them.
"""
function point_balance!(residual, state, parameters::PointModelParameters, time)
    _require_point_state(state, "state")
    _require_point_state(residual, "residual")
    excitatory_balance, inhibitory_balance = _point_balance_values(state, parameters, time)
    residual[1] = excitatory_balance
    residual[2] = inhibitory_balance
    return nothing
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
    excitatory_balance, inhibitory_balance = _point_balance_values(state, parameters, time)
    # Scale before assignment: the output buffer may have lower precision.
    derivative[1] = excitatory_balance / parameters.excitatory.timescale
    derivative[2] = inhibitory_balance / parameters.inhibitory.timescale

    return nothing
end

function _require_point_jacobian(jacobian)
    jacobian isa AbstractMatrix || throw(ArgumentError("jacobian must be a 2x2 matrix"))
    size(jacobian) == (2, 2) || throw(ArgumentError("jacobian must be a 2x2 matrix"))
    return jacobian
end

function _point_balance_jacobian_values(state, parameters::PointModelParameters, time)
    excitatory_activity, inhibitory_activity, excitatory_input, inhibitory_input =
        _point_inputs(state, parameters, time)
    coupling = parameters.coupling
    excitatory = parameters.excitatory
    inhibitory = parameters.inhibitory

    excitatory_rate = response(excitatory.response, excitatory_input)
    inhibitory_rate = response(inhibitory.response, inhibitory_input)
    excitatory_slope = response_derivative(excitatory.response, excitatory_input)
    inhibitory_slope = response_derivative(inhibitory.response, inhibitory_input)

    j11 = -one(excitatory_activity) - excitatory_rate +
          (one(excitatory_activity) - excitatory_activity) *
          excitatory_slope * coupling.e_to_e
    j12 = -(one(excitatory_activity) - excitatory_activity) *
          excitatory_slope * coupling.i_to_e
    j21 = (one(inhibitory_activity) - inhibitory_activity) *
          inhibitory_slope * coupling.e_to_i
    j22 = -one(inhibitory_activity) - inhibitory_rate -
          (one(inhibitory_activity) - inhibitory_activity) *
          inhibitory_slope * coupling.i_to_i
    return j11, j12, j21, j22
end

"""
    point_balance_jacobian!(jacobian, state, parameters, time)

Evaluate the analytical Jacobian `Dg` of `point_balance!` in place. The
result is dimensionless and is related to the ODE Jacobian by
`Dg = Diagonal([tau_E, tau_I]) * Df`.
"""
function point_balance_jacobian!(
    jacobian,
    state,
    parameters::PointModelParameters,
    time,
)
    _require_point_state(state, "state")
    _require_point_jacobian(jacobian)
    j11, j12, j21, j22 = _point_balance_jacobian_values(state, parameters, time)
    jacobian[1, 1] = j11
    jacobian[1, 2] = j12
    jacobian[2, 1] = j21
    jacobian[2, 2] = j22
    return jacobian
end

"""
    point_jacobian!(jacobian, state, parameters, time)

Evaluate the analytical Jacobian of `point_rhs!` in place. Like the RHS
kernel, this function evaluates the supplied state without clamping it.
"""
function point_jacobian!(jacobian, state, parameters::PointModelParameters, time)
    _require_point_state(state, "state")
    _require_point_jacobian(jacobian)
    j11, j12, j21, j22 = _point_balance_jacobian_values(state, parameters, time)
    # Retain promoted arithmetic through row scaling, then convert on assignment.
    jacobian[1, 1] = j11 / parameters.excitatory.timescale
    jacobian[1, 2] = j12 / parameters.excitatory.timescale
    jacobian[2, 1] = j21 / parameters.inhibitory.timescale
    jacobian[2, 2] = j22 / parameters.inhibitory.timescale

    return jacobian
end
