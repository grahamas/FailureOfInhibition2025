function _finite_model_parameter(value, name)
    value isa Real || throw(ArgumentError("$name must be real"))
    isfinite(value) || throw(ArgumentError("$name must be finite"))
    return value
end

"""
    PopulationParameters(; decay, saturation, timescale, response)

Parameters for one population in the provisional point-model equation.
"""
struct PopulationParameters{T<:Real,R<:AbstractPopulationResponse}
    decay::T
    saturation::T
    timescale::T
    response::R
end

function PopulationParameters(; decay, saturation, timescale, response::AbstractPopulationResponse)
    promoted_decay, promoted_saturation, promoted_timescale =
        promote(decay, saturation, timescale)
    _finite_model_parameter(promoted_decay, "decay")
    _finite_model_parameter(promoted_saturation, "saturation")
    _finite_model_parameter(promoted_timescale, "timescale")
    promoted_timescale > zero(promoted_timescale) ||
        throw(ArgumentError("timescale must be positive"))
    return PopulationParameters(
        promoted_decay,
        promoted_saturation,
        promoted_timescale,
        response,
    )
end

"""
    PointCoupling(; e_to_e, i_to_e, e_to_i, i_to_i)

Coupling weights named as `source_to_target`. Negative weights are allowed.
"""
struct PointCoupling{T<:Real}
    e_to_e::T
    i_to_e::T
    e_to_i::T
    i_to_i::T
end

function PointCoupling(; e_to_e, i_to_e, e_to_i, i_to_i)
    weights = promote(e_to_e, i_to_e, e_to_i, i_to_i)
    for (name, value) in zip(("e_to_e", "i_to_e", "e_to_i", "i_to_i"), weights)
        _finite_model_parameter(value, name)
    end
    return PointCoupling(weights...)
end

"""
    PointModelParameters(; excitatory, inhibitory, coupling, drive=NoDrive())

Typed parameters for a two-population point model. The state order is always
`[E, I]`. The equation remains provisional until the paper's mathematical
contract is approved.
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

Evaluate the provisional two-population Wilson-Cowan-type equation in place.
Both `state` and `derivative` must be two-element vectors ordered `[E, I]`.
"""
function point_rhs!(derivative, state, parameters::PointModelParameters, time)
    _require_point_state(state, "state")
    _require_point_state(derivative, "derivative")

    excitatory_activity, inhibitory_activity = state
    excitatory_drive, inhibitory_drive = drive_value(parameters.drive, time)
    coupling = parameters.coupling

    excitatory_input = excitatory_drive +
                       coupling.e_to_e * excitatory_activity +
                       coupling.i_to_e * inhibitory_activity
    inhibitory_input = inhibitory_drive +
                       coupling.e_to_i * excitatory_activity +
                       coupling.i_to_i * inhibitory_activity

    excitatory = parameters.excitatory
    inhibitory = parameters.inhibitory
    excitatory_rate = response(excitatory.response, excitatory_input)
    inhibitory_rate = response(inhibitory.response, inhibitory_input)

    derivative[1] = (
        -excitatory.decay * excitatory_activity +
        excitatory.saturation * (one(excitatory_activity) - excitatory_activity) * excitatory_rate
    ) / excitatory.timescale
    derivative[2] = (
        -inhibitory.decay * inhibitory_activity +
        inhibitory.saturation * (one(inhibitory_activity) - inhibitory_activity) * inhibitory_rate
    ) / inhibitory.timescale

    return nothing
end

"""
    point_jacobian!(jacobian, state, parameters, time)

Evaluate the analytical Jacobian of `point_rhs!` in place.
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
                       coupling.e_to_e * excitatory_activity +
                       coupling.i_to_e * inhibitory_activity
    inhibitory_input = inhibitory_drive +
                       coupling.e_to_i * excitatory_activity +
                       coupling.i_to_i * inhibitory_activity

    excitatory_rate = response(excitatory.response, excitatory_input)
    inhibitory_rate = response(inhibitory.response, inhibitory_input)
    excitatory_slope = response_derivative(excitatory.response, excitatory_input)
    inhibitory_slope = response_derivative(inhibitory.response, inhibitory_input)

    jacobian[1, 1] = (
        -excitatory.decay - excitatory.saturation * excitatory_rate +
        excitatory.saturation * (one(excitatory_activity) - excitatory_activity) *
        excitatory_slope * coupling.e_to_e
    ) / excitatory.timescale
    jacobian[1, 2] = (
        excitatory.saturation * (one(excitatory_activity) - excitatory_activity) *
        excitatory_slope * coupling.i_to_e
    ) / excitatory.timescale
    jacobian[2, 1] = (
        inhibitory.saturation * (one(inhibitory_activity) - inhibitory_activity) *
        inhibitory_slope * coupling.e_to_i
    ) / inhibitory.timescale
    jacobian[2, 2] = (
        -inhibitory.decay - inhibitory.saturation * inhibitory_rate +
        inhibitory.saturation * (one(inhibitory_activity) - inhibitory_activity) *
        inhibitory_slope * coupling.i_to_i
    ) / inhibitory.timescale

    return jacobian
end
