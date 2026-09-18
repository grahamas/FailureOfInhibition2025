abstract type AbstractPopulationResponse end

function _finite_response_parameter(value, name)
    value isa Real || throw(ArgumentError("$name must be real"))
    isfinite(value) || throw(ArgumentError("$name must be finite"))
    return value
end

function _positive_response_slope(value)
    _finite_response_parameter(value, "slope")
    value > zero(value) || throw(ArgumentError("slope must be positive"))
    return value
end

function _logistic_value(z)
    if z >= zero(z)
        return inv(one(z) + exp(-z))
    end

    exponential = exp(z)
    return exponential / (one(exponential) + exponential)
end

function _logistic_difference(first_argument, second_argument)
    if first_argument == second_argument
        return zero(_logistic_value(first_argument))
    elseif first_argument > second_argument
        separation = first_argument - second_argument
        return _logistic_value(first_argument) * _logistic_value(-second_argument) *
               (-expm1(-separation))
    end

    separation = second_argument - first_argument
    return -_logistic_value(second_argument) * _logistic_value(-first_argument) *
           (-expm1(-separation))
end

"""
    LogisticResponse(; slope, threshold)

Parameters for the logistic response
`1 / (1 + exp(-slope * (x - threshold)))`.
"""
struct LogisticResponse{T<:Real} <: AbstractPopulationResponse
    slope::T
    threshold::T

    function LogisticResponse(slope::T, threshold::T) where {T<:Real}
        _positive_response_slope(slope)
        _finite_response_parameter(threshold, "threshold")
        new{T}(slope, threshold)
    end
end

function LogisticResponse(slope::Real, threshold::Real)
    promoted_slope, promoted_threshold = promote(slope, threshold)
    return LogisticResponse(promoted_slope, promoted_threshold)
end

function LogisticResponse(; slope, threshold)
    _positive_response_slope(slope)
    _finite_response_parameter(threshold, "threshold")
    promoted_slope, promoted_threshold = promote(slope, threshold)
    return LogisticResponse(promoted_slope, promoted_threshold)
end

"""
    FailureOfInhibitionResponse(; slope, onset_threshold, failure_threshold)
    FailureOfInhibitionResponse(onset::LogisticResponse; failure_threshold)

Authoritative inhibitory response formed from the raw difference of two
equal-slope logistic responses. The slope is finite and positive, and the
finite onset threshold must be strictly below the failure threshold. The
response is neither normalized nor clamped.
"""
struct FailureOfInhibitionResponse{T<:Real} <: AbstractPopulationResponse
    slope::T
    onset_threshold::T
    failure_threshold::T

    function FailureOfInhibitionResponse(
        slope::T,
        onset_threshold::T,
        failure_threshold::T,
    ) where {T<:Real}
        _positive_response_slope(slope)
        _finite_response_parameter(onset_threshold, "onset_threshold")
        _finite_response_parameter(failure_threshold, "failure_threshold")
        onset_threshold < failure_threshold ||
            throw(ArgumentError("onset_threshold must be less than failure_threshold"))
        new{T}(slope, onset_threshold, failure_threshold)
    end
end

function FailureOfInhibitionResponse(
    slope::Real,
    onset_threshold::Real,
    failure_threshold::Real,
)
    promoted_slope, promoted_onset, promoted_failure =
        promote(slope, onset_threshold, failure_threshold)
    return FailureOfInhibitionResponse(promoted_slope, promoted_onset, promoted_failure)
end

function FailureOfInhibitionResponse(; slope, onset_threshold, failure_threshold)
    _positive_response_slope(slope)
    _finite_response_parameter(onset_threshold, "onset_threshold")
    _finite_response_parameter(failure_threshold, "failure_threshold")
    return FailureOfInhibitionResponse(slope, onset_threshold, failure_threshold)
end

function FailureOfInhibitionResponse(onset::LogisticResponse; failure_threshold)
    return FailureOfInhibitionResponse(
        onset.slope,
        onset.threshold,
        failure_threshold,
    )
end

"""
    RectifiedZeroedLogisticResponse(; slope, threshold)

Comparison response formed by subtracting the logistic value
at zero and rectifying negative results to zero.
"""
struct RectifiedZeroedLogisticResponse{T<:Real} <: AbstractPopulationResponse
    slope::T
    threshold::T

    function RectifiedZeroedLogisticResponse(slope::T, threshold::T) where {T<:Real}
        _positive_response_slope(slope)
        _finite_response_parameter(threshold, "threshold")
        new{T}(slope, threshold)
    end
end

function RectifiedZeroedLogisticResponse(slope::Real, threshold::Real)
    promoted_slope, promoted_threshold = promote(slope, threshold)
    return RectifiedZeroedLogisticResponse(promoted_slope, promoted_threshold)
end

function RectifiedZeroedLogisticResponse(; slope, threshold)
    _positive_response_slope(slope)
    _finite_response_parameter(threshold, "threshold")
    promoted_slope, promoted_threshold = promote(slope, threshold)
    return RectifiedZeroedLogisticResponse(promoted_slope, promoted_threshold)
end

"""
    DifferenceOfLogisticsCandidate(; activating_slope, activating_threshold,
                                     failing_slope, failing_threshold)

Comparison response equal to an activating logistic response
minus a failing logistic response. It is neither normalized nor clamped.
"""
struct DifferenceOfLogisticsCandidate{A<:LogisticResponse,F<:LogisticResponse} <: AbstractPopulationResponse
    activating::A
    failing::F
end

function DifferenceOfLogisticsCandidate(;
    activating_slope,
    activating_threshold,
    failing_slope,
    failing_threshold,
)
    activating = LogisticResponse(
        slope=activating_slope,
        threshold=activating_threshold,
    )
    failing = LogisticResponse(
        slope=failing_slope,
        threshold=failing_threshold,
    )
    return DifferenceOfLogisticsCandidate(activating, failing)
end

"""
    DifferenceOfRectifiedZeroedLogisticsCandidate(; activating_slope,
                                                    activating_threshold,
                                                    failing_slope,
                                                    failing_threshold)

Comparison response equal to an activating rectified, zeroed
logistic response minus a failing rectified, zeroed logistic response. The
difference itself is neither normalized nor clamped and may be negative.
"""
struct DifferenceOfRectifiedZeroedLogisticsCandidate{
    A<:RectifiedZeroedLogisticResponse,
    F<:RectifiedZeroedLogisticResponse,
} <: AbstractPopulationResponse
    activating::A
    failing::F
end

function DifferenceOfRectifiedZeroedLogisticsCandidate(;
    activating_slope,
    activating_threshold,
    failing_slope,
    failing_threshold,
)
    activating = RectifiedZeroedLogisticResponse(
        slope=activating_slope,
        threshold=activating_threshold,
    )
    failing = RectifiedZeroedLogisticResponse(
        slope=failing_slope,
        threshold=failing_threshold,
    )
    return DifferenceOfRectifiedZeroedLogisticsCandidate(activating, failing)
end

"""Evaluate a population response at effective input `x`."""
function response(parameters::LogisticResponse, x)
    argument = parameters.slope * (x - parameters.threshold)
    return _logistic_value(argument)
end

function response(parameters::FailureOfInhibitionResponse, x)
    onset_argument = parameters.slope * (x - parameters.onset_threshold)
    failure_argument = parameters.slope * (x - parameters.failure_threshold)
    scaled_separation =
        parameters.slope * (parameters.failure_threshold - parameters.onset_threshold)

    return _logistic_value(onset_argument) * _logistic_value(-failure_argument) *
           (-expm1(-scaled_separation))
end

function response(parameters::RectifiedZeroedLogisticResponse, x)
    logistic = LogisticResponse(parameters.slope, parameters.threshold)
    zeroed = response(logistic, x) - response(logistic, zero(x))
    return max(zero(zeroed), zeroed)
end

function response(parameters::DifferenceOfLogisticsCandidate, x)
    return response(parameters.activating, x) - response(parameters.failing, x)
end

function response(parameters::DifferenceOfRectifiedZeroedLogisticsCandidate, x)
    return response(parameters.activating, x) - response(parameters.failing, x)
end

"""Evaluate the derivative of a population response with respect to its input."""
function response_derivative(parameters::LogisticResponse, x)
    value = response(parameters, x)
    return parameters.slope * value * (one(value) - value)
end

function response_derivative(parameters::FailureOfInhibitionResponse, x)
    value = response(parameters, x)
    onset_complement_argument = -parameters.slope * (x - parameters.onset_threshold)
    failure_argument = parameters.slope * (x - parameters.failure_threshold)
    derivative_factor =
        _logistic_difference(onset_complement_argument, failure_argument)
    return parameters.slope * value * derivative_factor
end

function response_derivative(parameters::RectifiedZeroedLogisticResponse, x)
    logistic = LogisticResponse(parameters.slope, parameters.threshold)
    zeroed = response(logistic, x) - response(logistic, zero(x))
    return zeroed <= zero(zeroed) ? zero(zeroed) : response_derivative(logistic, x)
end

function response_derivative(parameters::DifferenceOfLogisticsCandidate, x)
    return response_derivative(parameters.activating, x) -
           response_derivative(parameters.failing, x)
end

function response_derivative(parameters::DifferenceOfRectifiedZeroedLogisticsCandidate, x)
    return response_derivative(parameters.activating, x) -
           response_derivative(parameters.failing, x)
end
