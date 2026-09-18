abstract type AbstractPopulationResponse end

function _finite_response_parameter(value, name)
    value isa Real || throw(ArgumentError("$name must be real"))
    isfinite(value) || throw(ArgumentError("$name must be finite"))
    return value
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
        _finite_response_parameter(slope, "slope")
        _finite_response_parameter(threshold, "threshold")
        new{T}(slope, threshold)
    end
end

function LogisticResponse(; slope, threshold)
    promoted_slope, promoted_threshold = promote(slope, threshold)
    return LogisticResponse(promoted_slope, promoted_threshold)
end

"""
    RectifiedZeroedLogisticResponse(; slope, threshold)

Candidate response formed by subtracting the logistic value at zero and
rectifying negative results to zero.
"""
struct RectifiedZeroedLogisticResponse{T<:Real} <: AbstractPopulationResponse
    slope::T
    threshold::T

    function RectifiedZeroedLogisticResponse(slope::T, threshold::T) where {T<:Real}
        _finite_response_parameter(slope, "slope")
        _finite_response_parameter(threshold, "threshold")
        new{T}(slope, threshold)
    end
end

function RectifiedZeroedLogisticResponse(; slope, threshold)
    promoted_slope, promoted_threshold = promote(slope, threshold)
    return RectifiedZeroedLogisticResponse(promoted_slope, promoted_threshold)
end

"""
    DifferenceOfLogisticsCandidate(; activating_slope, activating_threshold,
                                     failing_slope, failing_threshold)

Noncanonical candidate response equal to an activating logistic response minus
a failing logistic response. It is neither normalized nor clamped.
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

Noncanonical candidate response equal to an activating rectified, zeroed
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

function response(parameters::LogisticResponse, x)
    exponent = -parameters.slope * (x - parameters.threshold)
    return inv(one(exponent) + exp(exponent))
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

function response_derivative(parameters::LogisticResponse, x)
    value = response(parameters, x)
    return parameters.slope * value * (one(value) - value)
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
