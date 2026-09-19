@testset "Supported logistic response" begin
    logistic = LogisticResponse(slope=2.0, threshold=0.3)
    @test response(logistic, 0.3) == 0.5
    @test response_derivative(logistic, 0.3) == 0.5
    @test response(logistic, -20.0) < 1.0e-10
    @test response(logistic, 20.0) > 1.0 - 1.0e-10
    @test response(logistic, -Inf) == 0.0
    @test response(logistic, Inf) == 1.0
    @test all(diff(response.(Ref(logistic), -2.0:0.1:2.0)) .> 0.0)
    @test response_derivative(logistic, 0.7) ≈ central_difference(logistic, 0.7) rtol=1.0e-7

    unit_logistic = LogisticResponse(slope=1.0, threshold=0.0)
    negative_tail_reference = setprecision(BigFloat, 256) do
        Float64(inv(one(BigFloat) + exp(BigFloat(710))))
    end
    @test response(unit_logistic, -710.0) > 0.0
    @test response(unit_logistic, -710.0) ≈ negative_tail_reference rtol=1.0e-12
    @test response(unit_logistic, -1000.0) == 0.0
    @test response(unit_logistic, 1000.0) == 1.0

    @test_throws ArgumentError LogisticResponse(slope=0.0, threshold=0.0)
    @test_throws ArgumentError LogisticResponse(slope=-1.0, threshold=0.0)
    @test_throws ArgumentError LogisticResponse(slope=Inf, threshold=0.0)
    @test_throws ArgumentError LogisticResponse(slope=1.0, threshold=NaN)
    @test_throws ArgumentError LogisticResponse(0.0, 0.0)
    @test_throws ArgumentError LogisticResponse(-1.0, 0.0)
    @test_throws ArgumentError LogisticResponse(1.0, Inf)
end

@testset "Supported failure-of-inhibition response" begin
    onset = LogisticResponse(slope=2.0, threshold=-0.5)
    failure = FailureOfInhibitionResponse(onset; failure_threshold=1.5)
    equivalent = FailureOfInhibitionResponse(
        slope=2.0,
        onset_threshold=-0.5,
        failure_threshold=1.5,
    )
    @test failure == equivalent

    midpoint = 0.5
    peak = tanh(failure.slope * (failure.failure_threshold - failure.onset_threshold) / 4)
    @test response(failure, midpoint) ≈ peak
    @test response_derivative(failure, midpoint) == 0.0
    @test response(failure, midpoint - 0.7) ≈ response(failure, midpoint + 0.7)
    @test response_derivative(failure, midpoint - 0.7) > 0.0
    @test response_derivative(failure, midpoint + 0.7) < 0.0
    @test response(failure, -Inf) == 0.0
    @test response(failure, Inf) == 0.0
    @test 0.0 < response(failure, -20.0) < peak
    @test 0.0 < response(failure, 20.0) < peak
    @test response_derivative(failure, -1.2) ≈ central_difference(failure, -1.2) rtol=1.0e-7
    @test response_derivative(failure, 2.3) ≈ central_difference(failure, 2.3) rtol=1.0e-7

    tail_response_reference(x) = setprecision(BigFloat, 4096) do
        slope = BigFloat(1)
        onset_threshold = BigFloat(-2)
        failure_threshold = BigFloat(3)
        input = BigFloat(x)
        logistic(argument) = inv(one(argument) + exp(-argument))
        onset_value = logistic(slope * (input - onset_threshold))
        failure_value = logistic(slope * (input - failure_threshold))
        value = onset_value - failure_value
        derivative = slope * value * (one(value) - onset_value - failure_value)
        return Float64(value), Float64(derivative)
    end

    tail_response = FailureOfInhibitionResponse(
        slope=1.0,
        onset_threshold=-2.0,
        failure_threshold=3.0,
    )
    for input in (-700.0, 700.0)
        expected_value, expected_derivative = tail_response_reference(input)
        @test response(tail_response, input) ≈ expected_value rtol=2.0e-13
        @test response_derivative(tail_response, input) ≈
              expected_derivative rtol=2.0e-13
    end

    @test_throws ArgumentError FailureOfInhibitionResponse(
        slope=0.0,
        onset_threshold=0.0,
        failure_threshold=1.0,
    )
    @test_throws ArgumentError FailureOfInhibitionResponse(
        slope=-1.0,
        onset_threshold=0.0,
        failure_threshold=1.0,
    )
    @test_throws ArgumentError FailureOfInhibitionResponse(
        slope=Inf,
        onset_threshold=0.0,
        failure_threshold=1.0,
    )
    @test_throws ArgumentError FailureOfInhibitionResponse(
        slope=1.0,
        onset_threshold=0.0,
        failure_threshold=0.0,
    )
    @test_throws ArgumentError FailureOfInhibitionResponse(
        slope=1.0,
        onset_threshold=1.0,
        failure_threshold=0.0,
    )
    @test_throws ArgumentError FailureOfInhibitionResponse(
        slope=1.0,
        onset_threshold=NaN,
        failure_threshold=1.0,
    )
    @test_throws ArgumentError FailureOfInhibitionResponse(onset; failure_threshold=Inf)
    @test_throws ArgumentError FailureOfInhibitionResponse(1.0, 1.0, 1.0)
    @test_throws ArgumentError FailureOfInhibitionResponse(1.0, 0.0, Inf)
end

@testset "Comparison response candidates" begin

    rectified = RectifiedZeroedLogisticResponse(slope=2.0, threshold=0.3)
    @test response(rectified, -1.0) == 0.0
    @test response(rectified, 0.0) == 0.0
    @test response(rectified, 1.0) > 0.0
    rectified_limit = 1.0 - response(LogisticResponse(slope=2.0, threshold=0.3), 0.0)
    @test response(rectified, -20.0) == 0.0
    @test response(rectified, 20.0) ≈ rectified_limit atol=1.0e-10
    @test response_derivative(rectified, 0.7) ≈ central_difference(rectified, 0.7) rtol=1.0e-7

    ordinary_difference = DifferenceOfLogisticsCandidate(
        activating_slope=1.0,
        activating_threshold=0.0,
        failing_slope=5.0,
        failing_threshold=0.0,
    )
    @test response(ordinary_difference, 0.0) == 0.0
    @test response(ordinary_difference, 0.5) < 0.0
    @test abs(response(ordinary_difference, -50.0)) < 1.0e-20
    @test abs(response(ordinary_difference, 50.0)) < 1.0e-15
    @test response_derivative(ordinary_difference, 0.7) ≈
          central_difference(ordinary_difference, 0.7) rtol=1.0e-7

    rectified_difference = DifferenceOfRectifiedZeroedLogisticsCandidate(
        activating_slope=1.0,
        activating_threshold=0.0,
        failing_slope=5.0,
        failing_threshold=0.0,
    )
    @test response(rectified_difference, -0.5) == 0.0
    @test response(rectified_difference, 0.0) == 0.0
    @test response(rectified_difference, 0.5) < 0.0
    @test response(rectified_difference, -50.0) == 0.0
    @test abs(response(rectified_difference, 50.0)) < 1.0e-15
    @test response_derivative(rectified_difference, 0.7) ≈
          central_difference(rectified_difference, 0.7) rtol=1.0e-7

    @test_throws ArgumentError RectifiedZeroedLogisticResponse(slope=0.0, threshold=0.0)
    @test_throws ArgumentError RectifiedZeroedLogisticResponse(slope=1.0, threshold=NaN)
end
