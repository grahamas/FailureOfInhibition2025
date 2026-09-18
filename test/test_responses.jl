@testset "Response candidates" begin
    logistic = LogisticResponse(slope=2.0, threshold=0.3)
    @test response(logistic, 0.3) == 0.5
    @test response(logistic, -20.0) < 1.0e-10
    @test response(logistic, 20.0) > 1.0 - 1.0e-10
    @test response_derivative(logistic, 0.7) ≈ central_difference(logistic, 0.7) rtol=1.0e-7

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

    @test_throws ArgumentError LogisticResponse(slope=Inf, threshold=0.0)
    @test_throws ArgumentError RectifiedZeroedLogisticResponse(slope=1.0, threshold=NaN)
end
