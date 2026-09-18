function finite_difference_jacobian(model, state, time; step=1.0e-7)
    jacobian = zeros(2, 2)
    plus = zeros(2)
    minus = zeros(2)
    for column in 1:2
        upper = copy(state)
        lower = copy(state)
        upper[column] += step
        lower[column] -= step
        point_rhs!(plus, upper, model, time)
        point_rhs!(minus, lower, model, time)
        jacobian[:, column] .= (plus .- minus) ./ (2step)
    end
    return jacobian
end

@testset "Analytical Jacobian" begin
    candidates = (
        LogisticResponse(slope=2.0, threshold=0.1),
        RectifiedZeroedLogisticResponse(slope=2.0, threshold=-0.2),
        DifferenceOfLogisticsCandidate(
            activating_slope=2.0,
            activating_threshold=0.0,
            failing_slope=1.0,
            failing_threshold=0.8,
        ),
        DifferenceOfRectifiedZeroedLogisticsCandidate(
            activating_slope=2.0,
            activating_threshold=-0.1,
            failing_slope=1.0,
            failing_threshold=0.8,
        ),
    )

    state = [0.4, 0.2]
    for candidate in candidates
        model = synthetic_model(
            excitatory_response=candidate,
            inhibitory_response=candidate,
        )
        analytical = zeros(2, 2)
        point_jacobian!(analytical, state, model, 0.0)
        numerical = finite_difference_jacobian(model, state, 0.0)
        @test analytical ≈ numerical atol=1.0e-7 rtol=1.0e-6
    end

    model = synthetic_model()
    @test_throws ArgumentError point_jacobian!(zeros(4), state, model, 0.0)
    @test_throws ArgumentError point_jacobian!(zeros(3, 3), state, model, 0.0)
end
