function figure5b_anchor_model(; ratio=4.4, coupling=PointCoupling(
    e_to_e=19.0, i_to_e=13.0, e_to_i=19.0, i_to_i=6.0))
    return PointModelParameters(
        excitatory=PopulationParameters(timescale=7.8,
            response=LogisticResponse(slope=5.0, threshold=1.5)),
        inhibitory=PopulationParameters(timescale=7.8ratio,
            response=FailureOfInhibitionResponse(slope=5.0,
                onset_threshold=4.0, failure_threshold=8.0)),
        coupling=coupling,
        drive=NoDrive(),
    )
end

function figure5b_seeds(model, points)
    seeds = default_equilibrium_seeds(model)
    upper_e, upper_i = last(seeds)
    append!(seeds, [[e, i] for e in range(0.0, upper_e; length=points)
        for i in range(0.0, upper_i; length=points)])
    return unique!(seeds)
end

@testset "Figure-5b topology and anchor Hopf diagnostics" begin
    model = figure5b_anchor_model()
    searches = Tuple(find_equilibria(model; seeds=figure5b_seeds(model, points))
        for points in (11, 21, 41))
    search = first(searches)
    topology = classify_figure5b_topology(searches)
    @test topology.qualified
    @test isempty(topology.reasons)
    @test length(topology.root_tracks) == 7
    @test topology.central_track !== nothing
    @test topology.refinement_grid_points == (11, 21, 41)
    @test topology.central_state ≈ [0.329934171847263, 0.367295168265613] atol=1e-10
    @test topology.nullcline_slopes[1] ≈ 1.324446405145 rtol=1e-9
    @test topology.nullcline_slopes[2] ≈ 2.359771995363 rtol=1e-9

    diagnostics = hopf_diagnostics(model, topology.central_state)
    @test diagnostics.resolved
    @test diagnostics.classification == :supercritical_candidate
    @test diagnostics.critical_ratio ≈ 0.4302092308329947 rtol=1e-10
    @test diagnostics.critical_tau_i ≈ 3.355632000497359 rtol=1e-10
    @test diagnostics.determinant ≈ 69.90875041111174 rtol=1e-9
    @test diagnostics.frequency ≈ 1.634298208639488 rtol=1e-8
    @test diagnostics.linear_period ≈ 3.844577001898681 rtol=1e-8
    @test diagnostics.transversality ≈ 2.148326593590406 rtol=1e-8
    @test diagnostics.lyapunov_ad ≈ -114.166238098275 rtol=1e-7
    @test diagnostics.lyapunov_fd ≈ diagnostics.lyapunov_ad rtol=1e-4
    @test all(isfinite, (diagnostics.g21, diagnostics.c1,
        diagnostics.lyapunov_ad, diagnostics.lyapunov_fd,
        diagnostics.modal_radius_squared_slope))
    @test diagnostics.modal_radius_squared_slope ≈
        -diagnostics.transversality / real(diagnostics.c1) rtol=1e-12

    mismatched_model = figure5b_anchor_model(coupling=PointCoupling(
        e_to_e=19.0, i_to_e=13.0, e_to_i=19.5, i_to_i=6.0))
    mismatched = find_equilibria(mismatched_model;
        seeds=figure5b_seeds(mismatched_model, 41))
    rejected = classify_figure5b_topology((searches[1], searches[2], mismatched))
    @test !rejected.qualified
    @test :model_context_mismatch in rejected.reasons

    nonmatching = classify_figure5b_topology((searches[1], searches[2],
        find_equilibria(model; seeds=[[0.0, 0.0]])))
    @test !nonmatching.qualified
    @test :root_count_mismatch in nonmatching.reasons

    shortcut = classify_figure5b_topology((search, search, search))
    @test !shortcut.qualified
    @test :refinement_schedule_mismatch in shortcut.reasons
end

@testset "Independent first-Lyapunov implementations" begin
    function radial_field(a, b, omega)
        return state -> begin
            x, y = state
            radius_squared = x^2 + y^2
            [a * radius_squared * x - omega * y - b * radius_squared * y,
             omega * x + b * radius_squared * x + a * radius_squared * y]
        end
    end
    for (a, b, omega) in ((-0.7, 0.2, 1.3), (0.4, -0.3, 0.9))
        field = radial_field(a, b, omega)
        g21, _ = FailureOfInhibition2025._kuznetsov_l1(field, [0.0, 0.0], omega)
        ad = real(g21) / (2omega)
        fd = FailureOfInhibition2025._gh_l1(field, [0.0, 0.0], 7.5e-4)
        @test g21 ≈ 4complex(a, b) rtol=1e-10 atol=1e-10
        @test ad ≈ 2a / omega rtol=1e-10
        @test fd ≈ ad rtol=1e-5 atol=1e-6
    end

    omega, a, b = 1.7, 0.6, -0.8
    quadratic = state -> begin
        x, y = state
        [-omega * y + a * x^2, omega * x + b * x^2]
    end
    g21, _ = FailureOfInhibition2025._kuznetsov_l1(
        quadratic, [0.0, 0.0], omega)
    expected = -a * b / (2omega^2)
    @test real(g21) / (2omega) ≈ expected rtol=1e-10
    @test FailureOfInhibition2025._gh_l1(quadratic, [0.0, 0.0], 7.5e-4) ≈
        expected rtol=1e-5 atol=1e-7

    linear = state -> [-state[2], state[1]]
    g21_linear, _ = FailureOfInhibition2025._kuznetsov_l1(
        linear, [0.0, 0.0], 1.0)
    @test abs(g21_linear) <= 1e-12
    @test abs(FailureOfInhibition2025._gh_l1(
        linear, [0.0, 0.0], 7.5e-4)) <= 1e-8
end

@testset "Hopf diagnostics fail closed on invalid or marginal inputs" begin
    for kwargs in ((; balance_residual_atol=true), (; diagonal_atol=0.0),
                   (; determinant_atol=Inf), (; fd_steps=(1e-3, 1e-3, 2e-3)),
                   (; fd_steps=(1e-3, 2e-3)),
                   (; balance_residual_atol=big"1e1000"),
                   (; balance_residual_atol=big"1e-1000"),
                   (; balance_residual_atol=BigFloat(1)),
                   (; balance_residual_atol=1 // 10),
                   (; fd_steps=(big"1e-1000", big"2e-1000", big"3e-1000")),
                   (; fd_steps=(BigFloat(1), nextfloat(BigFloat(1)), BigFloat(2))))
        @test_throws ArgumentError HopfDiagnosticOptions(; kwargs...)
    end
    for kwargs in ((; coordinate_match_atol=true), (; slope_atol=0.0),
                   (; minimum_root_separation=1e-6,
                       coordinate_match_atol=1e-6),
                   (; residual_atol=big"1e1000"),
                   (; residual_atol=big"1e-1000"),
                   (; residual_atol=BigFloat(1)),
                   (; residual_atol=1 // 10))
        @test_throws ArgumentError Figure5bTopologyOptions(; kwargs...)
    end
    exact_options = HopfDiagnosticOptions(
        balance_residual_atol=Float32(0.5), fd_steps=(1 // 8, 1 // 4, 1 // 2))
    @test exact_options.balance_residual_atol == 0.5
    @test exact_options.fd_steps == (0.125, 0.25, 0.5)
    @test_throws ArgumentError hopf_diagnostics(x -> x, [0.0, 0.0], true)
    @test_throws ArgumentError hopf_diagnostics(x -> x, [0.0, NaN], 1.0)
    @test_throws ArgumentError hopf_diagnostics(x -> x, BigFloat[0.0, 0.0], 1.0)
    @test_throws ArgumentError hopf_diagnostics(x -> x, [0.0, 0.0], BigFloat(1.0))
    @test_throws ArgumentError hopf_diagnostics(
        x -> x, [big(2)^60 + 1, 0], 1.0)
    for (state, tau_e) in (([0, 0], 1),
            ([0 // 1, 0 // 1], 1 // 1),
            (Float32[0, 0], Float32(1)))
        supported = hopf_diagnostics(x -> [x[1], x[2]], state, tau_e)
        @test !supported.resolved
        @test :invalid_critical_ratio in supported.reasons
    end

    same_sign = hopf_diagnostics(x -> [x[1], x[2]], [0.0, 0.0], 1.0)
    @test !same_sign.resolved
    @test :invalid_critical_ratio in same_sign.reasons
    zero_diagonal = hopf_diagnostics(x -> [-x[2], x[1]], [0.0, 0.0], 1.0)
    @test !zero_diagonal.resolved
    @test :unresolved_diagonal in zero_diagonal.reasons
    negative_determinant = hopf_diagnostics(
        x -> [x[1] + 2x[2], 2x[1] - x[2]], [0.0, 0.0], 1.0)
    @test !negative_determinant.resolved
    @test :nonpositive_determinant in negative_determinant.reasons

    trace_zero_linear = x -> [-x[1] + 2x[2], -2x[1] + x[2]]
    extreme_finite = hopf_diagnostics(trace_zero_linear, [0.0, 0.0], 1e-200)
    @test !extreme_finite.resolved
    @test isfinite(extreme_finite.frequency)
    @test extreme_finite.frequency ≈ sqrt(3.0) * 1e200 rtol=1e-14
    @test :lyapunov_near_zero in extreme_finite.reasons
    extreme_large = hopf_diagnostics(trace_zero_linear, [0.0, 0.0], 1e200)
    @test !extreme_large.resolved
    @test isfinite(extreme_large.frequency)
    @test extreme_large.frequency ≈ sqrt(3.0) * 1e-200 rtol=1e-14
    @test :lyapunov_near_zero in extreme_large.reasons
    scaled_transversality = hopf_diagnostics(
        x -> [-x[1] + 1e100 * x[2], -2e100 * x[1] + 1e200 * x[2]],
        [0.0, 0.0], 1e-120)
    @test isfinite(scaled_transversality.transversality)
    @test scaled_transversality.transversality ≈ -5e-81 rtol=1e-14
    @test :nonfinite_transversality ∉ scaled_transversality.reasons
    @test FailureOfInhibition2025._half_quotient(
        floatmax(Float64), floatmax(Float64)) ≈ 0.5
    overflowing_transversality = hopf_diagnostics(
        x -> [1e200 * x[1] + 1e100 * x[2], -2e100 * x[1] - x[2]],
        [0.0, 0.0], 1e91)
    @test !overflowing_transversality.resolved
    @test :nonfinite_transversality in overflowing_transversality.reasons
    scaled_frequency = hopf_diagnostics(
        x -> [x[1] + 1e154 * x[2], -1e154 * x[1] - 1e-320 * x[2]],
        [0.0, 0.0], 1e200;
        options=HopfDiagnosticOptions(diagonal_atol=5e-324))
    frequency_oracle(determinant, ratio, tau_e) = setprecision(BigFloat, 256) do
        Float64(sqrt(BigFloat(determinant) / BigFloat(ratio)) / BigFloat(tau_e))
    end
    @test isfinite(scaled_frequency.frequency)
    @test scaled_frequency.frequency ≈ frequency_oracle(
        scaled_frequency.determinant, scaled_frequency.critical_ratio, 1e200) rtol=5e-16
    @test :nonfinite_frequency ∉ scaled_frequency.reasons
    minimum_subnormal = nextfloat(0.0)
    for (determinant, ratio, tau_e) in (
            (floatmax(Float64), minimum_subnormal, floatmax(Float64)),
            (minimum_subnormal, floatmax(Float64), minimum_subnormal))
        @test FailureOfInhibition2025._scaled_hopf_frequency(
            determinant, ratio, tau_e) ≈
            frequency_oracle(determinant, ratio, tau_e) rtol=5e-16
    end
    @test isinf(FailureOfInhibition2025._scaled_hopf_frequency(
        3.0, 1.0, 1e-320))
    @test iszero(FailureOfInhibition2025._scaled_hopf_frequency(
        minimum_subnormal, floatmax(Float64), floatmax(Float64)))
    @test FailureOfInhibition2025._rounded_power_of_two_scale(
        nextfloat(0.5), -1074) == minimum_subnormal
    minimum_frequency = hopf_diagnostics(
        x -> [1e-15 * x[1] + 1e-15 * x[2],
            -1.5e-15 * x[1] - 1e-15 * x[2]],
        [0.0, 0.0], floatmax(Float64);
        options=HopfDiagnosticOptions(
            diagonal_atol=minimum_subnormal,
            determinant_atol=minimum_subnormal))
    @test minimum_frequency.frequency == minimum_subnormal
    @test :nonfinite_frequency ∉ minimum_frequency.reasons
    @test FailureOfInhibition2025._scaled_mean(
        fill(floatmax(Float64), 3)) == floatmax(Float64)
    @test FailureOfInhibition2025._finite_difference_plateau(
        fill(floatmax(Float64), 3), 1e-4, 1e-4)
    @test !FailureOfInhibition2025._finite_difference_plateau(
        [-floatmax(Float64), 0.0, floatmax(Float64)], 1e-4, 1e-4)
    huge_cubic = 5e307
    nonfinite_lyapunov = hopf_diagnostics(
        x -> begin
            radius_squared = x[1]^2 + x[2]^2
            [x[1] - 2x[2] + huge_cubic * radius_squared * x[1],
             2x[1] - x[2] + huge_cubic * radius_squared * x[2]]
        end,
        [0.0, 0.0], 1.0)
    @test !nonfinite_lyapunov.resolved
    @test :nonfinite_automatic_differentiation_lyapunov in
        nonfinite_lyapunov.reasons
    tiny_diagonal = 1e-300
    large_cubic = 1e30
    underflowed_radius = hopf_diagnostics(
        x -> begin
            radius_squared = x[1]^2 + x[2]^2
            [tiny_diagonal * x[1] - x[2] +
                large_cubic * radius_squared * x[1],
             x[1] - tiny_diagonal * x[2] +
                large_cubic * radius_squared * x[2]]
        end,
        [0.0, 0.0], 1.0;
        options=HopfDiagnosticOptions(
            diagonal_atol=minimum_subnormal,
            transversality_atol=minimum_subnormal,
            lyapunov_atol=minimum_subnormal))
    @test !underflowed_radius.resolved
    @test underflowed_radius.modal_radius_squared_slope == 0.0
    @test underflowed_radius.reasons ==
        [:underflowed_modal_radius_squared_slope]
    @test all(isfinite, (underflowed_radius.c1,
        underflowed_radius.lyapunov_ad, underflowed_radius.lyapunov_fd))
    extreme_overflow = hopf_diagnostics(trace_zero_linear, [0.0, 0.0], 1e-320)
    @test !extreme_overflow.resolved
    @test :nonfinite_frequency in extreme_overflow.reasons
end
