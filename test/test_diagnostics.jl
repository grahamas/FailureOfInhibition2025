using SciMLBase: ReturnCode

function diagnostic_model(; timescale=1.0, drive=NoDrive(), recurrent=0.0, threshold=0.0)
    return PointModelParameters(
        excitatory=PopulationParameters(
            timescale=timescale,
            response=LogisticResponse(slope=1.0, threshold=threshold),
        ),
        inhibitory=PopulationParameters(
            timescale=timescale,
            response=LogisticResponse(slope=1.0, threshold=0.0),
        ),
        coupling=PointCoupling(recurrent, 0.0, 0.0, 0.0),
        drive=drive,
    )
end

function diagnostic_solution(state; times=collect(0.0:2.5:10.0), retcode=ReturnCode.Success)
    return (t=times, u=[copy(state) for _ in times], retcode=retcode)
end

function diagnostic_search_copy(search; equilibria=search.equilibria,
                                unresolved_nearby=search.unresolved_nearby)
    return EquilibriumSearchResult(
        search.model, search.frozen_model, search.frozen_drive, search.source_time,
        search.options, search.stability_options, search.attempts, equilibria,
        unresolved_nearby, search.completeness,
    )
end

@testset "Diagnostic options and API validation" begin
    @test DiagnosticOptions().window_duration == 5.0
    @test DiagnosticOptions().coordinate_atol == 1.0e-6
    @test DiagnosticOptions().balance_atol == 1.0e-8
    @test DiagnosticOptions().min_samples == 3
    @test DiagnosticOptions(window_duration=2, coordinate_atol=0, balance_atol=0).window_duration == 2.0
    for kwargs in ((; window_duration=0), (; window_duration=-1), (; window_duration=Inf),
                   (; coordinate_atol=-1), (; coordinate_atol=NaN), (; balance_atol=Inf),
                   (; balance_atol="small"), (; min_samples=1), (; min_samples=3.5))
        @test_throws ArgumentError DiagnosticOptions(; kwargs...)
    end
    @test_throws ArgumentError DiagnosticOptions(5.0, -1.0, 1e-8, 3)

    model = diagnostic_model()
    search = find_equilibria(model; seeds=[[1 / 3, 1 / 3]])
    solution = diagnostic_solution(only(search.equilibria).state)
    @test_throws ArgumentError diagnose_trajectory(solution, model; equilibria=search, options=nothing)
    @test_throws ArgumentError diagnose_trajectory(solution, model; equilibria=search.equilibria)
    @test_throws ArgumentError diagnose_trajectory(solution, diagnostic_model(timescale=2.0); equilibria=search)
    @test_throws ArgumentError diagnose_trajectory((t=solution.t, u=solution.u), model; equilibria=search)
    @test_throws ArgumentError diagnose_trajectory(merge(solution, (u=[[1.0]],)), model; equilibria=search)
    @test_throws ArgumentError diagnose_trajectory(merge(solution, (u=[zeros(3) for _ in solution.t],)), model; equilibria=search)
    @test_throws ArgumentError diagnose_trajectory(merge(solution, (t=fill("time", 5),)), model; equilibria=search)
    @test_throws ArgumentError diagnose_trajectory(merge(solution, (u=[ComplexF64[1, 1] for _ in solution.t],)), model; equilibria=search)
    @test_throws ArgumentError diagnose_trajectory(merge(solution, (retcode=nothing,)), model; equilibria=search)
    wrong_parameters = merge(solution, (prob=(p=diagnostic_model(timescale=2.0), tspan=(0.0, 10.0)),))
    @test_throws ArgumentError diagnose_trajectory(wrong_parameters, model; equilibria=search)
end

@testset "Finite-window equilibrium compatibility" begin
    model = diagnostic_model()
    search = find_equilibria(model; seeds=[[1 / 3, 1 / 3]])
    state = only(search.equilibria).state
    solution = diagnostic_solution(state)
    result = diagnose_trajectory(solution, model; equilibria=search)
    @test result.classification == EquilibriumCompatible
    @test isempty(result.reasons)
    @test result.window_bounds == ((0.0, 5.0), (5.0, 10.0))
    @test result.sample_counts == (3, 3)
    @test all(mean -> mean ≈ state, result.means)
    @test result.ranges == ([0.0, 0.0], [0.0, 0.0])
    @test all(value -> value <= 1e-8, result.max_balance_residuals)
    @test result.equilibrium_distances == [0.0]
    @test result.matched_equilibrium == 1
    @test result.integration_success
    @test search.completeness == CompletenessNotCertified

    integrated = solve_point_model([0.1, 0.15], (0.0, 30.0), model;
                                   saveat=0.5, abstol=1e-11, reltol=1e-11)
    @test diagnose_trajectory(integrated, model; equilibria=search).classification == EquilibriumCompatible

    # BigFloat observations are not silently narrowed to the root-search type.
    big_solution = diagnostic_solution(BigFloat.(state); times=BigFloat.(solution.t))
    big_result = diagnose_trajectory(big_solution, model; equilibria=search)
    @test eltype(first(big_result.means)) == BigFloat
    @test big_result.classification == EquilibriumCompatible

    # Range, root-distance, and residual comparisons include exact thresholds.
    varying = deepcopy(solution)
    varying.u[2][1] += 1.0e-6
    varying.u[4][1] -= 1.0e-6
    observed = diagnose_trajectory(varying, model; equilibria=search,
                                  options=DiagnosticOptions(coordinate_atol=1.0, balance_atol=1.0))
    range_limit = maximum(maximum, observed.ranges)
    balance_limit = maximum(observed.max_balance_residuals)
    exact_options = DiagnosticOptions(coordinate_atol=range_limit, balance_atol=balance_limit)
    exact = diagnose_trajectory(varying, model; equilibria=search, options=exact_options)
    @test exact.classification == EquilibriumCompatible
    narrow = diagnose_trajectory(varying, model; equilibria=search,
        options=DiagnosticOptions(coordinate_atol=prevfloat(range_limit), balance_atol=balance_limit))
    @test :large_coordinate_range in narrow.reasons
    strict_residual = diagnose_trajectory(varying, model; equilibria=search,
        options=DiagnosticOptions(coordinate_atol=range_limit, balance_atol=prevfloat(balance_limit)))
    @test :large_balance_residual in strict_residual.reasons

    shifted = diagnostic_solution(state .+ [1e-6, 0.0])
    distance_limit = maximum(abs.(first(shifted.u) .- state))
    at_distance = diagnose_trajectory(shifted, model; equilibria=search,
        options=DiagnosticOptions(coordinate_atol=distance_limit, balance_atol=1.0))
    @test at_distance.classification == EquilibriumCompatible
    beyond_distance = diagnose_trajectory(shifted, model; equilibria=search,
        options=DiagnosticOptions(coordinate_atol=prevfloat(distance_limit), balance_atol=1.0))
    @test :no_matching_equilibrium in beyond_distance.reasons
end

@testset "Compatibility is separate from attraction and root certainty" begin
    saddle_model = diagnostic_model(recurrent=12.0, threshold=3 + log(2))
    saddle_search = find_equilibria(saddle_model; seeds=[[0.25, 1 / 3]])
    saddle = only(saddle_search.equilibria)
    @test saddle.stability.classification == Saddle
    stationary_saddle = diagnose_trajectory(diagnostic_solution(saddle.state), saddle_model;
                                           equilibria=saddle_search)
    @test stationary_saddle.classification == EquilibriumCompatible
    @test saddle_search.equilibria[stationary_saddle.matched_equilibrium].stability.classification == Saddle

    slow_model = diagnostic_model(timescale=1.0e12)
    slow_search = find_equilibria(slow_model; seeds=[[1 / 3, 1 / 3]])
    slow_state = [0.1, 0.1]
    derivative = zeros(2)
    point_rhs!(derivative, slow_state, slow_model, 0.0)
    @test maximum(abs, derivative) < 1e-8
    slow = diagnose_trajectory(diagnostic_solution(slow_state), slow_model;
                               equilibria=slow_search)
    @test slow.classification == TrajectoryUnresolved
    @test :large_balance_residual in slow.reasons
    @test minimum(slow.max_balance_residuals) > 0.3

    model = diagnostic_model()
    search = find_equilibria(model; seeds=[[1 / 3, 1 / 3]])
    root = only(search.equilibria)
    solution = diagnostic_solution(root.state)
    duplicated = diagnostic_search_copy(search; equilibria=[root, root])
    ambiguous = diagnose_trajectory(solution, model; equilibria=duplicated)
    @test ambiguous.classification == TrajectoryUnresolved
    @test ambiguous.matched_equilibrium === nothing
    @test :ambiguous_equilibrium_match in ambiguous.reasons
    nearby = diagnostic_search_copy(search; unresolved_nearby=[copy(root.member_attempts)])
    uncertain = diagnose_trajectory(solution, model; equilibria=nearby)
    @test :unresolved_nearby_equilibria in uncertain.reasons
    @test uncertain.classification == TrajectoryUnresolved
    empty_search = diagnostic_search_copy(search; equilibria=empty(search.equilibria))
    undiscovered = diagnose_trajectory(solution, model; equilibria=empty_search)
    @test :no_discovered_equilibria in undiscovered.reasons

    singular_model = diagnostic_model(recurrent=8.0, threshold=2 + log(2))
    singular_search = find_equilibria(singular_model; seeds=[[0.25, 1 / 3]])
    singular = diagnose_trajectory(diagnostic_solution(only(singular_search.equilibria).state),
                                   singular_model; equilibria=singular_search)
    @test :near_singular_equilibrium in singular.reasons
    @test singular.classification == TrajectoryUnresolved
end

@testset "Closed-window drive context" begin
    model = diagnostic_model()
    search = find_equilibria(model; seeds=[[1 / 3, 1 / 3]])
    solution = diagnostic_solution(only(search.equilibria).state)
    for pulse in (
        DrivePulse(onset=1.0, offset=2.0, increment=(1.0, 0.0)),
        DrivePulse(onset=10.0, offset=11.0, increment=(1.0, 0.0)),
        DrivePulse(onset=4.0, offset=5.0, increment=(1.0, 0.0)),
    )
        driven = diagnostic_model(drive=PiecewiseConstantDrive(
            baseline=(0.0, 0.0), pulses=(pulse,), interpretation=AfferentExcitation))
        result = diagnose_trajectory(solution, driven; equilibria=search)
        @test :changing_drive in result.reasons
        @test result.classification == TrajectoryUnresolved
    end
    canceled = diagnostic_model(drive=PiecewiseConstantDrive(
        baseline=(0.0, 0.0), interpretation=AbstractIntervention,
        pulses=(
            DrivePulse(onset=1.0, offset=2.0, increment=(1.0, 0.0)),
            DrivePulse(onset=1.0, offset=2.0, increment=(-1.0, 0.0)),
        )))
    @test diagnose_trajectory(solution, canceled; equilibria=search).classification == EquilibriumCompatible

    constant = diagnostic_model(drive=PiecewiseConstantDrive(
        baseline=(1.0, 0.0), pulses=(), interpretation=AfferentExcitation))
    mismatch = diagnose_trajectory(solution, constant; equilibria=search)
    @test :frozen_drive_mismatch in mismatch.reasons
    @test mismatch.classification == TrajectoryUnresolved

    # A pulse ending exactly at the first window boundary is already absent.
    prior_pulse = diagnostic_model(drive=PiecewiseConstantDrive(
        baseline=(0.0, 0.0), interpretation=AfferentExcitation,
        pulses=(DrivePulse(onset=-1.0, offset=0.0, increment=(1.0, 0.0)),)))
    @test diagnose_trajectory(solution, prior_pulse; equilibria=search).classification == EquilibriumCompatible

    rational_model = diagnostic_model(drive=PiecewiseConstantDrive(
        baseline=(1 // 3, 1 // 3), pulses=(), interpretation=AfferentExcitation))
    rational_search = find_equilibria(rational_model; seeds=[[1 / 3, 1 / 3]])
    rational_solution = diagnostic_solution(only(rational_search.equilibria).state)
    @test rational_search.frozen_drive != drive_value(rational_model.drive, 0.0)
    @test diagnose_trajectory(rational_solution, rational_model;
                              equilibria=rational_search).classification == EquilibriumCompatible

    # A completed solution from a different earlier protocol is not evidence
    # for this model even if its two terminal windows share the same drive.
    earlier_pulse = diagnostic_model(drive=PiecewiseConstantDrive(
        baseline=(0.0, 0.0), interpretation=AfferentExcitation,
        pulses=(DrivePulse(onset=1.0, offset=2.0, increment=(1.0, 0.0)),)))
    long_solution = diagnostic_solution(only(search.equilibria).state; times=collect(0.0:2.5:20.0))
    wrong_protocol = merge(long_solution, (prob=(p=earlier_pulse, tspan=(0.0, 20.0)),))
    @test_throws ArgumentError diagnose_trajectory(wrong_protocol, model; equilibria=search)
    effective_same = merge(solution, (prob=(p=canceled, tspan=(0.0, 10.0)),))
    @test diagnose_trajectory(effective_same, model; equilibria=search).classification == EquilibriumCompatible
end

@testset "Unresolved numerical outcomes retain diagnostics" begin
    model = diagnostic_model()
    search = find_equilibria(model; seeds=[[1 / 3, 1 / 3]])
    state = only(search.equilibria).state
    solution = diagnostic_solution(state)
    failed = diagnose_trajectory(merge(solution, (retcode=ReturnCode.MaxIters,)), model; equilibria=search)
    @test !failed.integration_success
    @test failed.classification == TrajectoryUnresolved
    @test :integration_failed in failed.reasons
    @test all(isfinite, failed.max_balance_residuals)

    terminated = merge(solution, (retcode=ReturnCode.Terminated, prob=(tspan=(0.0, 20.0),)))
    truncated = diagnose_trajectory(terminated, model; equilibria=search)
    @test !truncated.integration_success
    @test :incomplete_integration in truncated.reasons
    @test truncated.classification == TrajectoryUnresolved

    too_short = diagnose_trajectory(diagnostic_solution(state; times=[0.0, 2.5, 5.0]), model;
                                    equilibria=search)
    @test :insufficient_window_coverage in too_short.reasons
    @test :insufficient_samples in too_short.reasons
    unsampled = diagnose_trajectory(diagnostic_solution(state; times=[0.0, 2.0, 4.0, 6.0, 8.0, 10.0]), model;
                                    equilibria=search)
    @test :unsampled_window_boundary in unsampled.reasons
    @test unsampled.sample_counts == (3, 3)

    for (times, reason) in (([0.0, 5.0, 2.5, 7.5, 10.0], :unordered_times),
                            ([0.0, 2.5, 2.5, 7.5, 10.0], :unordered_times),
                            ([0.0, 2.5, NaN, 7.5, 10.0], :nonfinite_time))
        result = diagnose_trajectory(diagnostic_solution(state; times=times), model; equilibria=search)
        @test result.classification == TrajectoryUnresolved
        @test reason in result.reasons
    end
    empty_result = diagnose_trajectory((t=Float64[], u=Vector{Float64}[], retcode=ReturnCode.Success),
                                       model; equilibria=search)
    @test :empty_solution in empty_result.reasons
    @test all(isnan, empty_result.max_balance_residuals)

    nonfinite_solution = deepcopy(solution)
    nonfinite_solution.u[2][1] = NaN
    nonfinite = diagnose_trajectory(nonfinite_solution, model; equilibria=search)
    @test :nonfinite_state in nonfinite.reasons
    @test isnan(first(nonfinite.max_balance_residuals))
    @test all(isnan, nonfinite.equilibrium_distances)
    @test isfinite(last(nonfinite.max_balance_residuals))
    outside = deepcopy(solution)
    outside.u[2][1] = 1.01
    invalid_domain = diagnose_trajectory(outside, model; equilibria=search)
    @test :outside_physical_domain in invalid_domain.reasons
    @test invalid_domain.classification == TrajectoryUnresolved

    # Invalid saved states remain evidence even when outside the terminal windows.
    earlier_invalid = diagnostic_solution(state; times=collect(0.0:2.5:20.0))
    earlier_invalid.u[1][1] = 1.01
    @test :outside_physical_domain in diagnose_trajectory(earlier_invalid, model; equilibria=search).reasons
end
