function parameterized_hopf_rhs!(du, u, parameters)
    growth, frequency = parameters
    radius_squared = sum(abs2, u)
    radial = growth - radius_squared
    du[1] = radial * u[1] - frequency * u[2]
    du[2] = frequency * u[1] + radial * u[2]
    return nothing
end

function parameterized_hopf_jacobian!(jacobian, u, parameters)
    growth, frequency = parameters
    radial = growth - sum(abs2, u)
    jacobian[1, 1] = radial - 2u[1]^2
    jacobian[1, 2] = -2u[1] * u[2] - frequency
    jacobian[2, 1] = -2u[1] * u[2] + frequency
    jacobian[2, 2] = radial - 2u[2]^2
    return jacobian
end

function cycle_fold_rhs!(du, u, parameters)
    parameter, frequency = parameters
    radius_squared = sum(abs2, u)
    radial = 0.1 * (parameter + (radius_squared - 1)^2)
    du[1] = radial * u[1] - frequency * u[2]
    du[2] = frequency * u[1] + radial * u[2]
    return nothing
end

function cycle_fold_jacobian!(jacobian, u, parameters)
    parameter, frequency = parameters
    radius_squared = sum(abs2, u)
    radial = 0.1 * (parameter + (radius_squared - 1)^2)
    radial_derivative = 0.4(radius_squared - 1)
    jacobian[1, 1] = radial + radial_derivative * u[1]^2
    jacobian[1, 2] = radial_derivative * u[1] * u[2] - frequency
    jacobian[2, 1] = radial_derivative * u[1] * u[2] + frequency
    jacobian[2, 2] = radial + radial_derivative * u[2]^2
    return jacobian
end

function harmonic_rhs!(du, u, _)
    du[1] = -u[2]
    du[2] = u[1]
    return nothing
end


function harmonic_jacobian!(jacobian, _, _)
    jacobian[1, 1] = 0.0
    jacobian[1, 2] = -1.0
    jacobian[2, 1] = 1.0
    jacobian[2, 2] = 0.0
    return jacobian
end

function frequency_harmonic_rhs!(du, u, frequency)
    du[1] = -frequency * u[2]
    du[2] = frequency * u[1]
    return nothing
end


function frequency_harmonic_jacobian!(jacobian, _, frequency)
    jacobian[1, 1] = 0.0
    jacobian[1, 2] = -frequency
    jacobian[2, 1] = frequency
    jacobian[2, 2] = 0.0
    return jacobian
end

@testset "Periodic pseudo-arclength follows the analytical Hopf branch" begin
    factory = parameter -> (parameter, 1.0)
    result = continue_periodic_orbit(parameterized_hopf_rhs!,
        parameterized_hopf_jacobian!, factory, [0.5, 0.0], 2π, 0.25;
        parameter_bounds=(0.04, 0.64),
        options=PeriodicContinuationOptions(initial_step=0.03,
            maximum_step=0.06, minimum_step=1e-6, max_steps=80))
    @test result.initial_orbit.validation == NumericallyValidatedPeriodicOrbit
    @test result.initial_orbit.stability == PeriodicOrbitAttracting
    @test result.negative.termination == :parameter_boundary
    @test result.positive.termination == :parameter_boundary
    @test isempty(result.negative.parameter_reversals)
    @test isempty(result.positive.parameter_reversals)
    for branch in (result.negative, result.positive)
        @test count(attempt -> attempt.accepted, branch.attempts) ==
            length(branch.points) - 1
        @test all(point -> point.orbit.validation == NumericallyValidatedPeriodicOrbit,
            branch.points)
        @test all(point -> point.orbit.stability == PeriodicOrbitAttracting,
            branch.points)
        for attempt in filter(attempt -> attempt.accepted, branch.attempts)
            @test abs(attempt.post_shoot_arclength_residual) <=
                result.options.corrector_atol
            @test attempt.post_shoot_constraint_residual <=
                result.options.corrector_atol
            @test attempt.candidate[1:2] == attempt.orbit.initial_state
            @test attempt.candidate[3] == log(attempt.orbit.period)
        end
        for point in branch.points
            @test point.state[1]^2 + point.state[2]^2 ≈ point.parameter atol=2e-6
            @test point.period ≈ 2π atol=2e-7
            @test point.component_half_ranges ≈ fill(sqrt(point.parameter), 2) atol=2e-4
            @test abs(point.signed_area) ≈ π * point.parameter rtol=2e-3
            @test point.primitive_period_check.primitive
            @test point.divergence_check.resolved
            @test sum(abs2, point.tangent) ≈ 1 atol=1e-10
        end
    end
end

@testset "Periodic parameter derivatives use bounded resolvable stencils" begin
    options = PeriodicContinuationOptions(max_steps=1)
    parameter = 0.25
    nominal = options.parameter_difference_step
    @test FailureOfInhibition2025._pc_parameter_stencil(
        parameter, (prevfloat(parameter), 0.64), options) ==
        (parameter, parameter + nominal)
    @test FailureOfInhibition2025._pc_parameter_stencil(
        parameter, (parameter, 0.64), options) ==
        (parameter, parameter + nominal)
    @test FailureOfInhibition2025._pc_parameter_stencil(
        parameter, (0.04, parameter), options) ==
        (parameter - nominal, parameter)

    balanced_width = 2e-6
    @test FailureOfInhibition2025._pc_parameter_stencil(parameter,
        (parameter - balanced_width, parameter + balanced_width), options) ==
        (parameter - balanced_width, parameter + balanced_width)
    @test FailureOfInhibition2025._pc_parameter_stencil(parameter,
        (parameter - 1e-8, parameter + balanced_width), options) ==
        (parameter, parameter + balanced_width)
    @test isnothing(FailureOfInhibition2025._pc_parameter_stencil(
        parameter, (parameter, nextfloat(parameter)), options))
    maximum_step_options = PeriodicContinuationOptions(
        parameter_difference_step=floatmax(Float64), max_steps=1)
    @test isnothing(FailureOfInhibition2025._pc_parameter_stencil(
        0.0, (-floatmax(Float64), floatmax(Float64)), maximum_step_options))

    probes = Float64[]
    bounded_factory = value -> begin
        push!(probes, value)
        return (value, 1.0)
    end
    lower_bound = prevfloat(parameter)
    result = continue_periodic_orbit(parameterized_hopf_rhs!,
        parameterized_hopf_jacobian!, bounded_factory,
        [sqrt(parameter), 0.0], 2π, parameter;
        parameter_bounds=(lower_bound, 0.64), options=options)
    @test result.initial_orbit.validation == NumericallyValidatedPeriodicOrbit
    @test result.negative.termination == :parameter_boundary
    @test result.positive.termination == :step_limit
    @test length(result.positive.points) == 2
    @test all(lower_bound <= probe <= 0.64 for probe in probes)
    @test parameter in probes
    @test parameter + nominal in probes
end

@testset "Periodic pseudo-arclength traverses a fold of cycles" begin
    factory = parameter -> (parameter, 1.0)
    outer_radius = sqrt(1 + sqrt(0.1))
    result = continue_periodic_orbit(cycle_fold_rhs!, cycle_fold_jacobian!,
        factory, [outer_radius, 0.0], 2π, -0.1;
        parameter_bounds=(-0.25, 0.02),
        options=PeriodicContinuationOptions(initial_step=0.015,
            maximum_step=0.025, minimum_step=1e-7, max_steps=100,
            tangent_alignment_min=0.0))
    branch = result.positive
    @test !isempty(branch.parameter_reversals)
    @test branch.termination == :parameter_boundary
    @test maximum(point.parameter for point in branch.points) > -1e-4
    @test branch.points[1].orbit.stability == PeriodicOrbitRepelling
    @test any(point -> point.orbit.stability == PeriodicOrbitAttracting,
        branch.points)
    @test any(diff([point.parameter for point in branch.points]) .< 0)
    for point in branch.points
        radius_squared = sum(abs2, point.state)
        @test point.parameter ≈ -(radius_squared - 1)^2 atol=3e-6
    end

    tolerance = result.options.rank_atol + result.options.rank_rtol
    for (components, expected_reversals) in (
        ((0.2, 0.0, 0.0, -0.2), [4]),
        ((0.2, tolerance / 2, -0.2), [3]),
        ((0.2, 0.0, 0.2), Int[]),
    )
        reversals = Int[]
        last_orientation = FailureOfInhibition2025._pc_parameter_orientation(
            first(components), result.options)
        for (index, component) in enumerate(components[2:end])
            last_orientation =
                FailureOfInhibition2025._pc_record_parameter_orientation!(
                    reversals, index + 1, last_orientation, component,
                    result.options)
        end
        @test reversals == expected_reversals
    end
end

@testset "Periodic observables distinguish phase, aliases, centers, and stability" begin
    outer_radius = sqrt(1 + sqrt(0.1))
    attracting = solve_periodic_orbit(parameterized_hopf_rhs!,
        parameterized_hopf_jacobian!, (0.25, 1.0), [0.5, 0.0], 2π)
    phase_shifted = solve_periodic_orbit(parameterized_hopf_rhs!,
        parameterized_hopf_jacobian!, (0.25, 1.0), [0.0, 0.5], 2π)
    arbitrary_angle = 2π * 0.17321
    arbitrarily_shifted = solve_periodic_orbit(parameterized_hopf_rhs!,
        parameterized_hopf_jacobian!, (0.25, 1.0),
        0.5 .* [cos(arbitrary_angle), sin(arbitrary_angle)], 2π)
    repeated = solve_periodic_orbit(parameterized_hopf_rhs!,
        parameterized_hopf_jacobian!, (0.25, 1.0), [0.5, 0.0], 4π)
    repelling = solve_periodic_orbit(cycle_fold_rhs!, cycle_fold_jacobian!,
        (-0.1, 1.0), [outer_radius, 0.0], 2π)
    near_neutral_parameter = 1e-6
    neutral = solve_periodic_orbit(parameterized_hopf_rhs!,
        parameterized_hopf_jacobian!, (near_neutral_parameter, 1.0),
        [sqrt(near_neutral_parameter), 0.0], 2π)
    @test attracting.stability == PeriodicOrbitAttracting
    @test repelling.stability == PeriodicOrbitRepelling
    @test neutral.validation == NumericallyValidatedPeriodicOrbit
    @test neutral.stability == PeriodicOrbitStabilityUnresolved
    @test periodic_orbit_half_ranges(attracting) ≈ [0.5, 0.5] atol=2e-4
    distances = periodic_orbit_distances(attracting, [0.0, 0.0])
    @test distances.minimum ≈ 0.5 atol=1e-7
    @test distances.rms ≈ 0.5 atol=1e-7
    @test distances.maximum ≈ 0.5 atol=1e-7
    @test periodic_orbit_signed_area(attracting) ≈ π / 4 rtol=2e-3
    approximately_closed = [[1.0e9, 1.0e9], [1.0e9 + 2, 1.0e9],
        [1.0e9 + 2, 1.0e9 + 1], [1.0e9, 1.0e9 + 1],
        [1.0e9 + 0.001, 1.0e9 - 0.001]]
    translated_area = FailureOfInhibition2025._periodic_signed_area(
        approximately_closed)
    local_area = FailureOfInhibition2025._periodic_signed_area(
        [state .- 1.0e9 for state in approximately_closed])
    reversed_area = FailureOfInhibition2025._periodic_signed_area(
        reverse(approximately_closed))
    @test translated_area ≈ local_area atol=1e-12
    @test translated_area ≈ 1.9995 atol=2e-7
    @test reversed_area ≈ -translated_area atol=1e-12
    winding = periodic_orbit_winding(attracting, [0.0, 0.0])
    @test winding.resolved
    @test winding.winding == 1
    @test !periodic_orbit_winding(attracting, attracting.states[1]).resolved
    @test periodic_orbit_primitive_check(attracting).primitive
    @test !periodic_orbit_primitive_check(repeated).primitive
    @test periodic_orbit_primitive_check(repeated).alias_divisor == 2
    equivalence = phase_invariant_orbit_equivalence(attracting, phase_shifted)
    @test equivalence.equivalent
    @test equivalence.distance < 1e-6
    arbitrary_equivalence = phase_invariant_orbit_equivalence(attracting,
        arbitrarily_shifted)
    @test arbitrary_equivalence.equivalent
    @test arbitrary_equivalence.distance < 1e-7
    @test min(abs(arbitrary_equivalence.phase_shift - (1 - 0.17321)),
        abs(arbitrary_equivalence.phase_shift + 0.17321)) < 1e-6
    @test !phase_invariant_orbit_equivalence(attracting, repeated).equivalent
    divergence = periodic_orbit_divergence_check(attracting,
        parameterized_hopf_jacobian!, (0.25, 1.0))
    @test divergence.resolved
    @test divergence.divergence_multiplier ≈ exp(-4π * 0.25) atol=2e-5

    high_turn = solve_periodic_orbit(frequency_harmonic_rhs!,
        frequency_harmonic_jacobian!, 60.0, [0.5, 0.0], 2π)
    @test high_turn.validation == NumericallyValidatedPeriodicOrbit
    resolved_high_turn = periodic_orbit_winding(high_turn, [0.0, 0.0])
    @test resolved_high_turn.resolved
    @test resolved_high_turn.winding == 60
    @test resolved_high_turn.samples >= 512
    rejected_high_turn = periodic_orbit_winding(high_turn, [0.0, 0.0];
        max_samples=512, max_segment_angle=0.5)
    @test !rejected_high_turn.resolved
    @test isnothing(rejected_high_turn.winding)
    @test rejected_high_turn.reason in (:angular_resolution,
        :sampling_disagreement, :doubled_grid_unavailable)
end

@testset "Periodic continuation retains failures, limits, and input errors" begin
    factory = parameter -> (parameter, 1.0)
    limited = continue_periodic_orbit(parameterized_hopf_rhs!,
        parameterized_hopf_jacobian!, factory, [0.5, 0.0], 2π, 0.25;
        parameter_bounds=(0.04, 0.64),
        options=PeriodicContinuationOptions(max_steps=1))
    @test limited.negative.termination == :step_limit
    @test limited.positive.termination == :step_limit
    @test length(limited.negative.points) == 2
    replayed = continue_periodic_orbit(parameterized_hopf_rhs!,
        parameterized_hopf_jacobian!, factory, [0.5, 0.0], 2π, 0.25;
        parameter_bounds=(0.04, 0.64),
        options=PeriodicContinuationOptions(max_steps=1))
    for (first_branch, second_branch) in
        ((limited.negative, replayed.negative), (limited.positive, replayed.positive))
        @test first_branch.termination == second_branch.termination
        @test [point.parameter for point in first_branch.points] ==
            [point.parameter for point in second_branch.points]
        @test [point.state for point in first_branch.points] ==
            [point.state for point in second_branch.points]
        @test [attempt.status for attempt in first_branch.attempts] ==
            [attempt.status for attempt in second_branch.attempts]
        @test [attempt.candidate for attempt in first_branch.attempts] ==
            [attempt.candidate for attempt in second_branch.attempts]
    end

    retried = continue_periodic_orbit(parameterized_hopf_rhs!,
        parameterized_hopf_jacobian!, factory, [0.5, 0.0], 2π, 0.25;
        parameter_bounds=(0.04, 0.64),
        options=PeriodicContinuationOptions(initial_step=0.8,
            maximum_step=0.8, minimum_step=1e-6, max_steps=2,
            max_corrector_iters=2))
    @test any(attempt -> !attempt.accepted,
        vcat(retried.negative.attempts, retried.positive.attempts))

    loose = continue_periodic_orbit(parameterized_hopf_rhs!,
        parameterized_hopf_jacobian!, factory, [0.5, 0.0], 2π, 0.25;
        parameter_bounds=(0.01, 0.8),
        options=PeriodicContinuationOptions(initial_step=0.4,
            maximum_step=0.4, minimum_step=1e-7, max_steps=1,
            corrector_atol=0.1, tangent_alignment_min=0.0))
    loose_attempt = only(filter(attempt -> attempt.accepted,
        loose.positive.attempts))
    @test loose_attempt.corrector_constraint_residual > 0.05
    @test abs(loose_attempt.post_shoot_arclength_residual) > 0.03
    @test loose_attempt.post_shoot_constraint_residual < 0.01
    @test loose_attempt.corrector_constraint_residual !=
        loose_attempt.post_shoot_constraint_residual
    @test max(abs(loose_attempt.post_shoot_arclength_residual),
        loose_attempt.post_shoot_constraint_residual) <= loose.options.corrector_atol

    unresolved = continue_periodic_orbit(parameterized_hopf_rhs!,
        parameterized_hopf_jacobian!, factory, [0.0, 0.0], 2π, 0.25;
        parameter_bounds=(0.04, 0.64))
    @test unresolved.initial_orbit.validation == PeriodicOrbitUnresolved
    @test unresolved.negative.termination == :initial_orbit_unresolved
    @test isempty(unresolved.negative.points)

    rank_unresolved = continue_periodic_orbit(harmonic_rhs!,
        harmonic_jacobian!, identity, [0.5, 0.0], 2π, 0.25;
        parameter_bounds=(0.04, 0.64))
    @test rank_unresolved.negative.termination == :initial_tangent_unresolved

    for kwargs in ((; initial_step=0.0), (; minimum_step=-1.0),
                   (; state_scales=[1.0, 1.0]), (; log_period_scale=0.0),
                   (; parameter_scale=Inf), (; max_steps=0),
                   (; max_corrector_iters=true), (; tangent_alignment_min=1.0))
        @test_throws ArgumentError PeriodicContinuationOptions(; kwargs...)
    end
    @test_throws ArgumentError PeriodicContinuationOptions(initial_step=big"0.02")
    @test_throws ArgumentError continue_periodic_orbit(parameterized_hopf_rhs!,
        parameterized_hopf_jacobian!, factory, [0.5, 0.0], 2π, 0.25;
        parameter_bounds=(0.5, 0.1))
    @test_throws ArgumentError continue_periodic_orbit(parameterized_hopf_rhs!,
        parameterized_hopf_jacobian!, factory, [0.5, 0.0], 2π, 0.25;
        parameter_bounds=(0.04, 0.64), options=nothing)
    @test_throws ArgumentError continue_periodic_orbit(parameterized_hopf_rhs!,
        parameterized_hopf_jacobian!, _ -> BigFloat(1), [0.5, 0.0], 2π, 0.25;
        parameter_bounds=(0.04, 0.64))
    point_factory_calls = Ref(0)
    guarded_point_factory = _ -> begin
        point_factory_calls[] += 1
        error("point factory reached")
    end
    @test_throws DomainError continue_periodic_orbit(guarded_point_factory,
        [-0.01, 0.5], 2π, 0.25; parameter_bounds=(0.04, 0.64))
    @test_throws DomainError continue_periodic_orbit(guarded_point_factory,
        [0.5, 1.01], 2π, 0.25; parameter_bounds=(0.04, 0.64))
    @test point_factory_calls[] == 0
    @test_throws ErrorException continue_periodic_orbit(guarded_point_factory,
        [0.0, 1.0], 2π, 0.25; parameter_bounds=(0.04, 0.64))
    @test point_factory_calls[] == 1

    callback_factory_calls = Ref(0)
    guarded_callback_factory = _ -> begin
        callback_factory_calls[] += 1
        error("callback factory reached")
    end
    @test_throws ErrorException continue_periodic_orbit(
        parameterized_hopf_rhs!, parameterized_hopf_jacobian!,
        guarded_callback_factory, [-0.25, 1.25], 2π, 0.25;
        parameter_bounds=(0.04, 0.64))
    @test callback_factory_calls[] == 1
end
