function hopf_test_rhs!(du, u, parameters)
    growth, frequency = parameters
    radius_squared = sum(abs2, u)
    du[1] = growth * (1 - radius_squared) * u[1] - frequency * u[2]
    du[2] = frequency * u[1] + growth * (1 - radius_squared) * u[2]
    return nothing
end

@testset "Finite-trace screening only proposes a shooting seed" begin
    include(joinpath(@__DIR__, "..", "scripts", "run_periodic_candidates.jl"))
    policy = PeriodicCandidateExperiment.screening_policy(false)
    times = collect(0.0:1.0:5000.0)
    states = [[0.25 + 0.1sin(2π * time / 100), 0.25 + 0.1cos(2π * time / 100)] for time in times]
    recurrence = PeriodicCandidateExperiment.recurrence_seed((t=times, u=states), policy)
    @test recurrence.status == "shooting_seed_only"
    @test recurrence.period_guess ≈ 100.0 atol=1e-9
    @test recurrence.relative_period_spread < 1e-9
    @test length(recurrence.crossings) >= policy.minimum_upward_crossings
    stationary = PeriodicCandidateExperiment.recurrence_seed(
        (t=times, u=[zeros(2) for _ in times]), policy)
    @test stationary.status == "no_seed_small_E_range"
    @test isnan(stationary.period_guess)
    config = PeriodicCandidateExperiment.Map.load_config(
        joinpath(@__DIR__, "..", "experiments", "coexistence.toml"))
    @test length(PeriodicCandidateExperiment.requested_cases(config, true)) == 1
    for restricted_config in (merge(config, (; e_to_i_values=[18.0])),
                              merge(config, (; failure_threshold_values=[10.0])))
        @test_throws ArgumentError PeriodicCandidateExperiment.requested_cases(restricted_config, true)
        @test_throws ArgumentError PeriodicCandidateExperiment.requested_cases(restricted_config, false)
    end
end

function hopf_test_jacobian!(jacobian, u, parameters)
    growth, frequency = parameters
    radial = growth * (1 - sum(abs2, u))
    jacobian[1, 1] = radial - 2growth * u[1]^2
    jacobian[1, 2] = -2growth * u[1] * u[2] - frequency
    jacobian[2, 1] = -2growth * u[1] * u[2] + frequency
    jacobian[2, 2] = radial - 2growth * u[2]^2
    return jacobian
end

@testset "Periodic shooting on an analytical attracting cycle" begin
    parameters = (0.1, 1.0)
    result = solve_periodic_orbit(hopf_test_rhs!, hopf_test_jacobian!, parameters,
                                 [1.02, 0.03], 6.2)
    @test result.validation == NumericallyValidatedPeriodicOrbit
    @test result.stability == PeriodicOrbitAttracting
    @test isempty(result.reasons)
    @test result.integration_success
    @test result.period ≈ 2π atol=1e-7
    @test sum(abs2, result.initial_state) ≈ 1 atol=1e-7
    @test result.closure_residual <= result.options.validation_atol
    @test result.phase_residual <= result.options.validation_atol
    @test result.equation_residual <= result.options.validation_atol
    @test maximum(abs, result.amplitudes .- 2) < 1e-3
    @test result.transverse_multiplier ≈ exp(-0.2 * 2π) atol=1e-7
    @test minimum(abs.(result.floquet_multipliers .- 1)) < 1e-7
    @test result.refinement_difference.waveform < 1e-7
    @test result.refinement_difference.period < 1e-7
    @test result.refinement_difference.monodromy < 1e-7
    phases = periodic_orbit_phases(result, [0.0, 0.25, 0.5, 0.75])
    @test phases[1] ≈ result.initial_state
    @test phases[3] ≈ -phases[1] atol=1e-7
    @test phases[2] ≈ [-phases[1][2], phases[1][1]] atol=1e-7
    phases[1][1] = 99.0
    @test result.initial_state[1] != 99.0
    @test_throws ArgumentError periodic_orbit_phases(result, [-0.1])
    @test_throws ArgumentError periodic_orbit_phases(result, [1.0])
end

@testset "Repelling and neutral periodic solutions" begin
    repelling = solve_periodic_orbit(hopf_test_rhs!, hopf_test_jacobian!, (-0.1, 1.0),
                                    [1.002, 0.0], 6.28)
    @test repelling.validation == NumericallyValidatedPeriodicOrbit
    @test repelling.stability == PeriodicOrbitRepelling
    @test repelling.transverse_multiplier ≈ exp(0.2 * 2π) atol=1e-6

    neutral = solve_periodic_orbit(hopf_test_rhs!, hopf_test_jacobian!, (0.0, 1.0),
                                  [1.0, 0.0], 2π)
    @test neutral.validation == NumericallyValidatedPeriodicOrbit
    @test neutral.stability == PeriodicOrbitStabilityUnresolved
end

@testset "Equilibria, transients and failed integrations are unresolved" begin
    equilibrium = solve_periodic_orbit(hopf_test_rhs!, hopf_test_jacobian!, (0.1, 1.0),
                                      [0.0, 0.0], 6.2)
    @test equilibrium.validation == PeriodicOrbitUnresolved
    @test :degenerate_phase_reference in equilibrium.reasons
    @test_throws ArgumentError periodic_orbit_phases(equilibrium, [0.0])

    # A damped spiral can resemble a cycle over a finite window but has no
    # nonconstant periodic solution. Shooting must not label it as one.
    function damped_rhs!(du, u, _)
        du[1] = -0.1u[1] - u[2]
        du[2] = u[1] - 0.1u[2]
        return nothing
    end
    function damped_jacobian!(J, u, _)
        J[1, 1], J[1, 2], J[2, 1], J[2, 2] = -0.1, -1.0, 1.0, -0.1
        return J
    end
    transient = solve_periodic_orbit(damped_rhs!, damped_jacobian!, nothing,
                                    [1.0, 0.0], 2π)
    @test transient.validation == PeriodicOrbitUnresolved
    @test transient.stability == PeriodicOrbitStabilityUnresolved

    incomplete = solve_periodic_orbit(hopf_test_rhs!, hopf_test_jacobian!, (0.1, 1.0),
                                     [1.0, 0.0], 2π;
                                     options=PeriodicOrbitOptions(ode_maxiters=1))
    @test incomplete.validation == PeriodicOrbitUnresolved
    @test !incomplete.integration_success
    @test :incomplete_integration in incomplete.reasons

    # Even an exactly circular solution is excluded when it is below the
    # explicitly configured nonconstant-amplitude threshold.
    small = solve_periodic_orbit(hopf_test_rhs!, hopf_test_jacobian!, (0.0, 1.0),
                                [1e-7, 0.0], 2π)
    @test small.validation == PeriodicOrbitUnresolved
    @test :constant_or_small_amplitude_solution in small.reasons
end

@testset "Periodic API validates precision, phase and autonomous context" begin
    for keywords in ((; ode_abstol=0.0), (; refinement_factor=1.0), (; maxiters=0),
                     (; samples=4), (; min_period=2, max_period=1),
                     (; floquet_atol=NaN), (; validation_atol=big"1e-6"),
                     (; validation_atol=true), (; maxiters=true),
                     (; ode_maxiters=true), (; samples=true))
        @test_throws ArgumentError PeriodicOrbitOptions(; keywords...)
    end
    @test_throws ArgumentError solve_periodic_orbit(hopf_test_rhs!, hopf_test_jacobian!,
                                                   (0.1, 1.0), BigFloat[1, 0], 6.2)
    @test_throws ArgumentError solve_periodic_orbit(hopf_test_rhs!, hopf_test_jacobian!,
                                                   (0.1, 1.0), [1.0, 0.0], big"6.2")
    @test_throws ArgumentError solve_periodic_orbit(hopf_test_rhs!, hopf_test_jacobian!,
                                                   (big"0.1", 1.0), [1.0, 0.0], 6.2)
    @test_throws ArgumentError solve_periodic_orbit(hopf_test_rhs!, hopf_test_jacobian!,
                                                   (0.1, 1.0), [1.0, 0.0], 0.0)
    @test_throws ArgumentError solve_periodic_orbit(hopf_test_rhs!, hopf_test_jacobian!,
                                                   (0.1, 1.0), [1.0, 0.0], true)
    model = synthetic_model()
    @test_throws DomainError solve_periodic_orbit(model, [1.1, 0.0], 2.0)
    @test_throws ArgumentError solve_periodic_orbit(model, [0.1, 0.0], 2.0; options=nothing)
    pulsed_model = synthetic_model(drive=PiecewiseConstantDrive(
        baseline=(0.0, 0.0), interpretation=AfferentExcitation,
        pulses=(DrivePulse(onset=1000.0, offset=1001.0, increment=(1.0, 0.0)),),
    ))
    @test_throws ArgumentError solve_periodic_orbit(pulsed_model, [0.1, 0.1], 2.0)
    high_precision = synthetic_model(excitatory_timescale=big"2.0")
    @test_throws ArgumentError solve_periodic_orbit(high_precision, [0.1, 0.1], 2.0)

    # The public model wrapper rejects an equilibrium as a cycle as well.
    solved = solve_equilibrium(model, [0.2, 0.2])
    stationary = solve_periodic_orbit(model, solved.attempt.candidate, 2.0)
    @test stationary.validation == PeriodicOrbitUnresolved
end
