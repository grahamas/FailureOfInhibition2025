@testset "Basin geometry on a known circular basin" begin
    options = BasinMeasurementOptions(grid_points=40, ray_samples=16,
        ray_refinements=10, angles=32,
        pulse_options=PulseExperimentOptions(amplitudes=[0], durations=[1],
            targets=[:I], followup_times=[20],
            diagnostic_options=DiagnosticOptions(window_duration=2, min_samples=5)))
    source = [0.5, 0.5]
    classify(point) = sum((point .- source) .^ 2) < 0.2^2 ? :source : :other
    result = FailureOfInhibition2025._measure_basin_geometry(classify, source, options)
    @test result.area.total_samples == 1600
    @test abs(result.area.observed_fraction - π * 0.2^2) < 0.01
    @test result.area.unresolved_samples == 0
    for direction in (:positive_E, :negative_E, :positive_I, :negative_I)
        boundary = result.directional[direction]
        @test boundary.status == :observed_exit
        @test boundary.lower <= 0.2 <= boundary.upper
        @test boundary.upper - boundary.lower < 1e-4
    end
    @test result.euclidean.status == :observed_exit
    @test result.euclidean.lower <= 0.2 <= result.euclidean.upper
    @test_throws ArgumentError BasinMeasurementOptions(grid_points=true)
    @test_throws ArgumentError BasinMeasurementOptions(angles=3)
    @test_throws ArgumentError FailureOfInhibition2025._measure_basin_geometry(
        _ -> :unresolved, source, options)
end

@testset "Tonic withdrawal keeps total drive nonnegative and returns to baseline" begin
    drive = PiecewiseConstantDrive(baseline=(1.0, 0.0), pulses=(),
        interpretation=AfferentExcitation)
    model = PointModelParameters(
        excitatory=PopulationParameters(timescale=1.0,
            response=LogisticResponse(slope=1.0, threshold=0.0)),
        inhibitory=PopulationParameters(timescale=1.0,
            response=LogisticResponse(slope=1.0, threshold=0.0)),
        coupling=PointCoupling(0.0, 0.0, 0.0, 0.0), drive=drive)
    search = find_equilibria(model; seeds=[[0.4, 1 / 3]])
    state = only(search.equilibria).state
    options = PulseExperimentOptions(amplitudes=[0], durations=[2], targets=[:I],
        followup_times=[50],
        diagnostic_options=DiagnosticOptions(window_duration=3, min_samples=7),
        refinement_levels=0)
    trial = run_tonic_rescue_trial(model, search, state;
        e_reduction=0.5, i_increment=0.25, duration=2, options)
    @test trial.status == :compatible
    @test trial.destination == 1
    @test trial.total_E == 0.5
    @test trial.total_I == 0.25
    @test trial.integrated_E == -1.0
    @test trial.integrated_I == 0.5
    @test length(trial.attempts) == 1
    @test basin_destination(model, search, state; options).destination == 1
    @test_throws ArgumentError run_tonic_rescue_trial(model, search, state;
        e_reduction=1.1, i_increment=0, duration=2, options)
    @test_throws ArgumentError run_tonic_rescue_trial(model, search, state;
        e_reduction=0, i_increment=true, duration=2, options)
    @test_throws ArgumentError run_tonic_rescue_trial(model, search, state;
        e_reduction=0, i_increment=0, duration=0, options)
end

@testset "Basin samples preserve source precision" begin
    options = BasinMeasurementOptions(grid_points=3, ray_samples=4,
        ray_refinements=2, angles=8)
    for T in (Float32, BigFloat)
        state = T[0.5, 0.5]
        samples = Vector{T}[]
        function classify(point)
            @test eltype(point) === T
            push!(samples, copy(point))
            return sum(abs2, point .- state) < (one(T) / 5)^2 ? :source : :other
        end
        result = FailureOfInhibition2025._measure_basin_geometry(classify, state, options)
        @test samples[2] == fill(one(T) / 6, 2)
        @test result.euclidean.status == :observed_exit
        @test result.euclidean.lower isa T
        @test result.euclidean.upper isa T
        @test all(item -> item.domain_limit isa T, values(result.directional))
    end
end
