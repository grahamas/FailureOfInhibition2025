using LinearAlgebra: Diagonal

@testset "Local spectral classification" begin
    fixtures = (
        ([-2.0 0.0; 0.0 -1.0], Attracting, RealDistinctSpectrum),
        ([-1.0 -2.0; 2.0 -1.0], Attracting, ComplexConjugateSpectrum),
        ([2.0 0.0; 0.0 1.0], Repelling, RealDistinctSpectrum),
        ([1.0 -2.0; 2.0 1.0], Repelling, ComplexConjugateSpectrum),
        ([-1.0 0.0; 0.0 2.0], Saddle, RealDistinctSpectrum),
        ([-1.0 1.0; 0.0 -1.0], Attracting, RealRepeatedSpectrum),
        ([1.0 1.0; 0.0 1.0], Repelling, RealRepeatedSpectrum),
        ([0.0 0.0; 0.0 -1.0], StabilityUnresolved, RealDistinctSpectrum),
        ([0.0 -1.0; 1.0 0.0], StabilityUnresolved, ComplexConjugateSpectrum),
    )

    for (jacobian, expected_class, expected_geometry) in fixtures
        result = classify_local_stability(jacobian)
        @test result.classification == expected_class
        @test result.geometry == expected_geometry
        @test result.jacobian == jacobian
        @test result.trace ≈ sum(Diagonal(jacobian))
        @test result.determinant ≈
              jacobian[1, 1] * jacobian[2, 2] - jacobian[1, 2] * jacobian[2, 1]
        @test result.spectral_abscissa == maximum(real, result.eigenvalues)
        @test length(result.thresholds) == 2
    end

    for abstract_jacobian in (
        Real[-1.0 0.0; 0.0 -2.0],
        Any[-1.0 0.0; 0.0 -2.0],
    )
        result = classify_local_stability(abstract_jacobian)
        @test result.jacobian isa Matrix{Float64}
        @test result.classification == Attracting
        @test result.geometry == RealDistinctSpectrum
    end

    tolerance = StabilityOptions(spectral_atol=1.0e-8, spectral_rtol=1.0e-6)
    unresolved = classify_local_stability([5.0e-9 -1.0; 1.0 5.0e-9]; options=tolerance)
    resolved = classify_local_stability([2.0e-6 -1.0; 1.0 2.0e-6]; options=tolerance)
    @test unresolved.classification == StabilityUnresolved
    @test resolved.classification == Repelling

    nearly_real = classify_local_stability([-1.0 -7.5e-9; 7.5e-9 -1.0])
    @test nearly_real.classification == Attracting
    @test nearly_real.geometry == RealRepeatedSpectrum

    @test_throws ArgumentError StabilityOptions(spectral_atol=-1.0)
    @test_throws ArgumentError StabilityOptions(spectral_rtol=Inf)
    @test_throws ArgumentError classify_local_stability(zeros(3, 3))
    @test_throws ArgumentError classify_local_stability([0.0 NaN; 0.0 1.0])
    @test_throws ArgumentError classify_local_stability(BigFloat[1 0; 0 1])
end

@testset "Model stability uses original time constants" begin
    coupling = PointCoupling(e_to_e=0.0, i_to_e=0.0, e_to_i=0.0, i_to_i=0.0)
    drive = PiecewiseConstantDrive(
        baseline=(0.7, 0.6),
        pulses=(),
        interpretation=AfferentExcitation,
    )
    excitatory_response = LogisticResponse(slope=2.0, threshold=0.3)
    inhibitory_response = LogisticResponse(slope=1.5, threshold=0.2)

    function scaled_model(scale)
        return PointModelParameters(
            excitatory=PopulationParameters(
                timescale=2.0scale,
                response=excitatory_response,
            ),
            inhibitory=PopulationParameters(
                timescale=5.0scale,
                response=inhibitory_response,
            ),
            coupling=coupling,
            drive=drive,
        )
    end

    unscaled = solve_equilibrium(scaled_model(1.0), [0.4, 0.4])
    scaled = solve_equilibrium(scaled_model(17.0), [0.4, 0.4])
    @test unscaled.attempt.validation == AdmissibleCandidate
    @test scaled.attempt.validation == AdmissibleCandidate
    @test scaled.attempt.candidate ≈ unscaled.attempt.candidate atol=1.0e-12
    @test scaled.stability.classification == unscaled.stability.classification == Attracting
    @test scaled.stability.eigenvalues ≈ unscaled.stability.eigenvalues ./ 17
    @test scaled.stability.trace ≈ unscaled.stability.trace / 17
    @test scaled.stability.determinant ≈ unscaled.stability.determinant / 17^2

    direct_jacobian = zeros(2, 2)
    point_jacobian!(direct_jacobian, unscaled.attempt.candidate, scaled_model(1.0), 0.0)
    @test unscaled.stability.jacobian ≈ direct_jacobian
end

@testset "Near-singular model equilibrium" begin
    model = PointModelParameters(
        excitatory=PopulationParameters(
            timescale=1.0,
            response=LogisticResponse(slope=1.0, threshold=2 + log(2)),
        ),
        inhibitory=PopulationParameters(
            timescale=1.0,
            response=LogisticResponse(slope=1.0, threshold=0.0),
        ),
        coupling=PointCoupling(e_to_e=8.0, i_to_e=0.0, e_to_i=0.0, i_to_i=0.0),
    )
    result = solve_equilibrium(model, [0.25, 1 / 3])
    @test result.attempt.validation == AdmissibleCandidate
    @test result.attempt.near_singular
    @test result.stability.classification == StabilityUnresolved
    @test result.stability.eigenvalues ≈ ComplexF64[-1.5, 0.0]
end

@testset "Exact coupled model spectra" begin
    logistic = LogisticResponse(slope=6.0, threshold=0.0)
    population = PopulationParameters(timescale=1.0, response=logistic)
    fixtures = (
        (
            PointCoupling(e_to_e=1.0, i_to_e=2.0, e_to_i=2.0, i_to_i=0.0),
            (1 / 3, -2 / 3),
            Attracting,
            [-0.5 -2.0; 2.0 -1.5],
        ),
        (
            PointCoupling(e_to_e=4.0, i_to_e=3.0, e_to_i=3.0, i_to_i=0.0),
            (-1 / 3, -1.0),
            Repelling,
            [2.5 -3.0; 3.0 -1.5],
        ),
    )
    for (coupling, baseline, classification, expected_jacobian) in fixtures
        drive = PiecewiseConstantDrive(
            baseline=baseline,
            pulses=(),
            interpretation=AbstractIntervention,
        )
        model = PointModelParameters(
            excitatory=population,
            inhibitory=population,
            coupling=coupling,
            drive=drive,
        )
        result = solve_equilibrium(model, [1 / 3, 1 / 3])
        @test result.attempt.validation == AdmissibleCandidate
        @test result.attempt.candidate ≈ [1 / 3, 1 / 3] atol=1.0e-12
        @test result.stability.jacobian ≈ expected_jacobian atol=1.0e-12
        @test result.stability.classification == classification
        @test result.stability.geometry == ComplexConjugateSpectrum
    end

    repelling_model = PointModelParameters(
        excitatory=PopulationParameters(
            timescale=1.0,
            response=LogisticResponse(slope=1.0, threshold=3 + log(2)),
        ),
        inhibitory=PopulationParameters(
            timescale=1.0,
            response=FailureOfInhibitionResponse(
                slope=1.0,
                onset_threshold=0.0,
                failure_threshold=log(9),
            ),
        ),
        coupling=PointCoupling(e_to_e=12.0, i_to_e=0.0, e_to_i=0.0, i_to_i=21.0),
        drive=PiecewiseConstantDrive(
            baseline=(0.0, 6 + log(9)),
            pulses=(),
            interpretation=AfferentExcitation,
        ),
    )
    repelling_result = solve_equilibrium(repelling_model, [0.25, 2 / 7])
    @test repelling_result.attempt.validation == AdmissibleCandidate
    @test repelling_result.stability.jacobian ≈ [2 / 3 0.0; 0.0 1.0] atol=1.0e-12
    @test repelling_result.stability.classification == Repelling
    @test repelling_result.stability.geometry == RealDistinctSpectrum
end
