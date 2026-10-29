import IntervalArithmetic

# Keep these fixtures separate from numerical multistart helpers: their root
# sets and references are known without a nonlinear solver or a study output.
function certification_test_model(inhibitory_response; drive=NoDrive())
    return PointModelParameters(
        excitatory=PopulationParameters(
            timescale=2.0,
            response=LogisticResponse(slope=2.0, threshold=0.3),
        ),
        inhibitory=PopulationParameters(timescale=5.0, response=inhibitory_response),
        coupling=PointCoupling(0.0, 0.0, 0.0, 0.0),
        drive=drive,
    )
end

function certification_test_polynomial(roots; inhibitory_root=3//8)
    function field(x)
        c(value) = IntervalArithmetic.interval(typeof(IntervalArithmetic.inf(x[1])), value)
        polynomial = one(x[1])
        for root in roots
            polynomial *= x[1] - c(root)
        end
        return [polynomial, x[2] - c(inhibitory_root)]
    end
    function jacobian(x)
        c(value) = IntervalArithmetic.interval(typeof(IntervalArithmetic.inf(x[1])), value)
        derivative = zero(x[1])
        for omitted in eachindex(roots)
            term = one(x[1])
            for index in eachindex(roots)
                index == omitted && continue
                term *= x[1] - c(roots[index])
            end
            derivative += term
        end
        return [derivative zero(x[1]); zero(x[1]) one(x[1])]
    end
    return field, jacobian
end

function certification_test_reference(parameters::LogisticResponse, input)
    argument = BigFloat(parameters.slope) * (BigFloat(input) - BigFloat(parameters.threshold))
    value = inv(1 + exp(-argument))
    derivative = BigFloat(parameters.slope) * value * (1 - value)
    return value, derivative
end

function certification_test_reference(parameters::FailureOfInhibitionResponse, input)
    slope = BigFloat(parameters.slope)
    onset = inv(1 + exp(-slope * (BigFloat(input) - BigFloat(parameters.onset_threshold))))
    failure = inv(1 + exp(-slope * (BigFloat(input) - BigFloat(parameters.failure_threshold))))
    return onset - failure, slope * (onset * (1 - onset) - failure * (1 - failure))
end

function certification_test_encloses(enclosure, value)
    return BigFloat(IntervalArithmetic.inf(enclosure)) <= value <=
           BigFloat(IntervalArithmetic.sup(enclosure))
end

@testset "Directed interval response enclosures" begin
    logistic = LogisticResponse(slope=1.0, threshold=0.0)
    failure = FailureOfInhibitionResponse(
        slope=1.0, onset_threshold=0.0, failure_threshold=1.0,
    )
    setprecision(BigFloat, 2048) do
        # Includes ordinary, near-equal saturated logistics, underflow-scale,
        # and zero-crossing arguments. References use independently evaluated
        # high-precision formulas and exact values of stored Float64 inputs.
        for parameters in (logistic, failure), input in (-1000.0, -710.0, -2.0, 0.0, 0.5, 2.0, 40.0, 710.0, 1000.0)
            rate, derivative = FailureOfInhibition2025._certification_response(
                parameters, IntervalArithmetic.interval(input),
            )
            reference, derivative_reference = certification_test_reference(parameters, input)
            @test certification_test_encloses(rate, reference)
            @test certification_test_encloses(derivative, derivative_reference)
            @test IntervalArithmetic.isguaranteed(rate)
            @test IntervalArithmetic.isguaranteed(derivative)
        end
        for parameters in (logistic, failure)
            rate, derivative = FailureOfInhibition2025._certification_response(
                parameters, IntervalArithmetic.interval(-2.0, 3.0),
            )
            for input in range(-2.0, 3.0; length=17)
                reference, derivative_reference = certification_test_reference(parameters, input)
                @test certification_test_encloses(rate, reference)
                @test certification_test_encloses(derivative, derivative_reference)
            end
        end
    end
end

function certification_test_box_contains(box, state)
    return all(box.lower[index] <= state[index] <= box.upper[index] for index in 1:2)
end

function certification_test_complete(certificate, roots)
    @test certificate.status == :certified
    @test certificate.equilibrium_count == length(roots)
    @test length(certificate.roots) == length(roots)
    @test !any(node -> node.kind == :unresolved, certificate.nodes)
    for root in roots
        # Membership in a unique certified enclosure also catches duplicate
        # assignments at an exact subdivision plane.
        @test count(result -> certification_test_box_contains(result.enclosure, root), certificate.roots) == 1
    end
end

@testset "Synthetic exhaustive root certificates" begin
    certify = FailureOfInhibition2025._certify_equilibria
    verify = FailureOfInhibition2025._verify_certificate
    domain = CertificationBox((0, 0), (1//2, 1//2))

    # Constant first component excludes every point, including boundaries.
    no_root(x) = [one(x[1]), x[2] - IntervalArithmetic.interval(typeof(IntervalArithmetic.inf(x[2])), 1//4)]
    no_root_jacobian(x) = [zero(x[1]) zero(x[1]); zero(x[1]) one(x[1])]
    empty = certify(no_root, no_root_jacobian; domain=domain)
    certification_test_complete(empty, ())
    @test verify(no_root, no_root_jacobian, empty)

    for expected_roots in ((1//4,), (1//8, 3//8), (1//8, 1//4, 3//8))
        field, jacobian = certification_test_polynomial(expected_roots)
        certificate = certify(field, jacobian; domain=domain)
        certification_test_complete(certificate, [(root, 3//8) for root in expected_roots])
        @test verify(field, jacobian, certificate)
        @test all(root -> root.norm_bound < 1, certificate.roots)
    end

    # A root on both outer boundaries is admissible and must be counted once.
    field, jacobian = certification_test_polynomial((0//1,); inhibitory_root=1//2)
    boundary = certify(field, jacobian; domain=domain)
    certification_test_complete(boundary, [(0//1, 1//2)])
    @test verify(field, jacobian, boundary)

    # Two nearby roots still require separate uniqueness enclosures.
    close_roots = (1//4 - 1//4096, 1//4 + 1//4096)
    field, jacobian = certification_test_polynomial(close_roots)
    close = certify(field, jacobian; domain=domain,
        options=CertificationOptions(maximum_boxes=20_000, maximum_depth=80))
    certification_test_complete(close, [(root, 3//8) for root in close_roots])
    @test verify(field, jacobian, close)

    # A repeated root has a singular derivative. Exhausting subdivisions must
    # preserve an unknown total count, never manufacture a uniqueness proof.
    field, jacobian = certification_test_polynomial((1//4, 1//4))
    degenerate = certify(field, jacobian; domain=domain,
        options=CertificationOptions(maximum_boxes=100, maximum_depth=12))
    @test degenerate.status == :unresolved
    @test isnothing(degenerate.equilibrium_count)
    @test any(node -> node.kind == :unresolved, degenerate.nodes)
    @test verify(field, jacobian, degenerate)
end

@testset "Certification resource limits cannot imply completeness" begin
    field, jacobian = certification_test_polynomial((1//8, 1//4, 3//8))
    for options in (
        CertificationOptions(maximum_boxes=0),
        CertificationOptions(maximum_depth=0),
        CertificationOptions(maximum_evaluations=0),
        CertificationOptions(maximum_records=1),
        CertificationOptions(time_limit=0.0),
    )
        certificate = FailureOfInhibition2025._certify_equilibria(field, jacobian; options=options)
        @test certificate.status == :unresolved
        @test isnothing(certificate.equilibrium_count)
        @test any(node -> node.kind == :unresolved, certificate.nodes)
    end
end

@testset "Real-model analytic equilibrium certificates" begin
    for inhibitory_response in (
        LogisticResponse(slope=1.5, threshold=0.2),
        FailureOfInhibitionResponse(slope=1.5, onset_threshold=0.2, failure_threshold=1.0),
    ), baseline in ((0.0, 0.0), (0.7, 0.6), (-100.0, -100.0), (100.0, 100.0))
        drive = baseline == (0.0, 0.0) ? NoDrive() : PiecewiseConstantDrive(
            baseline=baseline, pulses=(), interpretation=AbstractIntervention,
        )
        model = certification_test_model(inhibitory_response; drive=drive)
        certificate = certify_equilibria(model)
        expected = setprecision(BigFloat, 2048) do
            e_rate, _ = certification_test_reference(model.excitatory.response, baseline[1])
            i_rate, _ = certification_test_reference(model.inhibitory.response, baseline[2])
            (e_rate / (1 + e_rate), i_rate / (1 + i_rate))
        end
        certification_test_complete(certificate, [expected])
    end

    saturated_drive = PiecewiseConstantDrive(
        baseline=(40.0, 40.0), pulses=(), interpretation=AfferentExcitation,
    )
    saturated_model = certification_test_model(
        LogisticResponse(slope=1.5, threshold=0.2); drive=saturated_drive,
    )
    limited = certify_equilibria(saturated_model;
        options=CertificationOptions(maximum_precision=53))
    @test limited.status == :unresolved
    @test isnothing(limited.equilibrium_count)
    @test any(node -> node.reason == :saturation_precision_limit, limited.nodes)
    escalated = certify_equilibria(saturated_model;
        options=CertificationOptions(maximum_precision=256))
    saturated_expected = setprecision(BigFloat, 2048) do
        e_rate, _ = certification_test_reference(saturated_model.excitatory.response, 40.0)
        i_rate, _ = certification_test_reference(saturated_model.inhibitory.response, 40.0)
        (e_rate / (1 + e_rate), i_rate / (1 + i_rate))
    end
    certification_test_complete(escalated, [saturated_expected])
    @test any(root -> root.precision > 53, escalated.roots)

    nonlinear = PointModelParameters(
        excitatory=PopulationParameters(timescale=1.0,
            response=LogisticResponse(slope=1.0, threshold=3 + log(2))),
        inhibitory=PopulationParameters(timescale=1.0,
            response=LogisticResponse(slope=1.0, threshold=0.0)),
        coupling=PointCoupling(12.0, 0.0, 0.0, 0.0),
    )
    three = certify_equilibria(nonlinear)
    @test three.status == :certified
    @test three.equilibrium_count == 3
    @test length(three.roots) == 3
    field, jacobian = FailureOfInhibition2025._certification_callbacks(nonlinear)
    @test FailureOfInhibition2025._verify_certificate(field, jacobian, three)

    pulse_drive = PiecewiseConstantDrive(
        baseline=(0.0, 0.0),
        pulses=(DrivePulse(onset=1.0, offset=2.0, increment=(0.1, 0.2)),),
        interpretation=AfferentExcitation,
    )
    pulsed = certification_test_model(LogisticResponse(slope=1.5, threshold=0.2); drive=pulse_drive)
    @test_throws ArgumentError certify_equilibria(pulsed)
    @test find_equilibria(certification_test_model(LogisticResponse(slope=1.5, threshold=0.2))).completeness == CompletenessNotCertified
end

@testset "Certification validation and rigorous rounding" begin
    @test_throws ArgumentError CertificationBox((1, 0), (0, 1))
    @test_throws ArgumentError CertificationBox((NaN, 0), (1, 1))
    @test_throws ArgumentError CertificationBox((0, 0), (Inf, 1))
    @test_throws ArgumentError CertificationOptions(maximum_boxes=-1)
    @test_throws ArgumentError CertificationOptions(maximum_depth=-1)
    @test_throws ArgumentError CertificationOptions(maximum_evaluations=-1)
    @test_throws ArgumentError CertificationOptions(maximum_records=0)
    @test_throws ArgumentError CertificationOptions(maximum_precision=52)
    @test_throws ArgumentError CertificationOptions(time_limit=-1.0)
    @test_throws ArgumentError CertificationOptions(time_limit=Inf)

    rational_model = certification_test_model(LogisticResponse(slope=3//2, threshold=1//5))
    @test_throws ArgumentError certify_equilibria(rational_model)

    field, jacobian = certification_test_polynomial((1//4,))
    # The backend's fast, nondirected mode is a process-wide setting. Always
    # restore it so this negative check cannot weaken later package tests.
    original_rounding = IntervalArithmetic.configuration_options.rounding
    try
        IntervalArithmetic.configure(; rounding=:none)
        @test_throws ArgumentError FailureOfInhibition2025._certify_equilibria(field, jacobian)
    finally
        IntervalArithmetic.configure(; rounding=original_rounding)
    end

    # Ordinary scalar mixing loses the interval package's guarantee flag.
    # A user callback with that result must not be promoted to a certificate.
    invalid_field(x) = [x[1] - 0.25, x[2] - 0.375]
    for callback in (
        invalid_field,
        x -> [IntervalArithmetic.emptyinterval(typeof(IntervalArithmetic.inf(x[1]))), zero(x[2])],
        x -> [IntervalArithmetic.nai(typeof(IntervalArithmetic.inf(x[1]))), zero(x[2])],
        x -> [IntervalArithmetic.entireinterval(typeof(IntervalArithmetic.inf(x[1]))), zero(x[2])],
        x -> [IntervalArithmetic.interval(typeof(IntervalArithmetic.inf(x[1])), 0, 1, IntervalArithmetic.trv), zero(x[2])],
        x -> [sqrt(IntervalArithmetic.interval(typeof(IntervalArithmetic.inf(x[1])), -1, 1)), zero(x[2])],
    )
        invalid = FailureOfInhibition2025._certify_equilibria(callback, jacobian)
        @test invalid.status == :unresolved
        @test isnothing(invalid.equilibrium_count)
    end
end

@testset "Interval balance and Jacobian enclose independent high precision" begin
    setprecision(BigFloat, 2048) do
        for inhibitory_response in (
            LogisticResponse(slope=1.5, threshold=0.2),
            FailureOfInhibitionResponse(slope=1.5, onset_threshold=0.2, failure_threshold=1.0),
        ), baseline in ((0.0, 0.0), (-100.0, -100.0), (100.0, 100.0))
            model = PointModelParameters(
                excitatory=PopulationParameters(timescale=2.0,
                    response=LogisticResponse(slope=2.0, threshold=0.3)),
                inhibitory=PopulationParameters(timescale=5.0, response=inhibitory_response),
                coupling=PointCoupling(1.2, 0.6, 0.7, 0.4),
                drive=PiecewiseConstantDrive(baseline=baseline, pulses=(),
                    interpretation=AbstractIntervention),
            )
            field, jacobian = FailureOfInhibition2025._certification_callbacks(model)
            for state in ((0.0, 0.0), (0.125, 0.375), (0.5, 0.5))
                x, y = BigFloat.(state)
                a, b, c, d = BigFloat.((model.coupling.e_to_e, model.coupling.i_to_e,
                    model.coupling.e_to_i, model.coupling.i_to_i))
                input_e = BigFloat(baseline[1]) + a * x - b * y
                input_i = BigFloat(baseline[2]) + c * x - d * y
                e, de = certification_test_reference(model.excitatory.response, input_e)
                i, di = certification_test_reference(model.inhibitory.response, input_i)
                reference = [-1 - e + (1 - x) * de * a  -(1 - x) * de * b;
                    (1 - y) * di * c  -1 - i - (1 - y) * di * d]
                interval_state = IntervalArithmetic.interval.(collect(state))
                field_enclosure = field(interval_state)
                field_reference = [-x + (1 - x) * e, -y + (1 - y) * i]
                for index in eachindex(field_reference)
                    @test certification_test_encloses(field_enclosure[index], field_reference[index])
                end
                enclosure = jacobian(interval_state)
                for index in eachindex(reference)
                    @test certification_test_encloses(enclosure[index], reference[index])
                end
            end
        end
    end
end
