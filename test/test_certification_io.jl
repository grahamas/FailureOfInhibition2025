import TOML

@testset "Exact certificate numeric representation" begin
    exact_values = (Int8(-2), Int128(2)^100, UInt64(2)^63, big(2)^512,
        0.1f0, -0.0f0, nextfloat(0.0f0), 0.1, -0.0, nextfloat(0.0),
        BigFloat("0.1"), setprecision(128) do; BigFloat("0.1"); end,
        -BigFloat(0), ldexp(BigFloat(1), 1_000_000), ldexp(BigFloat(1), -1_000_000))
    for value in exact_values
        encoded = FailureOfInhibition2025._certificate_encode_number(value)
        recovered = FailureOfInhibition2025._certificate_decode_number(encoded)
        @test typeof(recovered) == typeof(value)
        @test isequal(recovered, value)
        value isa BigFloat && @test precision(recovered) == precision(value)
    end
    high_precision = setprecision(BigFloat, 512) do
        BigFloat("0.123456789012345678901234567890123456789012345678901234567890123456789")
    end
    low_ambient_encoding = setprecision(BigFloat, 64) do
        FailureOfInhibition2025._certificate_encode_number(high_precision)
    end
    recovered_high = FailureOfInhibition2025._certificate_decode_number(low_ambient_encoding)
    @test recovered_high == high_precision
    @test precision(recovered_high) == 512
    underflow = FailureOfInhibition2025._certificate_encode_number(BigFloat(1))
    underflow["exponent"] = typemin(Int)
    @test_throws ArgumentError FailureOfInhibition2025._certificate_decode_number(underflow)
    @test_throws ArgumentError FailureOfInhibition2025._certificate_encode_number(1//3)
    @test_throws ArgumentError FailureOfInhibition2025._certificate_encode_number(Inf)
    @test_throws ArgumentError FailureOfInhibition2025._certificate_decode_number(
        Dict("type" => "Float64", "bits" => "7ff8000000000000"))
    malformed = FailureOfInhibition2025._certificate_encode_number(BigFloat(1))
    malformed["precision"] = 1_048_577
    @test_throws ArgumentError FailureOfInhibition2025._certificate_decode_number(malformed)
    malformed["precision"] = 64
    malformed["significand"] = "123456789012345678901234567890"
    @test_throws ArgumentError FailureOfInhibition2025._certificate_decode_number(malformed)
end

@testset "Loaded certification source provenance" begin
    actual = FailureOfInhibition2025._certificate_check_loaded_sources()
    @test actual == FailureOfInhibition2025._CERTIFICATE_LOADED_SOURCE_HASHES
    stale_loaded_hashes = copy(actual)
    stale_loaded_hashes["src/point_model.jl"] = repeat("0", 64)
    @test_throws ArgumentError FailureOfInhibition2025._certificate_check_loaded_sources(stale_loaded_hashes)
    dependencies = FailureOfInhibition2025._certificate_check_loaded_dependencies()
    @test dependencies == FailureOfInhibition2025._CERTIFICATE_LOADED_DEPENDENCY_HASHES
    stale_dependencies = copy(dependencies)
    stale_dependencies["Manifest.toml"] = repeat("0", 64)
    @test_throws ArgumentError FailureOfInhibition2025._certificate_check_loaded_dependencies(stale_dependencies)
end

@testset "Equilibrium certificate archive and proof replay" begin
    model = PointModelParameters(;
        excitatory=PopulationParameters(; timescale=1,
            response=LogisticResponse(; slope=1f0, threshold=0f0)),
        inhibitory=PopulationParameters(; timescale=2.0,
            response=LogisticResponse(; slope=1.0, threshold=0.0)),
        coupling=PointCoupling(0, 0, 0, 0))
    certificate = certify_equilibria(model)
    @test certificate.status == :certified
    @test certificate.equilibrium_count == 1
    @test verify_equilibrium_certificate(certificate)
    rescaled = PointModelParameters(;
        excitatory=PopulationParameters(; timescale=3, response=model.excitatory.response),
        inhibitory=PopulationParameters(; timescale=7.0, response=model.inhibitory.response),
        coupling=model.coupling, drive=model.drive)
    rescaled_provenance = FailureOfInhibition2025._certificate_provenance(rescaled)
    @test rescaled_provenance["model_sha256"] != certificate.provenance["model_sha256"]
    @test rescaled_provenance["equilibrium_problem_sha256"] == certificate.provenance["equilibrium_problem_sha256"]
    for interpretation in (AfferentExcitation, AbstractIntervention)
        tonic_zero = PointModelParameters(; excitatory=model.excitatory, inhibitory=model.inhibitory,
            coupling=model.coupling, drive=PiecewiseConstantDrive(; baseline=(0, 0),
                pulses=(), interpretation))
        tonic_provenance = FailureOfInhibition2025._certificate_provenance(tonic_zero)
        @test tonic_provenance["model_sha256"] != certificate.provenance["model_sha256"]
        @test tonic_provenance["equilibrium_problem_sha256"] == certificate.provenance["equilibrium_problem_sha256"]
    end
    mktempdir() do directory
        path = joinpath(directory, "certificate.toml")
        @test write_equilibrium_certificate(path, certificate) == path
        recovered = read_equilibrium_certificate(path)
        @test verify_equilibrium_certificate(recovered)
        @test recovered.provenance == certificate.provenance
        @test recovered.options.maximum_boxes == certificate.options.maximum_boxes
        @test recovered.model.excitatory.timescale === 1
        @test recovered.model.excitatory.response.slope === 1f0
        @test recovered.model.inhibitory.timescale === 2.0
        @test recovered.model.drive isa NoDrive
        @test recovered.domain.lower == certificate.domain.lower
        @test recovered.domain.upper == certificate.domain.upper
        @test precision(recovered.domain.upper[1]) == precision(certificate.domain.upper[1])
        @test recovered.roots[1].enclosure.lower == certificate.roots[1].enclosure.lower
        @test recovered.roots[1].norm_bound == certificate.roots[1].norm_bound

        function changed_artifact(change)
            data = TOML.parsefile(path)
            change(data)
            changed_path = joinpath(directory, "changed.toml")
            open(changed_path, "w") do io
                TOML.print(io, data)
            end
            return read_equilibrium_certificate(changed_path)
        end
        changed = changed_artifact(data -> data["provenance"]["src/point_model.jl"] = repeat("0", 64))
        @test !verify_equilibrium_certificate(changed)
        # Timescales do not affect fixed points; the exact input hash must still
        # detect altered scientific context even when interval replay would pass.
        changed = changed_artifact(data -> data["model"]["excitatory"]["timescale"]["value"] = "2")
        @test !verify_equilibrium_certificate(changed)
        changed = changed_artifact(data -> empty!(data["nodes"]))
        @test !verify_equilibrium_certificate(changed)
        changed = changed_artifact(data -> data["roots"][1]["norm_bound"] =
            FailureOfInhibition2025._certificate_encode_number(BigFloat(99)))
        @test !verify_equilibrium_certificate(changed)
        @test_throws ArgumentError changed_artifact(data -> data["version"] = 999)

        incomplete = certify_equilibria(model; options=CertificationOptions(; maximum_boxes=0))
        @test incomplete.status == :unresolved
        write_equilibrium_certificate(path, incomplete)
        @test !verify_equilibrium_certificate(read_equilibrium_certificate(path))
        # A success flag cannot discharge a missing coverage obligation.
        forged = changed_artifact() do data
            data["status"] = "certified"
            data["has_equilibrium_count"] = true
            data["equilibrium_count"] = 0
        end
        @test !verify_equilibrium_certificate(forged)
    end

    # Reconstruct FoI responses and tonic drive metadata without changing any
    # stored BigFloat value or its nondefault precision.
    setprecision(BigFloat, 128) do
        tonic_model = PointModelParameters(;
            excitatory=PopulationParameters(; timescale=BigFloat("0.1"),
                response=LogisticResponse(; slope=BigFloat(1), threshold=BigFloat(0))),
            inhibitory=PopulationParameters(; timescale=BigFloat("0.2"),
                response=FailureOfInhibitionResponse(; slope=BigFloat(1),
                    onset_threshold=BigFloat(0), failure_threshold=BigFloat(2))),
            coupling=PointCoupling(BigFloat(0), BigFloat(0), BigFloat(0), BigFloat(0)),
            drive=PiecewiseConstantDrive(; baseline=(BigFloat("0.1"), BigFloat("0.2")),
                pulses=(), interpretation=AfferentExcitation))
        data = FailureOfInhibition2025._certificate_encode_model(tonic_model)
        recovered = setprecision(BigFloat, 64) do
            FailureOfInhibition2025._certificate_decode_model(data)
        end
        @test recovered.inhibitory.response isa FailureOfInhibitionResponse
        @test recovered.excitatory.timescale == tonic_model.excitatory.timescale
        @test precision(recovered.excitatory.timescale) == 128
        @test recovered.drive.baseline == tonic_model.drive.baseline
        @test all(x -> precision(x) == 128, recovered.drive.baseline)
        @test recovered.drive.interpretation == AfferentExcitation
    end
end

@testset "Certificate coverage and root identity mutation rejection" begin
    nonlinear = PointModelParameters(
        excitatory=PopulationParameters(timescale=1.0,
            response=LogisticResponse(slope=1.0, threshold=3 + log(2))),
        inhibitory=PopulationParameters(timescale=1.0,
            response=LogisticResponse(slope=1.0, threshold=0.0)),
        coupling=PointCoupling(12.0, 0.0, 0.0, 0.0),
    )
    original = certify_equilibria(nonlinear)
    @test original.status == :certified
    @test original.equilibrium_count == 3
    @test verify_equilibrium_certificate(original)

    # Copy the mutable proof nodes independently so each mutation exercises one
    # obligation and cannot contaminate the next mutation's baseline.
    function changed_certificate(; nodes=deepcopy(original.nodes),
            roots=deepcopy(original.roots), count=original.equilibrium_count)
        FailureOfInhibition2025.EquilibriumCertificate(original.model, original.options,
            original.domain, roots, nodes, original.status, count,
            original.processed_boxes, original.evaluations, original.elapsed_seconds,
            copy(original.provenance))
    end
    split_index = findfirst(n -> n.kind == :split, original.nodes)
    @test split_index !== nothing
    nodes = deepcopy(original.nodes)
    split_child = nodes[split_index].children[1]
    box = nodes[split_child].box
    nodes[split_child].box = CertificationBox(box.lower, box.lower)
    @test !verify_equilibrium_certificate(changed_certificate(; nodes))

    contraction_index = findfirst(n -> n.kind == :contract, original.nodes)
    @test contraction_index !== nothing
    nodes = deepcopy(original.nodes)
    contraction_child = nodes[contraction_index].children[1]
    box = nodes[contraction_child].box
    nodes[contraction_child].box = CertificationBox(box.lower, box.lower)
    @test !verify_equilibrium_certificate(changed_certificate(; nodes))

    assignment_index = findfirst(n -> n.kind == :root_assignment, original.nodes)
    @test assignment_index !== nothing
    nodes = deepcopy(original.nodes)
    nodes[assignment_index].root_id = mod1(nodes[assignment_index].root_id + 1, length(original.roots))
    @test !verify_equilibrium_certificate(changed_certificate(; nodes))

    roots = deepcopy(original.roots)
    push!(roots, deepcopy(first(roots)))
    @test !verify_equilibrium_certificate(changed_certificate(; roots, count=length(roots)))
    @test verify_equilibrium_certificate(original)

    # This is a valid local proof around the middle root. Even a complete proof
    # on that smaller box cannot establish the public physical-domain claim.
    field, jacobian = FailureOfInhibition2025._certification_callbacks(nonlinear)
    small = FailureOfInhibition2025._certify_equilibria(field, jacobian;
        model=nonlinear, domain=CertificationBox((0.23, 0.30), (0.27, 0.37)))
    @test small.status == :certified
    @test small.equilibrium_count == 1
    @test !verify_equilibrium_certificate(small)
end
