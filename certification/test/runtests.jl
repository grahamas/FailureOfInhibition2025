using Test
using IntervalArithmetic
using FailureOfInhibition2025
using TOML

include(joinpath(@__DIR__, "..", "certify_attractors.jl"))
using .AttractorCertification

const AC = AttractorCertification
const physical_domain = [interval(0.0, 1.0), interval(0.0, 1.0)]

@testset "Interval certificate gates" begin
    sink_field(box) = [interval(0.25) - box[1], interval(0.75) - box[2]]
    sink_jacobian(box) = [interval(-1) interval(0);
                          interval(0) interval(-1)]
    one_sink = AC.certify_attractor_count(sink_field, sink_jacobian,
        physical_domain)
    @test one_sink.status == :certified_exact
    @test one_sink.exact_count == 1
    @test one_sink.root_coverage.equilibrium_count == 1
    @test isempty(one_sink.root_coverage.unresolved)

    cubic_field(box) = begin
        E, Istate = box
        [-((E - interval(0.2)) * (E - interval(0.5)) *
            (E - interval(0.8))), interval(0.5) - Istate]
    end
    cubic_jacobian(box) = begin
        E = box[1]
        a, b, c = E - interval(0.2), E - interval(0.5),
            E - interval(0.8)
        [-(a*b + a*c + b*c) interval(0);
         interval(0) interval(-1)]
    end
    two_sinks = AC.certify_attractor_count(cubic_field, cubic_jacobian,
        physical_domain)
    @test two_sinks.status == :certified_exact
    @test two_sinks.exact_count == 2
    @test two_sinks.root_coverage.equilibrium_count == 3
    @test isempty(two_sinks.root_coverage.unresolved)

    rotation_field(box) = begin
        x, y = box .- interval(0.5)
        radius_term = interval(0.04) - x^2 - y^2
        [x * radius_term - y, y * radius_term + x]
    end
    rotation_jacobian(box) = begin
        x, y = box .- interval(0.5)
        radius_term = interval(0.04) - x^2 - y^2
        [radius_term - interval(2)*x^2  -interval(1) - interval(2)*x*y;
         interval(1) - interval(2)*x*y  radius_term - interval(2)*y^2]
    end
    cycle_candidate = AC.certify_attractor_count(rotation_field,
        rotation_jacobian, physical_domain; maximum_depth=24)
    @test cycle_candidate.status == :not_certified
    @test cycle_candidate.exact_count === nothing

    limited = AC.certify_attractor_count(cubic_field, cubic_jacobian,
        physical_domain; maximum_boxes=1)
    @test limited.status == :not_certified
    @test limited.root_coverage.equilibrium_count === nothing
    @test !isempty(limited.root_coverage.unresolved)

    unsafe_field(box) = [box[1] + 1, box[2] + 1]
    @test_throws ArgumentError AC.certify_roots(unsafe_field, sink_jacobian,
        physical_domain)
end

@testset "Model enclosure and selected cases" begin
    config = TOML.parsefile(joinpath(@__DIR__, "..", "..", "experiments",
        "basin_rescue.toml"))
    cases = Dict(item["name"] => item for item in config["cases"])
    for case_name in ("figure3", "figure4_rising")
        model = AC._case_model(cases[case_name])
        for state in ([0.0, 0.0], [0.17, 0.29], [0.5, 0.45], [1.0, 1.0])
            enclosure, jac_enclosure = AC.interval_field(model,
                interval.(state))
            derivative, jacobian = zeros(2), zeros(2, 2)
            point_rhs!(derivative, state, model, 0.0)
            point_jacobian!(jacobian, state, model, 0.0)
            @test all(isguaranteed, enclosure)
            @test all(isguaranteed, jac_enclosure)
            @test all(index -> inf(enclosure[index]) <= derivative[index] <=
                sup(enclosure[index]), eachindex(derivative))
            @test all(index -> inf(jac_enclosure[index]) <= jacobian[index] <=
                sup(jac_enclosure[index]), eachindex(jacobian))
        end
        field(box) = first(AC.interval_field(model, box))
        jacobian(box) = last(AC.interval_field(model, box))
        result = AC.certify_attractor_count(field, jacobian, physical_domain)
        @test isempty(result.root_coverage.unresolved)
        @test result.root_coverage.equilibrium_count ==
            (case_name == "figure3" ? 7 : 5)
        @test result.certified_attracting_equilibria == 3
        @test result.inward_boundary
        @test result.status == :not_certified
        @test result.exact_count === nothing
    end
end
