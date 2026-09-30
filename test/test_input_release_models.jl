import TOML
include(joinpath(@__DIR__, "..", "scripts", "input_release_models.jl"))

@testset "Matched autonomous release models" begin
    cases = TOML.parsefile(joinpath(@__DIR__, "..", "experiments", "exemplar_models.toml"))["cases"]
    for case in cases, e in (0.0, case["e_to_e"], 24.0)
        pair = InputReleaseModels.release_models(case; e_to_e=e)
        @test pair.off.excitatory === pair.on.excitatory
        @test pair.off.inhibitory === pair.on.inhibitory
        @test pair.off.coupling === pair.on.coupling
        @test pair.off.coupling.e_to_e == e
        @test pair.off.inhibitory.response.onset_threshold == case["theta_on"]
        @test pair.off.inhibitory.response.failure_threshold == case["theta_off"]
        for time in (-1.0, 0.0, 1.0, 20000.0)
            @test drive_value(pair.off.drive, time) == (0.0, 0.0)
            @test drive_value(pair.on.drive, time) == (8.0, 0.0)
        end
    end
    case = first(cases)
    for invalid in (-1.0, NaN, Inf, true, "1", big"1e1000")
        @test_throws ArgumentError InputReleaseModels.release_models(case; e_to_e=invalid)
        @test_throws ArgumentError InputReleaseModels.release_models(case; e_to_e=1, on_input=invalid)
    end
    @test_throws ArgumentError InputReleaseModels.release_models(case; e_to_e=1, on_input=0)
end
