include(joinpath(@__DIR__, "..", "scripts", "run_low_ratio_coexistence.jl"))

const LowRatio = LowRatioCoexistenceExperiment

@testset "Low-ratio coexistence contract" begin
    path = joinpath(@__DIR__, "..", "experiments", "low_ratio_coexistence.toml")
    config = LowRatio.load_config(path)
    @test (config.lower, config.upper) == (0.2, 4.4)
    @test length(LowRatio.BaseStudy.plane_cells(config.base)) == 1650
    @test length(LowRatio.BaseStudy.plane_cells(config.base; smoke=true)) == 2

    cell = first(LowRatio.BaseStudy.plane_cells(config.base; smoke=true))
    fast = LowRatio.model_at(config, cell, 0.2)
    manuscript = LowRatio.model_at(config, cell, 4.4)
    @test fast.inhibitory.timescale ≈ 1.56
    @test manuscript.inhibitory.timescale ≈ 34.32
    fast_balance, manuscript_balance = zeros(2), zeros(2)
    point_balance!(fast_balance, [0.3, 0.4], fast, 0.0)
    point_balance!(manuscript_balance, [0.3, 0.4], manuscript, 0.0)
    @test fast_balance == manuscript_balance

    mktempdir() do directory
        for mutate in (
            raw -> raw["ratios"]["minimum"] = 0.0,
            raw -> raw["ratios"]["maximum"] = 0.5,
            raw -> raw["ratios"]["minimum"] = true,
            raw -> raw["base_config"] = "../tetrastability.toml",
        )
            raw = deepcopy(config.raw)
            mutate(raw)
            # Keep the base configuration beside the temporary test copy.
            copied_base = joinpath(directory, "tetrastability.toml")
            cp(config.base_path, copied_base; force=true)
            invalid = joinpath(directory, "invalid-low-ratio.toml")
            LowRatio.Evidence.write_toml(invalid, raw)
            @test_throws ArgumentError LowRatio.load_config(invalid)
        end
    end
end

@testset "Low-ratio stability intervals and root identity" begin
    outer = [-1.0 0.0; 0.0 -1.0]
    intermediate = [1.0 -2.0; 1.0 -0.3]
    saddle = [-1.0 0.0; 0.0 1.0]
    @test LowRatio.critical_ratio(intermediate) ≈ 0.3
    @test LowRatio.attracting(intermediate, 7.8, 0.2)
    @test !LowRatio.attracting(intermediate, 7.8, 0.4)
    @test !LowRatio.attracting(saddle, 7.8, 0.2)
    intervals = LowRatio.candidate_intervals(
        [outer, outer, outer, intermediate, saddle], 7.8, 0.2, 4.4)
    @test length(intervals) == 1
    @test intervals[1].lower == 0.2
    @test intervals[1].upper ≈ 0.3
    @test intervals[1].indices == [1, 2, 3, 4]

    reverse_intermediate = [-1.0 -2.0; 1.0 0.3]
    @test !LowRatio.attracting(reverse_intermediate, 7.8, 0.2)
    @test LowRatio.attracting(reverse_intermediate, 7.8, 0.4)
    reverse_intervals = LowRatio.candidate_intervals(
        [outer, outer, outer, reverse_intermediate], 7.8, 0.2, 0.4)
    @test length(reverse_intervals) == 1
    @test reverse_intervals[1].lower ≈ 0.3
    @test reverse_intervals[1].upper == 0.4

    reference = [(; state=(0.1, 0.2)), (; state=(0.5, 0.4))]
    refined = reverse(reference)
    @test LowRatio.unique_match(reference[1], reference, refined, 1e-6) == 2
    @test isnothing(LowRatio.unique_match(reference[1], reference,
        [refined... , reference[1]], 1e-6))
end
