include(joinpath(@__DIR__, "..", "reproducibility", "selective_tonic_e_release_20261005", "replay.jl"))

@testset "Selective tonic-E release bookkeeping" begin
    study = SelectiveTonicERelease
    compatible(destination) = Dict("status" => "compatible", "destination" => destination)
    @test study.qualifies(compatible("herald"), compatible("herald"), compatible("active"))
    @test !study.qualifies(compatible("herald"), compatible("seizure"), compatible("active"))
    @test !study.qualifies(compatible("herald"), compatible("herald"), compatible("herald"))
    @test !study.qualifies(Dict("status" => "unresolved", "destination" => "herald"),
        compatible("herald"), compatible("active"))

    function point(success)
        source = Dict("status" => "compatible", "qualifies" =>
            Dict(study.return_key(0.35) => success))
        return Dict("sources" => Dict("rest" => source, "active" => source))
    end
    points = Dict(x => point(yes) for (x, yes) in
        ((0.35, false), (0.6, true), (0.85, false), (1.1, true), (1.35, true), (1.6, false)))
    runs = study.success_runs(points, "rest", 0.35)
    @test length(runs) == 2
    @test [(row["lower_sample"], row["first_success"], row["last_success"],
        row["next_sample"]) for row in runs] ==
        [(0.35, 0.6, 0.6, 0.85), (0.85, 1.1, 1.35, 1.6)]
    @test study.needs_extension(Dict(0.35 => point(false)))
    @test study.combine_roles(Dict("active" => 1, "herald" => 3),
        Dict("rest" => 1, "herald" => 3)) == Dict("active" => 1, "herald" => 3)
    @test !haskey(study.combine_roles(Dict("active" => 1),
        Dict("active" => 2)), "active")

    mktempdir() do parent
        output = joinpath(parent, "run")
        study.initialize(output)
        @test isfile(joinpath(output, "metadata.toml"))
        study.initialize(output; resume=true)
        @test_throws ErrorException study.initialize(output)
        meta = study.TOML.parsefile(joinpath(output, "metadata.toml"))
        meta["width"] = 0.1
        study.NS.write_record(joinpath(output, "metadata.toml"), meta)
        @test_throws ErrorException study.initialize(output; resume=true)
    end
end
