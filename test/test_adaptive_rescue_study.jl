include(joinpath(@__DIR__, "..", "scripts", "run_adaptive_rescue_study.jl"))

@testset "Adaptive rescue configuration, roles, and sampling" begin
    config = AdaptiveRescueStudy.load_config(joinpath(@__DIR__, "..",
        "experiments", "adaptive_rescue.toml"))
    @test length(config.cases) == 4
    @test config.targets == Set(("quiescent", "active_mid"))
    @test length(AdaptiveRescueStudy.parameter_cells(config.cases[1],
        config.sensitivity)) == 6 # nominal plus five in-bounds neighbors
    @test length(AdaptiveRescueStudy.parameter_cells(config.cases[2],
        config.sensitivity)) == 4 # three neighbors at configured lower bounds

    case = only(filter(x -> x["name"] == "four_sinks_central", config.cases))
    model = AdaptiveRescueStudy.model_for(case, 0.0)
    search = AdaptiveRescueStudy.search_model(model, 5)
    refs = Dict{String,Union{Nothing,Vector{Float64}}}(
        role => copy(state) for (role, state) in config.references[case["name"]])
    matches, reasons = AdaptiveRescueStudy.match_roles(search, refs, config.tolerance)
    @test all(role -> reasons[role] == "matched", keys(refs))
    @test length(unique(collect(values(matches)))) == 4
    mktempdir() do output
        cell = first(AdaptiveRescueStudy.parameter_cells(case, config.sensitivity))
        tracked = (; search, matches, reasons)
        AdaptiveRescueStudy.write_context_artifacts!(output, config, case,
            cell, 0.0, tracked)
        directory = joinpath(output, "contexts", case["name"], cell.name)
        context_path = joinpath(directory, "B_0p0.toml")
        branch_path = joinpath(directory, "B_0p0_branches.csv")
        expected_context, expected_branches = read(context_path), read(branch_path)
        write(context_path, "truncated TOML")
        write(branch_path, "truncated CSV")
        AdaptiveRescueStudy.write_context_artifacts!(output, config, case,
            cell, 0.0, tracked)
        @test read(context_path) == expected_context
        @test read(branch_path) == expected_branches
    end
    refs["seizure"] = nothing
    _, lost = AdaptiveRescueStudy.match_roles(search, refs, config.tolerance)
    @test lost["seizure"] == "tracking_lost"
    context = (matches=matches,)
    source = (index=matches["seizure"], role="seizure")
    trial = (destination=matches["active_mid"], status=:compatible,
        initial_state=search.equilibria[source.index].state,
        e_reduction=0.0, i_increment=2.0, total_E=0.0, total_I=2.0,
        integrated_E=0.0, integrated_I=20.0,
        attempts=[(followup_time=5000.0,)])
    row = AdaptiveRescueStudy.trial_row(case["name"], "nominal", 0.0,
        source, 10.0, trial, context, config.targets)
    @test row.rescue && row.destination_role == "active_mid"
    @test row.protocol == "positive_I"
    herald_trial = merge(trial, (destination=matches["herald"],))
    @test !AdaptiveRescueStudy.trial_row(case["name"], "nominal", 0.0,
        source, 10.0, herald_trial, context, config.targets).rescue
    @test AdaptiveRescueStudy.unit_status([row], "seizure", false) ==
        "target_unavailable"

    seen = Set{Float64}()
    line(x) = (push!(seen, x); (status=:compatible, destination=x < 0.3 ? 1 : 2))
    AdaptiveRescueStudy.sample_line!(line, 0.0, 1.0, 1.0, 0.0625)
    @test 0.5 in seen
    @test maximum(filter(x -> x < 0.3, collect(seen))) >= 0.25
    @test minimum(filter(x -> x >= 0.3, collect(seen))) -
        maximum(filter(x -> x < 0.3, collect(seen))) <= 0.0625
    @test length(seen) < 20

    sampled = Set{Tuple{Float64,Float64}}()
    island(e, i) = (push!(sampled, (e, i));
        (status=:compatible, destination=hypot(e - 0.5, i - 0.5) < 0.1 ? 2 : 1))
    AdaptiveRescueStudy.sample_cell!(island, 0.0, 1.0, 0.0, 1.0, 0.0625)
    @test (0.5, 0.5) in sampled
    @test (0.25, 0.25) in sampled
    @test length(sampled) < 500

    uniform(e, i) = (status=:compatible, destination=1)
    calls = Ref(0)
    AdaptiveRescueStudy.sample_cell!((e, i) -> (calls[] += 1; uniform(e, i)),
        0.0, 1.0, 0.0, 1.0, 0.0625)
    @test calls[] == 5

    unresolved(e, i) = (status=:unresolved, destination=nothing)
    calls[] = 0
    AdaptiveRescueStudy.sample_cell!((e, i) -> (calls[] += 1; unresolved(e, i)),
        0.0, 0.125, 0.0, 0.125, 0.0625)
    @test calls[] > 5
end

@testset "Resume verifies every recorded source" begin
    mktempdir() do root
        paths = ["src/diagnostics.jl", "src/simulation.jl",
            "scripts/run_minimal_experiment.jl", "Project.toml", "Manifest.toml",
            "scripts/run_adaptive_rescue_study.jl"]
        exemplar = joinpath(root, "custom_exemplars.toml")
        write(exemplar, "original exemplars")
        hashes = Dict("experiments/exemplar_models.toml" =>
            AdaptiveRescueStudy.file_hash(exemplar))
        for relative in paths
            path = joinpath(root, relative)
            mkpath(dirname(path))
            write(path, "original $relative")
            hashes[relative] = AdaptiveRescueStudy.file_hash(path)
        end
        metadata = Dict("source_sha256" => hashes)
        verify() = AdaptiveRescueStudy.verify_resume_sources(metadata, exemplar; root)
        @test verify() === nothing
        for path in [joinpath.(root, paths); exemplar]
            original = read(path)
            write(path, "changed")
            @test_throws ArgumentError verify()
            rm(path)
            @test_throws ArgumentError verify()
            write(path, original)
        end
        @test verify() === nothing
    end
end

@testset "Completion markers survive interrupted writes" begin
    mktempdir() do directory
        path = joinpath(directory, "done.toml")
        data = Dict("presence" => "observed_rescue")
        function interrupted(path, data)
            write(path, "partial TOML =")
            throw(InterruptException())
        end
        @test_throws InterruptException AdaptiveRescueStudy.write_completion_marker(
            path, data; writer=interrupted)
        @test isempty(readdir(directory))
        AdaptiveRescueStudy.write_completion_marker(path, data)
        @test AdaptiveRescueStudy.TOML.parsefile(path) == data
        @test_throws InterruptException AdaptiveRescueStudy.write_completion_marker(
            path, Dict("presence" => "changed"); writer=interrupted)
        @test AdaptiveRescueStudy.TOML.parsefile(path) == data
        @test readdir(directory) == ["done.toml"]
    end
end

@testset "Role collisions preserve independent matches and lost roles" begin
    search = (equilibria=[(state=[0.1, 0.1], stability=(classification=Attracting,)),
        (state=[0.5, 0.4], stability=(classification=Attracting,))],)
    refs = Dict("quiescent" => [0.1, 0.1], "seizure" => [0.5, 0.4],
        "active_mid" => [0.51, 0.4], "herald" => nothing)
    matches, reasons = AdaptiveRescueStudy.match_roles(search, refs, 0.15)
    @test matches["quiescent"] == 1
    @test reasons["quiescent"] == "matched"
    @test all(role -> matches[role] === nothing && reasons[role] == "ambiguous",
        ("seizure", "active_mid"))
    @test reasons["herald"] == "tracking_lost"
    next_refs = Dict(role => index === nothing ? nothing :
        search.equilibria[index].state for (role, index) in matches)
    next_matches, next_reasons = AdaptiveRescueStudy.match_roles(search, next_refs, 0.15)
    @test next_matches["quiescent"] == 1
    @test next_reasons["seizure"] == "tracking_lost"
end
