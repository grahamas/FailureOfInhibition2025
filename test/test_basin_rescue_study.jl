include(joinpath(@__DIR__, "..", "scripts", "run_basin_rescue_study.jl"))

@testset "Basin and tonic study configuration and branch identity" begin
    config = BasinRescueStudy.load_config(joinpath(@__DIR__, "..", "experiments",
        "basin_rescue.toml"))
    @test length(config.cases) == 2
    @test config.scan["baseline_stop"] == 8
    @test config.sensitivity["tau_ratio_minimum"] == 0.2
    case = config.cases[1]
    model = BasinRescueStudy.model_for(case, 0.0)
    search = BasinRescueStudy.search_model(model, 5)
    references = Dict(role => case[string(role)] for role in BasinRescueStudy.ROLES)
    matches, reasons = BasinRescueStudy.match_roles(search, references, 0.15)
    @test all(role -> reasons[role] == :matched, BasinRescueStudy.ROLES)
    @test length(unique(collect(values(matches)))) == 3
    lost_references = Dict{Symbol,Union{Nothing,Vector{Float64}}}(
        role => Float64.(case[string(role)]) for role in BasinRescueStudy.ROLES)
    lost_references[:low_activity] = nothing
    lost_matches, lost_reasons = BasinRescueStudy.match_roles(search,
        lost_references, 0.15)
    @test lost_matches[:low_activity] === nothing
    @test lost_reasons[:low_activity] == :tracking_lost
    @test search.completeness == CompletenessNotCertified
    shifted = BasinRescueStudy.model_for(case, 0.5; tau_ratio=0.2)
    @test shifted.inhibitory.timescale == case["tau_e"] * 0.2
    @test_throws ArgumentError BasinRescueStudy.main(String[])
    @test_throws ArgumentError BasinRescueStudy.main(["--output", "a", "--wrong"])
end

@testset "Basin runner preserves unrelated branch matches" begin
    search = (equilibria=[(state=[0.1, 0.1], stability=(classification=Attracting,)),
        (state=[0.5, 0.4], stability=(classification=Attracting,))],)
    references = Dict(:low_activity => [0.1, 0.1],
        :high_e_low_i => [0.5, 0.4], :high_e_high_i => [0.51, 0.4])
    matches, reasons = BasinRescueStudy.match_roles(search, references, 0.15)
    @test matches[:low_activity] == 1
    @test reasons[:low_activity] == :matched
    @test all(role -> matches[role] === nothing && reasons[role] == :ambiguous,
        (:high_e_low_i, :high_e_high_i))
end
