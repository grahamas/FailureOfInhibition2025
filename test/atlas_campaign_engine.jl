module AtlasCampaignEngineTests
using Test, TOML, SHA, LinearAlgebra
include(joinpath(@__DIR__,"..","scripts","atlas","engine.jl"))
include(joinpath(@__DIR__,"..","scripts","atlas","targeted.jl"))

@testset "Campaign worker transport preserves science without inventory copies" begin
    config=atlas_config()
    @test atlas_worker_config(config) === config
    merge!(config,Dict("protocol"=>"bounded_atlas_campaign_v1",
        "campaign_manifest"=>Dict("large_historical_inventory"=>["retained once"]),
        "campaign_manifest_hash"=>"bound-manifest",
        "campaign_calibration_sources"=>["frozen-calibration"],
        "numerical_policy"=>Dict("balance_atol"=>1e-12)))
    worker=atlas_worker_config(config)
    @test !haskey(worker,"campaign_manifest")
    @test !haskey(worker,"campaign_calibration_sources")
    @test worker["science"] == config["science"]
    @test worker["numerical_policy"] == config["numerical_policy"]
    @test worker["campaign_manifest_hash"] == "bound-manifest"
    @test haskey(config,"campaign_manifest")
    original=Dict{String,Any}("balance_atol"=>1e-12,"event_atol"=>1e-10,
        "seed_grid"=>5,"max_corrector_iters"=>10)
    historical=targeted_execution_config(original)
    @test historical["balance_atol"] == 1e-11
    @test historical["event_atol"] == 1e-9
    reviewed=targeted_execution_config(merge(original,Dict("protocol"=>"bounded_atlas_campaign_v1")))
    @test reviewed["balance_atol"] == 1e-12
    @test reviewed["event_atol"] == 1e-10
    @test reviewed["seed_grid"] == 9
    @test reviewed["max_corrector_iters"] == 100
end

@testset "Campaign CLI requires explicit frozen inputs and storage" begin
    mktempdir() do root
        output=joinpath(root,"must-not-launch")
        @test_throws ArgumentError atlas_main(["campaign",output])
        @test_throws ArgumentError atlas_main(["campaign",output,"--budget-seconds","1"])
        @test_throws ArgumentError atlas_main(["pilot",output,"--numerical-policy","policy.toml"])
        @test !ispath(output)
    end
end
@testset "Campaign storage frontier survives ordinary resume" begin
    state=Dict{String,Any}("protocol"=>"bounded_atlas_campaign_v1","status"=>"storage_budget","jobs"=>Any[],
        "campaign"=>Dict{String,Any}("protocol_complete"=>true))
    @test atlas_campaign_storage_stopped(state)
    mktempdir() do output
        save_atlas_state(output,state)
        saved=TOML.parsefile(joinpath(output,"state.toml"))
        @test saved["campaign_storage_stopped"]
        @test !saved["campaign"]["protocol_complete"]
        @test saved["campaign"]["storage_stop"]
        saved["status"]="running"
        @test_throws ArgumentError drive_atlas!(output,saved,Dict();additional_seconds=1)
        @test !isdir(joinpath(output,"workers"))
    end
    @test !atlas_campaign_storage_stopped(Dict("protocol"=>"targeted_followup_v1","status"=>"storage_budget"))
end
@testset "Calibration destination cannot mutate historical inputs through aliases" begin
    mktempdir() do root
        first=mkdir(joinpath(root,"first"));second=mkdir(joinpath(root,"second"))
        roots=[first,second]
        @test_throws ArgumentError atlas_calibration_destination(joinpath(first,"new"),roots)
        @test_throws ArgumentError atlas_calibration_destination(joinpath(second,"new","nested"),roots)
        alias=joinpath(root,"alias");symlink(first,alias)
        @test_throws ArgumentError atlas_calibration_destination(joinpath(alias,"new"),roots)
        destination=joinpath(root,"new","calibration")
        @test atlas_calibration_destination(destination,roots)==destination
        @test isempty(readdir(first)) && isempty(readdir(second))
        @test !ispath(destination)
    end
end
end
