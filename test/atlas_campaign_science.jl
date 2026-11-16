module AtlasCampaignScienceTests
using Test, TOML, SHA, LinearAlgebra, FailureOfInhibition2025
include(joinpath(@__DIR__,"..","scripts","atlas","core.jl"))
include(joinpath(@__DIR__,"..","scripts","atlas","events.jl"))
include(joinpath(@__DIR__,"..","scripts","atlas","science.jl"))
include(joinpath(@__DIR__,"..","scripts","atlas","pilot.jl"))
include(joinpath(@__DIR__,"..","scripts","atlas","targeted.jl"))
include(joinpath(@__DIR__,"..","scripts","atlas","targeted_events.jl"))
include(joinpath(@__DIR__,"..","scripts","atlas","campaign_science.jl"))
parameters()=Dict{String,Any}("A"=>0.,"B"=>0.,"C"=>0.,"D"=>0.,"a"=>0.,"b"=>0.,"h"=>2.,"r"=>1.)
root(x)=Dict("state"=>[Float64(x),0.2])
point(id,p;roots=[root(0.3)],signature="resolved")=Dict{String,Any}("id"=>id,"parameters"=>copy(p),"roots"=>roots,"signature"=>signature,"attempts"=>Any[])
function fake_point(p,c;id,extra_seeds=[])
    point(id,p)
end
function fake_tracks(p0,p1,initial,grid,origin,config,ids)
    attempts=[Dict("to_point_id"=>ids[div(index*128,grid-1)+1],"track_clean"=>true,"forward"=>Dict("candidate"=>copy(root["state"]))) for index in 0:grid-1 for root in initial]
    Dict{String,Any}("attempts"=>attempts,"endpoint_clean_tracks"=>deepcopy(initial),"adjacency_links"=>Any[],"origin"=>origin)
end
@testset "Strict root matching and exact baseline comparison" begin
    @test campaign_unique_matching([root(.3)],[root(.3)])["complete"]
    @test !campaign_unique_matching([root(.3)],[root(.3),root(.3+1e-8)])["complete"]
    @test !campaign_unique_matching([root(.3),root(.3+1e-8)],[root(.3)])["complete"]
    p=parameters();baseline=point("saved",p)
    result=execute_campaign_audit(Dict("params"=>p,"training_point"=>baseline),Dict();point_solver=fake_point)
    row=only(result["evidence"]["audits"])
    @test row["status"]=="agreement_observed"
    @test length(row["comparisons"])==2
    @test baseline==point("saved",p)
    contradicted=point("saved",p;roots=[root(.4)])
    @test execute_campaign_audit(Dict("params"=>p,"training_point"=>contradicted),Dict();point_solver=fake_point)["status"]=="unresolved"
    @test isempty(result["followups"])
    @test execute_campaign_audit(Dict("params"=>p),Dict();point_solver=fake_point)["status"]=="unresolved"
    @test campaign_science_settings(Dict("balance_atol"=>1e-13,"event_atol"=>1e-12))["balance_atol"]==1e-13
    @test campaign_science_settings(Dict("balance_atol"=>1e-13,"event_atol"=>1e-12))["event_atol"]==1e-12
    observations=campaign_observed_roots([("same_inventory",[root(.3),root(.3+1e-8)])])
    @test length(observations["roots"])==2
    @test observations["ambiguous"]
    altered_parameters=copy(p);altered_parameters["a"]=.1
    @test !campaign_search_comparison(baseline,point("other",altered_parameters),"original","changed")["parameters_match"]
    numerical_failure(args...;kwargs...)=throw(SingularException(1))
    @test campaign_science_point(p,Dict();id="singular",point_solver=numerical_failure)["signature"]=="unresolved_solver_failure"
    programmer_failure(args...;kwargs...)=throw(ErrorException("implementation defect"))
    @test_throws ErrorException campaign_science_point(p,Dict();id="defect",point_solver=programmer_failure)
end
@testset "Independent real grid inventories" begin
    p=parameters()
    result=execute_campaign_validation(Dict("kind"=>"campaign_validation","params"=>p),Dict())
    @test result["status"]=="completed"
    points=result["evidence"]["points"]
    @test length.(getindex.(points,"attempts"))==[81,169]
    @test all(point->length(point["roots"])==1,points)
    @test all(point->point["completeness"]=="CompletenessNotCertified",points)
    seeded_counts=Int[]
    seeded(p,c;id,extra_seeds=[])=begin push!(seeded_counts,length(extra_seeds));fake_point(p,c;id=id) end
    execute_campaign_validation(Dict("params"=>p,"campaign_chart"=>"equilibrium","chart_coordinates"=>[0.,0.,0.,0.,0.,0.,2.]),Dict();point_solver=seeded)
    @test seeded_counts==[1,1]
end
@testset "A sampled path requires clean complete forward evidence" begin
    p=parameters();left=point("left",p);right=point("right",p)
    job=Dict("kind"=>"campaign_path","from"=>left,"to"=>right)
    result=execute_campaign_job(job,Dict();point_solver=fake_point,track_solver=fake_tracks)
    edge=first(result["evidence"]["paths"])
    @test edge["from"]=="left" && edge["to"]=="right"
    @test edge["validated"] && edge["forward_tracking_unambiguous"] && edge["audit_agreement"]
    @test length(edge["sample_ids"])==129
    @test first(edge["sample_ids"])==edge["from"] && last(edge["sample_ids"])==edge["to"]
    @test length(result["evidence"]["points"])==131
    @test count(point->get(point,"audit_only",false),result["evidence"]["points"])==4
    @test length(only(result["evidence"]["campaign_paths"])["endpoint_repeats"])==2
    @test !edge["continuous_coverage_certified"]
    parsed=TOML.parse(sprint(io->TOML.print(io,result)))
    @test parsed["evidence"]["campaign_paths"][1]["sample_point_ids"]==only(result["evidence"]["campaign_paths"])["sample_point_ids"]
    bad_tracks(args...)=merge(fake_tracks(args...),Dict("attempts"=>Any[]))
    bad=execute_campaign_job(job,Dict();point_solver=fake_point,track_solver=bad_tracks)
    @test !first(bad["evidence"]["paths"])["validated"]
    @test first(bad["evidence"]["paths"])["kind"]=="unknown_connection"
    @test !isempty(first(bad["evidence"]["paths"])["tracking_guard_failures"])
    # Keep a synthetic signature fixed to verify the explicit per-point root
    # count guard, independently of how signatures encode an inventory.
    q=copy(p);q["a"]=1.
    multibase=point("multi-left",p;roots=[root(.3),root(.4)])
    multitarget=point("multi-right",q;roots=[root(.3),root(.4)])
    missing(p,c;id,extra_seeds=[])=point(id,p;roots=p["a"] in (0.,1.) ? [root(.3),root(.4)] : [root(.3)])
    missing_result=execute_campaign_job(Dict("kind"=>"campaign_path","from"=>multibase,"to"=>multitarget),Dict();point_solver=missing,track_solver=fake_tracks)
    @test !first(missing_result["evidence"]["paths"])["forward_tracking_unambiguous"]
end
@testset "Recovery denominator includes all endpoint roots and rejects ambiguity" begin
    p=parameters();base=point("base",p);base["point_id"]="base"
    target=point("target",p;roots=[root(.3),root(.4)])
    job=Dict("kind"=>"campaign_recovery","base"=>base,"target_parameters"=>p,
        "validation_evidence"=>Dict("points"=>[target]),"validation_inventory_status"=>"completed")
    result=execute_campaign_job(job,Dict();point_solver=fake_point,track_solver=fake_tracks)
    row=only(result["evidence"]["recoveries"])
    @test row["observed_target_root_count"]==2
    @test row["matched_target_roots"]==1
    @test !row["base_origin_target_coverage"]
    @test all(level->level["base_origin_forward"]["origin"]=="base_origin_forward",row["levels"])
    @test length(row["independent_point_ids"])==129
    @test TOML.parse(sprint(io->TOML.print(io,result)))["evidence"]["recoveries"][1]["observed_target_root_count"]==2
    @test result["status"]=="unresolved"
    changed=deepcopy(job);changed["validation_evidence"]["points"][1]["roots"]=[root(.1),root(.45)]
    altered=execute_campaign_job(changed,Dict();point_solver=fake_point,track_solver=fake_tracks)
    @test getindex.(row["levels"],"base_origin_forward")==getindex.(only(altered["evidence"]["recoveries"])["levels"],"base_origin_forward")
    @test execute_campaign_job(Dict("kind"=>"campaign_recovery"),Dict())["status"]=="unresolved"
end
@testset "Changed intervals honor finite localization and corrector limits" begin
    p=parameters();q=copy(p);q["a"]=1.
    changing(p,c;id,extra_seeds=[])=point(id,p;signature=isodd(floor(Int,p["a"]*128)) ? "odd" : "even")
    calls=Ref(0)
    unresolved(args...;kwargs...)=begin calls[]+=1;Dict{String,Any}("converged"=>false,"attempts"=>Any[]) end
    job=Dict("kind"=>"campaign_path","from"=>point("left",p),"to"=>point("right",q))
    result=execute_campaign_job(job,Dict();point_solver=changing,track_solver=fake_tracks,corrector=unresolved)
    row=only(result["evidence"]["campaign_paths"])
    @test row["localization_split_count"]==12
    @test row["corrector_count"]==calls[]==64
    @test row["inventory_location_count"]<=256
    @test any(interval->interval["corrector_status"]=="unresolved_corrector_cap",result["evidence"]["paths"][2:end])
    @test !first(result["evidence"]["paths"])["validated"]
end
@testset "Transverse sides preserve inadmissible and unexamined status" begin
    p=parameters();job=Dict("kind"=>"campaign_geometry_probe","params"=>p,"parameter_pair"=>["a","b"],
        "transverse_coordinate"=>"A","normalized_transverse_offset"=>-.01,"nexus_evidence"=>Dict("validated"=>true))
    result=execute_campaign_job(job,Dict();point_solver=fake_point)
    record=only(result["evidence"]["geometry_probes"])
    @test only(record["transverse_probes"])["status"]=="outside_admissible_domain"
    @test !isempty(record["unresolved_sides"])
    @test Set(record["unexamined_geometry_directions"])==Set(["B","C","D","h"])
    @test isempty(result["evidence"]["points"])
    positive=execute_campaign_job(merge(job,Dict("normalized_transverse_offset"=>.01)),Dict();point_solver=fake_point)
    @test only(only(positive["evidence"]["geometry_probes"])["transverse_probes"])["resolved"]
    @test length(positive["evidence"]["points"])==1
    @test_throws ArgumentError execute_campaign_job(merge(job,Dict("normalized_transverse_offset"=>.02)),Dict())
end
end
