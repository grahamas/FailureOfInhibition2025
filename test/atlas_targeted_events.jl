module AtlasTargetedEventsTests
using Test, TOML, SHA, LinearAlgebra, FailureOfInhibition2025
include(joinpath(@__DIR__,"..","scripts","atlas","core.jl"))
include(joinpath(@__DIR__,"..","scripts","atlas","events.jl"))
include(joinpath(@__DIR__,"..","scripts","atlas","science.jl"))
include(joinpath(@__DIR__,"..","scripts","atlas","pilot.jl"))
include(joinpath(@__DIR__,"..","scripts","atlas","targeted_events.jl"))

const TEST_TARGETED_CONFIG=Dict{String,Any}("seed_grid"=>9,"balance_atol"=>1e-11,
    "event_atol"=>1e-9,"max_corrector_iters"=>100,"max_evaluations"=>10000)
function fixture_parameters(;A=0.0,a=0.0)
    Dict{String,Any}("A"=>A,"B"=>0.0,"C"=>0.0,"D"=>0.0,"a"=>a,
        "b"=>0.0,"h"=>log(9.0),"r"=>1.0)
end
function frozen_fixture(first,last)
    samples=[atlas_point(targeted_affine_parameters(first,last,(index-1)/16),
        TEST_TARGETED_CONFIG;id="frozen-sample$index") for index in 1:17]
    Dict{String,Any}("id"=>"fixture-transition","source_job_id"=>"job-000946",
        "source_payload"=>Dict("base"=>Dict("parameters"=>first),"target"=>Dict("parameters"=>last)),
        "source_recovery"=>Dict("samples"=>samples))
end

@testset "Frozen affine path and fold corrector evidence" begin
    e=(1-sqrt(1-8/12))/4
    threshold=12*e-log(e/(1-2*e))
    first=fixture_parameters(A=12.0,a=threshold-0.044)
    last=fixture_parameters(A=12.0,a=threshold+0.156)
    @test targeted_affine_parameters(first,last,0.0)==first
    @test targeted_affine_parameters(first,last,1.0)==last
    @test_throws ArgumentError targeted_affine_parameters(first,last,-0.01)
    expected=vcat([e,2/7],0.22)
    corrected=targeted_corrector(first,last,vcat([e+0.005,2/7],0.21),"fold",(0.1875,0.25))
    @test corrected["converged"]
    @test corrected["coordinates"]≈expected atol=1e-7
    fine=targeted_corrector(first,last,corrected["coordinates"],"fold",(0.1875,0.25);tight=true)
    @test fine["converged"]
    coarse_coefficient=targeted_fold_coefficients(first,last,fine["coordinates"])
    fine_coefficient=targeted_fold_coefficients(first,last,fine["coordinates"];step=5e-4)
    @test coarse_coefficient["resolved"]&&fine_coefficient["resolved"]
    @test atlas_coefficient_resolved(coarse_coefficient["quadratic"],fine_coefficient["quadratic"])
    @test atlas_coefficient_resolved(coarse_coefficient["path_transversality"],fine_coefficient["path_transversality"])
    @test !targeted_fold_coefficients(first,last,vcat([e,2/7],0.0))["resolved"]
    @test !targeted_corrector(first,last,expected,"fold",(0.3,0.4))["converged"]
    independent=targeted_event_residual(first,last,fine["coordinates"],"fold")
    @test maximum(abs,independent[1:2])<=1e-11
    @test abs(independent[3])<=1e-9
    @test targeted_fold_validation(first,last,fine["coordinates"],corrected["coordinates"],true)["validated"]
    @test !targeted_fold_validation(first,last,fine["coordinates"],corrected["coordinates"],false)["validated"]
    # A cusp has a simple zero but vanishing quadratic coefficient.
    cusp=fixture_parameters(A=8.0,a=2+log(2.0))
    cusp_first=copy(cusp);cusp_first["a"]-=0.1
    cusp_last=copy(cusp);cusp_last["a"]+=0.1
    cusp_coordinates=[0.25,2/7,0.5]
    cusp_validation=targeted_fold_validation(cusp_first,cusp_last,cusp_coordinates,cusp_coordinates,true)
    @test !cusp_validation["validated"]
    @test !cusp_validation["quadratic_refinement_passed"]
    # Passing through no parameter variation is not a transverse fold.
    stationary=targeted_affine_parameters(first,last,0.22)
    stationary_validation=targeted_fold_validation(stationary,stationary,expected,expected,true)
    @test !stationary_validation["validated"]
    @test !stationary_validation["transversality_refinement_passed"]
end

@testset "Bounded transition refinement and independently saved events" begin
    e=(1-sqrt(1-8/12))/4
    threshold=12*e-log(e/(1-2*e))
    first=fixture_parameters(A=12.0,a=threshold-0.044)
    last=fixture_parameters(A=12.0,a=threshold+0.156)
    result=execute_targeted_transition(frozen_fixture(first,last),TEST_TARGETED_CONFIG)
    record=only(result["evidence"]["targeted_transitions"])
    @test record["inventory_location_count"]<=256
    @test record["corrector_count"]<=64
    @test length(record["initial_brackets"])==3
    @test all(bracket->length(bracket["source_endpoint_comparisons"])==2,record["initial_brackets"])
    @test all(bracket->haskey(bracket,"grid9_signature_change_observed")&&
        haskey(bracket,"source_endpoint_inventories_reproduced"),record["initial_brackets"])
    @test all(interval->!interval["validated"]&&!interval["continuous_coverage_certified"],record["intervals"])
    @test all(interval->get(interval,"split_count",0)<=12,record["intervals"])
    @test any(event->event["classification"]=="numerically_validated_fold",record["events"])
    @test record["event_estimate_count"]>=record["repeat_group_count"]
    @test all(group->group["representative_estimate_id"] in group["raw_estimate_ids"],record["events"])
    @test all(event->event["kind"]!="trace_zero"||!event["validated"],record["events"])
    point_ids=Set(point["id"] for point in result["evidence"]["points"])
    @test all(interval->interval["from"] in point_ids&&interval["to"] in point_ids,record["intervals"])
    buffer=IOBuffer();TOML.print(buffer,result);saved=TOML.parse(String(take!(buffer)))
    for event in saved["evidence"]["events"]
        reconstructed=targeted_affine_parameters(first,last,event["affine_t"])
        @test reconstructed==event["parameters"]
        residual=targeted_event_residual(first,last,vcat(event["state"],event["affine_t"]),event["kind"])
        @test maximum(abs,residual[1:2])==event["balance_residual"]
        @test abs(residual[3])==event["event_condition_residual"]
    end
    constant=fixture_parameters()
    unchanged=execute_targeted_transition(frozen_fixture(constant,constant),TEST_TARGETED_CONFIG)
    unchanged_record=only(unchanged["evidence"]["targeted_transitions"])
    @test all(interval->interval["status"]=="finite_same_signature_samples",unchanged_record["intervals"])
    @test all(interval->!interval["validated"],unchanged_record["intervals"])
    @test isempty(unchanged_record["events"])
    # Fault-inject an old saved inventory difference. Fresh dense endpoints
    # agree, so the result must expose that the source evidence was contradicted.
    contradicted_job=frozen_fixture(constant,constant)
    old_middle=contradicted_job["source_recovery"]["samples"][5]
    old_middle["signature"]*="|old_spurious_root"
    push!(old_middle["roots"],merge(old_middle["roots"][1],Dict("state"=>[0.2,0.2])))
    contradicted=execute_targeted_transition(contradicted_job,TEST_TARGETED_CONFIG)
    bracket=only(contradicted["evidence"]["targeted_transitions"])["initial_brackets"][1]
    @test !bracket["grid9_signature_change_observed"]
    @test !bracket["source_endpoint_inventories_reproduced"]
    @test bracket["source_root_counts"]!=bracket["grid9_root_counts"]
end

@testset "Repeat groups retain estimates without tolerance chaining" begin
    base=Dict{String,Any}("source_job_id"=>"fixture","kind"=>"fold","validated"=>true,
        "affine_t"=>0.5,"state"=>[0.1,0.2],"balance_residual"=>1e-13,"event_condition_residual"=>1e-12)
    estimates=[merge(base,Dict("id"=>"estimate$i","affine_t"=>0.5+delta,
        "state"=>[0.1+delta,0.2])) for (i,delta) in enumerate((0.0,0.7e-6,1.4e-6))]
    groups,mapping=targeted_repeat_groups(estimates)
    @test length(groups)==2
    @test length(mapping)==3
    @test sort(vcat([group["raw_estimate_ids"] for group in groups]...))==getindex.(estimates,"id")
    @test all(group->group["state"]==only(filter(estimate->estimate["id"]==group["representative_estimate_id"],estimates))["state"],groups)
    @test mapping["estimate1"]!=mapping["estimate3"]
end

@testset "Persisted prefix includes parameters and state" begin
    saved=Dict("points"=>[Dict("parameter"=>1.0,"state"=>[0.1,0.2],
        "balance_residual"=>0.0,"stability"=>"Attracting")],"attempts"=>Any[],"candidates"=>Any[])
    @test targeted_prefix_comparison(saved,deepcopy(saved))["status"]=="shared_prefix_consistent"
    altered=deepcopy(saved);altered["points"][1]["parameter"]+=1e-5
    @test targeted_prefix_comparison(saved,altered)["status"]=="shared_prefix_disagreement_or_unresolved"
    altered=deepcopy(saved);altered["points"][1]["state"][1]+=1e-5
    @test targeted_prefix_comparison(saved,altered)["status"]=="shared_prefix_disagreement_or_unresolved"
    @test targeted_prefix_comparison(saved,deepcopy(saved))["tangent_comparison"]=="unavailable_not_persisted_in_source"
    parameters=fixture_parameters();initial=occupancy(0.0,0.0,parameters["h"])
    curve=continue_equilibria(value->reduced_model(atlas_replace(parameters,"A",value)),initial,0.0;
        parameter_bounds=(0.0,0.5),options=ContinuationOptions(max_steps=24,corrector_atol=1e-11),
        equilibrium_options=atlas_equilibrium_options(TEST_TARGETED_CONFIG))
    source=Dict("deep_steps"=>24,"coordinate"=>"A","bounds"=>[0.0,0.5],
        "initial_state"=>initial,"deep"=>pilot_branch_summary(curve))
    job=Dict{String,Any}("id"=>"fixture-continuation","source_job_id"=>"job-000584",
        "source_payload"=>Dict("training_point"=>Dict("parameters"=>parameters)),"source_audit"=>source)
    result=execute_targeted_continuation(job,TEST_TARGETED_CONFIG)
    record=only(result["evidence"]["targeted_continuations"])
    @test record["extended_steps"]==96
    @test record["bounds"]==[0.0,0.5]
    @test length(record["branches"])==2
    @test all(branch->branch["prefix_comparison"]["status"]=="shared_prefix_consistent",record["branches"])
    @test all(branch->haskey(branch,"attempts")&&haskey(branch,"candidates"),record["branches"])
end
end
