module AtlasPilotTests
using Test, TOML, SHA, LinearAlgebra, FailureOfInhibition2025
include(joinpath(@__DIR__,"..","scripts","atlas","core.jl"))
include(joinpath(@__DIR__,"..","scripts","atlas","events.jl"))
include(joinpath(@__DIR__,"..","scripts","atlas","science.jl"))
include(joinpath(@__DIR__,"..","scripts","atlas","pilot.jl"))
include(joinpath(@__DIR__,"..","scripts","atlas","engine.jl"))
function fixture_result!(directory,job,result;status="completed")
    path=joinpath(directory,job["result_file"]);mkpath(dirname(path))
    atomic_toml(path,result);job["result_hash"]=bytes2hex(sha256(read(path)));job["status"]=status
end
function fixture_state()
    state=Dict{String,Any}("jobs"=>Any[])
    for payload in pilot_initial_jobs(Dict())
        register_atlas_job!(state,payload)
    end
    state
end
@testset "Finite declared pilot manifest and phase barriers" begin
    initial=pilot_initial_jobs(Dict());manifest=pilot_manifest(Dict())
    @test length(initial)==515
    @test manifest["mandatory_jobs"]==947
    @test manifest["maximum_jobs"]==971
    @test manifest["counts"]["audit"]==112
    @test length(filter(job->job["pilot_role"]=="training",initial))==512
    @test all(job->job["pilot_role"]!="heldout",initial)
    @test all(job->!get(job,"fixture",false),initial)
    training=filter(job->job["pilot_role"]=="training",initial)
    @test Set(job["pilot_mask"] for job in training)==Set(0:15)
    @test Set(job["pilot_design_position"] for job in training)==Set([1,2,3,4])
    @test manifest["selection_seed"]=="pilot-v1"
    @test manifest["selection_pool"]==collect(1:256)
    @test length(manifest["cohort_designs"])==128
    for design in manifest["cohort_designs"]
        @test design["indices"]==pilot_selected_indices(design["shell"],design["chart"],design["mask"])
        @test length(Set(design["indices"]))==6
        @test all(index->1<=index<=256,design["indices"])
        @test isempty(intersect(design["training_indices"],design["heldout_indices"]))
        @test design["audit_indices"]==design["indices"][[1,3]]
        @test [sample["accepted_index"] for sample in design["samples"]]==design["indices"]
        @test length(design["samples"])==6
    end
    @test length(manifest["support_ledger"])==4
    for row in manifest["support_ledger"]
        @test row["sample_count"]==(row["role"]=="training" ? 256 : 128)
        @test row["h_below_one_count"]>0 && row["h_at_least_one_count"]>0
        @test row["A_at_most_eight_count"]>0 && row["A_above_eight_count"]>0
        signed=row["chart"]=="parameter" ? ("a","b") : ("u","v")
        @test all(label->row["dimensions"][label]["negative_count"]>0&&row["dimensions"][label]["nonnegative_count"]>0,signed)
    end
    mktempdir() do directory
        state=fixture_state();config=Dict{String,Any}()
        @test pilot_role(pilot_next_job!(directory,state,config))=="control"
        for job in filter(job->pilot_role(job)=="control",state["jobs"])
            fixture_result!(directory,job,Dict("evidence"=>Dict("controls"=>[Dict("status"=>"passed")]),"followups"=>Any[]))
        end
        @test pilot_role(pilot_next_job!(directory,state,config))=="training"
        p=equilibrium_chart(0.,0.,0.,0.,0.,0.,log(9.))["params"]
        for job in filter(job->pilot_role(job)=="training",state["jobs"])
            point=Dict("id"=>job["id"],"parameters"=>p,"signature"=>"fixture_signature","roots"=>[Dict("state"=>[1/3,2/7])])
            fixture_result!(directory,job,Dict("evidence"=>Dict("points"=>[point]),"followups"=>Any[]))
        end
        @test pilot_role(pilot_next_job!(directory,state,config))=="audit"
        @test state["pilot"]["training_followup_registered_count"]==0
        @test length(state["pilot"]["followup_opportunities"])==24
        @test all(slot->slot["status"]=="not_observed",state["pilot"]["followup_opportunities"])
        @test count(job->pilot_role(job)=="audit",state["jobs"])==112
        @test Set(job["payload"]["pilot_design_position"] for job in state["jobs"] if pilot_role(job)=="audit")==Set([1,3])
        for audited in filter(job->pilot_role(job)=="audit",state["jobs"])
            source=only(filter(job->job["id"]==audited["payload"]["training_source_job"],state["jobs"]))
            @test source["payload"]["pilot_sample_index"]==audited["payload"]["pilot_sample_index"]
        end
        @test !any(job->pilot_role(job)=="heldout",state["jobs"])
        for (index,job) in enumerate(filter(job->pilot_role(job)=="audit",state["jobs"]))
            comparison=Dict("status"=>index==1 ? "disagreement_or_unresolved" : "agreement_observed","classification_resolved"=>true)
            fixture_result!(directory,job,Dict("evidence"=>Dict("audits"=>[Dict("comparisons"=>[comparison,comparison],"training_repeat_comparison"=>comparison)]),"followups"=>Any[]))
        end
        @test pilot_role(pilot_next_job!(directory,state,config))=="heldout"
        selection=state["pilot"]["base_selection"];digest=state["pilot"]["base_selection_hash"]
        @test length(selection["entries"])==1
        @test selection["entries"][1]["audit_status"]=="not_audited"
        @test selection["entries"][1]["cohort_audit_warning"]
        @test first(selection["excluded_candidates"])["point_id"]=="job-000004"
        @test selection["entries"][1]["point_id"]=="job-000005"
        @test length(selection["recovery_routes"])==64
        @test all(route->route["base_point_id"]=="job-000005",selection["recovery_routes"])
        @test count(job->pilot_role(job)=="heldout",state["jobs"])==256
        @test Set(route["design_position"] for route in selection["recovery_routes"])==Set([5,6])
        # These synthetic outcomes deliberately disagree with the training
        # signature. They cannot alter routes selected before the heldout phase.
        for job in filter(job->pilot_role(job)=="heldout",state["jobs"])
            point=Dict("id"=>job["id"],"parameters"=>pilot_sample_payload(job["payload"])["params"],"signature"=>"different_fixture_signature","roots"=>[Dict("state"=>[0.2,0.3])])
            fixture_result!(directory,job,Dict("evidence"=>Dict("points"=>[point]),"followups"=>Any[]))
        end
        @test pilot_role(pilot_next_job!(directory,state,config))=="recovery"
        recovery=filter(job->pilot_role(job)=="recovery",state["jobs"])
        @test length(recovery)==64
        @test all(job->job["payload"]["base"]["point_id"]=="job-000005",recovery)
        @test all(job->!job["payload"]["base_target_signature_equal"],recovery)
        @test state["pilot"]["base_selection_hash"]==digest
        for job in recovery
            fixture_result!(directory,job,Dict("evidence"=>Dict("recoveries"=>Any[]),"followups"=>Any[]);status="unresolved")
        end
        @test pilot_next_job!(directory,state,config)===nothing
        @test state["pilot"]["protocol_complete"]
        @test length(state["jobs"])==947
        state["pilot"]["base_selection"]["entries"][1]["signature"]="tampered_fixture"
        @test_throws ArgumentError pilot_next_job!(directory,state,config)
    end
    mktempdir() do directory
        state=fixture_state()
        for (index,job) in enumerate(filter(job->pilot_role(job)=="control",state["jobs"]))
            fixture_result!(directory,job,Dict("evidence"=>Dict("controls"=>[Dict("status"=>index==1 ? "failed" : "passed")]),"followups"=>Any[]);status=index==1 ? "unresolved" : "completed")
        end
        @test pilot_next_job!(directory,state,Dict())===nothing
        @test state["pilot"]["stop_reason"]=="control_validation_failed"
        @test !state["pilot"]["protocol_complete"]
        @test all(job->job["status"]=="pending",filter(job->pilot_role(job)=="training",state["jobs"]))
    end
end
@testset "Finite training followup opportunities" begin
    mktempdir() do directory
        state=fixture_state()
        p=equilibrium_chart(0.,0.,0.,0.,0.,0.,log(9.))["params"]
        for job in filter(job->pilot_role(job)=="training",state["jobs"])
            proposals=[Dict("kind"=>kind,"queue"=>kind=="periodic" ? "cycles" : "events","params"=>p,"state"=>[1/3,2/7]) for kind in ("continuation","nexus","periodic")]
            fixture_result!(directory,job,Dict("evidence"=>Dict(),"followups"=>proposals))
        end
        chosen=pilot_followup_jobs(directory,state)
        @test length(chosen)==24
        @test all(payload->payload["pilot_role"]=="training_followup"&&payload["pilot_followup_depth"]==1,chosen)
        @test all(slot->slot["status"]=="queued",state["pilot"]["followup_opportunities"])
        @test Set(payload["kind"] for payload in chosen)==Set(["continuation","nexus","periodic"])
    end
end
@testset "Protected actual-model readiness controls" begin
    for (index,job) in enumerate(pilot_initial_jobs(Dict())[1:3])
        job["id"]="control-test-$index"
        result=execute_job(job,Dict{String,Any}("seed_grid"=>5,"balance_atol"=>1e-9,"event_atol"=>1e-7))
        row=only(result["evidence"]["controls"])
        @test row["status"]=="passed"
        @test !row["discovery_yield_eligible"]
        @test isempty(result["followups"])
        @test !isempty(TOML.parse(sprint(io->TOML.print(io,result))))
    end
end
@testset "Real bounded audit and corrected-root recovery" begin
    config=Dict{String,Any}("seed_grid"=>5,"balance_atol"=>1e-9,"event_atol"=>1e-7)
    audit=merge(pilot_sample_job(1,"parameter",0,1,"audit"),Dict("id"=>"audit-test","kind"=>"pilot_audit"))
    result=execute_pilot_job(audit,config);TOML.parse(sprint(io->TOML.print(io,result)))
    record=only(result["evidence"]["audits"])
    @test length(record["variants"])==3
    @test length(record["comparisons"])==2
    @test length(record["variants"][1]["point"]["attempts"])==25
    @test length(record["variants"][2]["point"]["attempts"])==81
    @test record["variants"][3]["point"]["balance_atol"]≈1e-11
    @test record["continuation_pair"]["status"]=="attempted"
    @test record["continuation_pair"]["shared_prefix_count"]>0
    @test record["continuation_pair"]["comparison_status"]=="shared_prefix_consistent"
    @test length(record["continuation_pair"]["depth_extensions"])==2
    p=equilibrium_chart(0.,0.,0.,0.,0.,0.,log(9.))["params"]
    q=copy(p);q["a"]+=0.3;q["b"]+=0.2
    basepoint=atlas_point(p,config;id="base");target=atlas_point(q,config;id="target")
    job=Dict{String,Any}("kind"=>"pilot_recovery","id"=>"recovery-test","recovery_target_cohort"=>"test","pilot_target_id"=>"target-job",
        "base"=>Dict("point_id"=>"base","parameters"=>p,"roots"=>basepoint["roots"]),"target"=>target,
        "base_target_signature_equal"=>true,"heldout_inventory_available"=>true)
    recovery=execute_pilot_recovery(job,config);saved=TOML.parse(sprint(io->TOML.print(io,recovery)))
    row=only(saved["evidence"]["recoveries"])
    @test row["tracked_target_coverage"]
    @test row["independent_endpoint_solved"]
    @test row["observed_target_root_count"]==row["matched_target_roots"]==1
    @test length(row["samples"])==17
    @test !isempty(row["tracks"])
    @test all(track->haskey(track,"forward")&&haskey(track,"reverse"),row["tracks"])
    missing=copy(job);delete!(missing,"base");missing["unavailable_reason"]="fixture_no_base"
    @test execute_pilot_recovery(missing,config)["status"]=="unresolved"
    unknown=copy(job);unknown["heldout_inventory_available"]=false
    @test !only(execute_pilot_recovery(unknown,config)["evidence"]["recoveries"])["tracked_target_coverage"]
    far=copy(p);far["a"]=16.0
    far_job=copy(job);far_job["target"]=atlas_point(far,config;id="far-target")
    far_result=only(execute_pilot_recovery(far_job,config)["evidence"]["recoveries"])
    @test far_result["independent_endpoint_solved"]
    @test !far_result["tracked_target_coverage"]
    @test length(far_result["samples"])==17
    @test any(track->"large_step_diagnostic" in track["diagnostics"],far_result["tracks"])
    unresolved_point=merge(basepoint,Dict("signature"=>"StabilityUnresolved"))
    comparison=pilot_search_comparison(unresolved_point,unresolved_point,"a","b")
    @test comparison["inventory_match"]
    @test !comparison["classification_resolved"]
    @test comparison["status"]=="disagreement_or_unresolved"
    @test length(pilot_match_roots([Dict("state"=>[0.2,0.3]),Dict("state"=>[0.2,0.3])],[Dict("state"=>[0.2,0.3])]))==1
end
end
