module AtlasCampaignTests
using Test, TOML, SHA, LinearAlgebra
const ATLAS_LIGHTWEIGHT_COORDINATOR=true
include(joinpath(@__DIR__,"..","scripts","atlas","core.jl"))
include(joinpath(@__DIR__,"..","scripts","atlas","science.jl"))
include(joinpath(@__DIR__,"..","scripts","atlas","pilot.jl"))
include(joinpath(@__DIR__,"..","scripts","atlas","storage.jl"))
include(joinpath(@__DIR__,"..","scripts","atlas","targeted_protocol.jl"))
include(joinpath(@__DIR__,"..","scripts","atlas","engine.jl"))
include(joinpath(@__DIR__,"..","scripts","atlas","report.jl"))
include(joinpath(@__DIR__,"..","scripts","atlas","campaign.jl"))

fixture_parameters(a=0.)=Dict{String,Any}("A"=>0.,"B"=>0.,"C"=>0.,"D"=>0.,"a"=>a,"b"=>0.,"h"=>2.,"r"=>1.)
function fixture_sample(k,index;chart="parameter",mask=0)
    Dict{String,Any}("params"=>fixture_parameters(index/10000),"chart"=>chart,"shell"=>k,
        "sequence_index"=>index,"raw_sequence_index"=>index,"boundary_mask"=>mask,"chart_coordinates"=>[0.,0.,0.,0.,0.,0.,2.])
end
function fixture_manifest()
    replacements=[Dict{String,Any}("kind"=>i<=3 ? "nexus" : "targeted_diagnosis","queue"=>"events",
        "campaign_stage"=>i<=3 ? "replacement_controls" : "replacement","campaign_role"=>"replacement","campaign_round"=>0,"fixture_index"=>i) for i in 1:22]
    Dict{String,Any}("replacement_jobs"=>replacements,"design"=>campaign_design(;sampler=fixture_sample),"bounds"=>campaign_job_bounds())
end
function fixture_state(manifest)
    state=Dict{String,Any}("jobs"=>Dict{String,Any}[])
    foreach(payload->register_atlas_job!(state,deepcopy(payload)),manifest["replacement_jobs"])
    state
end
function save_result(output,job,result)
    result=merge(result,Dict("executed_job"=>merge(job["payload"],Dict("id"=>job["id"]))))
    path=joinpath(output,job["result_file"]);atomic_toml(path,result)
    job["result_hash"]=bytes2hex(sha256(read(path)));job["status"]=result["status"]
end
function fixture_point(id,a;signature="same",role="inventory",shell=1,chart="parameter",source=id,audit_only=false)
    Dict{String,Any}("id"=>id,"parameters"=>fixture_parameters(a),"roots"=>[Dict("state"=>[.3,.2])],"signature"=>signature,
        "campaign_role"=>role,"campaign_shell"=>shell,"campaign_chart"=>chart,"source_job"=>source,"audit_only"=>audit_only)
end

@testset "Historical targeted packet uses its own summary schema" begin
    mktempdir() do root
        config=Dict{String,Any}()
        state=Dict("config_hash"=>toml_digest(config),"source_hashes"=>Dict{String,String}(),
            "jobs"=>Any[],"protocol"=>TARGETED_PROTOCOL)
        atomic_toml(joinpath(root,"config.toml"),config)
        atomic_toml(joinpath(root,"state.toml"),state)
        atomic_toml(joinpath(root,"targeted_summary.toml"),Dict("readiness_stop"=>true))
        packet=campaign_packet(root,"historical_targeted")
        @test packet["summary"]["readiness_stop"]
        @test !isfile(joinpath(root,"pilot_summary.toml"))
        @test_throws ArgumentError campaign_packet(root,"unknown")
    end
end

@testset "Frozen disjoint SHA-ranked accepted-index design" begin
    design=campaign_design(;sampler=fixture_sample)
    @test length(design)==512
    @test count(row->row["round"]<4,design)==384
    for row in design
        r=row["round"];indices=row["selected_indices"]
        @test length(indices)==(r==4 ? 2 : 4)
        @test all(index->256r+1<=index<=256r+256,indices)
        @test indices==campaign_selected_indices(row["shell"],row["chart"],row["mask"],r)
    end
    @test length(campaign_sample_jobs(Dict("design"=>design),1))==512
    @test length(campaign_sample_jobs(Dict("design"=>design),1;audit=true))==128
    @test length(campaign_sample_jobs(Dict("design"=>design),4;validation=true))==256
    @test campaign_job_bounds()["round_maximum"]==1120
    @test campaign_job_bounds()["maximum_jobs"]==3894
    @test campaign_job_bounds()["mandatory_jobs"]==2454
    @test_throws ArgumentError campaign_selected_indices(5,"parameter",0,1)
    @test_throws ArgumentError campaign_selected_indices(1,"parameter",0,5)
end

@testset "Storage forecast fails closed and preserves fixed reserves" begin
    capacity=campaign_capacity_forecast(5_000_000_000,60_000_000;measured_result_bytes=1000,remaining_jobs=3894)
    @test capacity["new_packet_cap_bytes"]==4_940_000_000
    @test capacity["projected_capacity_sufficient"]
    @test capacity["dispatch_reserves_bytes"]>0
    @test !campaign_capacity_forecast(5_000_000_000,60_000_000)["projection_available"]
    @test_throws ArgumentError campaign_capacity_forecast(5_000_000_001,0)
    @test_throws ArgumentError campaign_capacity_forecast(1000,999)
    @test_throws OverflowError campaign_capacity_forecast(5_000_000_000,0;measured_result_bytes=typemax(Int),remaining_jobs=2)
end

@testset "Prescribed scientific limits and operational choices" begin
    policy=Dict("balance_atol"=>1e-9,"event_atol"=>1e-9)
    config=campaign_validate_configuration(atlas_config(),policy)
    @test config["science"]["seed_grid"]==9
    @test config["science"]["event_atol"]==1e-9
    @test config["pilot_max_jobs"]==3894
    bad=atlas_config();bad["science"]["continuation_steps"]=1000
    @test_throws ArgumentError campaign_validate_configuration(bad,policy)
    bad=atlas_config();bad["pilot_max_jobs"]=971
    @test_throws ArgumentError campaign_validate_configuration(bad,policy)
end

@testset "Controls gate the entire finite scheduler; traversal exceeds old cap" begin
    mktempdir() do output
        manifest=fixture_manifest();config=Dict("campaign_manifest"=>manifest,"campaign_manifest_hash"=>toml_digest(manifest));state=fixture_state(manifest)
        @test campaign_next_job!(output,state,config)["id"]=="job-000001"
        state["jobs"][1]["status"]="unresolved"
        @test campaign_next_job!(output,state,config)["id"]=="job-000002"
        state["jobs"][2]["status"]="unresolved";state["jobs"][3]["status"]="unresolved"
        @test campaign_next_job!(output,state,config)===nothing
        @test state["campaign"]["stop_reason"]=="control_validation_failed"
        @test length(state["jobs"])==22
        state=fixture_state(manifest)
        for job in state["jobs"][1:3]
            save_result(output,job,Dict("status"=>"completed","evidence"=>Dict("controls"=>[Dict("status"=>"passed")]),"followups"=>Any[]))
        end
        # This exercises orchestration only: no numerical job is run.
        visited=0
        while true
            job=campaign_next_job!(output,state,config)
            job===nothing && break
            visited+=1;job["status"]="unresolved"
            if campaign_role(job)=="validation_inventory"
                @test state["campaign"]["bases_frozen"]
                @test length(state["campaign"]["base_selection"]["routes"])==256
            end
        end
        @test visited==2451
        @test length(state["jobs"])==2454
        @test state["campaign"]["protocol_complete"]
        @test campaign_verify_schedule(output,state,manifest)===nothing
        storage_stopped=deepcopy(state);last(storage_stopped["jobs"])["status"]="storage_budget"
        @test_throws ArgumentError campaign_verify_schedule(output,storage_stopped,manifest)
        @test campaign_next_job!(output,storage_stopped,config)===nothing
        @test !storage_stopped["campaign"]["protocol_complete"]
        @test storage_stopped["campaign"]["storage_stop"]
        bad=deepcopy(state);bad["campaign"]["round1_inventory_registered_count"]=0
        @test_throws ArgumentError campaign_verify_schedule(output,bad,manifest)
        bad=deepcopy(state);bad["campaign"]["base_selection"]["routes"][1]["status"]="invented";bad["campaign"]["base_selection_hash"]=toml_digest(bad["campaign"]["base_selection"])
        @test_throws ArgumentError campaign_verify_schedule(output,bad,manifest)
        bad=deepcopy(state);filter!(j->campaign_stage(j)!="round1_inventory",bad["jobs"])
        @test_throws ArgumentError campaign_verify_schedule(output,bad,manifest)
        bad=deepcopy(state);push!(bad["jobs"],deepcopy(bad["jobs"][23]))
        @test_throws ArgumentError campaign_verify_schedule(output,bad,manifest)
        early=fixture_state(manifest);early["campaign"]=Dict("round2_inventory_registered"=>true,"round2_inventory_registered_count"=>0)
        @test_throws ArgumentError campaign_verify_schedule(output,early,manifest)
    end
end

@testset "Only explicit clean connections merge components" begin
    points=[fixture_point(string(i),i/10) for i in 1:5]
    records=Dict("points"=>points,"paths"=>Any[],"audit_exclusions"=>Any[])
    @test length(atlas_region_components(campaign_eligible_points(records),campaign_connection_paths(records)))==5
    path=Dict{String,Any}("from"=>"1","to"=>"5","sample_ids"=>string.(1:5),"status"=>"completed","validated"=>true,"signature_preserved"=>true,
        "forward_tracking_unambiguous"=>true,"audit_agreement"=>true,"tracking_guard_failures"=>String[])
    push!(records["paths"],path)
    @test length(atlas_region_components(campaign_eligible_points(records),campaign_connection_paths(records)))==1
    path["forward_tracking_unambiguous"]=false
    @test length(atlas_region_components(campaign_eligible_points(records),campaign_connection_paths(records)))==5
    path["forward_tracking_unambiguous"]=true;push!(records["audit_exclusions"],Dict("source_job"=>"3"))
    @test length(campaign_eligible_points(records))==4
    @test length(atlas_region_components(campaign_eligible_points(records),campaign_connection_paths(records)))==4
    empty!(records["audit_exclusions"]);push!(records["points"],fixture_point("audit-repeat",.1;audit_only=true))
    @test length(campaign_eligible_points(records))==5
end

@testset "Proposal quotas use stable source IDs and original proposal order" begin
    mktempdir() do output
        state=Dict{String,Any}("jobs"=>Dict{String,Any}[])
        for i in 1:3
            payload=Dict("kind"=>"discovery","queue"=>"discovery","campaign_role"=>"inventory","campaign_round"=>1,"campaign_shell"=>1,"campaign_chart"=>"parameter","index"=>i)
            job=register_atlas_job!(state,payload)
            follow=[Dict("kind"=>kind,"queue"=>"events","params"=>fixture_parameters(i/10),"state"=>[.3,.2],"proposal_marker"=>j) for (j,kind) in enumerate(("continuation","continuation","nexus","periodic"))]
            save_result(output,job,Dict("status"=>"completed","evidence"=>Dict(),"followups"=>follow))
        end
        reverse!(state["jobs"])
        jobs,slots=campaign_proposal_jobs(output,state,1)
        @test length(jobs)==6
        @test all(j->j["campaign_source_job"]=="job-000001",filter(j->j["kind"]=="continuation",jobs))
        @test [j["campaign_proposal_order"] for j in filter(j->j["kind"]=="continuation",jobs)]==[1,2]
        @test length(slots)==24
        @test count(slot->slot["selected"]==0,slots)==21
    end
end

@testset "Campaign marker leaves historical packets readable and fails closed" begin
    mktempdir() do output
        @test !campaign_run_marker(output,Dict("protocol"=>"targeted_followup_v1"),Dict("protocol"=>"targeted_followup_v1"))
        @test_throws ArgumentError campaign_run_marker(output,Dict(),Dict("campaign_manifest"=>Dict()))
        @test campaign_run_marker(output,Dict("protocol"=>CAMPAIGN_PROTOCOL),Dict("protocol"=>CAMPAIGN_PROTOCOL))
    end
end

@testset "Exact policy and calibration review binding" begin
    @test_throws ArgumentError campaign_required_calibration_controls(Dict("controls"=>Any[]))
    @test_throws ArgumentError campaign_required_calibration_controls(Dict("controls"=>[Dict("name"=>"failed_targeted_cusp_bias_reproduced","passed"=>false)]))
    @test_throws ArgumentError campaign_required_calibration_controls(Dict("controls"=>[Dict("name"=>"failed_targeted_cusp_bias_reproduced","passed"=>true),Dict("name"=>"failed_targeted_cusp_bias_reproduced","passed"=>true)]))
    mktempdir() do root
        policy=Dict{String,Any}("calibration_root"=>root,"balance_atol"=>1e-9)
        atomic_toml(joinpath(root,"policy.toml"),policy)
        atomic_toml(joinpath(root,"calibration.toml"),Dict("training"=>512,"controls"=>[Dict("name"=>"failed_targeted_cusp_bias_reproduced","passed"=>true)]))
        entries=filter(row->!(row["path"] in (".","review.toml")),targeted_inventory(root))
        review=Dict("status"=>"passed","reviewer"=>"independent-fixture","policy_hash"=>toml_digest(policy),
            "calibration_inventory_hash"=>toml_digest(Dict("entries"=>entries)))
        atomic_toml(joinpath(root,"review.toml"),review)
        @test campaign_calibration_review(policy)["record"]==review
        review["status"]="pending";atomic_toml(joinpath(root,"review.toml"),review)
        @test_throws ArgumentError campaign_calibration_review(policy)
        review["status"]="passed";atomic_toml(joinpath(root,"review.toml"),review)
        atomic_toml(joinpath(root,"calibration.toml"),Dict("training"=>511))
        @test_throws ArgumentError campaign_calibration_review(policy)
    end
end

@testset "Measured stage forecasts stop without shrinking the finite design" begin
    mktempdir() do output
        policy=storage_policy(4*1024^2)
        open(joinpath(output,".storage.toml"),"w") do io
            TOML.print(io,policy;sorted=true)
        end
        manifest=Dict("bounds"=>campaign_job_bounds(),"replacement_jobs"=>Any[],"source_packets"=>[Dict("inventory"=>[Dict("kind"=>"file","path"=>"results/old.toml","bytes"=>100_000)],"result_kinds"=>Dict("results/old.toml"=>"discovery"))])
        config=Dict("campaign_manifest"=>manifest);state=Dict{String,Any}("jobs"=>Any[],"campaign"=>Dict{String,Any}())
        @test_throws StorageBudgetExceeded campaign_stage_forecast!(output,state,config,"round1_inventory")
        @test state["campaign"]["stop_reason"]=="storage_budget"
        @test !state["campaign"]["round1_inventory_capacity_forecast"]["capacity_sufficient_for_known_estimates"]
        @test isempty(state["jobs"])
    end
end

@testset "One declared nexus section and four offsets per remaining geometry coordinate" begin
    mktempdir() do output
        state=Dict{String,Any}("jobs"=>Dict{String,Any}[])
        payload=Dict("kind"=>"nexus","queue"=>"events","campaign_role"=>"proposal","campaign_round"=>1,"campaign_shell"=>1,"campaign_chart"=>"equilibrium")
        job=register_atlas_job!(state,payload)
        section=Dict("kind"=>"section","queue"=>"events","params"=>fixture_parameters(0.),"state"=>[.3,.2],"parameter_pair"=>["A","r"],"nexus_type"=>"bt")
        save_result(output,job,Dict("status"=>"completed","evidence"=>Dict("events"=>[Dict("validated"=>true)]),"followups"=>[section]))
        jobs=campaign_nexus_jobs(output,state,1)
        @test length(jobs)==25
        @test count(j->j["campaign_role"]=="nexus_section",jobs)==1
        @test count(j->j["campaign_role"]=="geometry_probe",jobs)==24
        @test all(j->j["transverse_coordinate"]!="r",filter(j->j["campaign_role"]=="geometry_probe",jobs))
        @test all(j->j["transverse_coordinate"]!="A",filter(j->j["campaign_role"]=="geometry_probe",jobs))
    end
end

@testset "Result evidence is bound to canonical job and disposition" begin
    mktempdir() do output
        state=Dict{String,Any}("jobs"=>Dict{String,Any}[])
        job=register_atlas_job!(state,Dict("kind"=>"discovery","queue"=>"discovery"))
        job["status"]="completed"
        @test_throws ArgumentError campaign_verify_result(output,job)
        save_result(output,job,Dict("status"=>"completed","evidence"=>Dict()))
        @test campaign_verify_result(output,job)["status"]=="completed"
        job["status"]="unresolved";job["error"]="coordinator failed while committing evidence"
        @test_throws ArgumentError campaign_verify_result(output,job)
        job["status"]="completed"
        result=TOML.parsefile(joinpath(output,job["result_file"]))
        result["executed_job"]["kind"]="periodic";atomic_toml(joinpath(output,job["result_file"]),result)
        job["result_hash"]=bytes2hex(sha256(read(joinpath(output,job["result_file"]))))
        @test_throws ArgumentError campaign_verify_result(output,job)
    end
end

@testset "Component path distances minimize over all eligible endpoints" begin
    mktempdir() do output
        state=Dict{String,Any}("jobs"=>Dict{String,Any}[])
        groups=Any[]
        for (prefix,values) in (("a",[10.,.01,20.,30.,40.]),("b",[-10.,.011,-20.,-30.,-40.]))
            points=[fixture_point(prefix*string(i),value) for (i,value) in enumerate(values)]
            push!(groups,points)
            payload=Dict("kind"=>"discovery","queue"=>"discovery","campaign_role"=>"inventory","campaign_stage"=>"round1_inventory",
                "campaign_round"=>1,"campaign_shell"=>1,"campaign_chart"=>"parameter","fixture_group"=>prefix)
            job=register_atlas_job!(state,payload)
            save_result(output,job,Dict("status"=>"completed","evidence"=>Dict("points"=>points)))
        end
        for points in groups
            payload=Dict("kind"=>"campaign_path","queue"=>"events","campaign_role"=>"component_path","campaign_stage"=>"round1_component_paths",
                "campaign_round"=>1,"campaign_shell"=>1,"campaign_chart"=>"parameter","from"=>first(points),"to"=>last(points))
            job=register_atlas_job!(state,payload)
            path=Dict("from"=>first(points)["id"],"to"=>last(points)["id"],"sample_ids"=>getindex.(points,"id"),"status"=>"completed",
                "validated"=>true,"signature_preserved"=>true,"forward_tracking_unambiguous"=>true,"audit_agreement"=>true,"tracking_guard_failures"=>String[])
            save_result(output,job,Dict("status"=>"completed","evidence"=>Dict("paths"=>[path])))
        end
        jobs,slots=campaign_path_jobs(output,state,2)
        @test length(jobs)==1
        @test only(jobs)["from"]["id"]=="a2"
        @test only(jobs)["to"]["id"]=="b2"
        @test only(jobs)["component_pair_ids"]==["a1","b1"]
        @test only(filter(slot->slot["shell"]==1&&slot["chart"]=="parameter"&&slot["same_signature"],slots))["selected"]==1
    end
end

@testset "Every reviewed calibration root enters combined storage accounting" begin
    policy=Dict("calibration_sources"=>["/tmp/reviewed-calibration"])
    @test campaign_calibration_roots(Dict(),policy)==policy["calibration_sources"]
    @test campaign_calibration_roots(Dict("campaign_calibration_sources"=>policy["calibration_sources"]),policy)==policy["calibration_sources"]
    @test_throws ArgumentError campaign_calibration_roots(Dict("campaign_calibration_sources"=>["/tmp/other"]),policy)
end
end
