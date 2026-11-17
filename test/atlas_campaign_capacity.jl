module AtlasCampaignCapacityTests
using Test,TOML,SHA,LinearAlgebra,FailureOfInhibition2025
include(joinpath(@__DIR__,"..","scripts","zero_drive_atlas.jl"))
const Atlas=ZeroDriveAtlas
bytes(value)=ncodeunits(sprint(io->TOML.print(io,value;sorted=true)))
parameters()=Dict{String,Any}("A"=>0.,"B"=>0.,"C"=>0.,"D"=>0.,"a"=>0.,"b"=>0.,"h"=>2.,"r"=>1.)
function root_record(index,count=82;unresolved=false)
    state=[0.49999999999999994-index/(2count+2),0.22222222222222221]
    intervals=[Dict("lower"=>0.12345678901234567,"upper"=>1.2345678901234567e20,"probe"=>1.2345678901234567e10,
        "stability"=>unresolved ? "unresolved" : "attracting","spectrum"=>"complex") for _ in 1:4]
    Dict{String,Any}("state"=>state,"residual"=>1.2345678901234567e-14,"stability"=>unresolved ? "Unresolved" : "Attracting",
        "spectrum"=>"ComplexConjugateSpectrum","eigenvalue_real"=>[-1.2345678901234567,-1.2345678901234567],
        "eigenvalue_imag"=>[-1.2345678901234567,1.2345678901234567],"spectral_abscissa"=>-1.2345678901234567,
        "trace"=>-2.4691357802469134,"determinant"=>1.2345678901234567,"ratio_partition"=>Dict("intervals"=>intervals,
            "boundaries"=>[.12345678901234567,1.2345678901234567,12.345678901234567],"hopf_candidate"=>true,"hopf_ratio"=>1.2345678901234567,
            "determinant"=>1.2345678901234567,"completeness"=>"analytic linear classification only"),
        "inhibitory_response_branch"=>"falling","excitatory_chart_branch"=>"middle","inhibitory_self_chart_branch"=>"middle")
end
function full_point(id;count=2,unresolved=false)
    roots=[root_record(index,count;unresolved=unresolved) for index in 1:count]
    signature="Eturn:two/Iturn:two/zero:0/"*join(fill("Attracting:ComplexConjugateSpectrum:falling:middle:middle",count),"|")
    Dict{String,Any}("id"=>id,"parameters"=>parameters(),"signature"=>signature,"roots"=>roots,
        "attempts"=>[Dict("seed"=>[.22222222222222221,.33333333333333331],"candidate"=>[.22222222222222221,.33333333333333331],
            "residual"=>1.2345678901234567e-14,"validation"=>"AdmissibleCandidate","solver_status"=>"ReturnCode.Success","reasons"=>String[]) for _ in 1:81])
end
reference(id="job-000001";key="points",index=1,record_id="point")=Dict{String,Any}("source_job"=>id,"result_file"=>"results/$id.toml", "result_hash"=>repeat("a",64),"evidence_key"=>key,"record_index"=>index,"record_id"=>record_id)
function fake_tracks(p0,p1,initial,grid,origin,config,ids)
    attempts=Dict{String,Any}[];links=Dict{String,Any}[]
    for index in 0:grid-1, (track,root) in enumerate(initial)
        key=div(index*128,grid-1);prior=copy(root["state"])
        attempt=Dict("seed"=>prior,"candidate"=>copy(prior),"residual"=>1.2345678901234567e-14,"validation"=>"AdmissibleCandidate","solver_status"=>"ReturnCode.Success","reasons"=>String[])
        push!(attempts,Dict("origin"=>origin,"track"=>track,"t"=>index/(grid-1),"dyadic_key"=>key,"previous_state"=>prior,
            "forward"=>attempt,"reverse_check"=>deepcopy(attempt),"movement"=>0.,"reverse_returned_previous"=>true,"diagnostics"=>String[],
            "track_clean"=>true,"previous_track_clean"=>true,"from_point_id"=>ids[max(1,key)],"to_point_id"=>ids[key+1]))
        index==0 && continue
        push!(links,Dict("from"=>ids[max(1,key)],"to"=>ids[key+1],"kind"=>"tracked_root_link","origin"=>origin,"track"=>track,"grid_size"=>grid,
            "track_clean"=>true,"validated"=>true,"status"=>"clean_sampled_corrector_link","from_state"=>copy(prior),"to_state"=>copy(prior),"continuous_coverage_certified"=>false))
    end
    Dict{String,Any}("origin"=>origin,"grid_size"=>grid,"initial_track_count"=>length(initial),"attempts"=>attempts,
        "endpoint_clean_tracks"=>[Dict("track"=>index,"state"=>copy(root["state"])) for (index,root) in enumerate(initial)],"adjacency_links"=>links,"branch_identity_certified"=>false)
end
function fake_point(p,c;id,extra_seeds=[])
    point=full_point(id);point["parameters"]=deepcopy(p);point
end

@testset "Scientific input projections preserve exact identities and root order" begin
    original=full_point("original");original["roots"]=reverse(original["roots"]);before=deepcopy(original)
    compact=Atlas.campaign_point_input(original;reference=reference())
    @test original==before
    @test compact["id"]==original["id"] && compact["parameters"]==original["parameters"] && compact["signature"]==original["signature"]
    @test getindex.(compact["roots"],"state")==getindex.(original["roots"],"state")
    @test all(Set(keys(root))==Set(["state"]) for root in compact["roots"])
    @test !haskey(compact,"attempts") && compact["source_evidence_reference"]["result_hash"]==repeat("a",64)
    @test bytes(Dict("point"=>compact))<bytes(Dict("point"=>original))÷5
    validation=Atlas.campaign_point_input(original;reference=reference(),retain_root_records=true)
    @test validation["roots"]==original["roots"]
end

@testset "Audit, paths, recovery, and report frontiers preserve adapter semantics" begin
    p=parameters();left=full_point("left");right=full_point("right")
    # Reverse root order in both the saved input and deterministic inventories.
    left["roots"]=reverse(left["roots"]);right["roots"]=reverse(right["roots"])
    reversed_point(p,c;id,extra_seeds=[])=begin point=fake_point(p,c;id=id);point["roots"]=reverse(point["roots"]);point end
    original_audit=Dict("id"=>"audit","kind"=>"campaign_audit","params"=>p,"training_point"=>left)
    compact_audit=merge(original_audit,Dict("training_point"=>Atlas.campaign_point_input(left;reference=reference())))
    @test Atlas.execute_campaign_audit(original_audit,Dict();point_solver=reversed_point)==Atlas.execute_campaign_audit(compact_audit,Dict();point_solver=reversed_point)
    original_path=Dict("id"=>"path","kind"=>"campaign_path","from"=>left,"to"=>right)
    compact_path=merge(original_path,Dict("from"=>Atlas.campaign_point_input(left;reference=reference()),"to"=>Atlas.campaign_point_input(right;reference=reference())))
    @test Atlas.execute_campaign_path(original_path,Dict();point_solver=reversed_point,track_solver=fake_tracks)==Atlas.execute_campaign_path(compact_path,Dict();point_solver=reversed_point,track_solver=fake_tracks)
    base=merge(left,Dict("point_id"=>"left","members"=>["left","other"]));validation=full_point("validation";unresolved=true)
    original_recovery=Dict("id"=>"recovery","kind"=>"campaign_recovery","base"=>base,"target_parameters"=>p,"validation_inventory_status"=>"completed",
        "validation_evidence"=>Dict("points"=>[validation],"validations"=>[Dict("diagnostic"=>"not consumed by recovery")]))
    compact_recovery=merge(original_recovery,Dict("base"=>Atlas.campaign_point_input(base;reference=reference()),
        "validation_evidence"=>Dict("points"=>[Atlas.campaign_point_input(validation;reference=reference(),retain_root_records=true)])))
    old=Atlas.execute_campaign_recovery(original_recovery,Dict();point_solver=reversed_point,track_solver=fake_tracks)
    new=Atlas.execute_campaign_recovery(compact_recovery,Dict();point_solver=reversed_point,track_solver=fake_tracks)
    @test old==new
    job=Dict("id"=>"recovery","status"=>new["status"],"payload"=>Dict("campaign_role"=>"validation_recovery"))
    @test Atlas._campaign_unresolved_diagnostics(job,old)==Atlas._campaign_unresolved_diagnostics(job,new)
    @test !isempty(Atlas._campaign_unresolved_diagnostics(job,new))
    @test Atlas._campaign_fully_resolved(job,old)==Atlas._campaign_fully_resolved(job,new)
end

mutable struct ByteCounter <: IO
    count::Int
end
Base.unsafe_write(io::ByteCounter,p::Ptr{UInt8},n::UInt)=(io.count=Base.checked_add(io.count,Int(n));n)
Base.write(io::ByteCounter,value::UInt8)=(io.count=Base.checked_add(io.count,1);1)
Base.isopen(::ByteCounter)=true
function serialized_size(value)
    io=ByteCounter(0);TOML.print(io,value;sorted=true);io.count
end

function full_capacity_fixture()
    source_reference=reference("job-999999";record_id=repeat("p",64))
    source_point=full_point(repeat("p",64);count=82)
    compact=Atlas.campaign_point_input(source_point;reference=source_reference)
    scientific=Atlas.campaign_point_input(full_point(repeat("v",64);count=170);reference=source_reference,retain_root_records=true)
    job(id,kind,payload)=Dict{String,Any}("id"=>id,"kind"=>kind,"queue"=>"events","status"=>"completed","elapsed_seconds"=>123.45678901234567,
        "payload_hash"=>repeat("a",64),"payload"=>payload,"result_file"=>"results/$id.toml","result_hash"=>repeat("b",64),"error"=>"")
    payload=Dict{String,Any}("kind"=>"discovery","queue"=>"discovery","params"=>parameters(),"chart_coordinates"=>fill(.12345678901234567,7),
        "campaign_stage"=>"round3_inventory","campaign_role"=>"inventory","campaign_round"=>3,"campaign_stream"=>"shell4/equilibrium/mask15", "campaign_shell"=>4,"campaign_chart"=>"equilibrium","campaign_mask"=>15,"accepted_index"=>1024,"design_position"=>4)
    jobs=Dict{String,Any}[]
    add(kind,count,changes)=foreach(1:count) do _
        p=merge(payload,Dict("kind"=>kind),changes);id="job-"*lpad(string(length(jobs)+1),6,'0');push!(jobs,job(id,kind,p))
    end
    add("replacement",22,Dict("original_replacement_budget"=>repeat("x",45_000))) # Above the measured 904,797-byte original22 total.
    add("discovery",1536,Dict())
    add("campaign_audit",384,Dict("training_point"=>compact,"training_source_job"=>"job-999999","training_status"=>"completed"))
    for kind in ("continuation","nexus","periodic")
        add(kind,48,Dict("state"=>[.22222222222222221,.33333333333333331],"campaign_source_job"=>"job-999999","campaign_proposal_order"=>999))
    end
    section=Dict("state"=>[.22222222222222221,.33333333333333331],"parameter_pair"=>["A","r"],"nexus_type"=>"bt","campaign_source_job"=>"job-999999")
    add("section",48,section)
    add("campaign_geometry_probe",1152,merge(section,Dict("normalized_transverse_offset"=>.005,"transverse_coordinate"=>"b","nexus_evidence"=>Dict("validated"=>true,"source_evidence_reference"=>source_reference))))
    add("campaign_path",96,Dict("from"=>compact,"to"=>compact,"same_signature_pair"=>true,"component_pair_ids"=>[repeat("a",64),repeat("b",64)],"path_samples"=>129))
    add("campaign_validation",256,Dict("validation_no_followups"=>true))
    route=Dict("stream"=>"shell4/equilibrium/mask15","accepted_index"=>1280,"params"=>parameters(),"base_point_id"=>repeat("p",64),"status"=>"route_frozen", "selection_rule"=>"Nearest frozen explicit-component base by parameter-only pilot_distance, then point_id; no validation outcomes")
    base=merge(compact,Dict("point_id"=>repeat("p",64)))
    add("campaign_recovery",256,Dict("base"=>base,"route"=>route,"base_selection_hash"=>repeat("a",64),"target_parameters"=>parameters(),"path_samples"=>129,
        "validation_inventory_status"=>"completed","validation_evidence"=>Dict("points"=>[scientific,scientific])))
    entries=[merge(base,Dict("point_id"=>"point-"*lpad(string(index),58,'0'),"members"=>["point-"*lpad(string(index),58,'0')],"source_job"=>"job-999999")) for index in 1:14880]
    witnesses=[Dict("id"=>repeat("w",64),"kind"=>"local_neighborhood_witness","validated"=>true,"source_job"=>"job-999999","source_evidence_reference"=>source_reference) for _ in 1:144]
    selection=Dict("entries"=>entries,"routes"=>fill(route,256),"event_witnesses"=>witnesses,"frozen_before_validation"=>true,"component_rule"=>"Stable-ID representative per explicit examined signature-preserving sampled-path component; equal signatures alone do not connect points", "completeness"=>"CompletenessNotCertified")
    Dict("jobs"=>jobs,"campaign"=>Dict("base_selection"=>selection,"base_selection_hash"=>repeat("a",64),"bases_frozen"=>true,"protocol_complete"=>true),"metadata_allowance"=>repeat("m",2*1024^2))
end

@testset "Full protocol checkpoint and multiroot result serialization bounds" begin
    fixture=full_capacity_fixture();@test length(fixture["jobs"])==3894
    @test length(fixture["campaign"]["base_selection"]["entries"])==14880
    checkpoint_bytes=serialized_size(fixture)
    println("CAMPAIGN_CAPACITY_CHECKPOINT_BYTES=",checkpoint_bytes)
    p=parameters();base=full_point(repeat("b",64);count=82);base["point_id"]=base["id"]
    validation=[full_point("validation-grid9";count=82),full_point("validation-grid13";count=170)]
    job=Dict("id"=>"job-999999","kind"=>"campaign_recovery","base"=>Atlas.campaign_point_input(base;reference=reference()),"target_parameters"=>p,"validation_inventory_status"=>"completed",
        "validation_evidence"=>Dict("points"=>[Atlas.campaign_point_input(point;reference=reference(),retain_root_records=true) for point in validation]))
    maximum_point(p,c;id,extra_seeds=[])=begin point=full_point(id;count=c["seed_grid"]^2);point["parameters"]=deepcopy(p);point end
    result=Atlas.execute_campaign_recovery(job,Dict();point_solver=maximum_point,track_solver=fake_tracks)
    result["executed_job"]=job
    result_bytes=serialized_size(result)
    println("CAMPAIGN_CAPACITY_RECOVERY_RESULT_BYTES=",result_bytes)
    policy=Atlas.storage_policy(4_489_486_848)
    @test checkpoint_bytes*5<=policy["checkpoint_reserve_bytes"]*4 # at least25percent checkpoint margin
    @test result_bytes*5<=policy["result_reserve_bytes"]*4
    @test policy["result_reserve_bytes"]<=policy["file_limit_bytes"]
end

@testset "Compact inputs retain immutable raw evidence bindings" begin
    mktempdir() do output
        state=Dict{String,Any}("jobs"=>Dict{String,Any}[])
        payload=Dict("kind"=>"discovery","queue"=>"discovery","campaign_role"=>"inventory","campaign_stage"=>"round1_inventory",
            "campaign_round"=>1,"campaign_stream"=>"shell1/parameter/mask00","campaign_shell"=>1,"campaign_chart"=>"parameter","accepted_index"=>257)
        job=Atlas.register_atlas_job!(state,payload);point=full_point("saved-point")
        result=Dict("status"=>"completed","evidence"=>Dict("points"=>[point]),"executed_job"=>merge(payload,Dict("id"=>job["id"])))
        Atlas.atomic_toml(joinpath(output,job["result_file"]),result)
        job["status"]="completed";job["result_hash"]=bytes2hex(sha256(read(joinpath(output,job["result_file"]))))
        records=Atlas.campaign_records(output,state)
        projected=Atlas.campaign_point_input(only(records["points"]))
        ref=projected["source_evidence_reference"]
        @test ref==Dict("source_job"=>job["id"],"result_file"=>job["result_file"],"result_hash"=>job["result_hash"],"evidence_key"=>"points","record_index"=>1,"record_id"=>"saved-point")
        @test TOML.parsefile(joinpath(output,ref["result_file"]))["evidence"][ref["evidence_key"]][ref["record_index"]]["id"]==ref["record_id"]
        @test haskey(TOML.parsefile(joinpath(output,ref["result_file"]))["evidence"]["points"][1],"attempts")
        open(joinpath(output,job["result_file"]),"a") do io;write(io,"\n# changed raw evidence\n");end
        @test_throws ArgumentError Atlas.campaign_records(output,state)
    end
end

@testset "Geometry projection preserves optional nexus identity fallback" begin
    for has_id in (false,true)
        nexus=Dict{String,Any}("validated"=>true,"point_id"=>"unrelated-fallback","attempts"=>repeat("x",1000))
        has_id && (nexus["id"]="original-nexus")
        compact=Dict{String,Any}(key=>deepcopy(nexus[key]) for key in ("validated","id") if haskey(nexus,key))
        compact["source_evidence_reference"]=reference(;key="events")
        original=Dict("id"=>"geometry","kind"=>"campaign_geometry_probe","params"=>parameters(),"parameter_pair"=>["A","r"],
            "transverse_coordinate"=>"h","normalized_transverse_offset"=>.005,"campaign_parent_job"=>"parent-fallback","nexus_evidence"=>nexus)
        @test Atlas.execute_campaign_geometry_probe(original,Dict();point_solver=fake_point)==Atlas.execute_campaign_geometry_probe(merge(original,Dict("nexus_evidence"=>compact)),Dict();point_solver=fake_point)
    end
end

@testset "Completed conditional stages release only their own unused capacity" begin
    manifest=Dict("replacement_jobs"=>Any[])
    state=Dict{String,Any}("jobs"=>Any[],"campaign"=>Dict{String,Any}())
    initial=Atlas.campaign_remaining_kind_bounds(manifest,state)
    state["campaign"]["round1_proposals_registered"]=true
    after_proposals=Atlas.campaign_remaining_kind_bounds(manifest,state)
    @test after_proposals["nexus"]==initial["nexus"]-16
    @test after_proposals["campaign_geometry_probe"]==initial["campaign_geometry_probe"]
    state["campaign"]["round1_nexus_sections_registered"]=true
    push!(state["jobs"],Dict("kind"=>"campaign_geometry_probe","status"=>"pending","payload"=>Dict("campaign_stage"=>"round1_nexus_sections")))
    @test Atlas.campaign_remaining_kind_bounds(manifest,state)["campaign_geometry_probe"]==initial["campaign_geometry_probe"]
    state["jobs"][1]["status"]="completed"
    @test Atlas.campaign_remaining_kind_bounds(manifest,state)["campaign_geometry_probe"]==initial["campaign_geometry_probe"]-384
    for round in 1:3, role in ("proposals","nexus_sections","component_paths")
        state["campaign"]["round$(round)_$(role)_registered"]=true
    end
    boundary=Atlas.campaign_remaining_kind_bounds(manifest,state)
    @test all(boundary[kind]==0 for kind in ("continuation","nexus","periodic","section","campaign_geometry_probe","campaign_path"))
    @test boundary["campaign_validation"]==256 && boundary["campaign_recovery"]==256
end

@testset "Full practical three-root study graph fits the report reservation" begin
    nodes=Dict{String,Any}[];edges=Dict{String,Any}[]
    signature="Eturn:two/Iturn:two/zero:0/"*join(fill("Attracting:ComplexConjugateSpectrum:falling:middle:middle",3),"|")
    for index in 1:(256+96)
        job="job-"*lpad(string(index),6,'0');result="results/$job.toml"
        for sample in 0:130
            id="$job-inventory$sample"
            push!(nodes,Dict("id"=>id,"node_type"=>"inventory","signature"=>signature,"root_count"=>3,"audit_only"=>false,
                "eligible"=>true,"source_job"=>job,"source_result"=>result,"round"=>4))
        end
        # Component-path forward links are nested diagnostic evidence, not
        # top-level graph links. Only recovery publishes these track witnesses.
        index>256 && continue
        for track in 1:3, sample in 0:128
            context="$job-inventory$sample";id="root:$job:base_origin_forward:$track:$context"
            push!(nodes,Dict("id"=>id,"node_type"=>"root_witness","inventory_context"=>context,"inventory_membership_asserted"=>false,
                "origin"=>"base_origin_forward","source_job"=>job,"source_result"=>result))
            sample==0 && continue
            from="root:$job:base_origin_forward:$track:$job-inventory$(sample-1)"
            push!(edges,Dict("id"=>"$job-edge-$track-$sample","from"=>from,"to"=>id,"kind"=>"forward_root_track","sample_count"=>0,
                "source_job"=>job,"source_result"=>result))
        end
    end
    # Remaining bounded point-producing roles plus all12 path-localization
    # midpoints/their paired audits and conservative replacement observations.
    for index in 1:(1536+768+48+48*48+1152+512+96*36+2048)
        push!(nodes,Dict("id"=>"additional-inventory-$index","node_type"=>"inventory","signature"=>signature,"root_count"=>3,"audit_only"=>false,
            "eligible"=>true,"source_job"=>"job-999999","source_result"=>"results/job-999999.toml","round"=>3))
    end
    for index in 1:(96*140+96*64)
        push!(edges,Dict("id"=>"additional-context-$index","from"=>"additional-inventory-1","to"=>"additional-inventory-2","kind"=>"uncertainty",
            "sample_count"=>2,"source_job"=>"job-999999","source_result"=>"results/job-999999.toml"))
    end
    status=Dict("planned"=>3894,"attempted"=>3894,"resolved"=>3894,"completed_with_unresolved"=>0,"pending"=>0)
    summary=Dict("status"=>status,"progress"=>Any[],"graph"=>Dict("nodes"=>nodes,"edges"=>edges),"bases"=>Dict(),"coverage_ledger"=>Any[],
        "round_novelty"=>Any[],"recoveries"=>Any[],"unresolved_frontier"=>Any[],"issues"=>Any[])
    data=(state=Dict("campaign"=>Dict("protocol_complete"=>true)),)
    io=ByteCounter(0);Atlas._campaign_render_html(io,summary,data)
    println("CAMPAIGN_CAPACITY_PRACTICAL_REPORT_BYTES=",io.count)
    policy=Atlas.storage_policy(4_489_486_848)
    @test io.count*5<=policy["report_reserve_bytes"]*4
    @test serialized_size(summary)*5<=policy["report_reserve_bytes"]*4
    @test policy["checkpoint_reserve_bytes"]+policy["report_reserve_bytes"]+2policy["result_reserve_bytes"]+2policy["monitor_headroom_bytes"]<policy["cap_bytes"]
end
end
