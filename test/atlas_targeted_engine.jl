module TargetedEngineTests
using Test,TOML,SHA,LinearAlgebra
include(joinpath(@__DIR__,"..","scripts","atlas","storage.jl"))
include(joinpath(@__DIR__,"..","scripts","atlas","targeted_protocol.jl"))
include(joinpath(@__DIR__,"..","scripts","atlas","engine.jl"))

targeted_scientific_policy()=Dict("seed_grid"=>9,"balance_atol"=>1e-11)
function build_targeted_jobs(packet,config)
    jobs=Dict{String,Any}[]
    for role in ("control","diagnosis","recovery_refinement","multiroot","transition","audit_extension")
        for index in 1:TARGETED_COUNTS[role]
            push!(jobs,Dict{String,Any}("kind"=>"targeted_mock","queue"=>"events","target_role"=>role,
                "case_index"=>index,"source_job_ids"=>["job-000001"],"config_overrides"=>Dict("seed_grid"=>9)))
        end
    end
    jobs
end

function first_packet(root)
    mkdir(root);mkpath(joinpath(root,"results"))
    config=atlas_config()
    atomic_toml(joinpath(root,"config.toml"),config)
    payload=Dict("kind"=>"source_fixture","queue"=>"events")
    result=Dict("status"=>"completed","evidence"=>Dict("known"=>true),"followups"=>Any[],"executed_job"=>merge(payload,Dict("id"=>"job-000001")))
    relative=joinpath("results","job-000001.toml")
    atomic_toml(joinpath(root,relative),result)
    job=Dict("id"=>"job-000001","status"=>"completed","kind"=>"source_fixture","queue"=>"events","payload"=>payload,"payload_hash"=>toml_digest(payload),"result_file"=>relative,
        "result_hash"=>bytes2hex(sha256(read(joinpath(root,relative)))))
    state=Dict("mode"=>"pilot","status"=>"protocol_complete","pilot"=>Dict("protocol_complete"=>true),
        "config_hash"=>toml_digest(config),"source_hashes"=>Dict{String,String}(),"jobs"=>[job])
    atomic_toml(joinpath(root,"state.toml"),state)
    atomic_toml(joinpath(root,"pilot_summary.toml"),Dict("frozen"=>true))
    root
end

@testset "Frozen review sockets are metadata, never executable inputs" begin
    mktempdir() do parent
        root=first_packet(joinpath(parent,"first"))
        path=joinpath(root,"review-socket")
        ccall(:mknod,Cint,(Cstring,Cuint,Culong),path,Cuint(0o140600),Culong(0))==0 || Base.systemerror("create inert review socket fixture",true)
        packet=targeted_source_packet(root)
        @test only(filter(row->row["path"]=="review-socket",packet["inventory"]))["kind"]=="socket"
        @test packet["retained_bytes"]==storage_owned_bytes(root)
        @test_throws ArgumentError targeted_packet_file(packet,"review-socket")
    end
end

@testset "Frozen first packet and combined storage allowance" begin
    mktempdir() do parent
        root=first_packet(joinpath(parent,"first"))
        external=joinpath(parent,"external")
        write(external,"outside evidence")
        symlink(external,joinpath(root,"review-link"))
        packet=targeted_source_packet(root)
        @test packet["retained_bytes"]==storage_owned_bytes(root)
        @test only(filter(row->row["path"]=="review-link",packet["inventory"]))["kind"]=="symlink"
        @test targeted_read_result(packet,"job-000001")["status"]=="completed"
        @test_throws ArgumentError targeted_packet_file(packet,"review-link")
        @test_throws ArgumentError targeted_packet_file(packet,"../external")
        write(external,"external change must not affect the frozen packet")
        @test isequal(targeted_inventory(root),packet["inventory"])
        prepared,effective=targeted_prepare(targeted_config(),root,16*1024^2)
        @test effective+packet["retained_bytes"]==16*1024^2
        @test length(prepared["targeted_manifest"]["jobs"])==22
        @test prepared["targeted_manifest"]["counts"]==TARGETED_COUNTS
        @test prepared["job_timeout_seconds"]==900
        @test prepared["warmup_seconds"]==600
        @test prepared["pilot_guard_seconds"]==21600
        @test prepared["pilot_max_jobs"]==22
        invalid=targeted_config();invalid["science"]["seed_grid"]=7
        @test_throws ArgumentError targeted_prepare(invalid,root,16*1024^2)
        @test_throws ArgumentError targeted_prepare(targeted_config(),root,packet["retained_bytes"])
        @test_throws ArgumentError run_targeted_pilot(joinpath(root,"mutates-first"),root,targeted_config();combined_storage_bytes=16*1024^2)
        @test !ispath(joinpath(root,"mutates-first"))
        alias=joinpath(parent,"alias");symlink(root,alias)
        @test_throws ArgumentError run_targeted_pilot(joinpath(alias,"mutates-first"),root,targeted_config();combined_storage_bytes=16*1024^2)
        write(joinpath(root,"results","job-000001.toml"),"changed")
        @test_throws ArgumentError targeted_read_result(packet,"job-000001")
        @test_throws ArgumentError targeted_source_packet(root)
    end
end

@testset "Source payload and executed-job lineage" begin
    mktempdir() do parent
        root=first_packet(joinpath(parent,"first"))
        state_path=joinpath(root,"state.toml")
        state=TOML.parsefile(state_path)
        original=deepcopy(state)
        state["jobs"][1]["payload"]["kind"]="changed_without_hash"
        atomic_toml(state_path,state)
        @test_throws ArgumentError targeted_source_packet(root)
        state=deepcopy(original)
        raw_path=joinpath(root,state["jobs"][1]["result_file"])
        raw=TOML.parsefile(raw_path)
        raw["executed_job"]["kind"]="different_executed_job"
        atomic_toml(raw_path,raw)
        state["jobs"][1]["result_hash"]=bytes2hex(sha256(read(raw_path)))
        atomic_toml(state_path,state)
        @test_throws ArgumentError targeted_source_packet(root)
    end
end

@testset "Manifest freeze, readiness barrier, and explicit unknowns" begin
    mktempdir() do parent
        root=first_packet(joinpath(parent,"first"))
        prepared,effective=targeted_prepare(targeted_config(),root,32*1024^2)
        config=merge(prepared,Dict("mode"=>"pilot","storage_bytes"=>effective))
        output=joinpath(parent,"second")
        state=initialize_atlas(output,"pilot",config,10.0)
        @test state["protocol"]==TARGETED_PROTOCOL
        @test length(state["jobs"])==22
        @test verify_targeted_inputs(output,state,config)===nothing
        missing=deepcopy(state);delete!(missing,"protocol")
        @test_throws ArgumentError verify_targeted_inputs(output,missing,config)
        missing_config=deepcopy(config);delete!(missing_config,"protocol")
        @test_throws ArgumentError verify_targeted_inputs(output,state,missing_config)
        save_atlas_state(output,missing)
        @test_throws ArgumentError report_atlas(output)
        @test_throws ArgumentError validate_atlas_resume(output)
        save_atlas_state(output,state)
        stripped_state=deepcopy(state);stripped_config=deepcopy(config)
        for key in ("protocol","targeted_manifest_hash","targeted")
            pop!(stripped_state,key,nothing)
        end
        for key in ("protocol","targeted_manifest","targeted_manifest_hash")
            pop!(stripped_config,key,nothing)
        end
        path=joinpath(output,"targeted_manifest.toml")
        mv(path,path*".saved")
        try
            @test_throws ArgumentError verify_targeted_inputs(output,stripped_state,stripped_config)
        finally
            mv(path*".saved",path)
        end
        @test targeted_next_job!(output,state,config)["payload"]["target_role"]=="control"
        state["jobs"][1]["status"]="failed"
        @test targeted_next_job!(output,state,config)["id"]==state["jobs"][2]["id"]
        state["jobs"][1]["status"]="pending"
        for job in state["jobs"][1:3]
            result=Dict("status"=>"completed","evidence"=>Dict("controls"=>[Dict("status"=>"passed")]),"followups"=>Any[])
            atomic_toml(joinpath(output,job["result_file"]),result)
            job["status"]="completed"
            job["result_hash"]=bytes2hex(sha256(read(joinpath(output,job["result_file"]))))
        end
        @test targeted_next_job!(output,state,config)["payload"]["target_role"]=="diagnosis"
        state["jobs"][2]["status"]="unresolved"
        @test targeted_next_job!(output,state,config)===nothing
        @test state["targeted"]["stop_reason"]=="control_validation_failed"
        @test !state["targeted"]["protocol_complete"]
        state["jobs"][2]["status"]="completed"
        for job in state["jobs"][4:end]
            job["status"]="unresolved"
        end
        @test targeted_next_job!(output,state,config)===nothing
        @test state["targeted"]["protocol_complete"]
        @test state["targeted"]["controls_ready"]
        save_atlas_state(output,state)
        recovered,saved=validate_atlas_resume(output)
        @test recovered["protocol"]==TARGETED_PROTOCOL
        @test saved["targeted_manifest_hash"]==config["targeted_manifest_hash"]
        # Even internally rehashed metadata cannot change the prescribed job
        # construction while retaining the same first-packet provenance.
        forged_config=deepcopy(config)
        forged=forged_config["targeted_manifest"]
        forged["jobs"][4]["config_overrides"]["seed_grid"]=7
        forged["scientific_policy"]["per_job_config_overrides"][4]["seed_grid"]=7
        forged["job_payload_hashes"]=toml_digest.(forged["jobs"])
        forged_config["targeted_manifest_hash"]=toml_digest(forged)
        forged_state=deepcopy(state)
        forged_state["targeted_manifest_hash"]=forged_config["targeted_manifest_hash"]
        forged_state["config_hash"]=toml_digest(forged_config)
        forged_state["jobs"][4]["payload"]=deepcopy(forged["jobs"][4])
        atomic_toml(joinpath(output,"targeted_manifest.toml"),forged)
        @test_throws ArgumentError verify_targeted_inputs(output,forged_state,forged_config)
        atomic_toml(joinpath(output,"targeted_manifest.toml"),config["targeted_manifest"])
        @test isfile(joinpath(output,"targeted_manifest.toml"))
        frozen=TOML.parsefile(joinpath(output,"targeted_manifest.toml"))
        frozen["jobs"][4]["case_index"]=99
        atomic_toml(joinpath(output,"targeted_manifest.toml"),frozen)
        @test_throws ArgumentError verify_targeted_inputs(output,state,config)
        atomic_toml(joinpath(output,"targeted_manifest.toml"),config["targeted_manifest"])
        state["jobs"][4]["payload"]["case_index"]=100
        @test_throws ArgumentError verify_targeted_inputs(output,state,config)
        state["jobs"][4]["payload"]=deepcopy(config["targeted_manifest"]["jobs"][4])
        write(joinpath(root,"new-file"),"source packet changed")
        @test_throws ArgumentError verify_targeted_inputs(output,state,config)
    end
end
end
